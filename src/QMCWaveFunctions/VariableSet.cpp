//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//                    Jeremy McMinnis, jmcminis@gmail.com, University of Illinois at Urbana-Champaign
//                    Mark A. Berrill, berrillma@ornl.gov, Oak Ridge National Laboratory
//
// File created by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////

#include "VariableSet.h"
#include "Host/sysutil.h"
#include "io/hdf/hdf_archive.h"

#include <algorithm>
#include <iomanip>
#include <ios>
#include <sstream>
#include <stdexcept>

using std::setw;

namespace optimize
{
// Move all parallel arrays together and leave the source as a coherent empty set.
VariableSet::VariableSet(VariableSet&& other) noexcept
    : num_active_vars(other.num_active_vars),
      NameAndValue(std::move(other.NameAndValue)),
      ParameterType(std::move(other.ParameterType)),
      NameIndex(std::move(other.NameIndex)),
      Index(std::move(other.Index))
{
  other.clear();
}

// Move-assign all parallel arrays and leave the source ready for reuse.
VariableSet& VariableSet::operator=(VariableSet&& other) noexcept
{
  if (this != &other)
  {
    num_active_vars = other.num_active_vars;
    NameAndValue    = std::move(other.NameAndValue);
    ParameterType   = std::move(other.ParameterType);
    NameIndex       = std::move(other.NameIndex);
    Index           = std::move(other.Index);
    other.clear();
  }
  return *this;
}

// Return the ordered location of a name by probing the compact index table.
VariableSet::size_type VariableSet::findLocation(const std::string& vname) const
{
  if (NameIndex.empty())
    return missing_index_;

  const size_type slot_mask = NameIndex.size() - 1;
  size_type slot            = std::hash<std::string>{}(vname) & slot_mask;
  for (size_type probe_count = 0; probe_count < NameIndex.size(); ++probe_count)
  {
    const size_type location = NameIndex[slot];
    if (location == missing_index_)
      return missing_index_;
    if (NameAndValue[location].first == vname)
      return location;
    slot = (slot + 1) & slot_mask;
  }

  return missing_index_;
}

// Insert one ordered-storage location into an already-sized lookup table.
void VariableSet::indexName(size_type location)
{
  const size_type slot_mask = NameIndex.size() - 1;
  size_type slot            = std::hash<std::string>{}(NameAndValue[location].first) & slot_mask;
  while (NameIndex[slot] != missing_index_)
    slot = (slot + 1) & slot_mask;
  NameIndex[slot] = location;
}

// Maintain a power-of-two table below an approximately 80-percent load.
void VariableSet::ensureNameIndexCapacity(size_type entry_count)
{
  if (entry_count == 0)
    return;

  size_type table_size = NameIndex.empty() ? 8 : NameIndex.size();
  while (entry_count > table_size - table_size / 5)
  {
    if (table_size > std::numeric_limits<size_type>::max() / 2)
      throw std::length_error("VariableSet name index exceeds addressable size");
    table_size *= 2;
  }

  if (table_size == NameIndex.size())
    return;

  NameIndex.assign(table_size, missing_index_);
  for (size_type location = 0; location < NameAndValue.size(); ++location)
    indexName(location);
}

// Insert one value with the historical first-value-wins/sticky-disable rules.
void VariableSet::insertEntry(pair_type&& variable, bool enable, int type)
{
  size_type location = findLocation(variable.first);
  if (location == missing_index_)
  {
    reserve(NameAndValue.size() + 1);
    location = NameAndValue.size();
    Index.push_back(static_cast<int>(location));
    NameAndValue.push_back(std::move(variable));
    ParameterType.push_back(type);
    indexName(location);
  }

  if (!enable)
    Index[location] = -1;
}

// Find a mutable ordered entry through the indexed name lookup.
VariableSet::iterator VariableSet::find(const std::string& vname)
{
  const size_type location = findLocation(vname);
  return location == missing_index_ ? NameAndValue.end() : NameAndValue.begin() + location;
}

// Find a read-only ordered entry through the indexed name lookup.
VariableSet::const_iterator VariableSet::find(const std::string& vname) const
{
  const size_type location = findLocation(vname);
  return location == missing_index_ ? NameAndValue.end() : NameAndValue.begin() + location;
}

// Return a name's global active index, or -1 for absent/inactive entries.
int VariableSet::getIndex(const std::string& vname) const
{
  const size_type location = findLocation(vname);
  return location == missing_index_ ? -1 : Index[location];
}

// Return a name's canonical ordered-storage location.
int VariableSet::getLoc(const std::string& vname) const
{
  const size_type location = findLocation(vname);
  return location == missing_index_ ? -1 : static_cast<int>(location);
}

// Insert one parameter using the legacy duplicate and disable semantics.
void VariableSet::insert(const std::string& vname, real_type value, bool enable, int type)
{
  insertEntry(pair_type(vname, value), enable, type);
}

// Reserve all parallel storage and size the name index for a total entry count.
void VariableSet::reserve(size_type count)
{
  NameAndValue.reserve(count);
  ParameterType.reserve(count);
  Index.reserve(count);
  ensureNameIndexCapacity(count);
}

// Insert a batch sharing one enabled state and parameter category.
void VariableSet::insertBulk(std::vector<pair_type> variables, bool enable, int type)
{
  reserve(NameAndValue.size() + variables.size());
  for (pair_type& variable : variables)
    insertEntry(std::move(variable), enable, type);
}

// Insert a batch with independently specified enabled states and categories.
void VariableSet::insertBulk(std::vector<pair_type> variables,
                             const std::vector<bool>& enabled,
                             const std::vector<int>& types)
{
  if (variables.size() != enabled.size() || variables.size() != types.size())
    throw std::invalid_argument("VariableSet bulk metadata size does not match variable count");

  reserve(NameAndValue.size() + variables.size());
  for (size_type variable_index = 0; variable_index < variables.size(); ++variable_index)
    insertEntry(std::move(variables[variable_index]), enabled[variable_index], types[variable_index]);
}

// Provide map-like value access, inserting a disabled zero value when absent.
VariableSet::real_type& VariableSet::operator[](const std::string& vname)
{
  size_type location = findLocation(vname);
  if (location == missing_index_)
  {
    insertEntry(pair_type(vname, 0), false, OTHER_P);
    location = findLocation(vname);
  }
  return NameAndValue[location].second;
}

// Remove every parallel array and the derived name index.
void VariableSet::clear()
{
  num_active_vars = 0;
  Index.clear();
  NameAndValue.clear();
  ParameterType.clear();
  NameIndex.clear();
}

// Merge ordered entries, updating only values for names already present.
void VariableSet::insertFrom(const VariableSet& input)
{
  reserve(NameAndValue.size() + input.size());
  for (size_type input_index = 0; input_index < input.size(); ++input_index)
  {
    const size_type location = findLocation(input.name(input_index));
    if (location == missing_index_)
    {
      const size_type new_location = NameAndValue.size();
      Index.push_back(input.Index[input_index]);
      NameAndValue.push_back(input.NameAndValue[input_index]);
      ParameterType.push_back(input.ParameterType[input_index]);
      indexName(new_location);
    }
    else
      NameAndValue[location].second = input.NameAndValue[input_index].second;
  }
  num_active_vars = input.num_active_vars;
}

// Recompute dense active indices while preserving disabled entries.
void VariableSet::resetIndex()
{
  num_active_vars = 0;
  for (int& index : Index)
    index = index < 0 ? -1 : num_active_vars++;
}

// Map each local name to its cached index in the selected global set.
void VariableSet::getIndex(const VariableSet& selected)
{
  num_active_vars = 0;
  for (size_type variable_index = 0; variable_index < NameAndValue.size(); ++variable_index)
  {
    Index[variable_index] = selected.getIndex(NameAndValue[variable_index].first);
    if (Index[variable_index] >= 0)
      ++num_active_vars;
  }
}

// Return the selected index corresponding to this set's first parameter.
int VariableSet::findIndexOfFirstParam(const VariableSet& selected) const
{
  return NameAndValue.empty() ? -1 : selected.getIndex(NameAndValue.front().first);
}

// Assign each variable its canonical ordered location as a default index.
void VariableSet::setIndexDefault()
{
  for (size_type variable_index = 0; variable_index < Index.size(); ++variable_index)
    Index[variable_index] = static_cast<int>(variable_index);
}

// Print the complete variable set in its stable canonical order.
void VariableSet::print(std::ostream& os, int leftPadSpaces, bool printHeader) const
{
  const std::string pad_str(leftPadSpaces, ' ');
  int max_name_len = 0;
  if (!NameAndValue.empty())
    max_name_len =
        std::max_element(NameAndValue.begin(), NameAndValue.end(), [](const pair_type& lhs, const pair_type& rhs) {
          return lhs.first.length() < rhs.first.length();
        })->first.length();

  constexpr int max_value_len = 28; // precision plus sign, leading value, period, and exponent
  int max_type_len             = 1;
  constexpr int max_use_len    = 3;
  int max_index_len            = 1;
  if (printHeader)
  {
    max_name_len  = std::max(max_name_len, 4); // size of "Name" header
    max_type_len  = 4;
    max_index_len = 5;
    os << pad_str << setw(max_name_len) << "Name"
       << " " << setw(max_value_len) << "Value"
       << " " << setw(max_type_len) << "Type"
       << " " << setw(max_use_len) << "Use"
       << " " << setw(max_index_len) << "Index" << std::endl;
    os << pad_str << std::setfill('-') << setw(max_name_len) << ""
       << " " << setw(max_value_len) << ""
       << " " << setw(max_type_len) << ""
       << " " << setw(max_use_len) << ""
       << " " << setw(max_index_len) << "" << std::endl;
    os << std::setfill(' ');
  }

  for (size_type variable_index = 0; variable_index < NameAndValue.size(); ++variable_index)
  {
    os << pad_str << setw(max_name_len) << NameAndValue[variable_index].first << " " << std::setprecision(6)
       << std::scientific << setw(max_value_len) << NameAndValue[variable_index].second << " " << setw(max_type_len)
       << ParameterType[variable_index] << " " << std::defaultfloat;

    if (Index[variable_index] < 0)
      os << setw(max_use_len) << "OFF" << std::endl;
    else
      os << setw(max_use_len) << "ON"
         << " " << setw(max_index_len) << Index[variable_index] << std::endl;
  }
}

// Print only bounded edge diagnostics for large parameter collections.
void VariableSet::printSummary(std::ostream& os, int leftPadSpaces, size_type edgeEntries) const
{
  const std::string pad(leftPadSpaces, ' ');
  os << pad << NameAndValue.size() << " stored parameters, " << num_active_vars << " active\n";
  if (NameAndValue.empty() || edgeEntries == 0)
    return;

  const auto print_entry = [&](size_type variable_index) {
    os << pad << "  [" << variable_index << "] " << NameAndValue[variable_index].first << " = "
       << std::setprecision(6) << std::scientific << NameAndValue[variable_index].second << std::defaultfloat
       << " type=" << ParameterType[variable_index];
    if (Index[variable_index] < 0)
      os << " OFF\n";
    else
      os << " ON index=" << Index[variable_index] << '\n';
  };

  const size_type leading_entries = std::min(edgeEntries, NameAndValue.size());
  for (size_type variable_index = 0; variable_index < leading_entries; ++variable_index)
    print_entry(variable_index);

  if (NameAndValue.size() > 2 * edgeEntries)
    os << pad << "  ... " << NameAndValue.size() - 2 * edgeEntries << " parameters omitted ...\n";

  const size_type trailing_start = std::max(leading_entries, NameAndValue.size() - leading_entries);
  for (size_type variable_index = trailing_start; variable_index < NameAndValue.size(); ++variable_index)
    print_entry(variable_index);
}

// Save the stable ordered name/value representation used by VP restart files.
void VariableSet::writeToHDF(const std::string& filename, qmcplusplus::hdf_archive& hout) const
{
  hout.create(filename);

  // File Versioning
  // 1.0.0  Initial file version
  // 1.1.0  Files could have object-specific data from OptimizableObject::read/writeVariationalParameters
  std::vector<int> vp_file_version{1, 1, 0};
  hout.write(vp_file_version, "version");

  std::string timestamp(getDateAndTime("%Y-%m-%d %H:%M:%S %Z"));
  hout.write(timestamp, "timestamp");

  hout.push("name_value_lists");

  std::vector<qmcplusplus::QMCTraits::RealType> param_values;
  std::vector<std::string> param_names;
  param_values.reserve(NameAndValue.size());
  param_names.reserve(NameAndValue.size());
  for (const auto& [name, value] : NameAndValue)
  {
    param_names.push_back(name);
    param_values.push_back(value);
  }

  hout.write(param_names, "parameter_names");
  hout.write(param_values, "parameter_values");
  hout.pop();
}

// Load matching values without changing registration order or metadata.
void VariableSet::readFromHDF(const std::string& filename, qmcplusplus::hdf_archive& hin)
{
  if (!hin.open(filename, H5F_ACC_RDONLY))
  {
    std::ostringstream err_msg;
    err_msg << "Unable to open VP file: " << filename;
    throw std::runtime_error(err_msg.str());
  }

  try
  {
    hin.push("name_value_lists", false);
  }
  catch (std::runtime_error&)
  {
    std::ostringstream err_msg;
    err_msg << "The group name_value_lists in not present in file: " << filename;
    throw std::runtime_error(err_msg.str());
  }

  std::vector<qmcplusplus::QMCTraits::RealType> param_values;
  hin.read(param_values, "parameter_values");

  std::vector<std::string> param_names;
  hin.read(param_names, "parameter_names");

  for (size_type parameter_index = 0; parameter_index < param_names.size(); ++parameter_index)
  {
    // Values that are not already registered are deliberately ignored.
    if (iterator location = find(param_names[parameter_index]); location != end())
      location->second = param_values[parameter_index];
  }

  hin.pop();
}

} // namespace optimize
