//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//                    Jeremy McMinnis, jmcminis@gmail.com, University of Illinois at Urbana-Champaign
//                    Raymond Clay III, j.k.rofling@gmail.com, Lawrence Livermore National Laboratory
//                    Mark A. Berrill, berrillma@ornl.gov, Oak Ridge National Laboratory
//
// File created by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_OPTIMIZE_VARIABLESET_H
#define QMCPLUSPLUS_OPTIMIZE_VARIABLESET_H

#include "config.h"
#include "Configuration.h"

#include <algorithm>
#include <complex>
#include <functional>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace qmcplusplus
{
class hdf_archive;
}

namespace optimize
{
/** Identifies parameter categories that optimizers may handle specially. */
enum
{
  OTHER_P = 0,
  LOGLINEAR_P, // B-spline Jastrows
  LOGLINEAR_K, // K-space Jastrows
  LINEAR_P,    // Multi-determinant coefficients
  SPO_P,       // SPO-set parameters
  BACKFLOW_P   // Backflow parameters
};

/** Ordered collection of named variational parameters and global indices.
 *
 * NameAndValue remains the canonical iteration and serialization order. An
 * open-addressed table stores indices into that vector, providing expected
 * constant-time lookup without duplicating every parameter name. Iterator
 * clients may update values, but changing a parameter name through a mutable
 * iterator is unsupported because names are keys in the lookup table.
 */
struct VariableSet
{
  using real_type = qmcplusplus::QMCTraits::RealType;
  using pair_type = std::pair<std::string, real_type>;
  // Retained as a public alias for source compatibility with existing clients.
  using index_pair_type = std::pair<std::string, int>;
  using iterator        = std::vector<pair_type>::iterator;
  using const_iterator  = std::vector<pair_type>::const_iterator;
  using size_type       = std::vector<pair_type>::size_type;

private:
  static constexpr size_type missing_index_ = std::numeric_limits<size_type>::max();

  /// Number of entries whose global index is active.
  int num_active_vars;

  /// Canonical ordered parameter names and values.
  std::vector<pair_type> NameAndValue;

  /// Parameter category parallel to NameAndValue; names are not duplicated.
  std::vector<int> ParameterType;

  /// Open-addressed slots containing indices into NameAndValue.
  std::vector<size_type> NameIndex;

  /// Return the ordered-storage location for a name, or missing_index_.
  size_type findLocation(const std::string& vname) const;

  /// Resize and rebuild the lookup table for at least the requested entries.
  void ensureNameIndexCapacity(size_type entry_count);

  /// Insert one name's ordered-storage location into the lookup table.
  void indexName(size_type location);

  /// Apply scalar insertion semantics while moving the supplied name/value.
  void insertEntry(pair_type&& variable, bool enable, int type);

public:
  /** Stores the global locator of each named variable.
   *
   * If Index[i] == -1, the named variable is inactive.
   */
  std::vector<int> Index;

  /// Construct an empty variable set.
  VariableSet() : num_active_vars(0) {}

  /// Copy ordered values and the coherent lookup table.
  VariableSet(const VariableSet&) = default;

  /// Move ordered values and their index together.
  VariableSet(VariableSet&& other) noexcept;

  /// Copy all variable storage and lookup state.
  VariableSet& operator=(const VariableSet&) = default;

  /// Move all variable storage and lookup state.
  VariableSet& operator=(VariableSet&& other) noexcept;

  /// Virtual destructor retained for compatibility with derived containers.
  virtual ~VariableSet() = default;

  /// Return whether at least one variable is active.
  bool is_optimizable() const { return num_active_vars > 0; }

  /// Return the number of active variables.
  int size_of_active() const { return num_active_vars; }

  /// Return the first read-only ordered iterator.
  const_iterator begin() const { return NameAndValue.begin(); }

  /// Return the past-the-end read-only ordered iterator.
  const_iterator end() const { return NameAndValue.end(); }

  /// Return the first ordered iterator; parameter names must not be modified.
  iterator begin() { return NameAndValue.begin(); }

  /// Return the past-the-end ordered iterator.
  iterator end() { return NameAndValue.end(); }

  /// Return the number of stored variables, active and inactive.
  size_type size() const { return NameAndValue.size(); }

  /// Return the global locator of the i-th stored variable.
  int where(int i) const { return Index[i]; }

  /** Find a named parameter in expected constant time.
   *
   * Returns end() when the name is absent.
   */
  iterator find(const std::string& vname);

  /** Find a named parameter in a const variable set.
   *
   * Returns end() when the name is absent.
   */
  const_iterator find(const std::string& vname) const;

  /** Return the cached global index for a name, or -1 when absent/inactive. */
  int getIndex(const std::string& vname) const;

  /** Return a name's ordered-storage location independently of active state. */
  int getLoc(const std::string& vname) const;

  /** Insert one parameter while preserving the established duplicate rules.
   *
   * The first insertion establishes value and type. A duplicate leaves both
   * unchanged, although enable=false still disables the existing entry.
   */
  void insert(const std::string& vname, real_type value, bool enable = true, int type = OTHER_P);

  /** Preallocate ordered and indexed storage for the requested total size. */
  void reserve(size_type count);

  /** Insert a batch with common enabled state and parameter type.
   *
   * The input is passed by value so callers can move a prepared vector. The
   * operation has the same duplicate semantics as repeated scalar insertions.
   */
  void insertBulk(std::vector<pair_type> variables, bool enable = true, int type = OTHER_P);

  /** Insert a batch with per-entry enabled states and parameter types.
   *
   * enabled and types must have exactly one entry per supplied variable.
   */
  void insertBulk(std::vector<pair_type> variables,
                  const std::vector<bool>& enabled,
                  const std::vector<int>& types);

  /// Assign one parameter category to all stored variables.
  void setParameterType(int type) { std::fill(ParameterType.begin(), ParameterType.end(), type); }

  /// Copy all parameter categories in canonical order.
  void getParameterTypeList(std::vector<int>& types) const { types = ParameterType; }

  /** Return a named value, inserting a disabled zero value if absent. */
  real_type& operator[](const std::string& vname);

  /// Return the name of the i-th variable.
  const std::string& name(int i) const { return NameAndValue[i].first; }

  /// Return the i-th value.
  real_type operator[](int i) const { return NameAndValue[i].second; }

  /// Return a writable reference to the i-th value.
  real_type& operator[](int i) { return NameAndValue[i].second; }

  /// Return the i-th parameter category.
  int getType(int i) const { return ParameterType[i]; }

  /// Remove all ordered, index, type, and lookup-table data.
  void clear();

  /** Merge another VariableSet in its canonical order.
   *
   * Existing names receive incoming values while retaining destination
   * metadata. New names retain the source index and parameter category.
   */
  void insertFrom(const VariableSet& input);

  /// Assign dense global indices to enabled variables in canonical order.
  void resetIndex();

  /** Map this local set's names to cached indices in a selected global set. */
  void getIndex(const VariableSet& selected);

  /** Return the selected index of this set's first parameter, or -1. */
  int findIndexOfFirstParam(const VariableSet& selected) const;

  /// Set every stored variable's index to its ordered location.
  void setIndexDefault();

  /// Print parameters in canonical order using the established table format.
  void print(std::ostream& os, int leftPadSpaces = 0, bool printHeader = false) const;

  /** Print bounded diagnostics for a potentially large variable set.
   *
   * At most edgeEntries entries from each end are printed, so producing a
   * routine progress report does not scan or emit the full parameter vector.
   */
  void printSummary(std::ostream& os, int leftPadSpaces = 0, size_type edgeEntries = 3) const;

  /// Save variational parameter names and values to an HDF5 file.
  void writeToHDF(const std::string& filename, qmcplusplus::hdf_archive& hout) const;

  /** Load values for already-registered names from an HDF5 file. */
  void readFromHDF(const std::string& filename, qmcplusplus::hdf_archive& hin);
};
} // namespace optimize

#endif
