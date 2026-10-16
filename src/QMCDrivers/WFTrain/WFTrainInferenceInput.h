//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_WFTRAIN_INFERENCE_INPUT_H
#define QMCPLUSPLUS_WFTRAIN_INFERENCE_INPUT_H

#include "Estimators/EstimatorInputDelegates.h"
#include "QMCDrivers/QMCDriverInput.h"
#include "QMCDrivers/VMC/VMCDriverInput.h"
#include "Message/UniformCommunicateError.h"
#include "OhmmsData/XMLParsingString.h"
#include <limits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <set>
#include <string>

namespace qmcplusplus
{
/** Optional inference envelope; sampling semantics belong to the native VMC inputs. */
struct WFTrainInferenceInput
{
  bool enabled = false;
  bool last = false;
  std::size_t interval = 1000;
  QMCDriverInput sampling;
  VMCDriverInput vmc;

  bool due(std::size_t completed, bool final) const
  {
    return enabled && completed != 0 && (completed % interval == 0 || (last && final));
  }

  void readXML(xmlNodePtr qmc)
  {
    *this = WFTrainInferenceInput{};
    xmlNodePtr inference = nullptr;
    for (auto child = qmc->children; child; child = child->next)
      if (child->type == XML_ELEMENT_NODE && std::string(castXMLCharToChar(child->name)) == "inference")
      {
        if (inference)
          throw UniformCommunicateError("WFTrain inference: duplicate inference section");
        inference = child;
      }
    if (!inference)
      return;
    enabled = true;
    for (auto attr = inference->properties; attr; attr = attr->next)
    {
      const std::string name(castXMLCharToChar(attr->name));
      const std::string value = getXMLAttributeValue(inference, name);
      if (attr->ns)
        throw UniformCommunicateError("WFTrain inference: namespaced attributes are unsupported");
      if (name == "interval")
      {
        interval = 0;
        if (value.empty())
          throw UniformCommunicateError("WFTrain inference: interval must be positive");
        for (char digit : value)
        {
          if (digit < '0' || digit > '9' || interval > (std::numeric_limits<std::size_t>::max() - (digit - '0')) / 10)
            throw UniformCommunicateError("WFTrain inference: invalid interval");
          interval = interval * 10 + digit - '0';
        }
        if (!interval)
          throw UniformCommunicateError("WFTrain inference: interval must be positive");
      }
      else if (name == "last")
      {
        if (value != "yes" && value != "no")
          throw UniformCommunicateError("WFTrain inference: last must be yes or no");
        last = value == "yes";
      }
      else if (name != "move")
        throw UniformCommunicateError("WFTrain inference: unknown attribute " + name);
    }
    // Mirror native input names for strict spelling/duplicate admission only;
    // actual value parsing and all sampling behavior use QMCDriverInput/VMCDriverInput.
    const std::set<std::string> native_names{
        "checkProperties", "checkproperties", "check_properties", "blocks", "steps", "substeps", "sub_steps",
        "warmupsteps", "warmup_steps", "crowds", "crowd_serialize_walkers", "walkers_per_rank", "total_walkers",
        "estimator_period", "samples", "timestep", "time_step", "tau", "spin_mass", "blocks_between_recompute",
        "drift_modifier", "drift_UNR_a", "debug_checks", "measure_imbalance", "usedrift", "use_drift",
        "electrons_per_move", "electron_selection"};
    std::set<std::string> seen;
    for (auto child = inference->children; child; child = child->next)
    {
      if (child->type == XML_COMMENT_NODE)
        continue;
      if (child->type == XML_TEXT_NODE && std::string(castXMLCharToChar(child->content)).find_first_not_of(" \t\n\r") == std::string::npos)
        continue;
      const std::string tag(castXMLCharToChar(child->name));
      if (child->type == XML_ELEMENT_NODE && (tag == "estimators" || tag == "estimator"))
      {
        app_warning() << "WFTrain inference ignores supplemental estimators" << std::endl;
        continue;
      }
      if (child->type != XML_ELEMENT_NODE || tag != "parameter")
        throw UniformCommunicateError("WFTrain inference: expected parameter element");
      const std::string name = getXMLAttributeValue(child, "name");
      if (!native_names.count(name))
        throw UniformCommunicateError("WFTrain inference: unknown sampling parameter " + name);
      std::string canonical = name;
      if (name == "sub_steps") canonical = "substeps";
      if (name == "warmup_steps") canonical = "warmupsteps";
      if (name == "tau" || name == "time_step") canonical = "timestep";
      if (name == "use_drift") canonical = "usedrift";
      if (name == "checkProperties" || name == "checkproperties") canonical = "check_properties";
      if (!seen.insert(canonical).second)
        throw UniformCommunicateError("WFTrain inference: duplicate sampling parameter " + name);
      for (auto attr = child->properties; attr; attr = attr->next)
        if (attr->ns || std::string(castXMLCharToChar(attr->name)) != "name")
          throw UniformCommunicateError("WFTrain inference: parameter accepts only name");
      for (auto content = child->children; content; content = content->next)
        if (content->type != XML_TEXT_NODE && content->type != XML_CDATA_SECTION_NODE && content->type != XML_COMMENT_NODE)
          throw UniformCommunicateError("WFTrain inference: nested sampling parameter");
      xmlChar* raw = xmlNodeGetContent(child);
      std::string text(castXMLCharToChar(raw));
      xmlFree(raw);
      const auto first = text.find_first_not_of(" \t\r\n");
      const auto end = text.find_last_not_of(" \t\r\n");
      text = first == std::string::npos ? "" : text.substr(first, end - first + 1);
      if (canonical == "usedrift")
      {
        if (text != "yes" && text != "no")
          throw UniformCommunicateError("WFTrain inference: usedrift must be yes or no");
      }
      else if (canonical != "drift_modifier" && canonical != "debug_checks" &&
               canonical != "electron_selection" && canonical != "measure_imbalance" &&
               canonical != "crowd_serialize_walkers")
      {
        char* finish = nullptr;
        const double value = std::strtod(text.c_str(), &finish);
        std::uint64_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        if (text.empty() || finish != text.c_str() + text.size() ||
            (bits & UINT64_C(0x7ff0000000000000)) == UINT64_C(0x7ff0000000000000))
          throw UniformCommunicateError("WFTrain inference: invalid numeric sampling parameter " + name);
        const bool real = canonical == "timestep" || canonical == "spin_mass" || canonical == "drift_UNR_a";
        if (!real && (text.find_first_not_of("0123456789") != std::string::npos ||
                      value > std::numeric_limits<int>::max()))
          throw UniformCommunicateError("WFTrain inference: invalid integer sampling parameter " + name);
        if ((canonical == "estimator_period" || canonical == "blocks" || canonical == "substeps" ||
             canonical == "timestep" || canonical == "spin_mass") && value <= 0)
          throw UniformCommunicateError("WFTrain inference: sampling parameter must be positive: " + name);
      }
    }
    // Remove ignored estimators before invoking the native parser: even malformed
    // supplemental definitions must never instantiate or allocate an estimator.
    xmlNodePtr native = xmlCopyNode(inference, 1);
    for (auto child = native->children; child;)
    {
      auto next = child->next;
      const std::string tag(castXMLCharToChar(child->name));
      if (tag == "estimators" || tag == "estimator")
      {
        xmlUnlinkNode(child);
        xmlFreeNode(child);
      }
      child = next;
    }
    try
    {
      sampling.readXML(native);
      vmc.readXML(native);
    }
    catch (...)
    {
      xmlFreeNode(native);
      throw;
    }
    xmlFreeNode(native);
    if (sampling.get_max_blocks() <= 0 || sampling.get_sub_steps() <= 0 ||
        sampling.get_warmup_steps() < 0 || sampling.get_tau() <= 0 ||
        sampling.get_num_crowds() < 0 || sampling.get_walkers_per_rank() < 0 || sampling.get_total_walkers() < 0)
      throw UniformCommunicateError("WFTrain inference: invalid sampling counts or timestep");
  }
};
}
#endif
