//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//
// File developed by: QMCPACK developers
//////////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_BATCH_MEMORY_INPUT_H
#define QMCPLUSPLUS_BATCH_MEMORY_INPUT_H

#include "InputSection.h"
#include "QMCDrivers/BatchExecutionMemory.h"

#include <unordered_set>

namespace qmcplusplus
{

/** Strict parser for the section-local batch execution memory policy. */
class BatchMemoryInput : public InputSection
{
public:
  BatchMemoryInput();
  explicit BatchMemoryInput(xmlNodePtr cur);

  const BatchMemoryPolicy& getPolicy() const noexcept { return policy_; }

protected:
  void setFromStreamCustom(const std::string& element_name,
                           const std::string& name,
                           std::istringstream& value) override;

private:
  BatchMemoryPolicy policy_;
  std::unordered_set<std::string> seen_attributes_;
};

} // namespace qmcplusplus

#endif // QMCPLUSPLUS_BATCH_MEMORY_INPUT_H
