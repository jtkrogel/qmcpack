//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_WFTRAIN_DRIVER_INPUT_H
#define QMCPLUSPLUS_WFTRAIN_DRIVER_INPUT_H

#include "Configuration.h"

#include <cstddef>
#include <string>

namespace qmcplusplus
{

/** Strict, typed input for the high-parameter wavefunction-training driver. */
class WFTrainDriverInput
{
public:
  enum class Objective
  {
    ENERGY
  };

  enum class Optimizer
  {
    ADAM
  };

  /** Parse exactly one nested training section from a qmc driver node. */
  void readXML(xmlNodePtr qmc_node);

  Objective getObjective() const noexcept { return objective_; }
  Optimizer getOptimizer() const noexcept { return optimizer_; }
  std::size_t getParameterChunkSize() const noexcept { return parameter_chunk_size_; }
  const std::string& getCheckpointPath() const noexcept { return checkpoint_path_; }
  const std::string& getRestartPath() const noexcept { return restart_path_; }
  const std::string& getProviderID() const noexcept { return provider_id_; }

private:
  Objective objective_{Objective::ENERGY};
  Optimizer optimizer_{Optimizer::ADAM};
  std::size_t parameter_chunk_size_{65536};
  std::string checkpoint_path_;
  std::string restart_path_;
  std::string provider_id_;
};

} // namespace qmcplusplus

#endif // QMCPLUSPLUS_WFTRAIN_DRIVER_INPUT_H
