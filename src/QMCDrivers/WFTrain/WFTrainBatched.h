//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_WFTRAIN_BATCHED_H
#define QMCPLUSPLUS_WFTRAIN_BATCHED_H

#include "QMCDrivers/QMCDriverNew.h"
#include "QMCDrivers/VMC/VMCBatchedSampling.h"
#include "QMCDrivers/VMC/VMCDriverInput.h"
#include "QMCDrivers/WFTrain/WFTrainDriverInput.h"

namespace qmcplusplus
{

/** Batched high-parameter wavefunction-training driver scaffold. */
class WFTrainBatched : public QMCDriverNew
{
public:
  WFTrainBatched(const ProjectData& project_data,
                 QMCDriverInput&& qmcdriver_input,
                 UPtr<EstimatorManagerNew>&& estimator_manager,
                 VMCDriverInput&& vmc_input,
                 WFTrainDriverInput&& training_input,
                 WalkerConfigurations& walker_configurations,
                 MCPopulation&& population,
                 const RefVector<RandomBase<FullPrecRealType>>& random_generators,
                 Communicate* communicator);

  WFTrainBatched(const WFTrainBatched&)            = delete;
  WFTrainBatched& operator=(const WFTrainBatched&) = delete;

  /** Validate the provider and establish the native VMC walker population. */
  void process(xmlNodePtr node) override;

  /** Fail closed until the production gradient/update stage is installed. */
  void run() override;

private:
  QMCRunType getRunType() override { return QMCRunType::WF_TRAIN_BATCH; }

  VMCDriverInput vmcdriver_input_;
  WFTrainDriverInput training_input_;
  std::string selected_provider_id_;
};

} // namespace qmcplusplus

#endif // QMCPLUSPLUS_WFTRAIN_BATCHED_H
