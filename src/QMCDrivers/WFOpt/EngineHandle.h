//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2022 QMCPACK developers.
//
// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//                    Leon Otis, leon_otis@berkeley.edu, UC Berkeley
//
// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//////////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_ENGINE_HANDLE_HEADER
#define QMCPLUSPLUS_ENGINE_HANDLE_HEADER

#include "Containers/MinimalContainers/RecordArray.hpp"

#include "QMCDrivers/Optimizers/DescentEngine.h"

#ifdef HAVE_LMY_ENGINE
#include "formic/utils/lmyengine/engine.h"
#endif


namespace qmcplusplus
{
class EngineHandle
{
public:
  using Real          = QMCTraits::RealType;
  using Value         = QMCTraits::ValueType;
  using FullPrecReal  = QMCTraits::FullPrecRealType;
  using FullPrecValue = QMCTraits::FullPrecValueType;

  /** Describe how an optimizer consumes derivatives produced while sampling.
   *
   * The batched cost function uses this contract to avoid retaining an
   * additional samples-by-parameters matrix when an engine consumes every
   * crowd batch immediately.
   */
  struct SamplingRequirements
  {
    bool needs_log_derivatives       = true;
    bool needs_energy_derivatives    = true;
    bool persists_log_derivatives    = true;
    bool persists_energy_derivatives = true;
    bool consumes_batches_online     = false;
  };

  virtual ~EngineHandle() = default;

  /// Return the derivative-computation and retention policy for this engine.
  virtual SamplingRequirements getSamplingRequirements() const { return {}; }

  /** Function for preparing derivative ratio vectors used by optimizer engines
   *
   * \param[in] num_params Number of optimizable parameters
   * \param[in] num_samples Number of samples local to this MPI rank
   * \param[in] num_accumulators Number of concurrently executing crowds
   */
  virtual void prepareSampling(int num_params, int num_samples, int num_accumulators) = 0;
  /** Function for passing derivative ratios to optimizer engines
   *
   * \param[in] energy_list         Vector of local energy values
   * \param[in] dlogpsi_array       Parameter derivatives of log psi
   * \param[in] dhpsioverpsi_array  Parameter derivatives of local energy
   * \param[in] base_sample_index Index of the first sample on this MPI rank
   * \param[in] accumulator_index Stable crowd/accumulator index
   *
   */
  virtual void takeSample(const std::vector<FullPrecReal>& energy_list,
                          const RecordArray<Value>& dlogpsi_array,
                          const RecordArray<Value>& dhpsioverpsi_array,
                          int base_sample_index,
                          int accumulator_index) = 0;
  /** Function for having optimizer engines execute their sample_finish functions
   */
  virtual void finishSampling() = 0;
};

class NullEngineHandle : public EngineHandle
{
public:
  void prepareSampling(int num_params, int num_samples, int num_accumulators) override {}
  void takeSample(const std::vector<FullPrecReal>& energy_list,
                  const RecordArray<Value>& dlogpsi_array,
                  const RecordArray<Value>& dhpsioverpsi_array,
                  int base_sample_index,
                  int accumulator_index) override
  {}
  void finishSampling() override {}
};

class DescentEngineHandle : public EngineHandle
{
private:
  DescentEngine& engine_;
  std::vector<std::vector<FullPrecValue>> der_rat_samp_;
  std::vector<std::vector<FullPrecValue>> le_der_samp_;

public:
  DescentEngineHandle(DescentEngine& engine) : engine_(engine) {}

  /// Retrieve one crowd-local derivative-ratio buffer for testing.
  const std::vector<FullPrecValue>& getVector(int accumulator_index = 0) const
  {
    return der_rat_samp_.at(accumulator_index);
  }

  SamplingRequirements getSamplingRequirements() const override
  {
    return {true, true, false, false, true};
  }

  void prepareSampling(int num_params, int num_samples, int num_accumulators) override
  {
    engine_.prepareStorage(num_accumulators, num_params);
    der_rat_samp_.assign(num_accumulators, std::vector<FullPrecValue>(num_params + 1, 0.0));
    le_der_samp_.assign(num_accumulators, std::vector<FullPrecValue>(num_params + 1, 0.0));
  }

  void takeSample(const std::vector<FullPrecReal>& energy_list,
                  const RecordArray<Value>& dlogpsi_array,
                  const RecordArray<Value>& dhpsioverpsi_array,
                  int base_sample_index,
                  int accumulator_index) override
  {
    std::vector<FullPrecValue>& der_rat_samp = der_rat_samp_.at(accumulator_index);
    std::vector<FullPrecValue>& le_der_samp  = le_der_samp_.at(accumulator_index);
    const int current_batch_size = dlogpsi_array.getNumOfEntries();
    for (int local_index = 0; local_index < current_batch_size; local_index++)
    {
      der_rat_samp[0] = 1.0;
      le_der_samp[0]  = energy_list[local_index];

      int num_params = der_rat_samp.size() - 1;
      for (int j = 0; j < num_params; j++)
      {
        der_rat_samp[j + 1] = static_cast<FullPrecValue>(dlogpsi_array[local_index][j]);
        le_der_samp[j + 1]  = static_cast<FullPrecValue>(dhpsioverpsi_array[local_index][j]) +
            le_der_samp[0] * static_cast<FullPrecValue>(dlogpsi_array[local_index][j]);
      }
      engine_.takeSample(accumulator_index, der_rat_samp, le_der_samp, le_der_samp, 1.0, 1.0);
    }
  }

  void finishSampling() override { engine_.sample_finish(); }
};

class LMYEngineHandle : public EngineHandle
{
#ifdef HAVE_LMY_ENGINE
private:
  cqmc::engine::LMYEngine<Value>& lm_engine_;
  std::vector<std::vector<FullPrecValue>> der_rat_samp_;
  std::vector<std::vector<FullPrecValue>> le_der_samp_;

public:
  LMYEngineHandle(cqmc::engine::LMYEngine<Value>& lmyEngine) : lm_engine_(lmyEngine){};

  void prepareSampling(int num_params, int num_samples, int num_accumulators) override
  {
    der_rat_samp_.assign(num_accumulators, std::vector<FullPrecValue>(num_params + 1, 0.0));
    le_der_samp_.assign(num_accumulators, std::vector<FullPrecValue>(num_params + 1, 0.0));
    if (lm_engine_.getStoringSamples())
      lm_engine_.setUpStorage(num_params, num_samples);
  }
  void takeSample(const std::vector<FullPrecReal>& energy_list,
                  const RecordArray<Value>& dlogpsi_array,
                  const RecordArray<Value>& dhpsioverpsi_array,
                  int base_sample_index,
                  int accumulator_index) override
  {
    std::vector<FullPrecValue>& der_rat_samp = der_rat_samp_.at(accumulator_index);
    std::vector<FullPrecValue>& le_der_samp  = le_der_samp_.at(accumulator_index);
    int current_batch_size = dlogpsi_array.getNumOfEntries();
    for (int local_index = 0; local_index < current_batch_size; local_index++)
    {
      const int sample_index = base_sample_index + local_index;
      der_rat_samp[0]        = 1.0;
      le_der_samp[0]         = energy_list[local_index];

      int num_params = der_rat_samp.size() - 1;
      for (int j = 0; j < num_params; j++)
      {
        der_rat_samp[j + 1] = static_cast<FullPrecValue>(dlogpsi_array[local_index][j]);
        le_der_samp[j + 1]  = static_cast<FullPrecValue>(dhpsioverpsi_array[local_index][j]) +
            le_der_samp[0] * static_cast<FullPrecValue>(dlogpsi_array[local_index][j]);
      }


      if (lm_engine_.getStoringSamples())
        lm_engine_.store_sample(der_rat_samp, le_der_samp, le_der_samp, 1.0, 1.0, sample_index);
      else
        lm_engine_.take_sample(der_rat_samp, le_der_samp, le_der_samp, 1.0, 1.0);
    }
  }
  void finishSampling() override
  {
    if (!lm_engine_.getStoringSamples())
      lm_engine_.sample_finish();
  }
#endif
};


} // namespace qmcplusplus
#endif
