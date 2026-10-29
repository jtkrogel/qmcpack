//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_WFTRAIN_GPU_TRAINING_EPOCH_REPLAY_H
#define QMCPLUSPLUS_WFTRAIN_GPU_TRAINING_EPOCH_REPLAY_H

#include "QMCHamiltonians/NonLocalECPotential.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerSpatialLayout.h"
#include "Utilities/RandomGenerator.h"
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <typeinfo>
#include <utility>
#include <vector>

namespace qmcplusplus::wftrain
{
/** Preserve only native pre-EL random/grid state after the MC configurations are fixed.
 * The driver owns all referenced RNGs/potentials and the RefVector container;
 * all three must outlive this pending epoch (pass persistent driver rngs_, never a temporary).
 * Rewind repeats physical preparation, never MC moves. Acceptance follows publication.
 */
class GpuTrainingEpochReplay
{
public:
  using RNG = RandomBase<QMCTraits::FullPrecRealType>;

  /// Prepare independent retry/abort banks before any native EL draw or mixed arithmetic.
  GpuTrainingEpochReplay(const RefVector<RNG>& random_generators,
      const std::vector<NonLocalECPotential*>& potentials) : random_generators_(random_generators)
  {
    retry_rng_.reserve(random_generators.size());
    abort_rng_.reserve(random_generators.size());
    for (RNG& rng : random_generators)
    {
      auto retry = rng.makeClone(), abort = rng.makeClone();
      // The native driver uses StdRandom. Exact concrete clone size accounts for
      // its engine/distribution state without adding a generic RNG storage API.
      if (!retry || !abort || typeid(*retry) != typeid(RandomGenerator) || typeid(*abort) != typeid(RandomGenerator))
        throw std::invalid_argument("Native GPU epoch replay requires the configured StdRandom clone type");
      retry_rng_.push_back(std::move(retry));
      abort_rng_.push_back(std::move(abort));
    }
    quadrature_.reserve(potentials.size());
    for (auto* potential : potentials)
    {
      if (!potential || std::any_of(quadrature_.begin(), quadrature_.end(),
          [&](const auto& entry) { return entry.first == potential; }))
        throw std::invalid_argument("Native GPU epoch replay received a null or duplicate ECP owner");
      auto captured = potential->captureQuadratureState();
      potential->validateQuadratureState(captured);
      quadrature_.emplace_back(potential, std::move(captured));
    }
  }

  /// Restore predecessor state on terminal failure; no restore operation allocates or draws RNG.
  ~GpuTrainingEpochReplay()
  {
    if (!accepted_)
    {
      for (std::size_t i = 0; i < abort_rng_.size(); ++i)
        random_generators_[i].get().swapState(*abort_rng_[i]);
      restoreGrids();
    }
  }

  GpuTrainingEpochReplay(const GpuTrainingEpochReplay&) = delete;
  GpuTrainingEpochReplay& operator=(const GpuTrainingEpochReplay&) = delete;

  /// Rewind all rank-local EL streams exactly once before the collective FP64 replay pass.
  void rewindForFp64Replay()
  {
    if (accepted_ || rewound_)
      throw std::logic_error("Native GPU epoch permits one pending FP64 replay");
    for (std::size_t i = 0; i < retry_rng_.size(); ++i)
      random_generators_[i].get().swapState(*retry_rng_[i]);
    restoreGrids();
    rewound_ = true;
  }

  /// Retain the successful traversal's RNG/grid state only after the native update commits.
  void acceptPublishedEpoch() noexcept { accepted_ = true; }

  /// Charge requested HOST payloads, including both concrete RNG clone banks; allocator headers are excluded.
  std::size_t retainedBytes() const
  {
    const auto add = [](auto a, auto b) {
      return psiformer::spatial_detail::checkedAdd(a, b, "Native GPU replay HOST bytes overflow");
    };
    const auto mul = [](auto a, auto b) {
      return psiformer::spatial_detail::checkedProduct(a, b, "Native GPU replay HOST extent overflow");
    };
    auto bytes = mul(retry_rng_.size() + abort_rng_.size(), sizeof(RandomGenerator));
    bytes = add(bytes, mul(retry_rng_.capacity() + abort_rng_.capacity(), sizeof(std::unique_ptr<RNG>)));
    bytes = add(bytes, mul(quadrature_.capacity(), sizeof(decltype(quadrature_)::value_type)));
    for (const auto& entry : quadrature_)
    {
      bytes = add(bytes, mul(entry.second.knots_per_species.capacity(), sizeof(std::size_t)));
      bytes = add(bytes, mul(entry.second.coordinates.capacity(), sizeof(double)));
    }
    return bytes;
  }

private:
  /// The fixed PP owners/layouts are pinned by the enclosing native iteration.
  void restoreGrids() noexcept
  {
    for (const auto& entry : quadrature_)
      entry.first->restoreQuadratureStateUnchecked(entry.second);
  }

  const RefVector<RNG>& random_generators_;
  std::vector<std::unique_ptr<RNG>> retry_rng_, abort_rng_;
  std::vector<std::pair<NonLocalECPotential*, NonLocalECPotential::QuadratureState>> quadrature_;
  bool rewound_ = false, accepted_ = false;
};
} // namespace qmcplusplus::wftrain
#endif
