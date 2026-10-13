//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_NATIVE_LINE_SEARCH_H
#define QMCPLUSPLUS_NATIVE_LINE_SEARCH_H
#include "WFTrainDriverInput.h"
#include "QMCWaveFunctions/Optimization/StructuredParameterProvider.h"
#include <functional>
#include <vector>
class Communicate;
namespace qmcplusplus
{
class ParticleSet;
class TrialWaveFunction;
class QMCHamiltonian;
template<typename T> class RandomBase;
namespace wftrain
{
class DeterminantOverlapObjective;
/// Evaluate a bounded set of isolated inference candidates and return a relative scale.
double selectNativeStep(
    const WFTrainDriverInput& input, Communicate& communicator,
    TrialWaveFunction& prototype, QMCHamiltonian& hamiltonian,
    const std::string& provider_id,
    const std::vector<std::reference_wrapper<ParticleSet>>& configurations,
    const std::vector<std::size_t>& configuration_chain_ids,
    const std::vector<std::reference_wrapper<TrialWaveFunction>>& teachers,
    RandomBase<double>& production_rng, double timestep,
    const StructuredParameterSnapshot& old_parameters,
    const StructuredParameterSnapshot& full_candidate,
    const DeterminantOverlapObjective* overlap, double maximum_alpha,
    const std::function<void(double, double, double, double, double, double, bool)>& report);
}
}
#endif
