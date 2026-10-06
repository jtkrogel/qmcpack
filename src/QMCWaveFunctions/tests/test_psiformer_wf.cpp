//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_psiformer_wf.cpp
 * @brief Selected-parameter QMCPACK integration tests for PsiFormerWF.
 */
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include "Utilities/for_testing/Catch2Approx.h"

#include "Message/Communicate.h"
#include "OhmmsData/Libxml2Doc.h"
#include "Particle/ParticleSet.h"
#include "Particle/VirtualParticleBatch.h"
#include "Particle/VirtualParticleSet.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerDeterminant.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerInitialization.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerWaveFunctionBuilder.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerWF.h"
#include "QMCWaveFunctions/TrialWaveFunction.h"
#include "ResourceCollection.h"
#include "Utilities/RuntimeOptions.h"
#include "io/hdf/hdf_archive.h"
#include "psiformer_test_utils.h"

#include <array>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <future>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace qmcplusplus
{
namespace testing
{
/** Exact clone-local state used to prove that planned scalar guard failures
 * precede any mutation of accepted or proposed PsiFormer state. */
struct PsiFormerScalarStateSnapshot
{
  double current_sign;
  PsiFormerWF::LogValue log_value;
  bool restore_validation_pending;
  bool accepted_value_valid;
  std::uint64_t accepted_configuration_identity;
  std::size_t accepted_parameter_version;
  std::uint64_t accepted_state_requirement;
  std::size_t observed_parameter_version;
  double proposed_sign;
  PsiFormerWF::LogValue proposed_log_value;
  std::uint64_t proposed_configuration_identity;
  std::uint64_t proposed_descriptor_fingerprint;
  std::size_t proposed_parameter_version;
  int proposed_particle;
  std::uint8_t proposal_origin;
  bool has_proposal;
  int update_mode;
  std::size_t bytes_in_wf_buffer;
  const void* accepted_gradient_data;
  const void* accepted_laplacian_data;
  const void* proposed_gradient_data;
  const void* proposed_laplacian_data;
  std::size_t accepted_gradient_capacity;
  std::size_t accepted_laplacian_capacity;
  std::size_t proposed_gradient_capacity;
  std::size_t proposed_laplacian_capacity;
  const ParticleSet* bound_particle_set;
  std::vector<PsiFormerWF::GradType> accepted_gradient;
  std::vector<QMCTraits::ValueType> accepted_laplacian;
  std::vector<PsiFormerWF::GradType> proposed_gradient;
  std::vector<QMCTraits::ValueType> proposed_laplacian;
};

/** Access only the crowd-workspace ownership diagnostic used by this test. */
class TestPsiFormerWF
{
public:
  /// Public mirror of the private reversible scalar Phase-B fault selector.
  enum class PlannedScalarValueFault
  {
    NONE,
    RESULT_OWNER,
    RESULT_GENERATION,
    RESULT_SIZE,
    RESULT_VERSION,
    RESULT_SIGN,
    RESULT_LOG_MAGNITUDE,
    RESULT_RATIO,
    INPUT_FINGERPRINT,
    OUTPUT_IDENTITY,
    WORKSPACE_EVIDENCE,
    PUBLICATION_EVIDENCE
  };

  /// Public mirror of the private prepared scalar-capacity corruption seam.
  enum class PreparedScalarWorkspaceFault
  {
    NONE,
    LOGICAL_CAPACITY,
    TILE_CAPACITY
  };

  /// Identify one malformed caller range for direct typed-preflight coverage.
  enum class PlannedScalarOutputFault
  {
    NULL_STORAGE,
    RANGE_OVERFLOW,
    PUBLICATION_ALIAS,
    COMPONENT_ALIAS,
    PARTICLE_ALIAS,
    VIRTUAL_PARTICLE_ALIAS
  };

  /// Public spelling of each successful planned walker-record classification.
  enum class WalkerBufferClassification
  {
    RESTORABLE,
    VALID_ZERO,
    VALID_STALE
  };

  /// Public mirror of the private reversible walker-buffer fault selector.
  enum class PlannedWalkerBufferFault
  {
    NULL_BACKING,
    NULL_SCALAR,
    SIZE_EXCEEDS_CAPACITY,
    ATTACHED_STORAGE,
    MISALIGNED_BACKING,
    SCALAR_BEFORE_BACKING,
    SCALAR_AFTER_BACKING,
    MISALIGNED_SCALAR,
    MISALIGNED_BULK_CURSOR,
    BULK_CURSOR_BEYOND_DOMAIN,
    SCALAR_CURSOR_BEYOND_DOMAIN,
    BULK_CURSOR_OVERFLOW,
    SCALAR_CURSOR_OVERFLOW,
    TRUNCATED_BULK,
    TRUNCATED_SCALAR,
    ACCEPTED_STORAGE_ALIAS,
    PROPOSED_STORAGE_ALIAS,
    PARTICLE_STORAGE_ALIAS
  };

  /// Public mirror of each private typed walker-buffer preflight.
  enum class PlannedWalkerBufferOperation
  {
    REGISTER,
    READ,
    WRITE
  };

  /// Public mirror of inspection-only mutations injected between both phases.
  enum class PlannedWalkerBufferBetweenPhaseFault
  {
    STORAGE_POINTER,
    BULK_CURSOR,
    RECORD_CONTENT,
    PARTICLE_INPUT,
    PLAN_BINDING,
    LAYOUT_EVIDENCE,
    PREPARED_STORAGE_EVIDENCE
  };

  /// Public mirror of the final public buffer-transaction failure selector.
  enum class PlannedWalkerBufferLateFault
  {
    NONE,
    REGISTER,
    RESTORE,
    REFRESH
  };

  /// Copy the immutable cursor and fingerprint evidence from one typed preflight.
  struct WalkerBufferPreflightInspection
  {
    std::size_t bulk_cursor;
    std::size_t scalar_cursor;
    std::uint64_t storage_fingerprint;
    std::uint64_t input_fingerprint;
  };

  /// Copy only immutable parser observations needed by the public regression.
  struct WalkerBufferInspection
  {
    WalkerBufferClassification classification =
        WalkerBufferClassification::VALID_STALE;
    std::size_t next_bulk_cursor;
    std::size_t next_scalar_cursor;
    std::uint64_t schema;
    std::uint64_t requirement;
    std::uint64_t parameter_version;
    std::uint64_t configuration_identity;
    std::uint64_t storage_fingerprint;
    std::uint64_t input_fingerprint;
    std::uint64_t content_fingerprint;
  };

  /// Report the scalar evaluator workspaces currently owned by one component clone.
  static PsiFormerWorkspaceDiagnostics directWorkspaceDiagnostics(const PsiFormerWF& component)
  {
    return component.directWorkspaceDiagnosticsForTesting();
  }

  static std::array<std::size_t, 2> directKineticWorkspaceOwnership(
      const PsiFormerWF& leader,
      const RefVectorWithLeader<WaveFunctionComponent>& wfc_list)
  {
    return leader.directKineticWorkspaceOwnershipForTesting(wfc_list);
  }

  /// Report optimizer metadata sharing without exposing the implementation type.
  static PsiFormerOptimizationMetadataDiagnostics optimizationMetadataDiagnostics(
      const PsiFormerWF& component)
  {
    return component.optimizationMetadataDiagnosticsForTesting();
  }

  /// Report whether the component currently retains a nonempty participant view.
  static bool hasBatchExecutionPlan(const PsiFormerWF& component)
  {
    return static_cast<bool>(component.batch_execution_plan_);
  }

  /// Install a pending selected proposal without running an unrelated evaluator.
  static void markSelectedProposalPending(PsiFormerWF& component)
  {
    component.proposal_origin_ = PsiFormerWF::ProposalOrigin::MW_SELECTED_FULL_VGL;
    component.has_proposal_    = true;
  }

  /// Install only the visible planned-single marker needed by Phase-A rejection.
  static void markPlannedSingleProposalPending(PsiFormerWF& component,
                                               int particle)
  {
    component.proposed_particle_ = particle;
    component.proposal_origin_ =
        PsiFormerWF::ProposalOrigin::MW_CALC_RATIO_VALUE;
    component.has_proposal_ = true;
  }

  /// Install a scalar proposal whose legacy restore path would visibly clear state.
  static void markScalarProposalPending(PsiFormerWF& component, int particle)
  {
    component.proposed_sign_ = -component.current_sign_;
    component.proposed_log_value_ = component.log_value_ + PsiFormerWF::LogValue(0.125);
    component.proposed_configuration_identity_ = component.accepted_configuration_identity_ ^ 0x9e3779b97f4a7c15ULL;
    component.proposed_descriptor_fingerprint_ = 0x6a09e667f3bcc909ULL;
    component.proposed_parameter_version_ = component.observed_parameter_version_;
    component.proposed_particle_          = particle;
    component.proposal_origin_ = PsiFormerWF::ProposalOrigin::SCALAR_RATIO_VALUE;
    for (std::size_t electron = 0; electron < component.proposed_gradient_.size(); ++electron)
    {
      for (std::size_t dimension = 0; dimension < OHMMS_DIM; ++dimension)
        component.proposed_gradient_[electron][dimension] =
            QMCTraits::ValueType(0.25 * (1 + electron + dimension));
      component.proposed_laplacian_[electron] =
          QMCTraits::ValueType(-0.5 * (1 + electron));
    }
    component.has_proposal_ = true;
  }

  /// Restore the ordinary idle lifecycle after a transition-guard check.
  static void clearProposal(PsiFormerWF& component)
  {
    component.clearProposalState();
  }

  /// Inject or clear the late allocation failure used by the preparation test.
  static void failClonePreparationBeforePublish(PsiFormerWF& component,
                                                bool enabled)
  {
    component.fail_clone_preparation_before_publish_for_testing_ = enabled;
  }

  /// Replace only the scalar prepared-layout evidence for exact predicate tests.
  static void setPreparedWalkerBufferLayout(
      PsiFormerWF& component, const pf::WalkerBufferLayout& layout) noexcept
  {
    component.prepared_walker_buffer_layout_ = layout;
  }

  /// Expose the private typed-operation mapping without exposing its private enum.
  static BatchExecutionRequirements plannedWalkerOperationModes(
      BatchExecutionMode mode) noexcept
  {
    switch (mode)
    {
    case BatchExecutionMode::BUFFER_READ:
      return PsiFormerWF::plannedOperationRequiredModes(
          PsiFormerWF::PlannedOperation::BUFFER_READ);
    case BatchExecutionMode::BUFFER_WRITE:
      return PsiFormerWF::plannedOperationRequiredModes(
          PsiFormerWF::PlannedOperation::BUFFER_WRITE);
    case BatchExecutionMode::PREPARE_GROUP:
      return PsiFormerWF::plannedOperationRequiredModes(
          PsiFormerWF::PlannedOperation::PREPARE_GROUP);
    case BatchExecutionMode::COMPLETE_UPDATES:
      return PsiFormerWF::plannedOperationRequiredModes(
          PsiFormerWF::PlannedOperation::COMPLETE_UPDATES);
    default:
      return {};
    }
  }

  /// Inspect one planned walker record without exposing its nonowning byte ranges.
  static WalkerBufferInspection inspectPlannedWalkerBuffer(
      const PsiFormerWF& component,
      const ParticleSet& particles,
      const PsiFormerWF::WFBufferType& buffer)
  {
    const auto record =
        component.inspectPlannedWalkerBufferForTesting(particles, buffer);
    WalkerBufferClassification classification =
        WalkerBufferClassification::VALID_STALE;
    switch (record.classification)
    {
    case PsiFormerWF::WalkerBufferRecordClassification::RESTORABLE:
      classification = WalkerBufferClassification::RESTORABLE;
      break;
    case PsiFormerWF::WalkerBufferRecordClassification::VALID_ZERO:
      classification = WalkerBufferClassification::VALID_ZERO;
      break;
    case PsiFormerWF::WalkerBufferRecordClassification::VALID_STALE:
      classification = WalkerBufferClassification::VALID_STALE;
      break;
    case PsiFormerWF::WalkerBufferRecordClassification::MALFORMED:
      throw std::logic_error(
          "PsiFormer planned walker parser returned a malformed record");
    default:
      throw std::logic_error(
          "PsiFormer planned walker parser returned an unknown classification");
    }

    return {classification,
            record.next_bulk_cursor,
            record.next_scalar_cursor,
            record.schema,
            record.requirement,
            record.parameter_version,
            record.configuration_identity,
            record.cursor.storage_fingerprint,
            record.cursor.input_fingerprint,
            record.content_fingerprint};
  }

  /// Inspect one private REGISTER, READ, or WRITE preflight without parsing.
  static WalkerBufferPreflightInspection inspectPlannedWalkerBufferPreflight(
      const PsiFormerWF& component,
      const ParticleSet& particles,
      const PsiFormerWF::WFBufferType& buffer,
      PlannedWalkerBufferOperation operation)
  {
    using PrivateOperation = PsiFormerWF::PlannedWalkerBufferOperation;
    PrivateOperation private_operation = PrivateOperation::REGISTER;
    switch (operation)
    {
    case PlannedWalkerBufferOperation::REGISTER:
      private_operation = PrivateOperation::REGISTER;
      break;
    case PlannedWalkerBufferOperation::READ:
      private_operation = PrivateOperation::READ;
      break;
    case PlannedWalkerBufferOperation::WRITE:
      private_operation = PrivateOperation::WRITE;
      break;
    default:
      throw std::logic_error(
          "PsiFormer test received an unknown walker-buffer operation");
    }
    const auto snapshot =
        component.inspectPlannedWalkerBufferPreflightForTesting(
            private_operation, particles, buffer);
    return {snapshot.bulk_cursor, snapshot.scalar_cursor,
            snapshot.storage_fingerprint, snapshot.input_fingerprint};
  }

  /// Inject one reversible mutation inside the actual Phase-A/Phase-B flow.
  static void probePlannedWalkerBufferBetweenPhaseFault(
      PsiFormerWF& component,
      ParticleSet& particles,
      PsiFormerWF::WFBufferType& buffer,
      PlannedWalkerBufferBetweenPhaseFault fault)
  {
    using PrivateFault =
        PsiFormerWF::PlannedWalkerBufferBetweenPhaseFaultForTesting;
    PrivateFault private_fault = PrivateFault::NONE;
    switch (fault)
    {
    case PlannedWalkerBufferBetweenPhaseFault::STORAGE_POINTER:
      private_fault = PrivateFault::STORAGE_POINTER;
      break;
    case PlannedWalkerBufferBetweenPhaseFault::BULK_CURSOR:
      private_fault = PrivateFault::BULK_CURSOR;
      break;
    case PlannedWalkerBufferBetweenPhaseFault::RECORD_CONTENT:
      private_fault = PrivateFault::RECORD_CONTENT;
      break;
    case PlannedWalkerBufferBetweenPhaseFault::PARTICLE_INPUT:
      private_fault = PrivateFault::PARTICLE_INPUT;
      break;
    case PlannedWalkerBufferBetweenPhaseFault::PLAN_BINDING:
      private_fault = PrivateFault::PLAN_BINDING;
      break;
    case PlannedWalkerBufferBetweenPhaseFault::LAYOUT_EVIDENCE:
      private_fault = PrivateFault::LAYOUT_EVIDENCE;
      break;
    case PlannedWalkerBufferBetweenPhaseFault::PREPARED_STORAGE_EVIDENCE:
      private_fault = PrivateFault::PREPARED_STORAGE_EVIDENCE;
      break;
    default:
      throw std::logic_error(
          "PsiFormer test received an unknown between-phase fault");
    }
    component.probePlannedWalkerBufferBetweenPhaseFaultForTesting(
        particles, buffer, private_fault);
  }

  /// Select or clear one true late failure in a public buffer transaction.
  static void setPlannedWalkerBufferLateFault(
      PsiFormerWF& component,
      PlannedWalkerBufferLateFault fault) noexcept
  {
    using PrivateFault = PsiFormerWF::PlannedWalkerBufferLateFaultForTesting;
    PrivateFault private_fault = PrivateFault::NONE;
    switch (fault)
    {
    case PlannedWalkerBufferLateFault::NONE:
      private_fault = PrivateFault::NONE;
      break;
    case PlannedWalkerBufferLateFault::REGISTER:
      private_fault = PrivateFault::REGISTER;
      break;
    case PlannedWalkerBufferLateFault::RESTORE:
      private_fault = PrivateFault::RESTORE;
      break;
    case PlannedWalkerBufferLateFault::REFRESH:
      private_fault = PrivateFault::REFRESH;
      break;
    }
    component.setPlannedWalkerBufferLateFaultForTesting(private_fault);
  }

  /** Replace only restorable accepted-state fields with finite canaries.
   *
   * The fixed backing allocations and all proposal state remain untouched so
   * a subsequent public copyFromBuffer must reconstruct the accepted payload
   * rather than succeeding because the original cache was still resident.
   */
  static void poisonAcceptedStateForRestore(PsiFormerWF& component)
  {
    component.current_sign_ = -1.0;
    component.log_value_ = PsiFormerWF::LogValue(17.25, M_PI);
    component.accepted_value_valid_ = false;
    component.accepted_configuration_identity_ ^=
        UINT64_C(0x9e3779b97f4a7c15);
    component.accepted_parameter_version_ += 7;
    component.accepted_state_requirement_ =
        PsiFormerWF::AcceptedStateRequirement::INVALID;
    for (std::size_t electron = 0;
         electron < component.accepted_gradient_.size(); ++electron)
    {
      for (std::size_t dimension = 0; dimension < OHMMS_DIM; ++dimension)
        component.accepted_gradient_[electron][dimension] =
            QMCTraits::ValueType(40.0 + 3.0 * electron + dimension);
      component.accepted_laplacian_[electron] =
          QMCTraits::ValueType(-60.0 - electron);
    }
  }

  /// Replace only live accepted spatial payloads with unmistakable canaries.
  static void poisonAcceptedSpatialStateForStaleConsume(
      PsiFormerWF& component)
  {
    if (!component.accepted_value_valid_ ||
        component.accepted_state_requirement_ !=
            PsiFormerWF::AcceptedStateRequirement::FULL_SPATIAL)
      throw std::logic_error(
          "PsiFormer stale-consume test requires a live FULL_VGL state");
    for (std::size_t electron = 0;
         electron < component.accepted_gradient_.size(); ++electron)
    {
      for (std::size_t dimension = 0; dimension < OHMMS_DIM; ++dimension)
        component.accepted_gradient_[electron][dimension] =
            QMCTraits::ValueType(-140.0 - 3.0 * electron - dimension);
      component.accepted_laplacian_[electron] =
          QMCTraits::ValueType(260.0 + electron);
    }
  }

  /// Replace one accepted gradient element while retaining all cache metadata.
  static void setAcceptedGradientElement(PsiFormerWF& component,
                                         std::size_t electron,
                                         std::size_t dimension,
                                         QMCTraits::ValueType value)
  {
    if (electron >= component.accepted_gradient_.size() ||
        dimension >= OHMMS_DIM)
      throw std::out_of_range(
          "PsiFormer accepted-gradient test index is out of range");
    component.accepted_gradient_[electron][dimension] = value;
  }

  /// Compare only accepted spatial payloads when stale/zero metadata changes.
  static bool acceptedSpatialStateMatches(
      const PsiFormerWF& component,
      const PsiFormerScalarStateSnapshot& snapshot)
  {
    if (component.accepted_gradient_.size() !=
            snapshot.accepted_gradient.size() ||
        component.accepted_laplacian_.size() !=
            snapshot.accepted_laplacian.size())
      return false;
    for (std::size_t electron = 0;
         electron < snapshot.accepted_gradient.size(); ++electron)
    {
      for (std::size_t dimension = 0; dimension < OHMMS_DIM; ++dimension)
        if (std::memcmp(
                std::addressof(
                    component.accepted_gradient_[electron][dimension]),
                std::addressof(
                    snapshot.accepted_gradient[electron][dimension]),
                sizeof(QMCTraits::ValueType)) != 0)
          return false;
      if (std::memcmp(
              std::addressof(component.accepted_laplacian_[electron]),
              std::addressof(snapshot.accepted_laplacian[electron]),
              sizeof(QMCTraits::ValueType)) != 0)
        return false;
    }
    return true;
  }

  /// Exercise one private structural or alias fault through production checks.
  static void probePlannedWalkerBufferFault(
      const PsiFormerWF& component,
      const ParticleSet& particles,
      const PsiFormerWF::WFBufferType& buffer,
      PlannedWalkerBufferFault fault)
  {
    using PrivateFault = PsiFormerWF::PlannedWalkerBufferFaultForTesting;
    PrivateFault private_fault = PrivateFault::NULL_BACKING;
    switch (fault)
    {
    case PlannedWalkerBufferFault::NULL_BACKING:
      private_fault = PrivateFault::NULL_BACKING;
      break;
    case PlannedWalkerBufferFault::NULL_SCALAR:
      private_fault = PrivateFault::NULL_SCALAR;
      break;
    case PlannedWalkerBufferFault::SIZE_EXCEEDS_CAPACITY:
      private_fault = PrivateFault::SIZE_EXCEEDS_CAPACITY;
      break;
    case PlannedWalkerBufferFault::ATTACHED_STORAGE:
      private_fault = PrivateFault::ATTACHED_STORAGE;
      break;
    case PlannedWalkerBufferFault::MISALIGNED_BACKING:
      private_fault = PrivateFault::MISALIGNED_BACKING;
      break;
    case PlannedWalkerBufferFault::SCALAR_BEFORE_BACKING:
      private_fault = PrivateFault::SCALAR_BEFORE_BACKING;
      break;
    case PlannedWalkerBufferFault::SCALAR_AFTER_BACKING:
      private_fault = PrivateFault::SCALAR_AFTER_BACKING;
      break;
    case PlannedWalkerBufferFault::MISALIGNED_SCALAR:
      private_fault = PrivateFault::MISALIGNED_SCALAR;
      break;
    case PlannedWalkerBufferFault::MISALIGNED_BULK_CURSOR:
      private_fault = PrivateFault::MISALIGNED_BULK_CURSOR;
      break;
    case PlannedWalkerBufferFault::BULK_CURSOR_BEYOND_DOMAIN:
      private_fault = PrivateFault::BULK_CURSOR_BEYOND_DOMAIN;
      break;
    case PlannedWalkerBufferFault::SCALAR_CURSOR_BEYOND_DOMAIN:
      private_fault = PrivateFault::SCALAR_CURSOR_BEYOND_DOMAIN;
      break;
    case PlannedWalkerBufferFault::BULK_CURSOR_OVERFLOW:
      private_fault = PrivateFault::BULK_CURSOR_OVERFLOW;
      break;
    case PlannedWalkerBufferFault::SCALAR_CURSOR_OVERFLOW:
      private_fault = PrivateFault::SCALAR_CURSOR_OVERFLOW;
      break;
    case PlannedWalkerBufferFault::TRUNCATED_BULK:
      private_fault = PrivateFault::TRUNCATED_BULK;
      break;
    case PlannedWalkerBufferFault::TRUNCATED_SCALAR:
      private_fault = PrivateFault::TRUNCATED_SCALAR;
      break;
    case PlannedWalkerBufferFault::ACCEPTED_STORAGE_ALIAS:
      private_fault = PrivateFault::ACCEPTED_STORAGE_ALIAS;
      break;
    case PlannedWalkerBufferFault::PROPOSED_STORAGE_ALIAS:
      private_fault = PrivateFault::PROPOSED_STORAGE_ALIAS;
      break;
    case PlannedWalkerBufferFault::PARTICLE_STORAGE_ALIAS:
      private_fault = PrivateFault::PARTICLE_STORAGE_ALIAS;
      break;
    default:
      throw std::logic_error(
          "PsiFormer walker parser test received an unknown fault");
    }
    component.probePlannedWalkerBufferFaultForTesting(particles, buffer,
                                                       private_fault);
  }

  /// Capture every scalar accepted/proposal field and its fixed backing store.
  static PsiFormerScalarStateSnapshot scalarStateSnapshot(
      const PsiFormerWF& component)
  {
    PsiFormerScalarStateSnapshot snapshot{
        component.current_sign_,
        component.log_value_,
        component.restore_validation_pending_,
        component.accepted_value_valid_,
        component.accepted_configuration_identity_,
        component.accepted_parameter_version_,
        static_cast<std::uint64_t>(component.accepted_state_requirement_),
        component.observed_parameter_version_,
        component.proposed_sign_,
        component.proposed_log_value_,
        component.proposed_configuration_identity_,
        component.proposed_descriptor_fingerprint_,
        component.proposed_parameter_version_,
        component.proposed_particle_,
        static_cast<std::uint8_t>(component.proposal_origin_),
        component.has_proposal_,
        component.UpdateMode,
        component.Bytes_in_WFBuffer,
        component.accepted_gradient_.data(),
        component.accepted_laplacian_.data(),
        component.proposed_gradient_.data(),
        component.proposed_laplacian_.data(),
        component.accepted_gradient_.capacity(),
        component.accepted_laplacian_.capacity(),
        component.proposed_gradient_.capacity(),
        component.proposed_laplacian_.capacity(),
        component.bound_particle_set_,
        {},
        {},
        {},
        {}};
    snapshot.accepted_gradient.resize(component.accepted_gradient_.size());
    snapshot.accepted_laplacian.resize(component.accepted_laplacian_.size());
    snapshot.proposed_gradient.resize(component.proposed_gradient_.size());
    snapshot.proposed_laplacian.resize(component.proposed_laplacian_.size());
    for (std::size_t particle = 0; particle < component.accepted_gradient_.size(); ++particle)
      snapshot.accepted_gradient[particle] = component.accepted_gradient_[particle];
    for (std::size_t particle = 0; particle < component.accepted_laplacian_.size(); ++particle)
      snapshot.accepted_laplacian[particle] = component.accepted_laplacian_[particle];
    for (std::size_t particle = 0; particle < component.proposed_gradient_.size(); ++particle)
      snapshot.proposed_gradient[particle] = component.proposed_gradient_[particle];
    for (std::size_t particle = 0; particle < component.proposed_laplacian_.size(); ++particle)
      snapshot.proposed_laplacian[particle] = component.proposed_laplacian_[particle];
    return snapshot;
  }

  /// Force only the accepted log magnitude to exercise ratio range handling.
  static void setAcceptedLogMagnitude(PsiFormerWF& component, double log_magnitude)
  {
    component.log_value_ = PsiFormerWF::LogValue(
        log_magnitude, std::imag(component.log_value_));
  }

  /// Compare without tolerance: a rejected scalar entry must not change logical state.
  static bool scalarStateMatches(
      const PsiFormerWF& component,
      const PsiFormerScalarStateSnapshot& snapshot)
  {
    const auto scalar_bits_match = [](const auto& actual, const auto& expected) {
      return std::memcmp(std::addressof(actual), std::addressof(expected),
                         sizeof(actual)) == 0;
    };

    if (!scalar_bits_match(component.current_sign_, snapshot.current_sign) ||
        !scalar_bits_match(component.log_value_, snapshot.log_value) ||
        component.restore_validation_pending_ != snapshot.restore_validation_pending ||
        component.accepted_value_valid_ != snapshot.accepted_value_valid ||
        component.accepted_configuration_identity_ != snapshot.accepted_configuration_identity ||
        component.accepted_parameter_version_ != snapshot.accepted_parameter_version ||
        static_cast<std::uint64_t>(component.accepted_state_requirement_) != snapshot.accepted_state_requirement ||
        component.observed_parameter_version_ != snapshot.observed_parameter_version ||
        !scalar_bits_match(component.proposed_sign_, snapshot.proposed_sign) ||
        !scalar_bits_match(component.proposed_log_value_, snapshot.proposed_log_value) ||
        component.proposed_configuration_identity_ != snapshot.proposed_configuration_identity ||
        component.proposed_descriptor_fingerprint_ != snapshot.proposed_descriptor_fingerprint ||
        component.proposed_parameter_version_ != snapshot.proposed_parameter_version ||
        component.proposed_particle_ != snapshot.proposed_particle ||
        static_cast<std::uint8_t>(component.proposal_origin_) != snapshot.proposal_origin ||
        component.has_proposal_ != snapshot.has_proposal ||
        component.UpdateMode != snapshot.update_mode ||
        component.Bytes_in_WFBuffer != snapshot.bytes_in_wf_buffer ||
        component.accepted_gradient_.data() != snapshot.accepted_gradient_data ||
        component.accepted_laplacian_.data() != snapshot.accepted_laplacian_data ||
        component.proposed_gradient_.data() != snapshot.proposed_gradient_data ||
        component.proposed_laplacian_.data() != snapshot.proposed_laplacian_data ||
        component.accepted_gradient_.capacity() != snapshot.accepted_gradient_capacity ||
        component.accepted_laplacian_.capacity() != snapshot.accepted_laplacian_capacity ||
        component.proposed_gradient_.capacity() != snapshot.proposed_gradient_capacity ||
        component.proposed_laplacian_.capacity() != snapshot.proposed_laplacian_capacity ||
        component.bound_particle_set_ != snapshot.bound_particle_set ||
        component.accepted_gradient_.size() != snapshot.accepted_gradient.size() ||
        component.accepted_laplacian_.size() != snapshot.accepted_laplacian.size() ||
        component.proposed_gradient_.size() != snapshot.proposed_gradient.size() ||
        component.proposed_laplacian_.size() != snapshot.proposed_laplacian.size())
      return false;

    const auto gradients_match = [](const auto& actual, const auto& expected) {
      for (std::size_t particle = 0; particle < expected.size(); ++particle)
        for (std::size_t dimension = 0; dimension < OHMMS_DIM; ++dimension)
          if (std::memcmp(std::addressof(actual[particle][dimension]),
                          std::addressof(expected[particle][dimension]),
                          sizeof(actual[particle][dimension])) != 0)
            return false;
      return true;
    };
    if (!gradients_match(component.accepted_gradient_, snapshot.accepted_gradient) ||
        !gradients_match(component.proposed_gradient_, snapshot.proposed_gradient))
      return false;
    for (std::size_t particle = 0; particle < snapshot.accepted_laplacian.size(); ++particle)
      if (!scalar_bits_match(component.accepted_laplacian_[particle],
                             snapshot.accepted_laplacian[particle]))
        return false;
    for (std::size_t particle = 0; particle < snapshot.proposed_laplacian.size(); ++particle)
      if (!scalar_bits_match(component.proposed_laplacian_[particle],
                             snapshot.proposed_laplacian[particle]))
        return false;
    return true;
  }

  /// Report the model-wide proposal counters surrounding a scalar query.
  static std::array<std::size_t, 2> plannedProposalCounts(
      const PsiFormerWF& component) noexcept
  {
    return {component.plannedSingleTransactionCountForTesting(),
            component.plannedSelectedTransactionCountForTesting()};
  }

  /// Add one model-wide single-particle transaction owned by another crowd.
  static bool registerPlannedSingleTransaction(
      const PsiFormerWF& component) noexcept
  {
    return component.tryRegisterPlannedSingleTransaction();
  }

  /// Remove one model-wide single-particle transaction installed by a test.
  static void unregisterPlannedSingleTransaction(
      const PsiFormerWF& component) noexcept
  {
    component.unregisterPlannedSingleTransaction();
  }

  /// Add one model-wide selected-particle transaction owned by another crowd.
  static bool registerPlannedSelectedTransaction(
      const PsiFormerWF& component) noexcept
  {
    return component.tryRegisterPlannedSelectedTransaction();
  }

  /// Remove one model-wide selected-particle transaction installed by a test.
  static void unregisterPlannedSelectedTransaction(
      const PsiFormerWF& component) noexcept
  {
    component.unregisterPlannedSelectedTransaction();
  }

  /// Advance only the authoritative model version, leaving clone caches stale.
  static std::size_t advanceParameterVersion(PsiFormerWF& component)
  {
    return component.advanceParameterVersionForTesting();
  }

  /// Bind malformed reference evidence without invoking system validation.
  static void bindParticleSetForTesting(PsiFormerWF& component,
                                        const ParticleSet& particles)
  {
    component.bound_particle_set_ = &particles;
  }

  /// Toggle the deterministic final scalar-publication failure.
  static void failPlannedScalarValueBeforePublish(PsiFormerWF& component,
                                                  bool enabled)
  {
    component.fail_planned_scalar_value_before_publish_for_testing_ = enabled;
  }

  /// Select one reversible corruption between direct evaluation and Phase B.
  static void setPlannedScalarValueFault(PsiFormerWF& component,
                                         PlannedScalarValueFault fault)
  {
    using PrivateFault = PsiFormerWF::PlannedScalarValueFaultForTesting;
    switch (fault)
    {
    case PlannedScalarValueFault::NONE:
      component.planned_scalar_value_fault_for_testing_ = PrivateFault::NONE;
      break;
    case PlannedScalarValueFault::RESULT_OWNER:
      component.planned_scalar_value_fault_for_testing_ =
          PrivateFault::RESULT_OWNER;
      break;
    case PlannedScalarValueFault::RESULT_GENERATION:
      component.planned_scalar_value_fault_for_testing_ =
          PrivateFault::RESULT_GENERATION;
      break;
    case PlannedScalarValueFault::RESULT_SIZE:
      component.planned_scalar_value_fault_for_testing_ =
          PrivateFault::RESULT_SIZE;
      break;
    case PlannedScalarValueFault::RESULT_VERSION:
      component.planned_scalar_value_fault_for_testing_ =
          PrivateFault::RESULT_VERSION;
      break;
    case PlannedScalarValueFault::RESULT_SIGN:
      component.planned_scalar_value_fault_for_testing_ =
          PrivateFault::RESULT_SIGN;
      break;
    case PlannedScalarValueFault::RESULT_LOG_MAGNITUDE:
      component.planned_scalar_value_fault_for_testing_ =
          PrivateFault::RESULT_LOG_MAGNITUDE;
      break;
    case PlannedScalarValueFault::RESULT_RATIO:
      component.planned_scalar_value_fault_for_testing_ =
          PrivateFault::RESULT_RATIO;
      break;
    case PlannedScalarValueFault::INPUT_FINGERPRINT:
      component.planned_scalar_value_fault_for_testing_ =
          PrivateFault::INPUT_FINGERPRINT;
      break;
    case PlannedScalarValueFault::OUTPUT_IDENTITY:
      component.planned_scalar_value_fault_for_testing_ =
          PrivateFault::OUTPUT_IDENTITY;
      break;
    case PlannedScalarValueFault::WORKSPACE_EVIDENCE:
      component.planned_scalar_value_fault_for_testing_ =
          PrivateFault::WORKSPACE_EVIDENCE;
      break;
    case PlannedScalarValueFault::PUBLICATION_EVIDENCE:
      component.planned_scalar_value_fault_for_testing_ =
          PrivateFault::PUBLICATION_EVIDENCE;
      break;
    }
  }

  /// Exchange opaque batch owners without exposing their implementation type.
  static void swapBatchWorkspaces(PsiFormerWF& first,
                                  PsiFormerWF& second) noexcept
  {
    first.direct_batch_workspace_.swap(second.direct_batch_workspace_);
  }

  /// Corrupt or canonically restore the prepared scalar capacity record.
  static void setPreparedScalarWorkspaceFault(
      PsiFormerWF& component,
      PreparedScalarWorkspaceFault fault)
  {
    using PrivateFault = PsiFormerWF::PreparedScalarWorkspaceFaultForTesting;
    switch (fault)
    {
    case PreparedScalarWorkspaceFault::NONE:
      component.setPreparedScalarWorkspaceFaultForTesting(PrivateFault::NONE);
      break;
    case PreparedScalarWorkspaceFault::LOGICAL_CAPACITY:
      component.setPreparedScalarWorkspaceFaultForTesting(
          PrivateFault::LOGICAL_CAPACITY);
      break;
    case PreparedScalarWorkspaceFault::TILE_CAPACITY:
      component.setPreparedScalarWorkspaceFaultForTesting(
          PrivateFault::TILE_CAPACITY);
      break;
    }
  }

  /// Invoke typed scalar preflight with one otherwise unreachable caller range.
  static void probePlannedScalarOutput(
      PsiFormerWF& component,
      const ParticleSet& reference,
      const VirtualParticleSet* virtual_particles,
      std::vector<QMCTraits::ValueType>& ordinary_output,
      PlannedScalarOutputFault fault)
  {
    const bool is_virtual = virtual_particles != nullptr;
    const std::size_t output_count = is_virtual
        ? static_cast<std::size_t>(virtual_particles->getTotalNum())
        : static_cast<std::size_t>(reference.getTotalNum());
    PsiFormerWF::PlannedScalarValueRequest request{
        is_virtual
            ? PsiFormerWF::PlannedScalarValueOperation::VIRTUAL_PARTICLE_VALUE
            : PsiFormerWF::PlannedScalarValueOperation::ALL_TO_ONE,
        std::addressof(reference), virtual_particles, output_count + 1,
        ordinary_output.data(), output_count, ordinary_output.capacity()};

    switch (fault)
    {
    case PlannedScalarOutputFault::NULL_STORAGE:
      request.output_data = nullptr;
      break;
    case PlannedScalarOutputFault::RANGE_OVERFLOW:
      request.output_capacity =
          std::numeric_limits<std::size_t>::max() /
              sizeof(QMCTraits::ValueType) +
          1;
      break;
    case PlannedScalarOutputFault::PUBLICATION_ALIAS:
      request.output_data = component.scalar_value_publication_.data();
      request.output_capacity = output_count;
      break;
    case PlannedScalarOutputFault::COMPONENT_ALIAS:
      request.output_data = component.accepted_laplacian_.data();
      request.output_capacity = output_count;
      break;
    case PlannedScalarOutputFault::PARTICLE_ALIAS:
      request.output_data = const_cast<QMCTraits::ValueType*>(reference.L.data());
      request.output_capacity = output_count;
      break;
    case PlannedScalarOutputFault::VIRTUAL_PARTICLE_ALIAS:
      request.output_data = reinterpret_cast<QMCTraits::ValueType*>(
          const_cast<ParticleSet::PosType*>(virtual_particles->R.data()));
      request.output_capacity = output_count;
      break;
    }
    static_cast<void>(component.requirePlannedScalarValueOperation(request));
  }

  /// Copy the private publication payload for strict Phase-A atomicity checks.
  static std::vector<QMCTraits::ValueType> scalarValuePublication(
      const PsiFormerWF& component)
  {
    return component.scalar_value_publication_;
  }

  /// Temporarily detach the exact scalar publication allocation.
  static std::vector<QMCTraits::ValueType> takeScalarValuePublication(
      PsiFormerWF& component)
  {
    return std::move(component.scalar_value_publication_);
  }

  /// Install scalar publication storage, including deliberately malformed storage.
  static void restoreScalarValuePublication(
      PsiFormerWF& component,
      std::vector<QMCTraits::ValueType> publication)
  {
    component.scalar_value_publication_ = std::move(publication);
  }

  /// Change only the live publication extent while retaining its allocation.
  static void resizeScalarValuePublication(PsiFormerWF& component,
                                           std::size_t size)
  {
    component.scalar_value_publication_.resize(size);
  }
};
} // namespace testing

namespace
{
using namespace testing::psiformer;
using ValueType = QMCTraits::ValueType;

/// Temporarily select one PsiFormer migration backend for a self-contained test.
class ScopedEnvironmentVariable
{
public:
  ScopedEnvironmentVariable(std::string name, const char* value)
      : name_(std::move(name))
  {
    if (const char* previous = std::getenv(name_.c_str()))
      previous_ = previous;
    if (setenv(name_.c_str(), value, 1) != 0)
      throw std::runtime_error(
          "Unable to set PsiFormer test environment variable");
  }

  ScopedEnvironmentVariable(const ScopedEnvironmentVariable&) = delete;
  ScopedEnvironmentVariable& operator=(const ScopedEnvironmentVariable&) =
      delete;

  ~ScopedEnvironmentVariable()
  {
    if (previous_)
      setenv(name_.c_str(), previous_->c_str(), 1);
    else
      unsetenv(name_.c_str());
  }

private:
  std::string name_;
  std::optional<std::string> previous_;
};

/// Capture all public ParticleSet data and retained allocation identities read by scalar VALUE.
struct ScalarParticleStateSnapshot
{
  bool spinor = false;
  int total_particles = 0;
  int group_count = 0;
  std::vector<int> group_sizes;
  const void* position_data = nullptr;
  const void* soa_position_data = nullptr;
  const void* gradient_data = nullptr;
  const void* laplacian_data = nullptr;
  const void* group_data = nullptr;
  const void* spin_data = nullptr;
  std::array<std::size_t, 6> sizes{};
  std::array<std::size_t, 6> capacities{};
  ParticleSet::ParticlePos positions;
  std::vector<ParticleSet::PosType> soa_positions;
  ParticleSet::ParticleGradient gradients;
  ParticleSet::ParticleLaplacian laplacians;
  ParticleSet::ParticleIndex group_ids;
  ParticleSet::ParticleScalar spins;
  ParticleSet::Index_t active_particle = -1;
  ParticleSet::PosType active_position{};
  ParticleSet::RealType active_spin{};
};

/// Capture VirtualParticleSet provenance in addition to its ParticleSet base state.
struct ScalarVirtualParticleStateSnapshot
{
  ScalarParticleStateSnapshot particles;
  const ParticleSet* reference = nullptr;
  int reference_particle = -1;
  int reference_source_particle = -1;
  bool on_sphere = false;
};

/** Preserve both independent PooledMemory cursors, allocation identities, and
 * exact bytes around one nonmutating planned walker-record inspection. */
struct WalkerBufferStateSnapshot
{
  std::size_t bulk_cursor;
  std::size_t scalar_cursor;
  const char* data;
  const QMCTraits::FullPrecRealType* scalar_data;
  std::size_t size;
  std::size_t capacity;
  std::vector<char> bytes;
};

/// Capture a valid allocated walker buffer without invoking scalar_offset().
WalkerBufferStateSnapshot captureWalkerBufferState(
    const PsiFormerWF::WFBufferType& buffer)
{
  WalkerBufferStateSnapshot snapshot{buffer.current(),
                                     buffer.current_scalar(),
                                     buffer.myData.data(),
                                     buffer.Scalar_ptr,
                                     buffer.myData.size(),
                                     buffer.myData.capacity(),
                                     {}};
  if (snapshot.size != 0)
    snapshot.bytes.assign(snapshot.data, snapshot.data + snapshot.size);
  return snapshot;
}

/// Require that parsing or a rejected parser probe changed no caller storage.
void checkWalkerBufferState(const PsiFormerWF::WFBufferType& buffer,
                            const WalkerBufferStateSnapshot& expected)
{
  CHECK(buffer.current() == expected.bulk_cursor);
  CHECK(buffer.current_scalar() == expected.scalar_cursor);
  CHECK(buffer.myData.data() == expected.data);
  CHECK(buffer.Scalar_ptr == expected.scalar_data);
  CHECK(buffer.myData.size() == expected.size);
  CHECK(buffer.myData.capacity() == expected.capacity);
  REQUIRE(buffer.myData.size() == expected.bytes.size());
  for (std::size_t byte = 0; byte < expected.bytes.size(); ++byte)
    CHECK(buffer.myData[byte] == expected.bytes[byte]);
}

/** Write one persistent uint64 field in the exact two-limb representation used
 * by PsiFormer walker records. */
void setWalkerBufferInteger(PsiFormerWF::WFBufferType& buffer,
                            std::size_t scalar_entry,
                            std::size_t first_limb,
                            std::uint64_t value)
{
  using Scalar = QMCTraits::FullPrecRealType;
  static_assert(std::numeric_limits<Scalar>::digits >= 32);
  REQUIRE(buffer.Scalar_ptr != nullptr);
  buffer.Scalar_ptr[scalar_entry + first_limb] =
      static_cast<Scalar>(value & UINT64_C(0xffffffff));
  buffer.Scalar_ptr[scalar_entry + first_limb + 1] =
      static_cast<Scalar>(value >> 32);
}

/// Compare one floating or complex scalar without normalizing signed zero or NaN payloads.
template<class T>
bool sameScalarBits(const T& actual, const T& expected)
{
  return std::memcmp(std::addressof(actual), std::addressof(expected),
                     sizeof(T)) == 0;
}

/// Snapshot one reference or virtual ParticleSet before a scalar query.
ScalarParticleStateSnapshot captureScalarParticleState(
    const ParticleSet& particles)
{
  ScalarParticleStateSnapshot snapshot;
  const auto& soa = particles.getCoordinates().getAllParticlePos();
  snapshot.spinor = particles.isSpinor();
  snapshot.total_particles = particles.getTotalNum();
  snapshot.group_count = particles.groups();
  if (snapshot.group_count > 0)
  {
    snapshot.group_sizes.resize(static_cast<std::size_t>(snapshot.group_count));
    for (int group = 0; group < snapshot.group_count; ++group)
      snapshot.group_sizes[static_cast<std::size_t>(group)] =
          particles.groupsize(group);
  }
  snapshot.position_data = particles.R.data();
  snapshot.soa_position_data = soa.data();
  snapshot.gradient_data = particles.G.data();
  snapshot.laplacian_data = particles.L.data();
  snapshot.group_data = particles.GroupID.data();
  snapshot.spin_data = particles.spins.data();
  snapshot.sizes = {particles.R.size(), soa.size(), particles.G.size(),
                    particles.L.size(), particles.GroupID.size(),
                    particles.spins.size()};
  snapshot.capacities = {particles.R.capacity(), soa.capacity(),
                         particles.G.capacity(), particles.L.capacity(),
                         particles.GroupID.capacity(),
                         particles.spins.capacity()};
  snapshot.positions = particles.R;
  snapshot.soa_positions.resize(soa.size());
  for (std::size_t particle = 0; particle < soa.size(); ++particle)
    snapshot.soa_positions[particle] = soa[particle];
  snapshot.gradients = particles.G;
  snapshot.laplacians = particles.L;
  snapshot.group_ids = particles.GroupID;
  snapshot.spins = particles.spins;
  snapshot.active_particle = particles.getActivePtcl();
  snapshot.active_position = particles.getActivePos();
  snapshot.active_spin = particles.getActiveSpinVal();
  return snapshot;
}

/// Require exact scalar-query isolation for one ParticleSet.
void checkScalarParticleState(const ParticleSet& particles,
                              const ScalarParticleStateSnapshot& expected)
{
  const auto& soa = particles.getCoordinates().getAllParticlePos();
  CHECK(particles.isSpinor() == expected.spinor);
  CHECK(particles.getTotalNum() == expected.total_particles);
  REQUIRE(particles.groups() == expected.group_count);
  if (expected.group_count <= 0)
    REQUIRE(expected.group_sizes.empty());
  else
    REQUIRE(static_cast<std::size_t>(expected.group_count) ==
            expected.group_sizes.size());
  for (int group = 0; group < expected.group_count; ++group)
    CHECK(particles.groupsize(group) ==
          expected.group_sizes[static_cast<std::size_t>(group)]);
  CHECK(particles.R.data() == expected.position_data);
  CHECK(soa.data() == expected.soa_position_data);
  CHECK(particles.G.data() == expected.gradient_data);
  CHECK(particles.L.data() == expected.laplacian_data);
  CHECK(particles.GroupID.data() == expected.group_data);
  CHECK(particles.spins.data() == expected.spin_data);
  CHECK(std::array<std::size_t, 6>{particles.R.size(), soa.size(),
                                   particles.G.size(), particles.L.size(),
                                   particles.GroupID.size(),
                                   particles.spins.size()} == expected.sizes);
  CHECK(std::array<std::size_t, 6>{particles.R.capacity(), soa.capacity(),
                                   particles.G.capacity(),
                                   particles.L.capacity(),
                                   particles.GroupID.capacity(),
                                   particles.spins.capacity()} ==
        expected.capacities);
  CHECK(particles.getActivePtcl() == expected.active_particle);
  CHECK(sameScalarBits(particles.getActiveSpinVal(), expected.active_spin));
  for (std::size_t dimension = 0; dimension < OHMMS_DIM; ++dimension)
    CHECK(sameScalarBits(particles.getActivePos()[dimension],
                         expected.active_position[dimension]));

  REQUIRE(particles.R.size() == expected.positions.size());
  REQUIRE(soa.size() == expected.soa_positions.size());
  REQUIRE(particles.G.size() == expected.gradients.size());
  REQUIRE(particles.L.size() == expected.laplacians.size());
  REQUIRE(particles.GroupID.size() == expected.group_ids.size());
  REQUIRE(particles.spins.size() == expected.spins.size());
  for (std::size_t particle = 0; particle < particles.R.size(); ++particle)
    for (std::size_t dimension = 0; dimension < OHMMS_DIM; ++dimension)
      CHECK(sameScalarBits(particles.R[particle][dimension],
                           expected.positions[particle][dimension]));

  for (std::size_t particle = 0; particle < soa.size(); ++particle)
    for (std::size_t dimension = 0; dimension < OHMMS_DIM; ++dimension)
      CHECK(sameScalarBits(soa[particle][dimension],
                           expected.soa_positions[particle][dimension]));

  for (std::size_t particle = 0; particle < particles.G.size(); ++particle)
    for (std::size_t dimension = 0; dimension < OHMMS_DIM; ++dimension)
      CHECK(sameScalarBits(particles.G[particle][dimension],
                           expected.gradients[particle][dimension]));

  for (std::size_t particle = 0; particle < particles.L.size(); ++particle)
    CHECK(sameScalarBits(particles.L[particle],
                         expected.laplacians[particle]));

  for (std::size_t particle = 0; particle < particles.GroupID.size(); ++particle)
    CHECK(particles.GroupID[particle] == expected.group_ids[particle]);

  for (std::size_t particle = 0; particle < particles.spins.size(); ++particle)
    CHECK(sameScalarBits(particles.spins[particle],
                         expected.spins[particle]));
}

/// Snapshot a fully initialized VirtualParticleSet and its reference identity.
ScalarVirtualParticleStateSnapshot captureScalarVirtualParticleState(
    const VirtualParticleSet& virtual_particles)
{
  return {captureScalarParticleState(virtual_particles),
          std::addressof(virtual_particles.getRefPS()),
          virtual_particles.refPtcl, virtual_particles.refSourcePtcl,
          virtual_particles.isOnSphere()};
}

/// Require a VirtualParticleSet and its provenance to remain bitwise unchanged.
void checkScalarVirtualParticleState(
    const VirtualParticleSet& virtual_particles,
    const ScalarVirtualParticleStateSnapshot& expected)
{
  checkScalarParticleState(virtual_particles, expected.particles);
  CHECK(std::addressof(virtual_particles.getRefPS()) == expected.reference);
  CHECK(virtual_particles.refPtcl == expected.reference_particle);
  CHECK(virtual_particles.refSourcePtcl ==
        expected.reference_source_particle);
  CHECK(virtual_particles.isOnSphere() == expected.on_sphere);
}

/// Compare a real or complex wavefunction ratio to an independent reference.
void checkScalarRatio(ValueType actual, ValueType expected,
                      double tolerance = 3.0e-9)
{
  CHECK(std::real(actual) ==
        Catch::Approx(std::real(expected)).epsilon(tolerance).margin(tolerance));
  CHECK(std::imag(actual) ==
        Catch::Approx(std::imag(expected)).epsilon(tolerance).margin(tolerance));
}

/// Own a unique scratch directory used only for object-specific VP round trips.
struct ScopedTestDirectory
{
  std::filesystem::path path;

  explicit ScopedTestDirectory(const std::string& label)
  {
    static std::atomic<std::uint64_t> sequence{0};
    path = std::filesystem::temp_directory_path() /
        ("qmcpack_psiformer_" + label + "_" +
         std::to_string(static_cast<long long>(getpid())) + "_" +
         std::to_string(sequence.fetch_add(1, std::memory_order_relaxed)));
    std::filesystem::create_directories(path);
  }

  ~ScopedTestDirectory()
  {
    std::error_code error;
    std::filesystem::remove_all(path, error);
  }
};

/// Construct the electron ParticleSet matching one generated LiH fixture.
ParticleSet makeLiHElectrons(const SimulationCell& simulation_cell,
                             const std::string& system = "lih")
{
  const Geometry geometry = makeGeometry(system);
  ParticleSet electrons(simulation_cell);
  electrons.setName("e");
  electrons.create({static_cast<int>(geometry.nup),
                    static_cast<int>(geometry.electrons.size() / 3 - geometry.nup)});
  SpeciesSet& species = electrons.getSpeciesSet();
  species.addSpecies("u");
  species.addSpecies("d");
  const int mass = species.addAttribute("mass");
  species(mass, 0) = 1.0;
  species(mass, 1) = 1.0;
  electrons.resetGroups();
  for (int electron = 0; electron < electrons.getTotalNum(); ++electron)
    for (int dimension = 0; dimension < 3; ++dimension)
      electrons.R[electron][dimension] = geometry.electrons[3 * electron + dimension];
  electrons.update();
  return electrons;
}

/// Construct source ions whose positions and charges exactly match the generated export.
std::unique_ptr<ParticleSet> makeLiHIons(const SimulationCell& simulation_cell, const std::string& system = "lih")
{
  const Geometry geometry = makeGeometry(system);
  auto ions              = std::make_unique<ParticleSet>(simulation_cell);
  ions->setName("ion0");
  ions->create(std::vector<int>(geometry.charges.size(), 1));
  SpeciesSet& species = ions->getSpeciesSet();
  const int charge     = species.addAttribute("charge");
  for (std::size_t nucleus = 0; nucleus < geometry.charges.size(); ++nucleus)
  {
    species.addSpecies("ion_" + std::to_string(nucleus));
    species(charge, nucleus) = geometry.charges[nucleus];
    for (int dimension = 0; dimension < 3; ++dimension)
      ions->R[nucleus][dimension] = geometry.nuclei[3 * nucleus + dimension];
  }
  ions->resetGroups();
  ions->update();
  return ions;
}

/// Add the gradient of a fixed linear log factor to emulate composition with another component.
std::vector<double> addLinearLogGradient(ParticleSet& electrons)
{
  std::vector<double> extra_gradient(3 * electrons.getTotalNum());
  for (int electron = 0; electron < electrons.getTotalNum(); ++electron)
    for (int dimension = 0; dimension < 3; ++dimension)
    {
      const std::size_t coordinate   = 3 * electron + dimension;
      extra_gradient[coordinate]     = 0.01 * (dimension + 1);
      electrons.G[electron][dimension] += ValueType(extra_gradient[coordinate]);
    }
  return extra_gradient;
}

/// Evaluate the kinetic local energy from the logarithmic gradients and Laplacians in ParticleSet.
double kineticEnergy(const ParticleSet& electrons)
{
  double kinetic = 0.0;
  for (int electron = 0; electron < electrons.getTotalNum(); ++electron)
  {
    double squared_gradient = 0.0;
    for (int dimension = 0; dimension < 3; ++dimension)
    {
      const double real_part = std::real(electrons.G[electron][dimension]);
      const double imag_part = std::imag(electrons.G[electron][dimension]);
      squared_gradient += real_part * real_part + imag_part * imag_part;
    }
    kinetic -= 0.5 * (std::real(electrons.L[electron]) + squared_gradient);
  }
  return kinetic;
}

/// Evaluate the three straight-Coulomb potential terms for the LiH fixture.
double coulombPotential(const ParticleSet& electrons)
{
  const Geometry geometry = makeGeometry("lih");
  auto electronNucleusDistance = [&electrons, &geometry](int electron, std::size_t nucleus) {
    double squared_distance = 0.0;
    for (int dimension = 0; dimension < 3; ++dimension)
    {
      const double displacement =
          electrons.R[electron][dimension] - geometry.nuclei[3 * nucleus + dimension];
      squared_distance += displacement * displacement;
    }
    return std::sqrt(squared_distance);
  };

  double potential = 0.0;
  for (int first = 0; first < electrons.getTotalNum(); ++first)
    for (int second = first + 1; second < electrons.getTotalNum(); ++second)
    {
      double squared_distance = 0.0;
      for (int dimension = 0; dimension < 3; ++dimension)
      {
        const double displacement =
            electrons.R[first][dimension] - electrons.R[second][dimension];
        squared_distance += displacement * displacement;
      }
      potential += 1.0 / std::sqrt(squared_distance);
    }

  for (int electron = 0; electron < electrons.getTotalNum(); ++electron)
    for (std::size_t nucleus = 0; nucleus < geometry.charges.size(); ++nucleus)
      potential -= geometry.charges[nucleus] / electronNucleusDistance(electron, nucleus);

  for (std::size_t first = 0; first < geometry.charges.size(); ++first)
    for (std::size_t second = first + 1; second < geometry.charges.size(); ++second)
    {
      double squared_distance = 0.0;
      for (int dimension = 0; dimension < 3; ++dimension)
      {
        const double displacement =
            geometry.nuclei[3 * first + dimension] - geometry.nuclei[3 * second + dimension];
        squared_distance += displacement * displacement;
      }
      potential += geometry.charges[first] * geometry.charges[second] / std::sqrt(squared_distance);
    }
  return potential;
}

/** Select internally consistent evidence while deliberately overriding only
 * the stage-gating accounting claim.
 *
 * Production PsiFormer binding remains fail closed until later crowd-storage
 * stages are complete.  This test seam exercises the already implemented
 * clone owner without weakening that production gate.
 */
std::shared_ptr<const BatchExecutionPlan> makeClonePreparationTestPlan(
    PsiFormerWF& component,
    const BatchExecutionRequirements& requirements,
    const std::string& participant_id,
    const std::string& profile_id,
    std::size_t value_tile = 2,
    std::size_t ecp_outer_maximum = 0,
    BatchExecutionTargetCoordinate target_coordinate =
        BatchExecutionTargetCoordinate::POS_ONLY,
    const std::string& backend_id = "cpu",
    std::optional<std::size_t> device_id = std::nullopt,
    bool serialized_walkers = false)
{
  BatchExecutionSelectionInput selection;
  selection.requirements                       = requirements;
  selection.topology.initial_walkers_per_crowd = {1};
  selection.topology.reserve_walkers_per_crowd = {1};
  selection.topology.serialized_walkers         = serialized_walkers;
  selection.topology.run_kind = "psiformer-clone-preparation-test";
  selection.topology.backend_id    = backend_id;
  selection.topology.device_id     = device_id;
  selection.particle_count         = 4;
  selection.active_parameter_count = 2;
  selection.target_coordinate      = target_coordinate;
  selection.preference.id          = profile_id;
  selection.preference.preferred = {value_tile, 1, 1, ecp_outer_maximum};
  selection.logical_maximum = component.batchExecutionLogicalMaximum(
      {requirements, selection.topology, selection.particle_count,
       selection.active_parameter_count, selection.parameter_derivative_width,
       selection.target_coordinate});
  selection.logical_maximum.ecp_outer = ecp_outer_maximum;

  return std::make_shared<const BatchExecutionPlan>(
      selectBatchExecutionPlan(
          selection,
          [&component, &participant_id](
              const BatchExecutionPlanningContext& candidate) {
            BatchMemoryContribution fabricated =
                component.estimateBatchExecutionMemory(candidate);
            fabricated.fully_accounted = true;
            return std::vector<BatchMemoryParticipantContribution>{
                {participant_id, std::move(fabricated)}};
          }));
}

/// Keep a selected plan alive while one clone exercises scalar compatibility.
struct PreparedScalarPlan
{
  std::shared_ptr<const BatchExecutionPlan> plan;
  BatchExecutionParticipantPlan participant;
};

/// Select, bind, and prepare the exact clone-local scalar owner used by a test.
PreparedScalarPlan prepareScalarValuePlan(PsiFormerWF& component,
                                          const std::string& participant_id,
                                          const std::string& profile_id,
                                          bool select_ecp_outer = false)
{
  BatchExecutionRequirements requirements;
  component.contributeBatchExecutionRequirements(requirements);
  requirements.require(BatchExecutionMode::SCALAR_VALUE_COMPATIBILITY);
  if (select_ecp_outer)
  {
    requirements.require(BatchExecutionMode::VALUE);
    requirements.require(BatchExecutionMode::ECP_OUTER);
  }
  PreparedScalarPlan prepared;
  prepared.plan = makeClonePreparationTestPlan(
      component, requirements, participant_id, profile_id, 2,
      select_ecp_outer ? 2 : 0);
  prepared.participant =
      makeBatchExecutionParticipantPlan(prepared.plan, participant_id);
  component.bindBatchExecutionPlan(prepared.participant);
  component.prepareBatchExecutionClone(prepared.participant);
  return prepared;
}

/// Require that a scalar call changed no retained clone-local allocation evidence.
void checkScalarWorkspaceStorage(
    const testing::PsiFormerWorkspaceDiagnostics& actual,
    const testing::PsiFormerWorkspaceDiagnostics& expected)
{
  CHECK(actual.owns_value_workspace == expected.owns_value_workspace);
  CHECK(actual.owns_full_spatial_workspace ==
        expected.owns_full_spatial_workspace);
  CHECK(actual.owns_active_spatial_workspace ==
        expected.owns_active_spatial_workspace);
  CHECK(actual.owns_batch_workspace == expected.owns_batch_workspace);
  CHECK(actual.owns_score_workspace == expected.owns_score_workspace);
  CHECK(actual.owns_kinetic_workspace == expected.owns_kinetic_workspace);
  CHECK(actual.has_prepared_clone_plan == expected.has_prepared_clone_plan);
  CHECK(actual.value_bytes == expected.value_bytes);
  CHECK(actual.full_spatial_bytes == expected.full_spatial_bytes);
  CHECK(actual.active_spatial_bytes == expected.active_spatial_bytes);
  CHECK(actual.batch_bytes == expected.batch_bytes);
  CHECK(actual.score_bytes == expected.score_bytes);
  CHECK(actual.kinetic_bytes == expected.kinetic_bytes);
  CHECK(actual.total_log_gradient_bytes ==
        expected.total_log_gradient_bytes);
  CHECK(actual.scalar_value_publication_bytes ==
        expected.scalar_value_publication_bytes);
  CHECK(actual.accepted_spatial_bytes == expected.accepted_spatial_bytes);
  CHECK(actual.proposed_spatial_bytes == expected.proposed_spatial_bytes);
  CHECK(actual.batch_storage_fingerprint ==
        expected.batch_storage_fingerprint);
  CHECK(actual.batch_workspace_identity == expected.batch_workspace_identity);
  CHECK(actual.scalar_value_publication_identity ==
        expected.scalar_value_publication_identity);
  CHECK(actual.scalar_value_publication_size ==
        expected.scalar_value_publication_size);
  CHECK(actual.scalar_value_publication_capacity ==
        expected.scalar_value_publication_capacity);
  CHECK(actual.prepared_scalar_value_compatibility ==
        expected.prepared_scalar_value_compatibility);
  CHECK(actual.prepared_batch_workspace_identity ==
        expected.prepared_batch_workspace_identity);
  CHECK(actual.prepared_batch_storage_fingerprint ==
        expected.prepared_batch_storage_fingerprint);
  CHECK(actual.prepared_batch_bytes == expected.prepared_batch_bytes);
  CHECK(actual.prepared_scalar_value_publication_identity ==
        expected.prepared_scalar_value_publication_identity);
  CHECK(actual.prepared_scalar_value_publication_size ==
        expected.prepared_scalar_value_publication_size);
  CHECK(actual.prepared_scalar_value_publication_capacity ==
        expected.prepared_scalar_value_publication_capacity);
  CHECK(actual.prepared_walker_buffer_layout ==
        expected.prepared_walker_buffer_layout);
  CHECK(actual.accountedBytes() == expected.accountedBytes());
  CHECK(actual.ownedWorkspaceCount() == expected.ownedWorkspaceCount());
}

/// Snapshot every persistent or caller-owned object surrounding one scalar transaction.
struct PlannedScalarTransactionSnapshot
{
  testing::PsiFormerScalarStateSnapshot component;
  testing::PsiFormerWorkspaceDiagnostics workspace;
  std::vector<ValueType> publication;
  ScalarParticleStateSnapshot reference;
  std::optional<ScalarVirtualParticleStateSnapshot> virtual_particles;
  std::size_t parameter_version = 0;
  std::array<std::size_t, 2> proposal_counts{};
  const ValueType* output_data = nullptr;
  std::size_t output_size = 0;
  std::size_t output_capacity = 0;
  std::vector<ValueType> output;
};

/// Capture the complete externally observable boundary for one planned scalar call.
PlannedScalarTransactionSnapshot capturePlannedScalarTransaction(
    const PsiFormerWF& component,
    const ParticleSet& reference,
    const std::vector<ValueType>& output,
    const VirtualParticleSet* virtual_particles = nullptr)
{
  PlannedScalarTransactionSnapshot snapshot{
      testing::TestPsiFormerWF::scalarStateSnapshot(component),
      testing::TestPsiFormerWF::directWorkspaceDiagnostics(component),
      testing::TestPsiFormerWF::scalarValuePublication(component),
      captureScalarParticleState(reference),
      std::nullopt,
      component.parameterVersion(),
      testing::TestPsiFormerWF::plannedProposalCounts(component),
      output.data(),
      output.size(),
      output.capacity(),
      output};
  if (virtual_particles)
    snapshot.virtual_particles =
        captureScalarVirtualParticleState(*virtual_particles);
  return snapshot;
}

/// Check exact state isolation after either a successful scalar query or a rejected one.
void checkPlannedScalarTransaction(
    const PsiFormerWF& component,
    const ParticleSet& reference,
    const std::vector<ValueType>& output,
    const PlannedScalarTransactionSnapshot& expected,
    const VirtualParticleSet* virtual_particles = nullptr,
    bool check_output_contents = true,
    bool check_publication_contents = false)
{
  CHECK(testing::TestPsiFormerWF::scalarStateMatches(component,
                                                      expected.component));
  checkScalarWorkspaceStorage(
      testing::TestPsiFormerWF::directWorkspaceDiagnostics(component),
      expected.workspace);
  checkScalarParticleState(reference, expected.reference);
  CHECK(component.parameterVersion() == expected.parameter_version);
  CHECK(testing::TestPsiFormerWF::plannedProposalCounts(component) ==
        expected.proposal_counts);
  if (check_publication_contents)
  {
    const std::vector<ValueType> publication =
        testing::TestPsiFormerWF::scalarValuePublication(component);
    REQUIRE(publication.size() == expected.publication.size());
    for (std::size_t value = 0; value < publication.size(); ++value)
      CHECK(sameScalarBits(publication[value], expected.publication[value]));
  }
  CHECK(output.data() == expected.output_data);
  CHECK(output.size() == expected.output_size);
  CHECK(output.capacity() == expected.output_capacity);
  if (check_output_contents)
  {
    REQUIRE(output.size() == expected.output.size());
    for (std::size_t value = 0; value < output.size(); ++value)
      CHECK(sameScalarBits(output[value], expected.output[value]));
  }
  if (virtual_particles)
  {
    REQUIRE(expected.virtual_particles.has_value());
    checkScalarVirtualParticleState(*virtual_particles,
                                    *expected.virtual_particles);
  }
  else
    CHECK_FALSE(expected.virtual_particles.has_value());
}

/// Capture the high-level observables and selected derivatives of one component.
struct ComponentSnapshot
{
  double log_value;
  double phase;
  double wavefunction_value;
  double local_energy;
  std::vector<double> gradient;
  std::vector<double> laplacian;
  std::vector<double> log_parameter_derivative;
  std::vector<double> kinetic_parameter_derivative;
};

/// Register a component's selected parameters through the normal QMCPACK mapping path.
OptVariables registerSelectedParameters(PsiFormerWF& component)
{
  OptVariables active;
  component.checkInVariablesExclusive(active);
  active.resetIndex();
  component.checkOutVariables(active);
  return active;
}

/// Build a production-shape PsiFormer directly from XML and QMCPACK ParticleSets.
std::unique_ptr<PsiFormerWF> buildInternalPsiFormer(PsiFormerWaveFunctionBuilder& builder,
                                                   const std::string& name,
                                                   std::uint64_t seed,
                                                   const std::string& system = "all_electron",
                                                   const std::string& selected_indices = "0 514",
                                                   const std::string& feature_policy = "")
{
  std::ostringstream xml;
  xml << "<psiformer name=\"" << name
      << "\" initialization=\"" << psiformer::DEEPQMC_PSIFORMER_V1
      << "\" initialization_seed=\"" << seed
      << "\" source=\"ion0\" system=\"" << system << "\" optimize=\"yes\" "
         "optimize_scope=\"indices\" optimize_indices=\"" << selected_indices << "\"";
  if (!feature_policy.empty())
    xml << " feature_policy=\"" << feature_policy << "\"";
  xml << "/>";

  Libxml2Document document;
  if (!document.parseFromString(xml.str()))
    throw std::runtime_error("Unable to parse internally initialized PsiFormer test XML");
  std::unique_ptr<WaveFunctionComponent> component = builder.buildComponent(document.getRoot());
  auto* psiformer_component = dynamic_cast<PsiFormerWF*>(component.get());
  if (psiformer_component == nullptr)
    throw std::runtime_error("PsiFormer builder returned the wrong component type");
  component.release();
  return std::unique_ptr<PsiFormerWF>(psiformer_component);
}

/// Evaluate the component from scratch and flatten its public QMCPACK outputs.
ComponentSnapshot evaluateComponent(PsiFormerWF& component, ParticleSet& electrons, const OptVariables& active)
{
  electrons.G = ValueType(0);
  electrons.L = ValueType(0);
  const PsiFormerWF::LogValue log_value = component.evaluateLog(electrons, electrons.G, electrons.L);

  Vector<ValueType> dlogpsi(active.size());
  Vector<ValueType> dhpsioverpsi(active.size());
  dlogpsi      = ValueType(0);
  dhpsioverpsi = ValueType(0);
  component.evaluateDerivatives(electrons, active, dlogpsi, dhpsioverpsi);

  ComponentSnapshot snapshot;
  snapshot.log_value          = std::real(log_value);
  snapshot.phase              = std::imag(log_value);
  snapshot.wavefunction_value = std::real(std::exp(log_value));
  snapshot.local_energy       = kineticEnergy(electrons) + coulombPotential(electrons);
  snapshot.gradient.reserve(3 * electrons.getTotalNum());
  snapshot.laplacian.reserve(electrons.getTotalNum());
  snapshot.log_parameter_derivative.reserve(active.size());
  snapshot.kinetic_parameter_derivative.reserve(active.size());
  for (int electron = 0; electron < electrons.getTotalNum(); ++electron)
  {
    for (int dimension = 0; dimension < 3; ++dimension)
      snapshot.gradient.push_back(std::real(electrons.G[electron][dimension]));
    snapshot.laplacian.push_back(std::real(electrons.L[electron]));
  }
  for (int parameter = 0; parameter < active.size(); ++parameter)
  {
    snapshot.log_parameter_derivative.push_back(std::real(dlogpsi[parameter]));
    snapshot.kinetic_parameter_derivative.push_back(std::real(dhpsioverpsi[parameter]));
  }
  return snapshot;
}

/// Compare complete component snapshots at deterministic native-evaluator tolerance.
void checkComponentSnapshot(const ComponentSnapshot& actual, const ComponentSnapshot& expected)
{
  CHECK(actual.log_value == Catch::Approx(expected.log_value).epsilon(2e-10).margin(2e-10));
  CHECK(actual.phase == Catch::Approx(expected.phase).epsilon(2e-10).margin(2e-10));
  CHECK(actual.wavefunction_value ==
        Catch::Approx(expected.wavefunction_value).epsilon(2e-9).margin(1e-24));
  CHECK(actual.local_energy == Catch::Approx(expected.local_energy).epsilon(2e-9).margin(2e-9));
  REQUIRE(actual.gradient.size() == expected.gradient.size());
  REQUIRE(actual.laplacian.size() == expected.laplacian.size());
  REQUIRE(actual.log_parameter_derivative.size() == expected.log_parameter_derivative.size());
  REQUIRE(actual.kinetic_parameter_derivative.size() == expected.kinetic_parameter_derivative.size());

  for (std::size_t index = 0; index < actual.gradient.size(); ++index)
    CHECK(actual.gradient[index] == Catch::Approx(expected.gradient[index]).epsilon(2e-9).margin(2e-9));
  for (std::size_t index = 0; index < actual.laplacian.size(); ++index)
    CHECK(actual.laplacian[index] == Catch::Approx(expected.laplacian[index]).epsilon(2e-8).margin(2e-8));
  for (std::size_t index = 0; index < actual.log_parameter_derivative.size(); ++index)
    CHECK(actual.log_parameter_derivative[index] ==
          Catch::Approx(expected.log_parameter_derivative[index]).epsilon(2e-8).margin(2e-8));
  for (std::size_t index = 0; index < actual.kinetic_parameter_derivative.size(); ++index)
    CHECK(actual.kinetic_parameter_derivative[index] ==
          Catch::Approx(expected.kinetic_parameter_derivative[index]).epsilon(2e-8).margin(2e-8));
}

} // namespace

TEST_CASE("PsiFormer builder is fixed by default and parses selected indices", "[wavefunction][psiformer]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  ParticleSet electrons = makeLiHElectrons(simulation_cell);
  WaveFunctionComponentBuilder::PSetMap particle_sets;
  auto ions = makeLiHIons(simulation_cell);
  particle_sets.emplace(ions->getName(), std::move(ions));
  PsiFormerWaveFunctionBuilder builder(OHMMS::Controller, electrons, particle_sets);

  std::ostringstream fixed_xml;
  fixed_xml << "<psiformer name=\"pf_fixed\" parameters=\"" << files.parameters.string()
            << "\" configuration=\"" << files.configuration.string() << "\"/>";
  Libxml2Document fixed_document;
  REQUIRE(fixed_document.parseFromString(fixed_xml.str()));
  std::unique_ptr<WaveFunctionComponent> fixed = builder.buildComponent(fixed_document.getRoot());
  REQUIRE(fixed != nullptr);
  CHECK_FALSE(fixed->isOptimizable());
  UniqueOptObjRefs fixed_refs;
  fixed->extractOptimizableObjectRefs(fixed_refs);
  CHECK(fixed_refs.empty());

  std::ostringstream optimized_xml;
  optimized_xml << "<psiformer name=\"pf_selected\" parameters=\"" << files.parameters.string()
                << "\" configuration=\"" << files.configuration.string()
                << "\" system=\"all_electron\" optimize=\"yes\" optimize_scope=\"indices\" "
                   "optimize_indices=\"127, 0 1\"/>";
  Libxml2Document optimized_document;
  REQUIRE(optimized_document.parseFromString(optimized_xml.str()));
  std::unique_ptr<WaveFunctionComponent> optimized = builder.buildComponent(optimized_document.getRoot());
  REQUIRE(optimized != nullptr);
  CHECK(optimized->isOptimizable());
  UniqueOptObjRefs optimized_refs;
  optimized->extractOptimizableObjectRefs(optimized_refs);
  REQUIRE(optimized_refs.size() == 1);

  std::ostringstream all_xml;
  all_xml << "<psiformer name=\"pf_all\" parameters=\"" << files.parameters.string()
          << "\" configuration=\"" << files.configuration.string()
          << "\" system=\"all_electron\" optimize=\"yes\" optimize_scope=\"all\"/>";
  Libxml2Document all_document;
  REQUIRE(all_document.parseFromString(all_xml.str()));
  std::unique_ptr<WaveFunctionComponent> all = builder.buildComponent(all_document.getRoot());
  auto* all_psiformer = dynamic_cast<PsiFormerWF*>(all.get());
  REQUIRE(all_psiformer != nullptr);
  OptVariables all_active = registerSelectedParameters(*all_psiformer);
  const std::vector<Leaf> layout = makeLayout(4, 2);
  const std::size_t expected_parameter_count = std::accumulate(
      layout.begin(), layout.end(), std::size_t{0}, [](std::size_t count, const Leaf& leaf) {
        return count + product(leaf.shape);
      });
  CHECK(all_active.size() == expected_parameter_count);

  // Full-network clones share the O(P) selection, names, values, and global
  // indices. The inherited per-object VariableSet remains empty, preventing a
  // second O(P) copy from being created by OptimizableObject's copy constructor.
  std::unique_ptr<WaveFunctionComponent> all_clone_storage = all_psiformer->makeClone(electrons);
  auto* all_clone = dynamic_cast<PsiFormerWF*>(all_clone_storage.get());
  REQUIRE(all_clone != nullptr);
  const auto all_diagnostics = testing::TestPsiFormerWF::optimizationMetadataDiagnostics(*all_psiformer);
  const auto clone_diagnostics = testing::TestPsiFormerWF::optimizationMetadataDiagnostics(*all_clone);
  CHECK(all_diagnostics.identity == clone_diagnostics.identity);
  CHECK(all_diagnostics.shared_owner_count == 2);
  CHECK(clone_diagnostics.shared_owner_count == 2);
  CHECK(all_diagnostics.selected_index_count == expected_parameter_count);
  CHECK(all_diagnostics.shared_variable_count == expected_parameter_count);
  CHECK(all_diagnostics.mapped_variable_count == expected_parameter_count);
  CHECK(all_diagnostics.inherited_variable_count == 0);
  CHECK(clone_diagnostics.inherited_variable_count == 0);

  std::ostringstream unsupported_xml;
  unsupported_xml << "<psiformer parameters=\"" << files.parameters.string() << "\" configuration=\""
                  << files.configuration.string()
                  << "\" system=\"all_electron\" optimize=\"yes\" optimize_scope=\"all\" "
                     "optimize_indices=\"0\"/>";
  Libxml2Document unsupported_document;
  REQUIRE(unsupported_document.parseFromString(unsupported_xml.str()));
  CHECK_THROWS_AS(builder.buildComponent(unsupported_document.getRoot()), std::invalid_argument);

  std::ostringstream undeclared_system_xml;
  undeclared_system_xml << "<psiformer parameters=\"" << files.parameters.string() << "\" configuration=\""
                        << files.configuration.string()
                        << "\" optimize=\"yes\" optimize_indices=\"0\"/>";
  Libxml2Document undeclared_system_document;
  REQUIRE(undeclared_system_document.parseFromString(undeclared_system_xml.str()));
  CHECK_THROWS_WITH(builder.buildComponent(undeclared_system_document.getRoot()),
                    Catch::Matchers::ContainsSubstring("requires system="));

  ParticleSet spinor_electrons = makeLiHElectrons(simulation_cell);
  spinor_electrons.setSpinor(true);
  PsiFormerWaveFunctionBuilder spinor_builder(OHMMS::Controller, spinor_electrons, particle_sets);
  CHECK_THROWS_WITH(spinor_builder.buildComponent(fixed_document.getRoot()),
                    Catch::Matchers::ContainsSubstring("spinor"));
}

TEST_CASE("PsiFormer builder validates internal initialization XML", "[wavefunction][psiformer][initialization]")
{
  const SimulationCell simulation_cell;
  ParticleSet electrons = makeLiHElectrons(simulation_cell);
  WaveFunctionComponentBuilder::PSetMap particle_sets;
  auto ions = makeLiHIons(simulation_cell);
  particle_sets.emplace(ions->getName(), std::move(ions));
  PsiFormerWaveFunctionBuilder builder(OHMMS::Controller, electrons, particle_sets);

  auto check_invalid = [&builder](const std::string& xml, const std::string& message) {
    Libxml2Document document;
    REQUIRE(document.parseFromString(xml));
    CHECK_THROWS_WITH(builder.buildComponent(document.getRoot()),
                      Catch::Matchers::ContainsSubstring(message));
  };

  check_invalid(
      "<psiformer initialization=\"deepqmc_psiformer_v1\" initialization_seed=\"7\" "
      "parameters=\"parameters.h5\" source=\"ion0\" system=\"all_electron\"/>",
      "cannot be combined");
  check_invalid(
      "<psiformer parameters=\"parameters.h5\" configuration=\"configuration.h5\" "
      "initialization_seed=\"7\"/>",
      "requires internal initialization");
  check_invalid(
      "<psiformer initialization=\"deepqmc_psiformer_v1\" initialization_seed=\"7\" "
      "source=\"ion0\"/>",
      "requires explicit system");
  check_invalid(
      "<psiformer initialization=\"deepqmc_psiformer_v1\" initialization_seed=\"7\" "
      "system=\"all_electron\"/>",
      "requires an explicit source");
  check_invalid(
      "<psiformer initialization=\"unversioned\" initialization_seed=\"7\" "
      "source=\"ion0\" system=\"all_electron\"/>",
      "Unsupported PsiFormer initialization profile");
  check_invalid(
      "<psiformer initialization=\"deepqmc_psiformer_v1\" initialization_seed=\"-1\" "
      "source=\"ion0\" system=\"all_electron\"/>",
      "must be an unsigned integer");
}

TEST_CASE("PsiFormer internal initialization evaluates and restores without model files",
          "[wavefunction][psiformer][initialization]")
{
  constexpr std::uint64_t initialization_seed = 17;
  const SimulationCell simulation_cell;
  ParticleSet electrons = makeLiHElectrons(simulation_cell);
  WaveFunctionComponentBuilder::PSetMap particle_sets;
  auto ions = makeLiHIons(simulation_cell);
  particle_sets.emplace(ions->getName(), std::move(ions));
  PsiFormerWaveFunctionBuilder builder(OHMMS::Controller, electrons, particle_sets);

  std::unique_ptr<PsiFormerWF> original =
      buildInternalPsiFormer(builder, "pf_internal", initialization_seed);
  OptVariables active = registerSelectedParameters(*original);
  REQUIRE(active.size() == 2);

  // Identical seeds reproduce all public values, while changing the seed
  // changes a selected parameter in the first random tensor. Index zero is an
  // analytic cusp constant and remains seed independent.
  ParticleSet repeated_electrons = makeLiHElectrons(simulation_cell);
  std::unique_ptr<PsiFormerWF> repeated =
      buildInternalPsiFormer(builder, "pf_internal_repeated", initialization_seed);
  OptVariables repeated_active = registerSelectedParameters(*repeated);
  REQUIRE(repeated_active.size() == active.size());
  for (int parameter = 0; parameter < active.size(); ++parameter)
    CHECK(std::real(repeated_active[parameter]) == std::real(active[parameter]));

  std::unique_ptr<PsiFormerWF> changed_seed =
      buildInternalPsiFormer(builder, "pf_internal_changed", initialization_seed + 1);
  OptVariables changed_active = registerSelectedParameters(*changed_seed);
  REQUIRE(changed_active.size() == active.size());
  CHECK(std::real(changed_active[0]) == std::real(active[0]));
  CHECK(std::real(changed_active[1]) != std::real(active[1]));
  changed_seed.reset();

  const ComponentSnapshot baseline = evaluateComponent(*original, electrons, active);
  const ComponentSnapshot repeated_snapshot =
      evaluateComponent(*repeated, repeated_electrons, repeated_active);
  checkComponentSnapshot(repeated_snapshot, baseline);
  repeated.reset();

  CHECK(std::isfinite(baseline.log_value));
  CHECK(std::isfinite(baseline.phase));
  CHECK(std::isfinite(baseline.wavefunction_value));
  CHECK(std::isfinite(baseline.local_energy));
  for (double value : baseline.gradient)
    CHECK(std::isfinite(value));
  for (double value : baseline.laplacian)
    CHECK(std::isfinite(value));
  for (double value : baseline.log_parameter_derivative)
    CHECK(std::isfinite(value));
  for (double value : baseline.kinetic_parameter_derivative)
    CHECK(std::isfinite(value));

  // Update through normal optimizer registration, then persist both the
  // generic selected list and the complete object-specific model payload.
  active[0] += 1.5e-4;
  active[1] -= 2.5e-4;
  original->resetParametersExclusive(active);
  const ComponentSnapshot expected = evaluateComponent(*original, electrons, active);

  ScopedTestDirectory files("internal_restart");
  const std::filesystem::path state_path = files.path / "psiformer_internal.vp.h5";
  hdf_archive output;
  active.writeToHDF(state_path.string(), output);
  original->writeVariationalParameters(output);
  output.close();

  ParticleSet restored_electrons = makeLiHElectrons(simulation_cell);
  std::unique_ptr<PsiFormerWF> restored =
      buildInternalPsiFormer(builder, "pf_internal", initialization_seed);
  OptVariables restored_active = registerSelectedParameters(*restored);
  hdf_archive input;
  restored_active.readFromHDF(state_path.string(), input);
  restored->readVariationalParameters(input);
  input.close();
  restored->resetParametersExclusive(restored_active);
  const ComponentSnapshot restarted =
      evaluateComponent(*restored, restored_electrons, restored_active);
  checkComponentSnapshot(restarted, expected);

  // The seed is part of restart identity, so an otherwise compatible model
  // cannot silently accept a payload produced from another initial state.
  std::unique_ptr<PsiFormerWF> mismatched_seed =
      buildInternalPsiFormer(builder, "pf_internal", initialization_seed + 1);
  hdf_archive mismatch_input;
  REQUIRE(mismatch_input.open(state_path, H5F_ACC_RDONLY));
  CHECK_THROWS_WITH(mismatched_seed->readVariationalParameters(mismatch_input),
                    Catch::Matchers::ContainsSubstring("initialization seed"));
  mismatch_input.close();
}

TEST_CASE("PsiFormer internally initialized periodic model is lattice-image invariant",
          "[wavefunction][psiformer][initialization][periodic]")
{
  Lattice lattice;
  lattice.R         = {8.0, 0.0, 0.0, 0.6, 7.4, 0.0, -0.3, 0.5, 8.5};
  lattice.BoxBConds = {true, true, true};
  lattice.reset();
  const SimulationCell simulation_cell(lattice);

  ParticleSet electrons = makeLiHElectrons(simulation_cell);
  WaveFunctionComponentBuilder::PSetMap particle_sets;
  auto ions = makeLiHIons(simulation_cell);
  particle_sets.emplace(ions->getName(), std::move(ions));
  PsiFormerWaveFunctionBuilder builder(OHMMS::Controller, electrons, particle_sets);

  std::unique_ptr<PsiFormerWF> component = buildInternalPsiFormer(
      builder, "pf_periodic_internal", 29, "all_electron", "0 514", "periodic_torus_v1");
  OptVariables active = registerSelectedParameters(*component);
  const ComponentSnapshot baseline = evaluateComponent(*component, electrons, active);

  // Replacing one electron by an exact lattice image must leave the real
  // Gamma wavefunction, spatial derivatives, and parameter derivatives fixed.
  for (int dimension = 0; dimension < 3; ++dimension)
    electrons.R[0][dimension] += lattice.R(0, dimension) + lattice.R(2, dimension);
  electrons.update();
  const ComponentSnapshot image = evaluateComponent(*component, electrons, active);

  CHECK(image.log_value == Catch::Approx(baseline.log_value).epsilon(2e-10).margin(2e-10));
  CHECK(image.phase == Catch::Approx(baseline.phase).epsilon(2e-10).margin(2e-10));
  REQUIRE(image.gradient.size() == baseline.gradient.size());
  REQUIRE(image.laplacian.size() == baseline.laplacian.size());
  REQUIRE(image.log_parameter_derivative.size() == baseline.log_parameter_derivative.size());
  REQUIRE(image.kinetic_parameter_derivative.size() ==
          baseline.kinetic_parameter_derivative.size());
  for (std::size_t index = 0; index < image.gradient.size(); ++index)
    CHECK(image.gradient[index] ==
          Catch::Approx(baseline.gradient[index]).epsilon(2e-9).margin(2e-9));
  for (std::size_t index = 0; index < image.laplacian.size(); ++index)
    CHECK(image.laplacian[index] ==
          Catch::Approx(baseline.laplacian[index]).epsilon(2e-8).margin(2e-8));
  for (std::size_t index = 0; index < image.log_parameter_derivative.size(); ++index)
  {
    CHECK(image.log_parameter_derivative[index] ==
          Catch::Approx(baseline.log_parameter_derivative[index]).epsilon(2e-8).margin(2e-8));
    CHECK(image.kinetic_parameter_derivative[index] ==
          Catch::Approx(baseline.kinetic_parameter_derivative[index]).epsilon(2e-7).margin(2e-7));
  }
}

TEST_CASE("PsiFormer internal initialization uses the canonical pseudo-LiH layout",
          "[wavefunction][psiformer][initialization][ecp]")
{
  const SimulationCell simulation_cell;
  ParticleSet electrons = makeLiHElectrons(simulation_cell, "lih_pp");
  WaveFunctionComponentBuilder::PSetMap particle_sets;
  auto ions = makeLiHIons(simulation_cell, "lih_pp");
  particle_sets.emplace(ions->getName(), std::move(ions));
  PsiFormerWaveFunctionBuilder builder(OHMMS::Controller, electrons, particle_sets);

  std::unique_ptr<PsiFormerWF> component = buildInternalPsiFormer(
      builder, "pf_internal_pseudo", 23, "pseudopotential", "0 257");
  OptVariables active = registerSelectedParameters(*component);
  REQUIRE(active.size() == 2);

  electrons.G = ValueType(0);
  electrons.L = ValueType(0);
  const PsiFormerWF::LogValue log_value =
      component->evaluateLog(electrons, electrons.G, electrons.L);
  CHECK(std::isfinite(std::real(log_value)));
  CHECK(std::isfinite(std::imag(log_value)));
  for (int electron = 0; electron < electrons.getTotalNum(); ++electron)
  {
    for (int dimension = 0; dimension < 3; ++dimension)
      CHECK(std::isfinite(std::real(electrons.G[electron][dimension])));
    CHECK(std::isfinite(std::real(electrons.L[electron])));
  }

  Vector<ValueType> dlogpsi(active.size(), ValueType(0));
  component->evaluateDerivativesWF(electrons, active, dlogpsi);
  for (const ValueType derivative : dlogpsi)
    CHECK(std::isfinite(std::real(derivative)));
}

TEST_CASE("PsiFormer specialized public evaluation paths preserve high-level results", "[wavefunction][psiformer]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  ParticleSet electrons = makeLiHElectrons(simulation_cell);
  PsiFormerWF component(
      "pf_requests", files.parameters.string(), files.configuration.string(), true, {0, 127});
  OptVariables active = registerSelectedParameters(component);

  electrons.G = ValueType(0);
  electrons.L = ValueType(0);
  const PsiFormerWF::LogValue reference_log = component.evaluateLog(electrons, electrons.G, electrons.L);
  const ParticleSet::ParticleGradient reference_gradient = electrons.G;

  for (int electron = 0; electron < electrons.getTotalNum(); ++electron)
  {
    const PsiFormerWF::GradType active_gradient = component.evalGrad(electrons, electron);
    for (int dimension = 0; dimension < 3; ++dimension)
      CHECK(std::real(active_gradient[dimension]) ==
            Catch::Approx(std::real(reference_gradient[electron][dimension])).epsilon(2e-9).margin(2e-9));
  }

  constexpr int moved_electron = 1;
  const ParticleSet::SingleParticlePos displacement{0.08, -0.03, 0.02};
  ParticleSet moved = makeLiHElectrons(simulation_cell);
  moved.R[moved_electron] += displacement;
  moved.update();
  PsiFormerWF moved_reference("pf_requests_moved", files.parameters.string(), files.configuration.string());
  moved.G = ValueType(0);
  moved.L = ValueType(0);
  const PsiFormerWF::LogValue moved_log = moved_reference.evaluateLog(moved, moved.G, moved.L);
  const auto expected_ratio             = std::exp(moved_log - reference_log);

  electrons.makeMove(moved_electron, displacement);
  const ValueType ratio = component.ratio(electrons, moved_electron);
  CHECK(std::real(ratio) == Catch::Approx(std::real(expected_ratio)).epsilon(2e-9).margin(2e-12));
  CHECK(std::imag(ratio) == Catch::Approx(std::imag(expected_ratio)).epsilon(2e-9).margin(2e-12));
  component.restore(moved_electron);
  electrons.rejectMove(moved_electron);

  electrons.makeMove(moved_electron, displacement);
  PsiFormerWF::GradType proposed_gradient;
  const ValueType ratio_with_gradient = component.ratioGrad(electrons, moved_electron, proposed_gradient);
  CHECK(std::real(ratio_with_gradient) ==
        Catch::Approx(std::real(expected_ratio)).epsilon(2e-9).margin(2e-12));
  CHECK(std::imag(ratio_with_gradient) ==
        Catch::Approx(std::imag(expected_ratio)).epsilon(2e-9).margin(2e-12));
  for (int dimension = 0; dimension < 3; ++dimension)
    CHECK(std::real(proposed_gradient[dimension]) ==
          Catch::Approx(std::real(moved.G[moved_electron][dimension])).epsilon(2e-9).margin(2e-9));
  component.restore(moved_electron);
  electrons.rejectMove(moved_electron);

  Vector<ValueType> score_only(active.size());
  Vector<ValueType> score_with_kinetic(active.size());
  Vector<ValueType> kinetic(active.size());
  score_only         = ValueType(0.375);
  score_with_kinetic = ValueType(-0.125);
  kinetic            = ValueType(0.625);
  component.evaluateDerivativesWF(electrons, active, score_only);
  component.evaluateDerivatives(electrons, active, score_with_kinetic, kinetic);
  for (int parameter = 0; parameter < active.size(); ++parameter)
    CHECK(std::real(score_only[parameter]) - 0.375 ==
          Catch::Approx(std::real(score_with_kinetic[parameter]) + 0.125).epsilon(2e-10).margin(2e-10));
}

TEST_CASE("PsiFormer scalar ratio range failure preserves proposal state",
          "[wavefunction][psiformer][hardening][ratio][atomic]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  ParticleSet electrons = makeLiHElectrons(simulation_cell);
  PsiFormerWF component(
      "pf_ratio_range", files.parameters.string(), files.configuration.string());

  electrons.G = ValueType(0);
  electrons.L = ValueType(0);
  component.evaluateLog(electrons, electrons.G, electrons.L);
  testing::TestPsiFormerWF::setAcceptedLogMagnitude(component, -1000.0);
  const testing::PsiFormerScalarStateSnapshot before =
      testing::TestPsiFormerWF::scalarStateSnapshot(component);

  constexpr int moved_electron = 0;
  electrons.makeMove(
      moved_electron, ParticleSet::SingleParticlePos{0.01, -0.005, 0.002});
  CHECK_THROWS_WITH(
      component.ratio(electrons, moved_electron),
      Catch::Matchers::ContainsSubstring("ratio is non-finite"));
  CHECK(testing::TestPsiFormerWF::scalarStateMatches(component, before));
  electrons.rejectMove(moved_electron);

  // The rejected range failure does not strand a proposal and an immediate
  // ordinary evaluation remains usable.
  testing::TestPsiFormerWF::setAcceptedLogMagnitude(
      component, std::real(component.evaluateLog(
                     electrons, electrons.G, electrons.L)));
  electrons.makeMove(
      moved_electron, ParticleSet::SingleParticlePos{0.01, -0.005, 0.002});
  CHECK_NOTHROW(component.ratio(electrons, moved_electron));
  component.restore(moved_electron);
  electrons.rejectMove(moved_electron);
}

TEST_CASE("PsiFormer clone-local evaluator workspaces are allocated on demand",
          "[wavefunction][psiformer][memory]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  ParticleSet source_electrons = makeLiHElectrons(simulation_cell);
  PsiFormerWF source("pf_lazy_source", files.parameters.string(), files.configuration.string());

  // Establish accepted state once. Cloning preserves that state but deliberately
  // does not copy the source's now-populated full-VGL evaluator scratch.
  source_electrons.G = ValueType(0);
  source_electrons.L = ValueType(0);
  source.evaluateLog(source_electrons, source_electrons.G, source_electrons.L);

  auto make_lazy_clone = [&](ParticleSet& electrons) {
    std::unique_ptr<WaveFunctionComponent> storage = source.makeClone(electrons);
    auto* component = dynamic_cast<PsiFormerWF*>(storage.get());
    REQUIRE(component != nullptr);
    return storage;
  };
  auto as_psiformer = [](std::unique_ptr<WaveFunctionComponent>& storage) -> PsiFormerWF& {
    auto* component = dynamic_cast<PsiFormerWF*>(storage.get());
    REQUIRE(component != nullptr);
    return *component;
  };
  auto check_empty = [](const PsiFormerWF& component) {
    const auto diagnostics = testing::TestPsiFormerWF::directWorkspaceDiagnostics(component);
    CHECK(diagnostics.ownedWorkspaceCount() == 0);
    CHECK(diagnostics.accountedBytes() == 0);
  };

  ParticleSet value_electrons  = makeLiHElectrons(simulation_cell);
  ParticleSet full_electrons   = makeLiHElectrons(simulation_cell);
  ParticleSet active_electrons = makeLiHElectrons(simulation_cell);
  ParticleSet batch_electrons  = makeLiHElectrons(simulation_cell);
  ParticleSet untouched_electrons = makeLiHElectrons(simulation_cell);
  auto value_storage     = make_lazy_clone(value_electrons);
  auto full_storage      = make_lazy_clone(full_electrons);
  auto active_storage    = make_lazy_clone(active_electrons);
  auto batch_storage     = make_lazy_clone(batch_electrons);
  auto untouched_storage = make_lazy_clone(untouched_electrons);
  PsiFormerWF& value_component     = as_psiformer(value_storage);
  PsiFormerWF& full_component      = as_psiformer(full_storage);
  PsiFormerWF& active_component    = as_psiformer(active_storage);
  PsiFormerWF& batch_component     = as_psiformer(batch_storage);
  PsiFormerWF& untouched_component = as_psiformer(untouched_storage);

  check_empty(value_component);
  check_empty(full_component);
  check_empty(active_component);
  check_empty(batch_component);
  check_empty(untouched_component);

  value_electrons.makeMove(0, ParticleSet::SingleParticlePos{0.01, -0.02, 0.015});
  value_component.ratio(value_electrons, 0);
  value_component.restore(0);
  value_electrons.rejectMove(0);
  const auto value_diagnostics =
      testing::TestPsiFormerWF::directWorkspaceDiagnostics(value_component);
  CHECK(value_diagnostics.owns_value_workspace);
  CHECK(value_diagnostics.ownedWorkspaceCount() == 1);
  CHECK(value_diagnostics.value_bytes > 0);
  CHECK(value_diagnostics.accountedBytes() == value_diagnostics.value_bytes);

  full_electrons.G = ValueType(0);
  full_electrons.L = ValueType(0);
  full_component.evaluateLog(full_electrons, full_electrons.G, full_electrons.L);
  const auto full_diagnostics =
      testing::TestPsiFormerWF::directWorkspaceDiagnostics(full_component);
  CHECK(full_diagnostics.owns_full_spatial_workspace);
  CHECK(full_diagnostics.ownedWorkspaceCount() == 1);
  CHECK(full_diagnostics.full_spatial_bytes > 0);
  CHECK(full_diagnostics.accountedBytes() == full_diagnostics.full_spatial_bytes);

  active_component.evalGrad(active_electrons, 0);
  const auto active_diagnostics =
      testing::TestPsiFormerWF::directWorkspaceDiagnostics(active_component);
  CHECK(active_diagnostics.owns_active_spatial_workspace);
  CHECK(active_diagnostics.ownedWorkspaceCount() == 1);
  CHECK(active_diagnostics.active_spatial_bytes > 0);
  CHECK(active_diagnostics.accountedBytes() == active_diagnostics.active_spatial_bytes);

  batch_electrons.makeVirtualMoves(ParticleSet::SingleParticlePos{0.37, -0.22, 0.41});
  std::vector<ValueType> ratios(batch_electrons.getTotalNum());
  batch_component.evaluateRatiosAlltoOne(batch_electrons, ratios);
  const auto batch_diagnostics =
      testing::TestPsiFormerWF::directWorkspaceDiagnostics(batch_component);
  CHECK(batch_diagnostics.owns_batch_workspace);
  CHECK(batch_diagnostics.ownedWorkspaceCount() == 1);
  CHECK(batch_diagnostics.batch_bytes > 0);
  CHECK(batch_diagnostics.accountedBytes() == batch_diagnostics.batch_bytes);

  // An entirely untouched clone remains free of evaluator scratch after other
  // clones sharing the same immutable model exercise every scalar inference mode.
  check_empty(untouched_component);

  ParticleSet crowd_electrons0 = makeLiHElectrons(simulation_cell);
  ParticleSet crowd_electrons1 = makeLiHElectrons(simulation_cell);
  auto crowd_storage0 = make_lazy_clone(crowd_electrons0);
  auto crowd_storage1 = make_lazy_clone(crowd_electrons1);
  PsiFormerWF& crowd_component0 = as_psiformer(crowd_storage0);
  PsiFormerWF& crowd_component1 = as_psiformer(crowd_storage1);
  RefVectorWithLeader<WaveFunctionComponent> components(
      crowd_component0, {crowd_component0, crowd_component1});
  RefVectorWithLeader<ParticleSet> particles(
      crowd_electrons0, {crowd_electrons0, crowd_electrons1});
  std::array<ParticleSet::ParticleGradient, 2> gradients{
      ParticleSet::ParticleGradient(crowd_electrons0.getTotalNum()),
      ParticleSet::ParticleGradient(crowd_electrons1.getTotalNum())};
  std::array<ParticleSet::ParticleLaplacian, 2> laplacians{
      ParticleSet::ParticleLaplacian(crowd_electrons0.getTotalNum()),
      ParticleSet::ParticleLaplacian(crowd_electrons1.getTotalNum())};
  RefVector<ParticleSet::ParticleGradient> gradient_list{gradients[0], gradients[1]};
  RefVector<ParticleSet::ParticleLaplacian> laplacian_list{laplacians[0], laplacians[1]};

  ResourceCollection resource_template("psiformer_lazy_workspace_template");
  crowd_component0.createResource(resource_template);
  ResourceCollection crowd_resource(resource_template);
  {
    ResourceCollectionTeamLock<WaveFunctionComponent> lock(crowd_resource, components);
    crowd_component0.mw_evaluateLog(
        components, particles, gradient_list, laplacian_list);
    check_empty(crowd_component0);
    check_empty(crowd_component1);
  }
}

TEST_CASE("PsiFormer exposes fail-closed batch planning hooks",
          "[wavefunction][psiformer][batch_memory]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  ParticleSet electrons = makeLiHElectrons(simulation_cell);
  PsiFormerWF component("pf_batch_policy", files.parameters.string(),
                        files.configuration.string(), true, {0, 1});

  BatchExecutionRequirements requirements;
  component.contributeBatchExecutionRequirements(requirements);
  CHECK(requirements.requires(BatchExecutionMode::FULL_VGL));
  CHECK_FALSE(requirements.requires(BatchExecutionMode::VALUE));
  CHECK_FALSE(requirements.requires(BatchExecutionMode::ACTIVE_GRADIENT));
  CHECK_FALSE(requirements.requires(BatchExecutionMode::SCORE));
  CHECK_FALSE(requirements.requires(BatchExecutionMode::KINETIC));

  BatchExecutionRequirements broad_requirements = requirements;
  broad_requirements.require(BatchExecutionMode::VALUE);
  broad_requirements.require(BatchExecutionMode::ACTIVE_GRADIENT);
  broad_requirements.require(BatchExecutionMode::SCALAR_VALUE_COMPATIBILITY);
  BatchExecutionTopology topology;
  topology.initial_walkers_per_crowd = {1, 2};
  topology.reserve_walkers_per_crowd = {3, 2};
  topology.run_kind                  = "psiformer-hook-test";

  const BatchExecutionWorkloadContext workload{
      broad_requirements, topology, 0, 2, 0,
      BatchExecutionTargetCoordinate::POS_ONLY};
  const BatchTileCapacities logical_maximum =
      component.batchExecutionLogicalMaximum(workload);
  CHECK(logical_maximum == BatchTileCapacities{5, 3, 3, 0});

  const BatchExecutionPlanningContext context{
      broad_requirements, topology, logical_maximum,
      BatchTileCapacities{2, 2, 1, 0}, 0, 2, 0,
      BatchExecutionTargetCoordinate::POS_ONLY};
  const BatchMemoryContribution contribution =
      component.estimateBatchExecutionMemory(context);
  CHECK(contribution.logical_maximum == logical_maximum);
  CHECK(contribution.owner_multiplicity == 1);
  CHECK_FALSE(contribution.fully_accounted);
  CHECK(contribution.per_owner.total().host > 0);
  CHECK(contribution.per_owner.total().device == 0);

  // A real plan cannot be selected from partial evidence.  Later preparation
  // stages will enable accounting claims only as the corresponding ownership
  // and runtime guards become complete.
  BatchExecutionSelectionInput selection;
  selection.requirements                       = requirements;
  selection.topology                           = topology;
  selection.active_parameter_count             = 2;
  selection.target_coordinate                  = BatchExecutionTargetCoordinate::POS_ONLY;
  const BatchExecutionWorkloadContext selection_workload{
      requirements, topology, selection.particle_count, 2,
      selection.parameter_derivative_width, selection.target_coordinate};
  selection.logical_maximum =
      component.batchExecutionLogicalMaximum(selection_workload);
  CHECK_THROWS_WITH(
      selectBatchExecutionPlan(
          selection,
          [&component](const BatchExecutionPlanningContext& candidate) {
            return std::vector<BatchMemoryParticipantContribution>{
                {"twf/component/0/PsiFormerWF/pf_batch_policy",
                 component.estimateBatchExecutionMemory(candidate)}};
          }),
      Catch::Matchers::ContainsSubstring("not fully accounted"));

  // Even internally consistent evidence cannot authorize a plan that omitted
  // the component-owned FULL_VGL initialization requirement.
  BatchExecutionSelectionInput missing_requirement_selection = selection;
  missing_requirement_selection.requirements = {};
  const BatchExecutionWorkloadContext missing_workload{
      {}, topology, missing_requirement_selection.particle_count, 2,
      missing_requirement_selection.parameter_derivative_width,
      missing_requirement_selection.target_coordinate};
  missing_requirement_selection.logical_maximum =
      component.batchExecutionLogicalMaximum(missing_workload);
  const std::string participant_id =
      "twf/component/0/PsiFormerWF/pf_batch_policy";
  auto missing_plan = std::make_shared<const BatchExecutionPlan>(
      selectBatchExecutionPlan(
          missing_requirement_selection,
          [&component, &participant_id](
              const BatchExecutionPlanningContext& candidate) {
            BatchMemoryContribution fabricated =
                component.estimateBatchExecutionMemory(candidate);
            fabricated.fully_accounted = true;
            return std::vector<BatchMemoryParticipantContribution>{
                {participant_id, std::move(fabricated)}};
          }));
  const BatchExecutionParticipantPlan missing_view =
      makeBatchExecutionParticipantPlan(missing_plan, participant_id);
  CHECK_THROWS_WITH(
      component.validateBatchExecutionPlanBinding(missing_view),
      Catch::Matchers::ContainsSubstring("component-owned requirement"));

  // The explicit empty view is always safe while idle and is copied as empty
  // state rather than causing any evaluator scratch to be materialized.
  BatchExecutionParticipantPlan empty_plan;
  component.validateBatchExecutionPlanBinding(empty_plan);
  component.bindBatchExecutionPlan(empty_plan);
  CHECK_FALSE(testing::TestPsiFormerWF::hasBatchExecutionPlan(component));
  std::unique_ptr<WaveFunctionComponent> clone_storage =
      component.makeClone(electrons);
  auto* clone = dynamic_cast<PsiFormerWF*>(clone_storage.get());
  REQUIRE(clone != nullptr);
  CHECK_FALSE(testing::TestPsiFormerWF::hasBatchExecutionPlan(*clone));
}

TEST_CASE("PsiFormer prepares bounded clone scalar storage transactionally",
          "[wavefunction][psiformer][batch_memory]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  ParticleSet electrons = makeLiHElectrons(simulation_cell);
  std::unique_ptr<ParticleSet> ions = makeLiHIons(simulation_cell);
  const std::size_t electrons_count =
      static_cast<std::size_t>(electrons.getTotalNum());
  PsiFormerWF component("pf_clone_prepare", files.parameters.string(),
                        files.configuration.string(), true, {0, 1});
  component.validateSystem(electrons, *ions, "all_electron");

  BatchExecutionRequirements requirements;
  component.contributeBatchExecutionRequirements(requirements);
  requirements.require(BatchExecutionMode::SCALAR_VALUE_COMPATIBILITY);
  const std::string participant_id = "test/psiformer/clone-prepare";
  const auto plan = makeClonePreparationTestPlan(
      component, requirements, participant_id, "clone-prepare-v1");
  const BatchExecutionParticipantPlan participant_plan =
      makeBatchExecutionParticipantPlan(plan, participant_id);

  // Fabricated completeness permits selection only inside this test.  The
  // production validator still compares it with PsiFormer's false stage gate.
  CHECK_THROWS_WITH(
      component.validateBatchExecutionPlanBinding(participant_plan),
      Catch::Matchers::ContainsSubstring(
          "accounting evidence is stale"));
  component.bindBatchExecutionPlan(participant_plan);
  const auto before = testing::TestPsiFormerWF::directWorkspaceDiagnostics(component);
  CHECK_FALSE(before.has_prepared_clone_plan);
  CHECK_FALSE(before.owns_batch_workspace);
  CHECK(before.proposed_spatial_bytes == 0);

  electrons.makeVirtualMoves(
      ParticleSet::SingleParticlePos{0.11, -0.07, 0.19});
  std::vector<ValueType> unprepared_ratios(electrons_count, ValueType(11));
  CHECK_THROWS_WITH(
      component.evaluateRatiosAlltoOne(electrons, unprepared_ratios),
      Catch::Matchers::ContainsSubstring(
          "was not prepared for the bound batch plan"));
  CHECK(std::all_of(unprepared_ratios.begin(), unprepared_ratios.end(),
                    [](ValueType value) { return value == ValueType(11); }));
  CHECK_FALSE(testing::TestPsiFormerWF::directWorkspaceDiagnostics(component)
                  .owns_batch_workspace);

  testing::TestPsiFormerWF::failClonePreparationBeforePublish(component, true);
  CHECK_THROWS_AS(component.prepareBatchExecutionClone(participant_plan),
                  std::bad_alloc);
  const auto after_failure =
      testing::TestPsiFormerWF::directWorkspaceDiagnostics(component);
  CHECK_FALSE(after_failure.has_prepared_clone_plan);
  CHECK_FALSE(after_failure.owns_batch_workspace);
  CHECK(after_failure.scalar_value_publication_bytes == 0);
  CHECK(after_failure.proposed_spatial_bytes == 0);
  testing::TestPsiFormerWF::failClonePreparationBeforePublish(component, false);

  component.prepareBatchExecutionClone(participant_plan);
  const auto prepared =
      testing::TestPsiFormerWF::directWorkspaceDiagnostics(component);
  const std::size_t one_spatial_state =
      electrons_count *
      (sizeof(PsiFormerWF::GradType) + sizeof(ValueType));
  CHECK(prepared.has_prepared_clone_plan);
  CHECK(prepared.owns_batch_workspace);
  CHECK(prepared.batch_bytes > 0);
  CHECK(prepared.accepted_spatial_bytes == one_spatial_state);
  CHECK(prepared.proposed_spatial_bytes == one_spatial_state);
  CHECK(prepared.scalar_value_publication_bytes ==
        (electrons_count + 1) * sizeof(ValueType));
  CHECK(prepared.accountedBytes() ==
        prepared.batch_bytes + prepared.scalar_value_publication_bytes);

  // A scalar-only plan cannot authorize buffer restoration.  The typed
  // preflight rejects the foreign binding before any clone or cursor mutation.
  ParticleSet wrong_electron_count(simulation_cell);
  wrong_electron_count.setName("wrong_electron_count");
  wrong_electron_count.create(
      {static_cast<int>(electrons_count + 1)});
  PsiFormerWF::WFBufferType empty_buffer;
  CHECK_THROWS_WITH(
      component.copyFromBuffer(wrong_electron_count, empty_buffer),
      Catch::Matchers::ContainsSubstring(
          "received a foreign ParticleSet"));
  const auto after_wrong_restore =
      testing::TestPsiFormerWF::directWorkspaceDiagnostics(component);
  CHECK(after_wrong_restore.accepted_spatial_bytes ==
        prepared.accepted_spatial_bytes);
  CHECK(after_wrong_restore.proposed_spatial_bytes ==
        prepared.proposed_spatial_bytes);
  CHECK(after_wrong_restore.batch_workspace_identity ==
        prepared.batch_workspace_identity);
  CHECK(after_wrong_restore.batch_storage_fingerprint ==
        prepared.batch_storage_fingerprint);

  // Repeating the same preparation is a strict no-op with stable backing.
  component.prepareBatchExecutionClone(participant_plan);
  const auto repeated =
      testing::TestPsiFormerWF::directWorkspaceDiagnostics(component);
  CHECK(repeated.batch_workspace_identity ==
        prepared.batch_workspace_identity);
  CHECK(repeated.batch_storage_fingerprint ==
        prepared.batch_storage_fingerprint);
  CHECK(repeated.accountedBytes() == prepared.accountedBytes());

  // A clone inherits only immutable plan identity and accepted physical state;
  // its mutable proposal, scalar publication, and batch workspace start empty.
  std::unique_ptr<WaveFunctionComponent> clone_storage =
      component.makeClone(electrons);
  auto* clone = dynamic_cast<PsiFormerWF*>(clone_storage.get());
  REQUIRE(clone != nullptr);
  CHECK(testing::TestPsiFormerWF::hasBatchExecutionPlan(*clone));
  const auto clone_before =
      testing::TestPsiFormerWF::directWorkspaceDiagnostics(*clone);
  CHECK_FALSE(clone_before.has_prepared_clone_plan);
  CHECK_FALSE(clone_before.owns_batch_workspace);
  CHECK(clone_before.scalar_value_publication_bytes == 0);
  CHECK(clone_before.proposed_spatial_bytes == 0);
  clone->prepareBatchExecutionClone(participant_plan);
  const auto clone_prepared =
      testing::TestPsiFormerWF::directWorkspaceDiagnostics(*clone);
  CHECK(clone_prepared.has_prepared_clone_plan);
  CHECK(clone_prepared.batch_workspace_identity !=
        prepared.batch_workspace_identity);
  CHECK(clone_prepared.batch_bytes == prepared.batch_bytes);

  // Reject a one-element logical overrun before touching retained scratch or
  // caller output.
  std::vector<ParticleSet::SingleParticlePos> too_many_moves(
      electrons_count + 1,
      ParticleSet::SingleParticlePos{0.01, -0.02, 0.03});
  VirtualParticleSet virtual_particles(electrons);
  virtual_particles.makeMoves(electrons, 1, too_many_moves);
  std::vector<ValueType> overflow_ratios(too_many_moves.size(),
                                         ValueType(7));
  CHECK_THROWS_WITH(
      component.evaluateRatios(virtual_particles, overflow_ratios),
      Catch::Matchers::ContainsSubstring("exceeds the planned scalar VALUE envelope"));
  CHECK(std::all_of(overflow_ratios.begin(), overflow_ratios.end(),
                    [](ValueType value) { return value == ValueType(7); }));
  const auto after_overflow =
      testing::TestPsiFormerWF::directWorkspaceDiagnostics(component);
  CHECK(after_overflow.batch_storage_fingerprint ==
        prepared.batch_storage_fingerprint);

  // The largest admitted scalar virtual-ratio batch uses all Ne + 1 logical
  // configurations without changing the prepared capacity fingerprint.
  std::vector<ParticleSet::SingleParticlePos> boundary_moves;
  boundary_moves.reserve(electrons_count);
  for (std::size_t move = 0; move < electrons_count; ++move)
    boundary_moves.emplace_back(0.01 * (move + 1),
                                -0.015 * (move + 1),
                                0.02 * (move + 1));
  VirtualParticleSet boundary_virtual_particles(electrons);
  boundary_virtual_particles.makeMoves(electrons, 1, boundary_moves);
  std::vector<ValueType> boundary_ratios(electrons_count);
  component.evaluateRatios(boundary_virtual_particles, boundary_ratios);
  const std::vector<ValueType> planned_boundary_ratios = boundary_ratios;
  CHECK(testing::TestPsiFormerWF::directWorkspaceDiagnostics(component)
            .batch_storage_fingerprint == prepared.batch_storage_fingerprint);

  // The admitted all-to-one path copies through persistent staging and does
  // not replace that storage with the caller's vector.
  electrons.makeVirtualMoves(
      ParticleSet::SingleParticlePos{0.37, -0.22, 0.41});
  std::vector<ValueType> ratios(electrons_count);
  component.evaluateRatiosAlltoOne(electrons, ratios);
  const std::vector<ValueType> planned_all_to_one_ratios = ratios;
  const auto after_value =
      testing::TestPsiFormerWF::directWorkspaceDiagnostics(component);
  CHECK(after_value.batch_workspace_identity ==
        prepared.batch_workspace_identity);
  CHECK(after_value.batch_storage_fingerprint ==
        prepared.batch_storage_fingerprint);
  CHECK(after_value.scalar_value_publication_bytes ==
        prepared.scalar_value_publication_bytes);

  Vector<ValueType> score(2);
  Vector<ValueType> kinetic(2);
  score   = ValueType(0);
  kinetic = ValueType(0);
  CHECK_THROWS_WITH(
      component.evaluateDerivativesWF(electrons, OptVariables{}, score),
      Catch::Matchers::ContainsSubstring("not admitted as a clone-local operation"));
  CHECK_THROWS_WITH(
      component.evaluateDerivatives(electrons, OptVariables{}, score, kinetic),
      Catch::Matchers::ContainsSubstring("not admitted as a clone-local operation"));

  // Lifecycle transitions cannot invalidate storage beneath an in-flight
  // proposal, and a nonempty replan must pass through an explicit null clear.
  testing::TestPsiFormerWF::markSelectedProposalPending(component);
  CHECK_THROWS_WITH(
      component.prepareBatchExecutionClone(participant_plan),
      Catch::Matchers::ContainsSubstring("proposal is pending"));
  BatchExecutionParticipantPlan empty_plan;
  CHECK_THROWS_WITH(
      component.validateBatchExecutionPlanBinding(empty_plan),
      Catch::Matchers::ContainsSubstring("proposal is pending"));
  testing::TestPsiFormerWF::clearProposal(component);

  const auto replacement_plan = makeClonePreparationTestPlan(
      component, requirements, participant_id, "clone-prepare-v2", 1);
  const BatchExecutionParticipantPlan replacement_view =
      makeBatchExecutionParticipantPlan(replacement_plan, participant_id);
  CHECK_THROWS_WITH(
      component.validateBatchExecutionPlanBinding(replacement_view),
      Catch::Matchers::ContainsSubstring("explicit null-plan clear"));

  // Clearing the plan removes bounded and unaccounted evaluator scratch.  A
  // subsequent scalar call follows the original unrestricted lazy behavior.
  component.validateBatchExecutionPlanBinding(empty_plan);
  component.bindBatchExecutionPlan(empty_plan);
  const auto cleared =
      testing::TestPsiFormerWF::directWorkspaceDiagnostics(component);
  CHECK_FALSE(testing::TestPsiFormerWF::hasBatchExecutionPlan(component));
  CHECK_FALSE(cleared.has_prepared_clone_plan);
  CHECK_FALSE(cleared.owns_batch_workspace);
  CHECK(cleared.scalar_value_publication_bytes == 0);
  component.evaluateRatiosAlltoOne(electrons, ratios);
  for (std::size_t electron = 0; electron < electrons_count; ++electron)
    CHECK(std::abs(ratios[electron] - planned_all_to_one_ratios[electron]) ==
          Catch::Approx(0.0).margin(2e-12));

  std::vector<ValueType> legacy_boundary_ratios(electrons_count);
  component.evaluateRatios(boundary_virtual_particles,
                           legacy_boundary_ratios);
  for (std::size_t move = 0; move < electrons_count; ++move)
    CHECK(std::abs(legacy_boundary_ratios[move] -
                   planned_boundary_ratios[move]) ==
          Catch::Approx(0.0).margin(2e-12));
  CHECK(testing::TestPsiFormerWF::directWorkspaceDiagnostics(component)
            .owns_batch_workspace);

  // First binding from legacy mode canonicalizes the lazy workspace without
  // allocation; preparation remains a separate explicit lifecycle step.
  component.bindBatchExecutionPlan(replacement_view);
  const auto rebound =
      testing::TestPsiFormerWF::directWorkspaceDiagnostics(component);
  CHECK(testing::TestPsiFormerWF::hasBatchExecutionPlan(component));
  CHECK_FALSE(rebound.has_prepared_clone_plan);
  CHECK_FALSE(rebound.owns_batch_workspace);
}

TEST_CASE("PsiFormer prepares exact Boundary-32 walker layout evidence",
          "[wavefunction][psiformer][batch_memory][walker_layout]")
{
  static_assert(noexcept(
      std::declval<const PsiFormerWF&>().hasPreparedBatchExecutionClone(
          std::declval<const BatchExecutionParticipantPlan&>())));

  constexpr std::array<BatchExecutionMode, 4> planned_operations{
      BatchExecutionMode::BUFFER_READ, BatchExecutionMode::BUFFER_WRITE,
      BatchExecutionMode::PREPARE_GROUP,
      BatchExecutionMode::COMPLETE_UPDATES};
  for (const BatchExecutionMode mode : planned_operations)
    CHECK(testing::TestPsiFormerWF::plannedWalkerOperationModes(mode).mask() ==
          static_cast<std::uint32_t>(mode));

  struct LayoutCase
  {
    const char* name;
    bool buffer_read;
    bool buffer_write;
    bool prepare_group;
    bool complete_updates;
  };
  constexpr std::array<LayoutCase, 6> cases{
      LayoutCase{"read-only", true, false, false, false},
      LayoutCase{"write-only", false, true, false, false},
      LayoutCase{"read-write", true, true, false, false},
      LayoutCase{"prepare-group-only", false, false, true, false},
      LayoutCase{"complete-updates-only", false, false, false, true},
      LayoutCase{"no-buffer-reference", false, false, false, false}};

  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  const ParticleSet reference_electrons = makeLiHElectrons(simulation_cell);
  const std::size_t electron_count =
      static_cast<std::size_t>(reference_electrons.getTotalNum());
  const pf::WalkerBufferLayout build_layout =
      pf::walkerBufferLayoutRequirement(
          electron_count, sizeof(ValueType), sizeof(PsiFormerWF::GradType),
          sizeof(QMCTraits::FullPrecRealType), QMC_SIMD_ALIGNMENT);
  REQUIRE(build_layout.fingerprint != 0);
  CHECK(build_layout.electrons == electron_count);
  CHECK(build_layout.value_type_bytes == sizeof(ValueType));
  CHECK(build_layout.gradient_type_bytes == sizeof(PsiFormerWF::GradType));
  CHECK(build_layout.full_precision_real_type_bytes ==
        sizeof(QMCTraits::FullPrecRealType));

  for (std::size_t case_index = 0; case_index < cases.size(); ++case_index)
  {
    const LayoutCase& layout_case = cases[case_index];
    CAPTURE(layout_case.name);
    ParticleSet electrons = reference_electrons;
    std::unique_ptr<ParticleSet> ions = makeLiHIons(simulation_cell);
    PsiFormerWF component("pf_walker_layout_" + std::to_string(case_index),
                          files.parameters.string(),
                          files.configuration.string(), true, {0, 1});
    component.validateSystem(electrons, *ions, "all_electron");

    BatchExecutionRequirements requirements;
    component.contributeBatchExecutionRequirements(requirements);
    if (layout_case.buffer_read)
      requirements.require(BatchExecutionMode::BUFFER_READ);
    if (layout_case.buffer_write)
      requirements.require(BatchExecutionMode::BUFFER_WRITE);
    if (layout_case.prepare_group)
      requirements.require(BatchExecutionMode::PREPARE_GROUP);
    if (layout_case.complete_updates)
      requirements.require(BatchExecutionMode::COMPLETE_UPDATES);
    CHECK(requirements.requires(BatchExecutionMode::BUFFER_READ) ==
          layout_case.buffer_read);
    CHECK(requirements.requires(BatchExecutionMode::BUFFER_WRITE) ==
          layout_case.buffer_write);
    CHECK(requirements.requires(BatchExecutionMode::PREPARE_GROUP) ==
          layout_case.prepare_group);
    CHECK(requirements.requires(BatchExecutionMode::COMPLETE_UPDATES) ==
          layout_case.complete_updates);

    const bool buffer_selected =
        layout_case.buffer_read || layout_case.buffer_write;
    const pf::WalkerBufferLayout expected_layout =
        buffer_selected ? build_layout : pf::WalkerBufferLayout{};
    const std::string participant_id =
        "test/psiformer/walker-layout/" + std::to_string(case_index);
    const auto plan = makeClonePreparationTestPlan(
        component, requirements, participant_id,
        "walker-layout-v" + std::to_string(case_index));
    const BatchExecutionParticipantPlan participant_plan =
        makeBatchExecutionParticipantPlan(plan, participant_id);

    const BatchExecutionPlanningContext production_context{
        requirements,
        plan->topology(),
        plan->logicalMaximum(),
        plan->selectedCapacities(),
        plan->particleCount(),
        plan->activeParameterCount(),
        plan->parameterDerivativeWidth(),
        plan->targetCoordinate()};
    CHECK_FALSE(component.estimateBatchExecutionMemory(production_context)
                    .fully_accounted);
    CHECK_FALSE(component.supportsAtomicBatchPublication());

    component.bindBatchExecutionPlan(participant_plan);
    const auto before_failure =
        testing::TestPsiFormerWF::directWorkspaceDiagnostics(component);
    const testing::PsiFormerScalarStateSnapshot component_before_failure =
        testing::TestPsiFormerWF::scalarStateSnapshot(component);
    const ScalarParticleStateSnapshot particles_before_failure =
        captureScalarParticleState(electrons);
    CHECK(before_failure.prepared_walker_buffer_layout.empty());
    CHECK_FALSE(before_failure.has_prepared_clone_plan);
    CHECK_FALSE(component.hasPreparedBatchExecutionClone(participant_plan));

    testing::TestPsiFormerWF::failClonePreparationBeforePublish(component,
                                                                 true);
    CHECK_THROWS_AS(component.prepareBatchExecutionClone(participant_plan),
                    std::bad_alloc);
    testing::TestPsiFormerWF::failClonePreparationBeforePublish(component,
                                                                 false);
    const auto after_failure =
        testing::TestPsiFormerWF::directWorkspaceDiagnostics(component);
    checkScalarWorkspaceStorage(after_failure, before_failure);
    CHECK(testing::TestPsiFormerWF::scalarStateMatches(
        component, component_before_failure));
    checkScalarParticleState(electrons, particles_before_failure);
    CHECK_FALSE(component.hasPreparedBatchExecutionClone(participant_plan));

    component.prepareBatchExecutionClone(participant_plan);
    const auto prepared =
        testing::TestPsiFormerWF::directWorkspaceDiagnostics(component);
    CHECK(prepared.has_prepared_clone_plan);
    CHECK(component.hasPreparedBatchExecutionClone(participant_plan));
    CHECK(prepared.prepared_walker_buffer_layout == expected_layout);
    CHECK(prepared.prepared_walker_buffer_layout.empty() == !buffer_selected);
    if (buffer_selected)
      CHECK(prepared.prepared_walker_buffer_layout.fingerprint ==
            build_layout.fingerprint);
    CHECK(prepared.accountedBytes() == before_failure.accountedBytes());
    CHECK(prepared.ownedWorkspaceCount() ==
          before_failure.ownedWorkspaceCount());

    const testing::PsiFormerScalarStateSnapshot component_before_repeat =
        testing::TestPsiFormerWF::scalarStateSnapshot(component);
    component.prepareBatchExecutionClone(participant_plan);
    const auto repeated =
        testing::TestPsiFormerWF::directWorkspaceDiagnostics(component);
    checkScalarWorkspaceStorage(repeated, prepared);
    CHECK(testing::TestPsiFormerWF::scalarStateMatches(
        component, component_before_repeat));

    pf::WalkerBufferLayout corrupted_layout = expected_layout;
    if (buffer_selected)
      ++corrupted_layout.total_bytes;
    else
      corrupted_layout = build_layout;
    testing::TestPsiFormerWF::setPreparedWalkerBufferLayout(
        component, corrupted_layout);
    bool corrupted_is_prepared = true;
    CHECK_NOTHROW(corrupted_is_prepared =
                      component.hasPreparedBatchExecutionClone(
                          participant_plan));
    CHECK_FALSE(corrupted_is_prepared);
    testing::TestPsiFormerWF::setPreparedWalkerBufferLayout(
        component, expected_layout);
    CHECK(component.hasPreparedBatchExecutionClone(participant_plan));

    std::unique_ptr<WaveFunctionComponent> clone_storage =
        component.makeClone(electrons);
    auto* clone = dynamic_cast<PsiFormerWF*>(clone_storage.get());
    REQUIRE(clone != nullptr);
    CHECK(testing::TestPsiFormerWF::hasBatchExecutionPlan(*clone));
    const auto clone_diagnostics =
        testing::TestPsiFormerWF::directWorkspaceDiagnostics(*clone);
    CHECK_FALSE(clone_diagnostics.has_prepared_clone_plan);
    CHECK(clone_diagnostics.prepared_walker_buffer_layout.empty());
    CHECK_FALSE(clone->hasPreparedBatchExecutionClone(participant_plan));
    CHECK(clone_diagnostics.accountedBytes() == 0);
    CHECK(clone_diagnostics.ownedWorkspaceCount() == 0);

    PsiFormerWF::WFBufferType guarded_buffer;
    PsiFormerWF::GradType buffer_gradient;
    for (std::size_t dimension = 0; dimension < OHMMS_DIM; ++dimension)
      buffer_gradient[dimension] = ValueType(0.125 * (dimension + 1));
    QMCTraits::FullPrecRealType buffer_scalar = -3.75;
    guarded_buffer.add(&buffer_gradient, &buffer_gradient + 1);
    guarded_buffer.add(buffer_scalar);
    guarded_buffer.allocate();
    guarded_buffer.rewind();
    guarded_buffer.put(&buffer_gradient, &buffer_gradient + 1);
    guarded_buffer.put(buffer_scalar);
    const auto buffer_bulk_cursor   = guarded_buffer.current();
    const auto buffer_scalar_cursor = guarded_buffer.current_scalar();
    const auto buffer_storage       = guarded_buffer.myData;
    const auto buffer_capacity      = guarded_buffer.myData.capacity();
    const auto* buffer_data         = guarded_buffer.myData.data();
    const auto* buffer_scalar_data  = guarded_buffer.Scalar_ptr;
    const testing::PsiFormerScalarStateSnapshot component_before_guard =
        testing::TestPsiFormerWF::scalarStateSnapshot(component);
    const ScalarParticleStateSnapshot particles_before_guard =
        captureScalarParticleState(electrons);
    const auto workspace_before_guard =
        testing::TestPsiFormerWF::directWorkspaceDiagnostics(component);
    const std::size_t parameter_version_before_guard =
        component.parameterVersion();

    const auto check_guarded_state = [&]() {
      CHECK(testing::TestPsiFormerWF::scalarStateMatches(
          component, component_before_guard));
      checkScalarParticleState(electrons, particles_before_guard);
      checkScalarWorkspaceStorage(
          testing::TestPsiFormerWF::directWorkspaceDiagnostics(component),
          workspace_before_guard);
      CHECK(component.parameterVersion() == parameter_version_before_guard);
      CHECK(guarded_buffer.current() == buffer_bulk_cursor);
      CHECK(guarded_buffer.current_scalar() == buffer_scalar_cursor);
      CHECK(guarded_buffer.myData.data() == buffer_data);
      CHECK(guarded_buffer.myData.capacity() == buffer_capacity);
      CHECK(guarded_buffer.Scalar_ptr == buffer_scalar_data);
      REQUIRE(guarded_buffer.myData.size() == buffer_storage.size());
      for (std::size_t byte = 0; byte < buffer_storage.size(); ++byte)
        CHECK(guarded_buffer.myData[byte] == buffer_storage[byte]);
    };
    const auto expect_lifecycle_guard = [&](auto&& operation,
                                            const char* diagnostic) {
      CHECK_THROWS_WITH(operation(),
                        Catch::Matchers::ContainsSubstring(diagnostic));
      check_guarded_state();
    };
    const auto expect_lifecycle_success = [&](auto&& operation) {
      CHECK_NOTHROW(operation());
      check_guarded_state();
    };
    const auto expect_buffer_guard = [&](auto&& operation) {
      CHECK_THROWS(operation());
      check_guarded_state();
    };

    // Selected buffer modes enter typed Stage-3 preflight, while malformed
    // storage and wrong modes remain atomic. Selected scalar lifecycle hooks
    // are now validated no-ops; crowd hooks still require an acquired resource.
    expect_buffer_guard(
        [&]() { component.registerData(electrons, guarded_buffer); });
    expect_buffer_guard([&]() {
      component.updateBuffer(electrons, guarded_buffer, false);
    });
    expect_buffer_guard(
        [&]() { component.copyFromBuffer(electrons, guarded_buffer); });
    if (layout_case.prepare_group)
      expect_lifecycle_success(
          [&]() { component.prepareGroup(electrons, 0); });
    else
      expect_lifecycle_guard(
          [&]() { component.prepareGroup(electrons, 0); },
          "lifecycle operation is not admitted by its explicit batch mode");
    if (layout_case.complete_updates)
      expect_lifecycle_success([&]() { component.completeUpdates(); });
    else
      expect_lifecycle_guard(
          [&]() { component.completeUpdates(); },
          "lifecycle operation is not admitted by its explicit batch mode");
    RefVectorWithLeader<WaveFunctionComponent> components(component,
                                                           {component});
    RefVectorWithLeader<ParticleSet> particles(electrons, {electrons});
    expect_lifecycle_guard(
        [&]() { component.mw_prepareGroup(components, particles, 0); },
        layout_case.prepare_group
            ? "requires an acquired crowd resource"
            : "lifecycle operation is not admitted by its explicit batch mode");
    expect_lifecycle_guard(
        [&]() { component.mw_completeUpdates(components); },
        layout_case.complete_updates
            ? "requires an acquired crowd resource"
            : "lifecycle operation is not admitted by its explicit batch mode");

    BatchExecutionParticipantPlan empty_plan;
    component.validateBatchExecutionPlanBinding(empty_plan);
    component.bindBatchExecutionPlan(empty_plan);
    const auto cleared =
        testing::TestPsiFormerWF::directWorkspaceDiagnostics(component);
    CHECK_FALSE(testing::TestPsiFormerWF::hasBatchExecutionPlan(component));
    CHECK_FALSE(cleared.has_prepared_clone_plan);
    CHECK(cleared.prepared_walker_buffer_layout.empty());
    CHECK(cleared.accountedBytes() == prepared.accountedBytes());
    CHECK(cleared.ownedWorkspaceCount() == prepared.ownedWorkspaceCount());
    CHECK_FALSE(component.hasPreparedBatchExecutionClone(participant_plan));

    component.bindBatchExecutionPlan(participant_plan);
    const auto rebound =
        testing::TestPsiFormerWF::directWorkspaceDiagnostics(component);
    CHECK(testing::TestPsiFormerWF::hasBatchExecutionPlan(component));
    CHECK_FALSE(rebound.has_prepared_clone_plan);
    CHECK(rebound.prepared_walker_buffer_layout.empty());
    CHECK(rebound.accountedBytes() == prepared.accountedBytes());
    CHECK(rebound.ownedWorkspaceCount() == prepared.ownedWorkspaceCount());
    CHECK_FALSE(component.hasPreparedBatchExecutionClone(participant_plan));
  }
}

TEST_CASE("PsiFormer planned scalar lifecycle hooks obey independent modes",
          "[wavefunction][psiformer][batch_memory][lifecycle]")
{
  struct LifecycleCase
  {
    const char* name;
    bool prepare_group;
    bool complete_updates;
    bool scalar_value_compatibility;
  };
  constexpr std::array<LifecycleCase, 6> cases{
      LifecycleCase{"prepare-group-only", true, false, false},
      LifecycleCase{"complete-updates-only", false, true, false},
      LifecycleCase{"both-lifecycle-modes", true, true, false},
      LifecycleCase{"scalar-value-and-both-lifecycle-modes", true, true, true},
      LifecycleCase{"scalar-value-only", false, false, true},
      LifecycleCase{"base-full-vgl-only", false, false, false}};

  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  const ParticleSet reference_electrons = makeLiHElectrons(simulation_cell);

  for (std::size_t case_index = 0; case_index < cases.size(); ++case_index)
  {
    const LifecycleCase& lifecycle_case = cases[case_index];
    CAPTURE(lifecycle_case.name);
    ParticleSet electrons = reference_electrons;
    ParticleSet foreign_electrons = makeLiHElectrons(simulation_cell);
    std::unique_ptr<ParticleSet> ions = makeLiHIons(simulation_cell);
    PsiFormerWF component("pf_lifecycle_" + std::to_string(case_index),
                          files.parameters.string(),
                          files.configuration.string(), true, {0, 1});
    component.validateSystem(electrons, *ions, "all_electron");

    BatchExecutionRequirements requirements;
    component.contributeBatchExecutionRequirements(requirements);
    if (lifecycle_case.prepare_group)
      requirements.require(BatchExecutionMode::PREPARE_GROUP);
    if (lifecycle_case.complete_updates)
      requirements.require(BatchExecutionMode::COMPLETE_UPDATES);
    if (lifecycle_case.scalar_value_compatibility)
      requirements.require(BatchExecutionMode::SCALAR_VALUE_COMPATIBILITY);
    const std::string participant_id =
        "test/psiformer/lifecycle/" + std::to_string(case_index);
    const auto plan = makeClonePreparationTestPlan(
        component, requirements, participant_id,
        "lifecycle-v" + std::to_string(case_index));
    const BatchExecutionParticipantPlan participant_plan =
        makeBatchExecutionParticipantPlan(plan, participant_id);
    component.bindBatchExecutionPlan(participant_plan);
    component.prepareBatchExecutionClone(participant_plan);
    REQUIRE(component.hasPreparedBatchExecutionClone(participant_plan));

    // Lifecycle hooks must remain valid when the accepted numerical cache is
    // deliberately INVALID: these operations validate sequencing, not values.
    testing::TestPsiFormerWF::poisonAcceptedStateForRestore(component);
    REQUIRE_FALSE(testing::TestPsiFormerWF::scalarStateSnapshot(component)
                      .accepted_value_valid);

    const auto exercise_lifecycle = [&](PsiFormerWF& target,
                                        ParticleSet& particles,
                                        auto&& operation,
                                        const char* expected_diagnostic) {
      const testing::PsiFormerScalarStateSnapshot component_before =
          testing::TestPsiFormerWF::scalarStateSnapshot(target);
      const ScalarParticleStateSnapshot particles_before =
          captureScalarParticleState(particles);
      const testing::PsiFormerWorkspaceDiagnostics workspace_before =
          testing::TestPsiFormerWF::directWorkspaceDiagnostics(target);
      const std::size_t parameter_version_before = target.parameterVersion();
      const std::array<std::size_t, 2> counters_before =
          testing::TestPsiFormerWF::plannedProposalCounts(target);

      if (expected_diagnostic)
        CHECK_THROWS_WITH(
            operation(),
            Catch::Matchers::ContainsSubstring(expected_diagnostic));
      else
        CHECK_NOTHROW(operation());

      CHECK(testing::TestPsiFormerWF::scalarStateMatches(target,
                                                          component_before));
      checkScalarParticleState(particles, particles_before);
      checkScalarWorkspaceStorage(
          testing::TestPsiFormerWF::directWorkspaceDiagnostics(target),
          workspace_before);
      CHECK(target.parameterVersion() == parameter_version_before);
      CHECK(testing::TestPsiFormerWF::plannedProposalCounts(target) ==
            counters_before);
    };

    // Lifecycle authority is deliberately structural.  An unrelated scalar
    // VALUE workspace may be unusable without disabling either lifecycle hook.
    if (lifecycle_case.scalar_value_compatibility &&
        lifecycle_case.prepare_group && lifecycle_case.complete_updates)
    {
      using WorkspaceFault =
          testing::TestPsiFormerWF::PreparedScalarWorkspaceFault;
      testing::TestPsiFormerWF::setPreparedScalarWorkspaceFault(
          component, WorkspaceFault::LOGICAL_CAPACITY);
      REQUIRE_FALSE(component.hasPreparedBatchExecutionClone(participant_plan));
      exercise_lifecycle(
          component, electrons,
          [&]() { component.prepareGroup(electrons, 0); }, nullptr);
      exercise_lifecycle(
          component, electrons, [&]() { component.completeUpdates(); },
          nullptr);
      testing::TestPsiFormerWF::setPreparedScalarWorkspaceFault(
          component, WorkspaceFault::NONE);
      REQUIRE(component.hasPreparedBatchExecutionClone(participant_plan));
    }

    constexpr const char* mode_diagnostic =
        "lifecycle operation is not admitted by its explicit batch mode";
    for (const int group : {0, 1, 0, 1})
      exercise_lifecycle(
          component, electrons,
          [&]() { component.prepareGroup(electrons, group); },
          lifecycle_case.prepare_group ? nullptr : mode_diagnostic);
    for (int repeat = 0; repeat < 3; ++repeat)
      exercise_lifecycle(
          component, electrons, [&]() { component.completeUpdates(); },
          lifecycle_case.complete_updates ? nullptr : mode_diagnostic);

    if (lifecycle_case.prepare_group && lifecycle_case.complete_updates &&
        !lifecycle_case.scalar_value_compatibility)
    {
      // Model-wide counters can describe transactions in another crowd.  They
      // neither revoke authority from nor get consumed by this idle clone.
      REQUIRE(testing::TestPsiFormerWF::plannedProposalCounts(component) ==
              std::array<std::size_t, 2>{0, 0});
      REQUIRE(testing::TestPsiFormerWF::registerPlannedSingleTransaction(
          component));
      REQUIRE(testing::TestPsiFormerWF::registerPlannedSelectedTransaction(
          component));
      exercise_lifecycle(
          component, electrons,
          [&]() { component.prepareGroup(electrons, 1); }, nullptr);
      exercise_lifecycle(
          component, electrons, [&]() { component.completeUpdates(); },
          nullptr);
      CHECK(testing::TestPsiFormerWF::plannedProposalCounts(component) ==
            std::array<std::size_t, 2>{1, 1});
      testing::TestPsiFormerWF::unregisterPlannedSelectedTransaction(component);
      testing::TestPsiFormerWF::unregisterPlannedSingleTransaction(component);
      REQUIRE(testing::TestPsiFormerWF::plannedProposalCounts(component) ==
              std::array<std::size_t, 2>{0, 0});

      // Deliberate model-version drift proves that lifecycle validation does
      // not acquire numerical authority or synchronize stale clone caches.
      const testing::PsiFormerScalarStateSnapshot state_before_drift =
          testing::TestPsiFormerWF::scalarStateSnapshot(component);
      const testing::PsiFormerWorkspaceDiagnostics workspace_before_drift =
          testing::TestPsiFormerWF::directWorkspaceDiagnostics(component);
      const std::size_t parameter_version_before_drift =
          component.parameterVersion();
      const std::size_t drifted_parameter_version =
          testing::TestPsiFormerWF::advanceParameterVersion(component);
      REQUIRE(drifted_parameter_version > parameter_version_before_drift);
      REQUIRE(component.parameterVersion() == drifted_parameter_version);
      REQUIRE(state_before_drift.observed_parameter_version !=
              drifted_parameter_version);
      REQUIRE(testing::TestPsiFormerWF::scalarStateMatches(
          component, state_before_drift));
      checkScalarWorkspaceStorage(
          testing::TestPsiFormerWF::directWorkspaceDiagnostics(component),
          workspace_before_drift);
      exercise_lifecycle(
          component, electrons,
          [&]() { component.prepareGroup(electrons, 0); }, nullptr);
      exercise_lifecycle(
          component, electrons, [&]() { component.completeUpdates(); },
          nullptr);
      CHECK(component.parameterVersion() == drifted_parameter_version);
      CHECK(testing::TestPsiFormerWF::scalarStateMatches(
          component, state_before_drift));
      checkScalarWorkspaceStorage(
          testing::TestPsiFormerWF::directWorkspaceDiagnostics(component),
          workspace_before_drift);
    }

    if (lifecycle_case.prepare_group)
    {
      for (const int invalid_group : {-1, electrons.groups()})
        exercise_lifecycle(
            component, electrons,
            [&]() { component.prepareGroup(electrons, invalid_group); },
            "received an invalid group index");
      exercise_lifecycle(
          component, foreign_electrons,
          [&]() { component.prepareGroup(foreign_electrons, 0); },
          "received a foreign ParticleSet");
    }

    // Rebinding after preparation invalidates the retained exact ParticleSet
    // evidence even when the replacement has an otherwise compatible shape.
    if (lifecycle_case.prepare_group || lifecycle_case.complete_updates)
    {
      testing::TestPsiFormerWF::bindParticleSetForTesting(component,
                                                           foreign_electrons);
      if (lifecycle_case.prepare_group)
        exercise_lifecycle(
            component, foreign_electrons,
            [&]() { component.prepareGroup(foreign_electrons, 0); },
            "requires an exactly prepared clone");
      if (lifecycle_case.complete_updates)
        exercise_lifecycle(
            component, foreign_electrons,
            [&]() { component.completeUpdates(); },
            "requires an exactly prepared clone");
      testing::TestPsiFormerWF::bindParticleSetForTesting(component,
                                                           electrons);
      REQUIRE(component.hasPreparedBatchExecutionClone(participant_plan));
    }

    if (lifecycle_case.prepare_group || lifecycle_case.complete_updates)
    {
      electrons.makeMove(
          0, ParticleSet::SingleParticlePos{0.013, -0.009, 0.007});
      if (lifecycle_case.prepare_group)
        exercise_lifecycle(
            component, electrons,
            [&]() { component.prepareGroup(electrons, 0); },
            "requires an inactive ParticleSet move");
      if (lifecycle_case.complete_updates)
        exercise_lifecycle(
            component, electrons, [&]() { component.completeUpdates(); },
            "requires an inactive ParticleSet move");
      electrons.rejectMove(0);

      testing::TestPsiFormerWF::markScalarProposalPending(component, 0);
      if (lifecycle_case.prepare_group)
        exercise_lifecycle(
            component, electrons,
            [&]() { component.prepareGroup(electrons, 0); },
            "requires absent proposal state");
      if (lifecycle_case.complete_updates)
        exercise_lifecycle(
            component, electrons, [&]() { component.completeUpdates(); },
            "requires absent proposal state");
      testing::TestPsiFormerWF::clearProposal(component);
    }

    std::unique_ptr<WaveFunctionComponent> clone_storage =
        component.makeClone(electrons);
    auto* clone = dynamic_cast<PsiFormerWF*>(clone_storage.get());
    REQUIRE(clone != nullptr);
    REQUIRE(testing::TestPsiFormerWF::hasBatchExecutionPlan(*clone));
    REQUIRE_FALSE(clone->hasPreparedBatchExecutionClone(participant_plan));
    if (lifecycle_case.prepare_group)
      exercise_lifecycle(
          *clone, electrons, [&]() { clone->prepareGroup(electrons, 0); },
          "requires an exactly prepared clone");
    if (lifecycle_case.complete_updates)
      exercise_lifecycle(
          *clone, electrons, [&]() { clone->completeUpdates(); },
          "requires an exactly prepared clone");
  }
}

TEST_CASE("PsiFormer planned walker-buffer public APIs obey exact selected modes",
          "[wavefunction][psiformer][batch_memory][walker_modes]")
{
  const auto require_atomic_mode_rejection =
      [](PsiFormerWF& component,
         ParticleSet& particles,
         PsiFormerWF::WFBufferType& buffer,
         auto&& operation) {
        const testing::PsiFormerScalarStateSnapshot component_before =
            testing::TestPsiFormerWF::scalarStateSnapshot(component);
        const ScalarParticleStateSnapshot particles_before =
            captureScalarParticleState(particles);
        const testing::PsiFormerWorkspaceDiagnostics workspace_before =
            testing::TestPsiFormerWF::directWorkspaceDiagnostics(component);
        const WalkerBufferStateSnapshot buffer_before =
            captureWalkerBufferState(buffer);
        const std::size_t parameter_version_before = component.parameterVersion();

        CHECK_THROWS_WITH(
            operation(),
            Catch::Matchers::ContainsSubstring(
                "walker-buffer operation is not admitted by its explicit batch mode"));

        CHECK(testing::TestPsiFormerWF::scalarStateMatches(component,
                                                            component_before));
        checkScalarParticleState(particles, particles_before);
        checkScalarWorkspaceStorage(
            testing::TestPsiFormerWF::directWorkspaceDiagnostics(component),
            workspace_before);
        checkWalkerBufferState(buffer, buffer_before);
        CHECK(component.parameterVersion() == parameter_version_before);
      };

  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;

  SECTION("BUFFER_WRITE alone admits registration and refresh")
  {
    ParticleSet electrons = makeLiHElectrons(simulation_cell);
    std::unique_ptr<ParticleSet> ions = makeLiHIons(simulation_cell);
    PsiFormerWF component("pf_walker_modes_write", files.parameters.string(),
                          files.configuration.string());
    component.validateSystem(electrons, *ions, "all_electron");
    electrons.update();
    electrons.G = ValueType(0);
    electrons.L = ValueType(0);
    component.evaluateLog(electrons, electrons.G, electrons.L);

    BatchExecutionRequirements requirements;
    component.contributeBatchExecutionRequirements(requirements);
    requirements.require(BatchExecutionMode::BUFFER_WRITE);
    const std::string participant_id =
        "test/psiformer/walker-modes/write";
    const auto plan = makeClonePreparationTestPlan(
        component, requirements, participant_id, "walker-modes-write-v1");
    REQUIRE(plan->requirements().requires(BatchExecutionMode::BUFFER_WRITE));
    REQUIRE_FALSE(
        plan->requirements().requires(BatchExecutionMode::BUFFER_READ));
    const BatchExecutionParticipantPlan participant =
        makeBatchExecutionParticipantPlan(plan, participant_id);
    component.bindBatchExecutionPlan(participant);
    component.prepareBatchExecutionClone(participant);
    const testing::PsiFormerScalarStateSnapshot accepted_cache =
        testing::TestPsiFormerWF::scalarStateSnapshot(component);
    REQUIRE(accepted_cache.accepted_value_valid);

    const pf::WalkerBufferLayout layout =
        testing::TestPsiFormerWF::directWorkspaceDiagnostics(component)
            .prepared_walker_buffer_layout;
    REQUIRE_FALSE(layout.empty());

    PsiFormerWF::WFBufferType buffer;
    component.registerData(electrons, buffer);
    REQUIRE(buffer.current() == layout.bulk_bytes);
    REQUIRE(buffer.current_scalar() == layout.scalar_count);
    buffer.allocate();
    buffer.zero();
    buffer.rewind();

    electrons.G = ValueType(0);
    electrons.L = ValueType(0);
    const PsiFormerWF::LogValue refreshed_log =
        component.updateBuffer(electrons, buffer, false);
    CHECK(sameScalarBits(refreshed_log, accepted_cache.log_value));
    CHECK(buffer.current() == layout.bulk_bytes);
    CHECK(buffer.current_scalar() == layout.scalar_count);
    CHECK(testing::TestPsiFormerWF::scalarStateMatches(component,
                                                        accepted_cache));
    for (std::size_t electron = 0; electron < electrons.G.size(); ++electron)
    {
      for (std::size_t dimension = 0; dimension < OHMMS_DIM; ++dimension)
        CHECK(electrons.G[electron][dimension] ==
              accepted_cache.accepted_gradient[electron][dimension]);
      CHECK(electrons.L[electron] ==
            accepted_cache.accepted_laplacian[electron]);
    }

    buffer.rewind();
    require_atomic_mode_rejection(component, electrons, buffer, [&]() {
      component.copyFromBuffer(electrons, buffer);
    });
  }

  SECTION("BUFFER_READ alone admits restoration")
  {
    ParticleSet electrons = makeLiHElectrons(simulation_cell);
    std::unique_ptr<ParticleSet> ions = makeLiHIons(simulation_cell);
    PsiFormerWF component("pf_walker_modes_read", files.parameters.string(),
                          files.configuration.string());
    component.validateSystem(electrons, *ions, "all_electron");
    electrons.update();
    electrons.G = ValueType(0);
    electrons.L = ValueType(0);
    component.evaluateLog(electrons, electrons.G, electrons.L);

    PsiFormerWF::WFBufferType buffer;
    component.registerData(electrons, buffer);
    const std::size_t record_bulk_end = buffer.current();
    const std::size_t record_scalar_end = buffer.current_scalar();
    buffer.allocate();
    buffer.zero();
    buffer.rewind();
    electrons.G = ValueType(0);
    electrons.L = ValueType(0);
    component.updateBuffer(electrons, buffer, false);
    REQUIRE(buffer.current() == record_bulk_end);
    REQUIRE(buffer.current_scalar() == record_scalar_end);

    BatchExecutionRequirements requirements;
    component.contributeBatchExecutionRequirements(requirements);
    requirements.require(BatchExecutionMode::BUFFER_READ);
    const std::string participant_id =
        "test/psiformer/walker-modes/read";
    const auto plan = makeClonePreparationTestPlan(
        component, requirements, participant_id, "walker-modes-read-v1");
    REQUIRE(plan->requirements().requires(BatchExecutionMode::BUFFER_READ));
    REQUIRE_FALSE(
        plan->requirements().requires(BatchExecutionMode::BUFFER_WRITE));
    const BatchExecutionParticipantPlan participant =
        makeBatchExecutionParticipantPlan(plan, participant_id);
    component.bindBatchExecutionPlan(participant);
    component.prepareBatchExecutionClone(participant);
    const testing::PsiFormerScalarStateSnapshot accepted_cache =
        testing::TestPsiFormerWF::scalarStateSnapshot(component);
    REQUIRE(accepted_cache.accepted_value_valid);

    testing::TestPsiFormerWF::poisonAcceptedStateForRestore(component);
    buffer.rewind();
    component.copyFromBuffer(electrons, buffer);
    CHECK(buffer.current() == record_bulk_end);
    CHECK(buffer.current_scalar() == record_scalar_end);
    CHECK(testing::TestPsiFormerWF::scalarStateMatches(component,
                                                        accepted_cache));

    PsiFormerWF::WFBufferType sizing_buffer;
    require_atomic_mode_rejection(component, electrons, sizing_buffer, [&]() {
      component.registerData(electrons, sizing_buffer);
    });

    buffer.rewind();
    require_atomic_mode_rejection(component, electrons, buffer, [&]() {
      static_cast<void>(component.updateBuffer(electrons, buffer, false));
    });
  }

  SECTION("a hard plan without buffer modes rejects every public buffer API")
  {
    ParticleSet electrons = makeLiHElectrons(simulation_cell);
    std::unique_ptr<ParticleSet> ions = makeLiHIons(simulation_cell);
    PsiFormerWF component("pf_walker_modes_none", files.parameters.string(),
                          files.configuration.string());
    component.validateSystem(electrons, *ions, "all_electron");
    electrons.update();
    electrons.G = ValueType(0);
    electrons.L = ValueType(0);
    component.evaluateLog(electrons, electrons.G, electrons.L);

    PsiFormerWF::WFBufferType buffer;
    component.registerData(electrons, buffer);
    buffer.allocate();
    buffer.zero();
    buffer.rewind();
    electrons.G = ValueType(0);
    electrons.L = ValueType(0);
    component.updateBuffer(electrons, buffer, false);

    BatchExecutionRequirements requirements;
    component.contributeBatchExecutionRequirements(requirements);
    const std::string participant_id =
        "test/psiformer/walker-modes/none";
    const auto plan = makeClonePreparationTestPlan(
        component, requirements, participant_id, "walker-modes-none-v1");
    REQUIRE_FALSE(
        plan->requirements().requires(BatchExecutionMode::BUFFER_READ));
    REQUIRE_FALSE(
        plan->requirements().requires(BatchExecutionMode::BUFFER_WRITE));
    const BatchExecutionParticipantPlan participant =
        makeBatchExecutionParticipantPlan(plan, participant_id);
    component.bindBatchExecutionPlan(participant);
    component.prepareBatchExecutionClone(participant);
    REQUIRE(testing::TestPsiFormerWF::directWorkspaceDiagnostics(component)
                .prepared_walker_buffer_layout.empty());

    PsiFormerWF::WFBufferType sizing_buffer;
    require_atomic_mode_rejection(component, electrons, sizing_buffer, [&]() {
      component.registerData(electrons, sizing_buffer);
    });

    buffer.rewind();
    require_atomic_mode_rejection(component, electrons, buffer, [&]() {
      static_cast<void>(component.updateBuffer(electrons, buffer, false));
    });

    buffer.rewind();
    require_atomic_mode_rejection(component, electrons, buffer, [&]() {
      component.copyFromBuffer(electrons, buffer);
    });
  }
}

TEST_CASE("PsiFormer planned walker-buffer refresh rejects finite additive overflow",
          "[wavefunction][psiformer][batch_memory][walker_transaction][closure]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  ParticleSet electrons = makeLiHElectrons(simulation_cell);
  std::unique_ptr<ParticleSet> ions = makeLiHIons(simulation_cell);
  PsiFormerWF component("pf_walker_refresh_closure",
                        files.parameters.string(),
                        files.configuration.string());
  component.validateSystem(electrons, *ions, "all_electron");
  electrons.update();
  electrons.G = ValueType(0);
  electrons.L = ValueType(0);
  component.evaluateLog(electrons, electrons.G, electrons.L);

  BatchExecutionRequirements requirements;
  component.contributeBatchExecutionRequirements(requirements);
  requirements.require(BatchExecutionMode::BUFFER_WRITE);
  const std::string participant_id =
      "test/psiformer/walker-refresh-closure";
  const auto plan = makeClonePreparationTestPlan(
      component, requirements, participant_id, "walker-refresh-closure-v1");
  const BatchExecutionParticipantPlan participant_plan =
      makeBatchExecutionParticipantPlan(plan, participant_id);
  component.bindBatchExecutionPlan(participant_plan);
  component.prepareBatchExecutionClone(participant_plan);

  PsiFormerWF::WFBufferType buffer;
  component.registerData(electrons, buffer);
  buffer.allocate();
  buffer.zero();
  buffer.rewind();

  const ValueType largest_finite(
      std::numeric_limits<QMCTraits::RealType>::max());
  testing::TestPsiFormerWF::setAcceptedGradientElement(
      component, 0, 0, largest_finite);
  electrons.G[0][0] = largest_finite;
  const testing::PsiFormerScalarStateSnapshot component_before =
      testing::TestPsiFormerWF::scalarStateSnapshot(component);
  const ScalarParticleStateSnapshot particles_before =
      captureScalarParticleState(electrons);
  const WalkerBufferStateSnapshot buffer_before =
      captureWalkerBufferState(buffer);

  CHECK_THROWS_WITH(
      component.updateBuffer(electrons, buffer, false),
      Catch::Matchers::ContainsSubstring(
          "non-finite gradient input or sum"));
  CHECK(testing::TestPsiFormerWF::scalarStateMatches(component,
                                                      component_before));
  checkScalarParticleState(electrons, particles_before);
  checkWalkerBufferState(buffer, buffer_before);
}

TEST_CASE("PsiFormer plan clear restores the legacy walker-buffer round trip",
          "[wavefunction][psiformer][batch_memory][buffer][closure]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  ParticleSet electrons = makeLiHElectrons(simulation_cell);
  std::unique_ptr<ParticleSet> ions = makeLiHIons(simulation_cell);
  PsiFormerWF component("pf_walker_plan_clear", files.parameters.string(),
                        files.configuration.string());
  component.validateSystem(electrons, *ions, "all_electron");
  electrons.update();

  BatchExecutionRequirements requirements;
  component.contributeBatchExecutionRequirements(requirements);
  const std::string participant_id = "test/psiformer/walker-plan-clear";
  const auto plan = makeClonePreparationTestPlan(
      component, requirements, participant_id, "walker-plan-clear-v1");
  const BatchExecutionParticipantPlan participant_plan =
      makeBatchExecutionParticipantPlan(plan, participant_id);
  component.bindBatchExecutionPlan(participant_plan);
  component.prepareBatchExecutionClone(participant_plan);
  REQUIRE(testing::TestPsiFormerWF::hasBatchExecutionPlan(component));

  component.bindBatchExecutionPlan({});
  REQUIRE_FALSE(testing::TestPsiFormerWF::hasBatchExecutionPlan(component));

  PsiFormerWF::WFBufferType buffer;
  component.registerData(electrons, buffer);
  const std::size_t bulk_end = buffer.current();
  const std::size_t scalar_end = buffer.current_scalar();
  REQUIRE(bulk_end > 0);
  REQUIRE(scalar_end > 0);
  buffer.allocate();
  buffer.zero();

  electrons.G = ValueType(0);
  electrons.L = ValueType(0);
  buffer.rewind();
  const PsiFormerWF::LogValue expected_log =
      component.updateBuffer(electrons, buffer, true);
  const ParticleSet::ParticleGradient expected_gradient = electrons.G;
  const ParticleSet::ParticleLaplacian expected_laplacian = electrons.L;
  const testing::PsiFormerScalarStateSnapshot expected_cache =
      testing::TestPsiFormerWF::scalarStateSnapshot(component);
  REQUIRE(buffer.current() == bulk_end);
  REQUIRE(buffer.current_scalar() == scalar_end);

  testing::TestPsiFormerWF::poisonAcceptedStateForRestore(component);
  buffer.rewind();
  component.copyFromBuffer(electrons, buffer);
  REQUIRE(buffer.current() == bulk_end);
  REQUIRE(buffer.current_scalar() == scalar_end);
  CHECK(testing::TestPsiFormerWF::acceptedSpatialStateMatches(
      component, expected_cache));
  const testing::PsiFormerScalarStateSnapshot restored_cache =
      testing::TestPsiFormerWF::scalarStateSnapshot(component);
  CHECK(restored_cache.accepted_value_valid);
  CHECK(sameScalarBits(restored_cache.current_sign,
                       expected_cache.current_sign));
  CHECK(sameScalarBits(restored_cache.log_value,
                       expected_cache.log_value));

  electrons.G = ValueType(0);
  electrons.L = ValueType(0);
  buffer.rewind();
  const PsiFormerWF::LogValue restored_log =
      component.updateBuffer(electrons, buffer, false);
  CHECK(sameScalarBits(restored_log, expected_log));
  REQUIRE(electrons.G.size() == expected_gradient.size());
  REQUIRE(electrons.L.size() == expected_laplacian.size());
  for (std::size_t electron = 0; electron < expected_gradient.size();
       ++electron)
  {
    for (std::size_t dimension = 0; dimension < OHMMS_DIM; ++dimension)
      CHECK(sameScalarBits(electrons.G[electron][dimension],
                           expected_gradient[electron][dimension]));
    CHECK(sameScalarBits(electrons.L[electron],
                         expected_laplacian[electron]));
  }
  CHECK(buffer.current() == bulk_end);
  CHECK(buffer.current_scalar() == scalar_end);
}

TEST_CASE("PsiFormer planned walker parser classifies records without mutation",
          "[wavefunction][psiformer][batch_memory][walker_parser]")
{
  using Classification =
      testing::TestPsiFormerWF::WalkerBufferClassification;
  using Fault = testing::TestPsiFormerWF::PlannedWalkerBufferFault;
  using BetweenPhaseFault =
      testing::TestPsiFormerWF::PlannedWalkerBufferBetweenPhaseFault;
  using Operation =
      testing::TestPsiFormerWF::PlannedWalkerBufferOperation;
  using Scalar = QMCTraits::FullPrecRealType;

  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  ParticleSet electrons = makeLiHElectrons(simulation_cell);
  std::unique_ptr<ParticleSet> ions = makeLiHIons(simulation_cell);
  PsiFormerWF component("pf_planned_walker_parser",
                        files.parameters.string(),
                        files.configuration.string());
  component.validateSystem(electrons, *ions, "all_electron");
  electrons.update();

  // Build a legacy record between independent prefix and suffix records.  The
  // parser must hash only this component slice while preserving outer bytes.
  PsiFormerWF::WFBufferType buffer;
  PsiFormerWF::GradType prefix_gradient;
  PsiFormerWF::GradType suffix_gradient;
  for (std::size_t dimension = 0; dimension < OHMMS_DIM; ++dimension)
  {
    prefix_gradient[dimension] = ValueType(0.125 * (dimension + 1));
    suffix_gradient[dimension] = ValueType(-0.25 * (dimension + 1));
  }
  std::array<Scalar, 2> prefix_scalars{Scalar(3.25), Scalar(-1.75)};
  std::array<Scalar, 2> suffix_scalars{Scalar(-7.5), Scalar(9.125)};
  buffer.add(&prefix_gradient, &prefix_gradient + 1);
  for (Scalar& scalar : prefix_scalars)
    buffer.add(scalar);
  const std::size_t component_bulk_entry = buffer.current();
  const std::size_t component_scalar_entry = buffer.current_scalar();
  component.registerData(electrons, buffer);
  const std::size_t component_bulk_end = buffer.current();
  const std::size_t component_scalar_end = buffer.current_scalar();
  buffer.add(&suffix_gradient, &suffix_gradient + 1);
  for (Scalar& scalar : suffix_scalars)
    buffer.add(scalar);
  buffer.allocate();

  buffer.rewind();
  buffer.put(&prefix_gradient, &prefix_gradient + 1);
  for (Scalar& scalar : prefix_scalars)
    buffer.put(scalar);
  REQUIRE(buffer.current() == component_bulk_entry);
  REQUIRE(buffer.current_scalar() == component_scalar_entry);
  electrons.G = ValueType(0);
  electrons.L = ValueType(0);
  component.updateBuffer(electrons, buffer, false);
  REQUIRE(buffer.current() == component_bulk_end);
  REQUIRE(buffer.current_scalar() == component_scalar_end);
  buffer.put(&suffix_gradient, &suffix_gradient + 1);
  for (Scalar& scalar : suffix_scalars)
    buffer.put(scalar);
  buffer.rewind(component_bulk_entry, component_scalar_entry);

  BatchExecutionRequirements requirements;
  component.contributeBatchExecutionRequirements(requirements);
  requirements.require(BatchExecutionMode::BUFFER_READ);
  requirements.require(BatchExecutionMode::BUFFER_WRITE);
  const std::string participant_id = "test/psiformer/walker-parser";
  const auto plan = makeClonePreparationTestPlan(
      component, requirements, participant_id, "walker-parser-v1");
  const BatchExecutionParticipantPlan participant_plan =
      makeBatchExecutionParticipantPlan(plan, participant_id);
  component.bindBatchExecutionPlan(participant_plan);
  component.prepareBatchExecutionClone(participant_plan);

  const auto workspace =
      testing::TestPsiFormerWF::directWorkspaceDiagnostics(component);
  const pf::WalkerBufferLayout layout = workspace.prepared_walker_buffer_layout;
  REQUIRE_FALSE(layout.empty());
  REQUIRE(component_bulk_end - component_bulk_entry == layout.bulk_bytes);
  REQUIRE(component_scalar_end - component_scalar_entry ==
          layout.scalar_count);

  // Exercise all three typed preflights. Registration uses a nonzero aligned
  // sizing prefix, while READ and WRITE observe the same allocated record but
  // retain operation-distinct storage identities.
  PsiFormerWF::WFBufferType sizing_buffer;
  sizing_buffer.add(&prefix_gradient, &prefix_gradient + 1);
  for (Scalar& scalar : prefix_scalars)
    sizing_buffer.add(scalar);
  const auto registration_preflight =
      testing::TestPsiFormerWF::inspectPlannedWalkerBufferPreflight(
          component, electrons, sizing_buffer, Operation::REGISTER);
  const auto read_preflight =
      testing::TestPsiFormerWF::inspectPlannedWalkerBufferPreflight(
          component, electrons, buffer, Operation::READ);
  const auto write_preflight =
      testing::TestPsiFormerWF::inspectPlannedWalkerBufferPreflight(
          component, electrons, buffer, Operation::WRITE);
  CHECK(registration_preflight.bulk_cursor == component_bulk_entry);
  CHECK(registration_preflight.scalar_cursor == component_scalar_entry);
  CHECK(registration_preflight.storage_fingerprint != 0);
  CHECK(registration_preflight.input_fingerprint != 0);
  CHECK(read_preflight.bulk_cursor == component_bulk_entry);
  CHECK(read_preflight.scalar_cursor == component_scalar_entry);
  CHECK(write_preflight.bulk_cursor == component_bulk_entry);
  CHECK(write_preflight.scalar_cursor == component_scalar_entry);
  CHECK(registration_preflight.storage_fingerprint !=
        write_preflight.storage_fingerprint);
  CHECK(read_preflight.storage_fingerprint !=
        write_preflight.storage_fingerprint);

  PsiFormerWF::WFBufferType second_sizing_buffer;
  second_sizing_buffer.add(&prefix_gradient, &prefix_gradient + 1);
  for (Scalar& scalar : prefix_scalars)
    second_sizing_buffer.add(scalar);
  const auto second_registration_preflight =
      testing::TestPsiFormerWF::inspectPlannedWalkerBufferPreflight(
          component, electrons, second_sizing_buffer, Operation::REGISTER);
  CHECK(second_registration_preflight.storage_fingerprint !=
        registration_preflight.storage_fingerprint);

  sizing_buffer.rewind(component_bulk_entry + layout.alignment,
                       component_scalar_entry + 1);
  const auto shifted_registration_preflight =
      testing::TestPsiFormerWF::inspectPlannedWalkerBufferPreflight(
          component, electrons, sizing_buffer, Operation::REGISTER);
  CHECK(shifted_registration_preflight.storage_fingerprint !=
        registration_preflight.storage_fingerprint);
  sizing_buffer.rewind(component_bulk_entry, component_scalar_entry);

  const WalkerBufferStateSnapshot sizing_before =
      captureWalkerBufferState(sizing_buffer);
  sizing_buffer.rewind(component_bulk_entry + 1, component_scalar_entry);
  CHECK_THROWS_WITH(
      testing::TestPsiFormerWF::inspectPlannedWalkerBufferPreflight(
          component, electrons, sizing_buffer, Operation::REGISTER),
      Catch::Matchers::ContainsSubstring("registration cursor is misaligned"));
  sizing_buffer.rewind(component_bulk_entry, component_scalar_entry);
  checkWalkerBufferState(sizing_buffer, sizing_before);

  const testing::PsiFormerScalarStateSnapshot component_before =
      testing::TestPsiFormerWF::scalarStateSnapshot(component);
  const ScalarParticleStateSnapshot particles_before =
      captureScalarParticleState(electrons);
  const auto require_isolated = [&](const PsiFormerWF::WFBufferType& candidate,
                                    const WalkerBufferStateSnapshot& before) {
    CHECK(testing::TestPsiFormerWF::scalarStateMatches(component,
                                                        component_before));
    checkScalarParticleState(electrons, particles_before);
    checkWalkerBufferState(candidate, before);
  };

  const WalkerBufferStateSnapshot valid_before =
      captureWalkerBufferState(buffer);
  const auto current = testing::TestPsiFormerWF::inspectPlannedWalkerBuffer(
      component, electrons, buffer);
  CHECK(current.classification == Classification::RESTORABLE);
  CHECK(current.next_bulk_cursor == component_bulk_end);
  CHECK(current.next_scalar_cursor == component_scalar_end);
  CHECK(current.storage_fingerprint != 0);
  CHECK(current.input_fingerprint != 0);
  CHECK(current.content_fingerprint != 0);
  require_isolated(buffer, valid_before);

  // Bytes owned by neighboring components are outside the content identity.
  const std::size_t scalar_offset = static_cast<std::size_t>(
      reinterpret_cast<const char*>(buffer.Scalar_ptr) -
      buffer.myData.data());
  REQUIRE(component_bulk_entry > 0);
  REQUIRE(component_bulk_end < scalar_offset);
  const std::size_t scalar_capacity =
      (buffer.myData.size() - scalar_offset) / sizeof(Scalar);
  REQUIRE(component_scalar_entry == prefix_scalars.size());
  REQUIRE(component_scalar_end + suffix_scalars.size() <= scalar_capacity);
  const char prefix_byte = buffer.myData[0];
  const char suffix_byte = buffer.myData[component_bulk_end];
  const std::array<Scalar, 2> outer_prefix_scalars{
      buffer.Scalar_ptr[0], buffer.Scalar_ptr[1]};
  const std::array<Scalar, 2> outer_suffix_scalars{
      buffer.Scalar_ptr[component_scalar_end],
      buffer.Scalar_ptr[component_scalar_end + 1]};
  buffer.myData[0] ^= char{0x1};
  buffer.myData[component_bulk_end] ^= char{0x2};
  buffer.Scalar_ptr[0] += Scalar(1);
  buffer.Scalar_ptr[1] -= Scalar(2);
  buffer.Scalar_ptr[component_scalar_end] -= Scalar(3);
  buffer.Scalar_ptr[component_scalar_end + 1] += Scalar(4);
  const auto outside_slice =
      testing::TestPsiFormerWF::inspectPlannedWalkerBuffer(
          component, electrons, buffer);
  CHECK(outside_slice.storage_fingerprint == current.storage_fingerprint);
  CHECK(outside_slice.input_fingerprint == current.input_fingerprint);
  CHECK(outside_slice.content_fingerprint == current.content_fingerprint);
  buffer.myData[0] = prefix_byte;
  buffer.myData[component_bulk_end] = suffix_byte;
  buffer.Scalar_ptr[0] = outer_prefix_scalars[0];
  buffer.Scalar_ptr[1] = outer_prefix_scalars[1];
  buffer.Scalar_ptr[component_scalar_end] = outer_suffix_scalars[0];
  buffer.Scalar_ptr[component_scalar_end + 1] = outer_suffix_scalars[1];
  require_isolated(buffer, valid_before);

  // An entirely zero component record is the one canonical registration
  // sentinel.  Every aligned bulk byte, including padding, participates.
  PsiFormerWF::WFBufferType zero_record(buffer);
  std::memset(zero_record.myData.data() + component_bulk_entry, 0,
              layout.bulk_bytes);
  std::memset(zero_record.Scalar_ptr + component_scalar_entry, 0,
              layout.scalar_count * sizeof(Scalar));
  const WalkerBufferStateSnapshot zero_before =
      captureWalkerBufferState(zero_record);
  const auto zero = testing::TestPsiFormerWF::inspectPlannedWalkerBuffer(
      component, electrons, zero_record);
  CHECK(zero.classification == Classification::VALID_ZERO);
  CHECK(zero.next_bulk_cursor == component_bulk_end);
  CHECK(zero.next_scalar_cursor == component_scalar_end);
  require_isolated(zero_record, zero_before);

  // Structurally valid but nonrestorable metadata must classify as stale,
  // including both INVALID/VALUE requirements and canonical zero amplitude.
  const auto require_stale = [&](PsiFormerWF::WFBufferType& stale_record) {
    const WalkerBufferStateSnapshot stale_before =
        captureWalkerBufferState(stale_record);
    const auto inspected =
        testing::TestPsiFormerWF::inspectPlannedWalkerBuffer(
            component, electrons, stale_record);
    CHECK(inspected.classification == Classification::VALID_STALE);
    require_isolated(stale_record, stale_before);
  };

  PsiFormerWF::WFBufferType invalid_requirement(buffer);
  setWalkerBufferInteger(invalid_requirement, component_scalar_entry, 4, 0);
  require_stale(invalid_requirement);

  PsiFormerWF::WFBufferType value_requirement(buffer);
  setWalkerBufferInteger(value_requirement, component_scalar_entry, 4, 1);
  require_stale(value_requirement);

  PsiFormerWF::WFBufferType stale_version(buffer);
  setWalkerBufferInteger(stale_version, component_scalar_entry, 8,
                         current.parameter_version + 1);
  require_stale(stale_version);

  PsiFormerWF::WFBufferType stale_configuration(buffer);
  setWalkerBufferInteger(stale_configuration, component_scalar_entry, 10,
                         current.configuration_identity ^ UINT64_C(1));
  require_stale(stale_configuration);

  PsiFormerWF::WFBufferType zero_amplitude(buffer);
  zero_amplitude.Scalar_ptr[component_scalar_entry + 14] = Scalar(0);
  zero_amplitude.Scalar_ptr[component_scalar_entry + 15] =
      -std::numeric_limits<Scalar>::infinity();
  zero_amplitude.Scalar_ptr[component_scalar_entry + 16] = Scalar(0);
  require_stale(zero_amplitude);

  // A live coordinate change leaves the record structurally valid but stale,
  // and changes only the input fingerprint domain after AoS/SoA resynchronization.
  const ParticleSet::PosType original_position = electrons.R[0];
  electrons.R[0][0] = original_position[0] + ParticleSet::RealType(0.03125);
  electrons.update();
  const ScalarParticleStateSnapshot moved_particles =
      captureScalarParticleState(electrons);
  const auto moved_configuration =
      testing::TestPsiFormerWF::inspectPlannedWalkerBuffer(
          component, electrons, buffer);
  CHECK(moved_configuration.classification == Classification::VALID_STALE);
  CHECK(moved_configuration.storage_fingerprint == current.storage_fingerprint);
  CHECK(moved_configuration.content_fingerprint == current.content_fingerprint);
  CHECK(moved_configuration.input_fingerprint != current.input_fingerprint);
  CHECK(testing::TestPsiFormerWF::scalarStateMatches(component,
                                                      component_before));
  checkScalarParticleState(electrons, moved_particles);
  checkWalkerBufferState(buffer, valid_before);
  electrons.R[0] = original_position;
  electrons.update();
  require_isolated(buffer, valid_before);

  // Wrong schemas, partial zero sentinels, malformed limbs, nonfinite bulk
  // products, and incoherent amplitudes are malformed rather than stale.
  const auto require_malformed = [&](PsiFormerWF::WFBufferType& malformed,
                                     const char* message) {
    const WalkerBufferStateSnapshot malformed_before =
        captureWalkerBufferState(malformed);
    CHECK_THROWS_WITH(
        testing::TestPsiFormerWF::inspectPlannedWalkerBuffer(
            component, electrons, malformed),
        Catch::Matchers::ContainsSubstring(message));
    require_isolated(malformed, malformed_before);
  };

  PsiFormerWF::WFBufferType wrong_schema(buffer);
  setWalkerBufferInteger(wrong_schema, component_scalar_entry, 2,
                         current.schema + 1);
  require_malformed(wrong_schema, "schema is unsupported");

  PsiFormerWF::WFBufferType partial_zero(zero_record);
  partial_zero.myData[component_bulk_entry] = char{1};
  require_malformed(partial_zero, "partial zero sentinel");

  const std::size_t gradient_payload_bytes =
      layout.electrons * sizeof(PsiFormerWF::GradType);
  const std::size_t laplacian_payload_bytes =
      layout.electrons * sizeof(ValueType);
  if (layout.gradient_bytes > gradient_payload_bytes ||
      layout.laplacian_bytes > laplacian_payload_bytes)
  {
    PsiFormerWF::WFBufferType nonzero_padding(zero_record);
    const std::size_t padding_offset =
        layout.gradient_bytes > gradient_payload_bytes
        ? gradient_payload_bytes
        : layout.laplacian_offset + laplacian_payload_bytes;
    nonzero_padding.myData[component_bulk_entry + padding_offset] = char{1};
    require_malformed(nonzero_padding, "partial zero sentinel");
  }

  PsiFormerWF::WFBufferType fractional_magic(buffer);
  fractional_magic.Scalar_ptr[component_scalar_entry] = Scalar(0.5);
  require_malformed(fractional_magic, "fractional magic limb");

  PsiFormerWF::WFBufferType negative_zero_magic(buffer);
  const std::uint64_t negative_zero_bits = UINT64_C(0x8000000000000000);
  static_assert(sizeof(negative_zero_bits) == sizeof(Scalar));
  std::memcpy(negative_zero_magic.Scalar_ptr + component_scalar_entry,
              std::addressof(negative_zero_bits), sizeof(negative_zero_bits));
  require_malformed(negative_zero_magic, "invalid magic limb");

  PsiFormerWF::WFBufferType negative_zero_amplitude(buffer);
  std::memcpy(
      negative_zero_amplitude.Scalar_ptr + component_scalar_entry + 14,
      std::addressof(negative_zero_bits), sizeof(negative_zero_bits));
  negative_zero_amplitude.Scalar_ptr[component_scalar_entry + 15] =
      -std::numeric_limits<Scalar>::infinity();
  negative_zero_amplitude.Scalar_ptr[component_scalar_entry + 16] =
      Scalar(0);
  require_malformed(negative_zero_amplitude, "amplitude is invalid");

  PsiFormerWF::WFBufferType nonfinite_gradient(buffer);
  PsiFormerWF::GradType bad_gradient;
  bad_gradient = ValueType(0);
  bad_gradient[0] =
      ValueType(std::numeric_limits<double>::quiet_NaN());
  std::memcpy(nonfinite_gradient.myData.data() + component_bulk_entry,
              std::addressof(bad_gradient), sizeof(bad_gradient));
  require_malformed(nonfinite_gradient, "non-finite gradient");

  PsiFormerWF::WFBufferType invalid_amplitude(buffer);
  invalid_amplitude.Scalar_ptr[component_scalar_entry + 14] = Scalar(0);
  invalid_amplitude.Scalar_ptr[component_scalar_entry + 15] = Scalar(-1);
  invalid_amplitude.Scalar_ptr[component_scalar_entry + 16] = Scalar(0);
  require_malformed(invalid_amplitude, "amplitude is invalid");

  // Each inspection-only mutation occurs after the real Phase-A parse and
  // before Phase B recaptures its evidence. RAII restoration makes the same
  // record immediately retryable without rewinding or re-preparing.
  struct BetweenPhaseCase
  {
    BetweenPhaseFault fault;
    const char* message;
  };
  constexpr std::array<BetweenPhaseCase, 7> between_phase_cases{
      BetweenPhaseCase{BetweenPhaseFault::STORAGE_POINTER,
                       "storage has no scalar region"},
      BetweenPhaseCase{BetweenPhaseFault::BULK_CURSOR,
                       "bulk cursor overflowed"},
      BetweenPhaseCase{BetweenPhaseFault::RECORD_CONTENT,
                       "record content changed after Phase A"},
      BetweenPhaseCase{BetweenPhaseFault::PARTICLE_INPUT,
                       "input evidence changed after Phase A"},
      BetweenPhaseCase{BetweenPhaseFault::PLAN_BINDING,
                       "has no bound batch plan"},
      BetweenPhaseCase{BetweenPhaseFault::LAYOUT_EVIDENCE,
                       "requires an exactly prepared clone"},
      BetweenPhaseCase{BetweenPhaseFault::PREPARED_STORAGE_EVIDENCE,
                       "requires an exactly prepared clone"}};
  const auto workspace_before_between_phase =
      testing::TestPsiFormerWF::directWorkspaceDiagnostics(component);
  for (const BetweenPhaseCase& phase_case : between_phase_cases)
  {
    CAPTURE(static_cast<int>(phase_case.fault));
    CHECK_THROWS_WITH(
        testing::TestPsiFormerWF::probePlannedWalkerBufferBetweenPhaseFault(
            component, electrons, buffer, phase_case.fault),
        Catch::Matchers::ContainsSubstring(phase_case.message));
    require_isolated(buffer, valid_before);
    checkScalarWorkspaceStorage(
        testing::TestPsiFormerWF::directWorkspaceDiagnostics(component),
        workspace_before_between_phase);
    CHECK(testing::TestPsiFormerWF::hasBatchExecutionPlan(component));

    const auto retry =
        testing::TestPsiFormerWF::inspectPlannedWalkerBuffer(
            component, electrons, buffer);
    CHECK(retry.classification == Classification::RESTORABLE);
    CHECK(retry.storage_fingerprint == current.storage_fingerprint);
    CHECK(retry.input_fingerprint == current.input_fingerprint);
    CHECK(retry.content_fingerprint == current.content_fingerprint);
    require_isolated(buffer, valid_before);
  }

  // Prepared-layout corruption and every structural/alias seam are rejected
  // against a snapshot, leaving the live PooledMemory object untouched.
  pf::WalkerBufferLayout wrong_layout = layout;
  ++wrong_layout.total_bytes;
  testing::TestPsiFormerWF::setPreparedWalkerBufferLayout(component,
                                                           wrong_layout);
  CHECK_THROWS(testing::TestPsiFormerWF::inspectPlannedWalkerBuffer(
      component, electrons, buffer));
  require_isolated(buffer, valid_before);
  testing::TestPsiFormerWF::setPreparedWalkerBufferLayout(component, layout);
  CHECK(testing::TestPsiFormerWF::inspectPlannedWalkerBuffer(
            component, electrons, buffer)
            .classification == Classification::RESTORABLE);
  require_isolated(buffer, valid_before);

  const std::array<Fault, 18> faults{
      Fault::NULL_BACKING,
      Fault::NULL_SCALAR,
      Fault::SIZE_EXCEEDS_CAPACITY,
      Fault::ATTACHED_STORAGE,
      Fault::MISALIGNED_BACKING,
      Fault::SCALAR_BEFORE_BACKING,
      Fault::SCALAR_AFTER_BACKING,
      Fault::MISALIGNED_SCALAR,
      Fault::MISALIGNED_BULK_CURSOR,
      Fault::BULK_CURSOR_BEYOND_DOMAIN,
      Fault::SCALAR_CURSOR_BEYOND_DOMAIN,
      Fault::BULK_CURSOR_OVERFLOW,
      Fault::SCALAR_CURSOR_OVERFLOW,
      Fault::TRUNCATED_BULK,
      Fault::TRUNCATED_SCALAR,
      Fault::ACCEPTED_STORAGE_ALIAS,
      Fault::PROPOSED_STORAGE_ALIAS,
      Fault::PARTICLE_STORAGE_ALIAS};
  for (const Fault fault : faults)
  {
    CAPTURE(static_cast<int>(fault));
    CHECK_THROWS(testing::TestPsiFormerWF::probePlannedWalkerBufferFault(
        component, electrons, buffer, fault));
    require_isolated(buffer, valid_before);
  }
}

TEST_CASE("PsiFormer planned public walker-buffer transactions are atomic",
          "[wavefunction][psiformer][batch_memory][walker_transaction]")
{
  using Classification =
      testing::TestPsiFormerWF::WalkerBufferClassification;
  using LateFault = testing::TestPsiFormerWF::PlannedWalkerBufferLateFault;
  using Scalar = QMCTraits::FullPrecRealType;

  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  ParticleSet electrons = makeLiHElectrons(simulation_cell);
  std::unique_ptr<ParticleSet> ions = makeLiHIons(simulation_cell);
  PsiFormerWF component("pf_planned_walker_transaction",
                        files.parameters.string(),
                        files.configuration.string());
  component.validateSystem(electrons, *ions, "all_electron");
  electrons.update();

  // Establish the current FULL_VGL cache required by the no-evaluation write
  // route before binding the hard plan. Clone preparation must retain it.
  electrons.G = ValueType(0);
  electrons.L = ValueType(0);
  component.evaluateLog(electrons, electrons.G, electrons.L);

  BatchExecutionRequirements requirements;
  component.contributeBatchExecutionRequirements(requirements);
  requirements.require(BatchExecutionMode::BUFFER_READ);
  requirements.require(BatchExecutionMode::BUFFER_WRITE);
  const std::string participant_id =
      "test/psiformer/walker-transaction";
  const auto plan = makeClonePreparationTestPlan(
      component, requirements, participant_id, "walker-transaction-v1");
  const BatchExecutionParticipantPlan participant_plan =
      makeBatchExecutionParticipantPlan(plan, participant_id);
  component.bindBatchExecutionPlan(participant_plan);
  component.prepareBatchExecutionClone(participant_plan);

  const testing::PsiFormerWorkspaceDiagnostics workspace =
      testing::TestPsiFormerWF::directWorkspaceDiagnostics(component);
  const pf::WalkerBufferLayout layout = workspace.prepared_walker_buffer_layout;
  REQUIRE_FALSE(layout.empty());
  const testing::PsiFormerScalarStateSnapshot expected_cache =
      testing::TestPsiFormerWF::scalarStateSnapshot(component);
  REQUIRE(expected_cache.accepted_value_valid);

  // Planned registration starts after independent bulk/scalar prefixes.  Its
  // true late seam proves that neither cursor moves until the final publish.
  PsiFormerWF::GradType prefix_gradient;
  PsiFormerWF::GradType suffix_gradient;
  for (std::size_t dimension = 0; dimension < OHMMS_DIM; ++dimension)
  {
    prefix_gradient[dimension] = ValueType(0.375 * (dimension + 1));
    suffix_gradient[dimension] = ValueType(-0.625 * (dimension + 1));
  }
  std::array<Scalar, 2> prefix_scalars{Scalar(3.5), Scalar(-2.25)};
  std::array<Scalar, 2> suffix_scalars{Scalar(-8.75), Scalar(11.5)};

  PsiFormerWF::WFBufferType buffer;
  buffer.add(&prefix_gradient, &prefix_gradient + 1);
  for (Scalar& scalar : prefix_scalars)
    buffer.add(scalar);
  const std::size_t component_bulk_entry = buffer.current();
  const std::size_t component_scalar_entry = buffer.current_scalar();
  const WalkerBufferStateSnapshot registration_before =
      captureWalkerBufferState(buffer);
  const ScalarParticleStateSnapshot registration_particles_before =
      captureScalarParticleState(electrons);

  testing::TestPsiFormerWF::setPlannedWalkerBufferLateFault(
      component, LateFault::REGISTER);
  CHECK_THROWS_WITH(
      component.registerData(electrons, buffer),
      Catch::Matchers::ContainsSubstring(
          "injected late planned walker-buffer registration failure"));
  checkWalkerBufferState(buffer, registration_before);
  checkScalarParticleState(electrons, registration_particles_before);
  CHECK(testing::TestPsiFormerWF::scalarStateMatches(component,
                                                      expected_cache));

  testing::TestPsiFormerWF::setPlannedWalkerBufferLateFault(
      component, LateFault::NONE);
  component.registerData(electrons, buffer);
  const std::size_t component_bulk_end = buffer.current();
  const std::size_t component_scalar_end = buffer.current_scalar();
  CHECK(component_bulk_end == component_bulk_entry + layout.bulk_bytes);
  CHECK(component_scalar_end ==
        component_scalar_entry + layout.scalar_count);
  checkScalarParticleState(electrons, registration_particles_before);
  CHECK(testing::TestPsiFormerWF::scalarStateMatches(component,
                                                      expected_cache));

  buffer.add(&suffix_gradient, &suffix_gradient + 1);
  for (Scalar& scalar : suffix_scalars)
    buffer.add(scalar);
  buffer.allocate();
  buffer.zero();

  // Materialize neighboring records before any component write so every
  // success and failure below can prove strict prefix/suffix isolation.
  buffer.rewind();
  buffer.put(&prefix_gradient, &prefix_gradient + 1);
  for (Scalar& scalar : prefix_scalars)
    buffer.put(scalar);
  REQUIRE(buffer.current() == component_bulk_entry);
  REQUIRE(buffer.current_scalar() == component_scalar_entry);
  buffer.rewind(component_bulk_end, component_scalar_end);
  buffer.put(&suffix_gradient, &suffix_gradient + 1);
  for (Scalar& scalar : suffix_scalars)
    buffer.put(scalar);

  const auto check_outer_records = [&](const PsiFormerWF::WFBufferType& value) {
    REQUIRE(value.Scalar_ptr != nullptr);
    CHECK(std::memcmp(value.myData.data(), std::addressof(prefix_gradient),
                      sizeof(prefix_gradient)) == 0);
    CHECK(std::memcmp(value.myData.data() + component_bulk_end,
                      std::addressof(suffix_gradient),
                      sizeof(suffix_gradient)) == 0);
    for (std::size_t scalar = 0; scalar < prefix_scalars.size(); ++scalar)
      CHECK(sameScalarBits(value.Scalar_ptr[scalar],
                           prefix_scalars[scalar]));
    for (std::size_t scalar = 0; scalar < suffix_scalars.size(); ++scalar)
      CHECK(sameScalarBits(
          value.Scalar_ptr[component_scalar_end + scalar],
          suffix_scalars[scalar]));
  };
  check_outer_records(buffer);

  // WRITE owns this complete slice and must not parse arbitrary destination
  // bytes. Seed independent corruption in bulk, magic, and another scalar
  // field; the successful refresh below must replace all of it canonically.
  std::memset(buffer.myData.data() + component_bulk_entry, 0x5a,
              layout.bulk_bytes);
  PsiFormerWF::GradType nonfinite_destination_gradient;
  nonfinite_destination_gradient = ValueType(0);
  nonfinite_destination_gradient[0] =
      ValueType(std::numeric_limits<double>::quiet_NaN());
  std::memcpy(buffer.myData.data() + component_bulk_entry,
              std::addressof(nonfinite_destination_gradient),
              sizeof(nonfinite_destination_gradient));
  setWalkerBufferInteger(buffer, component_scalar_entry, 0,
                         UINT64_C(0x0123456789abcdef));
  buffer.Scalar_ptr[component_scalar_entry + 4] = Scalar(0.5);
  PsiFormerWF::GradType observed_destination_gradient;
  std::memcpy(std::addressof(observed_destination_gradient),
              buffer.myData.data() + component_bulk_entry,
              sizeof(observed_destination_gradient));
  CHECK(sameScalarBits(observed_destination_gradient[0],
                       nonfinite_destination_gradient[0]));
  CHECK(buffer.Scalar_ptr[component_scalar_entry] ==
        Scalar(UINT64_C(0x89abcdef)));
  CHECK(buffer.Scalar_ptr[component_scalar_entry + 4] == Scalar(0.5));
  check_outer_records(buffer);

  // Seed nonzero aggregate derivatives.  A from-scratch request and the true
  // late refresh seam must leave them, the cache, and all bytes unchanged.
  for (std::size_t electron = 0; electron < electrons.G.size(); ++electron)
  {
    for (std::size_t dimension = 0; dimension < OHMMS_DIM; ++dimension)
      electrons.G[electron][dimension] =
          ValueType(0.03125 * (1 + 3 * electron + dimension));
    electrons.L[electron] = ValueType(-0.046875 * (1 + electron));
  }
  buffer.rewind(component_bulk_entry, component_scalar_entry);
  const WalkerBufferStateSnapshot from_scratch_buffer_before =
      captureWalkerBufferState(buffer);
  const ScalarParticleStateSnapshot from_scratch_particles_before =
      captureScalarParticleState(electrons);
  CHECK_THROWS_WITH(
      component.updateBuffer(electrons, buffer, true),
      Catch::Matchers::ContainsSubstring("does not support from_scratch"));
  checkWalkerBufferState(buffer, from_scratch_buffer_before);
  checkScalarParticleState(electrons, from_scratch_particles_before);
  CHECK(testing::TestPsiFormerWF::scalarStateMatches(component,
                                                      expected_cache));

  testing::TestPsiFormerWF::setPlannedWalkerBufferLateFault(
      component, LateFault::REFRESH);
  CHECK_THROWS_WITH(
      component.updateBuffer(electrons, buffer, false),
      Catch::Matchers::ContainsSubstring(
          "injected late planned walker-buffer refresh failure"));
  checkWalkerBufferState(buffer, from_scratch_buffer_before);
  checkScalarParticleState(electrons, from_scratch_particles_before);
  CHECK(testing::TestPsiFormerWF::scalarStateMatches(component,
                                                      expected_cache));
  check_outer_records(buffer);

  testing::TestPsiFormerWF::setPlannedWalkerBufferLateFault(
      component, LateFault::NONE);
  const PsiFormerWF::LogValue update_log =
      component.updateBuffer(electrons, buffer, false);
  CHECK(sameScalarBits(update_log, expected_cache.log_value));
  CHECK(buffer.current() == component_bulk_end);
  CHECK(buffer.current_scalar() == component_scalar_end);
  CHECK(testing::TestPsiFormerWF::scalarStateMatches(component,
                                                      expected_cache));
  for (std::size_t electron = 0; electron < electrons.G.size(); ++electron)
  {
    for (std::size_t dimension = 0; dimension < OHMMS_DIM; ++dimension)
      CHECK(electrons.G[electron][dimension] ==
            from_scratch_particles_before.gradients[electron][dimension] +
                expected_cache.accepted_gradient[electron][dimension]);
    CHECK(electrons.L[electron] ==
          from_scratch_particles_before.laplacians[electron] +
              expected_cache.accepted_laplacian[electron]);
  }
  check_outer_records(buffer);
  const ScalarParticleStateSnapshot refreshed_particles =
      captureScalarParticleState(electrons);

  buffer.rewind(component_bulk_entry, component_scalar_entry);
  const auto valid_record =
      testing::TestPsiFormerWF::inspectPlannedWalkerBuffer(component,
                                                            electrons,
                                                            buffer);
  REQUIRE(valid_record.classification == Classification::RESTORABLE);

  // Destroy the resident accepted cache, then fail after Phase B.  Exact state
  // and cursors must survive; clearing the seam makes the same record retryable.
  testing::TestPsiFormerWF::poisonAcceptedStateForRestore(component);
  const testing::PsiFormerScalarStateSnapshot poisoned_cache =
      testing::TestPsiFormerWF::scalarStateSnapshot(component);
  const WalkerBufferStateSnapshot restore_buffer_before =
      captureWalkerBufferState(buffer);

  CHECK_THROWS_WITH(
      component.updateBuffer(electrons, buffer, false),
      Catch::Matchers::ContainsSubstring(
          "requires a current FULL_VGL accepted state"));
  checkWalkerBufferState(buffer, restore_buffer_before);
  checkScalarParticleState(electrons, refreshed_particles);
  CHECK(testing::TestPsiFormerWF::scalarStateMatches(component,
                                                      poisoned_cache));

  testing::TestPsiFormerWF::setPlannedWalkerBufferLateFault(
      component, LateFault::RESTORE);
  CHECK_THROWS_WITH(
      component.copyFromBuffer(electrons, buffer),
      Catch::Matchers::ContainsSubstring(
          "injected late planned walker-buffer restoration failure"));
  checkWalkerBufferState(buffer, restore_buffer_before);
  checkScalarParticleState(electrons, refreshed_particles);
  CHECK(testing::TestPsiFormerWF::scalarStateMatches(component,
                                                      poisoned_cache));

  testing::TestPsiFormerWF::setPlannedWalkerBufferLateFault(
      component, LateFault::NONE);
  component.copyFromBuffer(electrons, buffer);
  CHECK(buffer.current() == component_bulk_end);
  CHECK(buffer.current_scalar() == component_scalar_end);
  checkScalarParticleState(electrons, refreshed_particles);
  check_outer_records(buffer);
  const testing::PsiFormerScalarStateSnapshot restored_cache =
      testing::TestPsiFormerWF::scalarStateSnapshot(component);
  CHECK(restored_cache.accepted_value_valid);
  CHECK(sameScalarBits(restored_cache.current_sign,
                       expected_cache.current_sign));
  CHECK(sameScalarBits(restored_cache.log_value,
                       expected_cache.log_value));
  CHECK(restored_cache.accepted_configuration_identity ==
        expected_cache.accepted_configuration_identity);
  CHECK(restored_cache.accepted_parameter_version ==
        expected_cache.accepted_parameter_version);
  CHECK(restored_cache.accepted_state_requirement ==
        expected_cache.accepted_state_requirement);
  CHECK(restored_cache.observed_parameter_version ==
        expected_cache.observed_parameter_version);
  CHECK(testing::TestPsiFormerWF::acceptedSpatialStateMatches(
      component, expected_cache));

  // A malformed public restore is rejected before any component, particle, or
  // cursor publication, and therefore remains byte-for-byte retry isolated.
  PsiFormerWF::WFBufferType malformed_record(buffer);
  malformed_record.rewind(component_bulk_entry, component_scalar_entry);
  setWalkerBufferInteger(malformed_record, component_scalar_entry, 2,
                         valid_record.schema + 1);
  const WalkerBufferStateSnapshot malformed_buffer_before =
      captureWalkerBufferState(malformed_record);
  const testing::PsiFormerScalarStateSnapshot malformed_cache_before =
      testing::TestPsiFormerWF::scalarStateSnapshot(component);
  CHECK_THROWS_WITH(
      component.copyFromBuffer(electrons, malformed_record),
      Catch::Matchers::ContainsSubstring("schema is unsupported"));
  checkWalkerBufferState(malformed_record, malformed_buffer_before);
  checkScalarParticleState(electrons, refreshed_particles);
  CHECK(testing::TestPsiFormerWF::scalarStateMatches(
      component, malformed_cache_before));
  check_outer_records(malformed_record);

  // Canonical zero and structurally valid stale records consume their exact
  // slices but publish only the invalid marker; spatial payload stays reusable.
  PsiFormerWF::WFBufferType zero_record(buffer);
  zero_record.rewind(component_bulk_entry, component_scalar_entry);
  std::memset(zero_record.myData.data() + component_bulk_entry, 0,
              layout.bulk_bytes);
  std::memset(zero_record.Scalar_ptr + component_scalar_entry, 0,
              layout.scalar_count * sizeof(Scalar));
  const testing::PsiFormerScalarStateSnapshot before_zero =
      testing::TestPsiFormerWF::scalarStateSnapshot(component);
  component.copyFromBuffer(electrons, zero_record);
  CHECK(zero_record.current() == component_bulk_end);
  CHECK(zero_record.current_scalar() == component_scalar_end);
  checkScalarParticleState(electrons, refreshed_particles);
  check_outer_records(zero_record);
  CHECK(testing::TestPsiFormerWF::acceptedSpatialStateMatches(component,
                                                               before_zero));
  const testing::PsiFormerScalarStateSnapshot after_zero =
      testing::TestPsiFormerWF::scalarStateSnapshot(component);
  CHECK_FALSE(after_zero.accepted_value_valid);
  CHECK(after_zero.current_sign == 1.0);
  CHECK(sameScalarBits(after_zero.log_value, PsiFormerWF::LogValue(0)));
  CHECK(after_zero.accepted_configuration_identity == 0);
  CHECK(after_zero.accepted_state_requirement == 0);
  CHECK(after_zero.accepted_parameter_version ==
        after_zero.observed_parameter_version);

  // Reestablish a live key, then replace its resident G/L with distinct
  // canaries. A stale consume must actively publish the canonical invalid
  // metadata while leaving these preexisting spatial arrays byte-exact.
  buffer.rewind(component_bulk_entry, component_scalar_entry);
  component.copyFromBuffer(electrons, buffer);
  testing::TestPsiFormerWF::poisonAcceptedSpatialStateForStaleConsume(
      component);
  const testing::PsiFormerScalarStateSnapshot before_stale =
      testing::TestPsiFormerWF::scalarStateSnapshot(component);
  REQUIRE(before_stale.accepted_value_valid);
  REQUIRE_FALSE(before_stale.accepted_gradient.empty());
  CHECK_FALSE(sameScalarBits(before_stale.accepted_gradient.front()[0],
                             expected_cache.accepted_gradient.front()[0]));

  PsiFormerWF::WFBufferType stale_record(buffer);
  stale_record.rewind(component_bulk_entry, component_scalar_entry);
  setWalkerBufferInteger(stale_record, component_scalar_entry, 8,
                         valid_record.parameter_version + 1);
  component.copyFromBuffer(electrons, stale_record);
  CHECK(stale_record.current() == component_bulk_end);
  CHECK(stale_record.current_scalar() == component_scalar_end);
  checkScalarParticleState(electrons, refreshed_particles);
  check_outer_records(stale_record);
  CHECK(testing::TestPsiFormerWF::acceptedSpatialStateMatches(component,
                                                               before_stale));
  const testing::PsiFormerScalarStateSnapshot after_stale =
      testing::TestPsiFormerWF::scalarStateSnapshot(component);
  CHECK_FALSE(after_stale.accepted_value_valid);
  CHECK(after_stale.current_sign == 1.0);
  CHECK(sameScalarBits(after_stale.log_value, PsiFormerWF::LogValue(0)));
  CHECK(after_stale.accepted_configuration_identity == 0);
  CHECK(after_stale.accepted_state_requirement == 0);
  CHECK(after_stale.accepted_parameter_version ==
        after_stale.observed_parameter_version);
}

TEST_CASE("PsiFormer planned scalar VALUE matches independent direct evaluation",
          "[wavefunction][psiformer][batch_memory][scalar_value]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  ParticleSet planned_electrons = makeLiHElectrons(simulation_cell);
  ParticleSet legacy_electrons = makeLiHElectrons(simulation_cell);
  std::unique_ptr<ParticleSet> ions = makeLiHIons(simulation_cell);
  const std::size_t electron_count =
      static_cast<std::size_t>(planned_electrons.getTotalNum());
  REQUIRE(electron_count > 2);

  PsiFormerWF planned("pf_planned_scalar_value", files.parameters.string(),
                      files.configuration.string());
  PsiFormerWF legacy("pf_legacy_scalar_value", files.parameters.string(),
                     files.configuration.string());
  planned.validateSystem(planned_electrons, *ions, "all_electron");

  // Give the component nontrivial accepted spatial state before binding the
  // plan so the read-only scalar queries must preserve meaningful payload.
  planned_electrons.G = ValueType(0);
  planned_electrons.L = ValueType(0);
  planned.evaluateLog(planned_electrons, planned_electrons.G,
                      planned_electrons.L);
  const PreparedScalarPlan prepared = prepareScalarValuePlan(
      planned, "test/psiformer/scalar-value-correctness",
      "scalar-value-correctness-v1");
  static_cast<void>(prepared);

  const ParticleSet::SingleParticlePos common_position{0.43, -0.31, 0.27};
  planned_electrons.makeVirtualMoves(common_position);
  legacy_electrons.makeVirtualMoves(common_position);
  std::vector<ValueType> planned_all_to_one(electron_count,
                                             ValueType(17));
  std::vector<ValueType> legacy_all_to_one(electron_count, ValueType(0));
  const PlannedScalarTransactionSnapshot all_to_one_before =
      capturePlannedScalarTransaction(planned, planned_electrons,
                                      planned_all_to_one);
  legacy.evaluateRatiosAlltoOne(legacy_electrons, legacy_all_to_one);
  planned.evaluateRatiosAlltoOne(planned_electrons, planned_all_to_one);
  checkPlannedScalarTransaction(planned, planned_electrons,
                                planned_all_to_one, all_to_one_before,
                                nullptr, false);
  bool changed_all_to_one = false;
  for (std::size_t electron = 0; electron < electron_count; ++electron)
  {
    checkScalarRatio(planned_all_to_one[electron],
                     legacy_all_to_one[electron]);
    changed_all_to_one = changed_all_to_one ||
        std::abs(planned_all_to_one[electron] - ValueType(1)) > 1.0e-8;
  }
  CHECK(changed_all_to_one);

  // Reuse the same prepared VALUE owner at a singleton, an interior prefix,
  // and its exact N_e replacement envelope. VirtualParticleSet cannot safely
  // construct an empty move set; the direct adapter tests cover q == 0.
  // Alternating with all-to-one catches unsafe retained logical extents.
  const std::array<std::size_t, 3> virtual_counts{1, 2, electron_count};
  for (std::size_t cycle = 0; cycle < 2; ++cycle)
    for (const std::size_t virtual_count : virtual_counts)
    {
      CAPTURE(cycle, virtual_count);
      std::vector<ParticleSet::SingleParticlePos> displacements;
      displacements.reserve(virtual_count);
      for (std::size_t move = 0; move < virtual_count; ++move)
      {
        const double scale = static_cast<double>((cycle + 1) * (move + 1));
        displacements.emplace_back(0.019 * scale, -0.013 * scale,
                                   0.011 * scale);
      }
      VirtualParticleSet planned_virtual(planned_electrons);
      VirtualParticleSet legacy_virtual(legacy_electrons);
      planned_virtual.makeMoves(planned_electrons, 1, displacements);
      legacy_virtual.makeMoves(legacy_electrons, 1, displacements);
      std::vector<ValueType> planned_ratios(virtual_count, ValueType(-23));
      std::vector<ValueType> legacy_ratios(virtual_count, ValueType(0));
      const PlannedScalarTransactionSnapshot before =
          capturePlannedScalarTransaction(planned, planned_electrons,
                                          planned_ratios, &planned_virtual);

      legacy.evaluateRatios(legacy_virtual, legacy_ratios);
      planned.evaluateRatios(planned_virtual, planned_ratios);
      checkPlannedScalarTransaction(planned, planned_electrons,
                                    planned_ratios, before,
                                    &planned_virtual, false);
      bool changed_virtual = false;
      for (std::size_t move = 0; move < virtual_count; ++move)
      {
        checkScalarRatio(planned_ratios[move], legacy_ratios[move]);
        changed_virtual = changed_virtual ||
            std::abs(planned_ratios[move] - ValueType(1)) > 1.0e-10;
      }
      CHECK(changed_virtual);

      std::fill(planned_all_to_one.begin(), planned_all_to_one.end(),
                ValueType(29));
      legacy.evaluateRatiosAlltoOne(legacy_electrons, legacy_all_to_one);
      const PlannedScalarTransactionSnapshot alternating_before =
          capturePlannedScalarTransaction(planned, planned_electrons,
                                          planned_all_to_one);
      planned.evaluateRatiosAlltoOne(planned_electrons,
                                     planned_all_to_one);
      checkPlannedScalarTransaction(planned, planned_electrons,
                                    planned_all_to_one,
                                    alternating_before, nullptr, false);
      for (std::size_t electron = 0; electron < electron_count; ++electron)
        checkScalarRatio(planned_all_to_one[electron],
                         legacy_all_to_one[electron]);
    }
}

TEST_CASE("PsiFormer planned scalar VALUE late failures are atomic and retryable",
          "[wavefunction][psiformer][batch_memory][scalar_value][atomic]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  ParticleSet electrons = makeLiHElectrons(simulation_cell);
  ParticleSet reference_electrons = makeLiHElectrons(simulation_cell);
  std::unique_ptr<ParticleSet> ions = makeLiHIons(simulation_cell);
  const std::size_t electron_count =
      static_cast<std::size_t>(electrons.getTotalNum());

  PsiFormerWF component("pf_scalar_value_late_failure",
                        files.parameters.string(),
                        files.configuration.string());
  PsiFormerWF reference("pf_scalar_value_late_reference",
                        files.parameters.string(),
                        files.configuration.string());
  component.validateSystem(electrons, *ions, "all_electron");
  electrons.G = ValueType(0);
  electrons.L = ValueType(0);
  component.evaluateLog(electrons, electrons.G, electrons.L);
  const PreparedScalarPlan prepared = prepareScalarValuePlan(
      component, "test/psiformer/scalar-value-late-failure",
      "scalar-value-late-failure-v1");
  static_cast<void>(prepared);

  const ParticleSet::SingleParticlePos common_position{0.39, -0.23, 0.34};
  electrons.makeVirtualMoves(common_position);
  reference_electrons.makeVirtualMoves(common_position);
  std::vector<ValueType> expected_all_to_one(electron_count);
  reference.evaluateRatiosAlltoOne(reference_electrons,
                                   expected_all_to_one);
  std::vector<ValueType> all_to_one(electron_count, ValueType(-31));
  const PlannedScalarTransactionSnapshot all_to_one_before =
      capturePlannedScalarTransaction(component, electrons, all_to_one);
  testing::TestPsiFormerWF::failPlannedScalarValueBeforePublish(component,
                                                                true);
  CHECK_THROWS_WITH(
      component.evaluateRatiosAlltoOne(electrons, all_to_one),
      Catch::Matchers::ContainsSubstring(
          "planned scalar VALUE pre-publication failure"));
  testing::TestPsiFormerWF::failPlannedScalarValueBeforePublish(component,
                                                                false);
  checkPlannedScalarTransaction(component, electrons, all_to_one,
                                all_to_one_before);
  component.evaluateRatiosAlltoOne(electrons, all_to_one);
  for (std::size_t electron = 0; electron < electron_count; ++electron)
    checkScalarRatio(all_to_one[electron], expected_all_to_one[electron]);

  const std::vector<ParticleSet::SingleParticlePos> displacements{
      {0.023, -0.017, 0.012}, {-0.031, 0.014, 0.027}};
  VirtualParticleSet virtual_particles(electrons);
  VirtualParticleSet reference_virtual(reference_electrons);
  virtual_particles.makeMoves(electrons, 2, displacements);
  reference_virtual.makeMoves(reference_electrons, 2, displacements);
  std::vector<ValueType> expected_virtual(displacements.size());
  reference.evaluateRatios(reference_virtual, expected_virtual);
  std::vector<ValueType> virtual_ratios(displacements.size(), ValueType(37));
  const PlannedScalarTransactionSnapshot virtual_before =
      capturePlannedScalarTransaction(component, electrons, virtual_ratios,
                                      &virtual_particles);
  testing::TestPsiFormerWF::failPlannedScalarValueBeforePublish(component,
                                                                true);
  CHECK_THROWS_WITH(
      component.evaluateRatios(virtual_particles, virtual_ratios),
      Catch::Matchers::ContainsSubstring(
          "planned scalar VALUE pre-publication failure"));
  testing::TestPsiFormerWF::failPlannedScalarValueBeforePublish(component,
                                                                false);
  checkPlannedScalarTransaction(component, electrons, virtual_ratios,
                                virtual_before, &virtual_particles);
  component.evaluateRatios(virtual_particles, virtual_ratios);
  for (std::size_t move = 0; move < virtual_ratios.size(); ++move)
    checkScalarRatio(virtual_ratios[move], expected_virtual[move]);
}

TEST_CASE("PsiFormer planned scalar VALUE rejects malformed provenance atomically",
          "[wavefunction][psiformer][batch_memory][scalar_value][preflight]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  ParticleSet electrons = makeLiHElectrons(simulation_cell);
  ParticleSet foreign_electrons = makeLiHElectrons(simulation_cell);
  std::unique_ptr<ParticleSet> ions = makeLiHIons(simulation_cell);
  const std::size_t electron_count =
      static_cast<std::size_t>(electrons.getTotalNum());
  REQUIRE(electron_count > 1);

  PsiFormerWF component("pf_scalar_value_preflight",
                        files.parameters.string(),
                        files.configuration.string());
  component.validateSystem(electrons, *ions, "all_electron");
  electrons.G = ValueType(0);
  electrons.L = ValueType(0);
  component.evaluateLog(electrons, electrons.G, electrons.L);
  const PreparedScalarPlan prepared = prepareScalarValuePlan(
      component, "test/psiformer/scalar-value-preflight",
      "scalar-value-preflight-v1");
  static_cast<void>(prepared);

  const ParticleSet::SingleParticlePos common_position{0.41, -0.26, 0.33};
  electrons.makeVirtualMoves(common_position);
  foreign_electrons.makeVirtualMoves(common_position);

  const auto require_failure = [&](ParticleSet& reference,
                                   std::vector<ValueType>& output,
                                   const VirtualParticleSet* virtual_particles,
                                   auto&& invocation) {
    const PlannedScalarTransactionSnapshot before =
        capturePlannedScalarTransaction(component, reference, output,
                                        virtual_particles);
    CHECK_THROWS(invocation());
    checkPlannedScalarTransaction(component, reference, output, before,
                                  virtual_particles, true, true);
  };

  std::vector<ValueType> wrong_size(electron_count - 1, ValueType(5));
  require_failure(electrons, wrong_size, nullptr, [&]() {
    component.evaluateRatiosAlltoOne(electrons, wrong_size);
  });

  std::vector<ValueType> all_to_one(electron_count, ValueType(7));
  require_failure(foreign_electrons, all_to_one, nullptr, [&]() {
    component.evaluateRatiosAlltoOne(foreign_electrons, all_to_one);
  });

  electrons.setSpinor(true);
  require_failure(electrons, all_to_one, nullptr, [&]() {
    component.evaluateRatiosAlltoOne(electrons, all_to_one);
  });
  electrons.setSpinor(false);

  const ParticleSet::RealType saved_coordinate = electrons.R[0][0];
  electrons.R[0][0] += ParticleSet::RealType(0.125);
  require_failure(electrons, all_to_one, nullptr, [&]() {
    component.evaluateRatiosAlltoOne(electrons, all_to_one);
  });
  electrons.R[0][0] = saved_coordinate;

  electrons.R[0][0] =
      std::numeric_limits<ParticleSet::RealType>::infinity();
  electrons.update();
  require_failure(electrons, all_to_one, nullptr, [&]() {
    component.evaluateRatiosAlltoOne(electrons, all_to_one);
  });
  electrons.R[0][0] = saved_coordinate;
  electrons.update();
  electrons.makeVirtualMoves(common_position);

  const int saved_group = electrons.GroupID[0];
  electrons.GroupID[0] = 1;
  require_failure(electrons, all_to_one, nullptr, [&]() {
    component.evaluateRatiosAlltoOne(electrons, all_to_one);
  });
  electrons.GroupID[0] = saved_group;

  electrons.makeMove(0, ParticleSet::SingleParticlePos{0.01, -0.02, 0.03});
  require_failure(electrons, all_to_one, nullptr, [&]() {
    component.evaluateRatiosAlltoOne(electrons, all_to_one);
  });
  electrons.rejectMove(0);
  electrons.makeVirtualMoves(common_position);

  electrons.makeVirtualMoves(ParticleSet::SingleParticlePos{
      std::numeric_limits<ParticleSet::RealType>::infinity(), 0.0, 0.0});
  require_failure(electrons, all_to_one, nullptr, [&]() {
    component.evaluateRatiosAlltoOne(electrons, all_to_one);
  });
  electrons.makeVirtualMoves(common_position);

  testing::TestPsiFormerWF::markScalarProposalPending(component, 0);
  require_failure(electrons, all_to_one, nullptr, [&]() {
    component.evaluateRatiosAlltoOne(electrons, all_to_one);
  });
  testing::TestPsiFormerWF::clearProposal(component);
  testing::TestPsiFormerWF::markPlannedSingleProposalPending(component, 0);
  require_failure(electrons, all_to_one, nullptr, [&]() {
    component.evaluateRatiosAlltoOne(electrons, all_to_one);
  });
  testing::TestPsiFormerWF::clearProposal(component);
  testing::TestPsiFormerWF::markSelectedProposalPending(component);
  require_failure(electrons, all_to_one, nullptr, [&]() {
    component.evaluateRatiosAlltoOne(electrons, all_to_one);
  });
  testing::TestPsiFormerWF::clearProposal(component);

  const auto initialize_two_spin_species = [](ParticleSet& particles) {
    SpeciesSet& species = particles.getSpeciesSet();
    species.addSpecies("u");
    species.addSpecies("d");
    const int mass = species.addAttribute("mass");
    species(mass, 0) = 1.0;
    species(mass, 1) = 1.0;
    particles.resetGroups();
  };

  ParticleSet wrong_count(simulation_cell);
  wrong_count.setName("wrong_scalar_reference");
  wrong_count.create({3, 2});
  initialize_two_spin_species(wrong_count);
  wrong_count.update();
  wrong_count.makeVirtualMoves(common_position);
  testing::TestPsiFormerWF::bindParticleSetForTesting(component, wrong_count);
  std::vector<ValueType> wrong_count_output(
      static_cast<std::size_t>(wrong_count.getTotalNum()), ValueType(11));
  require_failure(wrong_count, wrong_count_output, nullptr, [&]() {
    component.evaluateRatiosAlltoOne(wrong_count, wrong_count_output);
  });
  testing::TestPsiFormerWF::bindParticleSetForTesting(component, electrons);

  ParticleSet wrong_partition(simulation_cell);
  wrong_partition.setName("wrong_scalar_spin_partition");
  wrong_partition.create({1, 3});
  initialize_two_spin_species(wrong_partition);
  for (std::size_t electron = 0; electron < electron_count; ++electron)
    wrong_partition.R[electron] = electrons.R[electron];
  wrong_partition.update();
  wrong_partition.makeVirtualMoves(common_position);
  testing::TestPsiFormerWF::bindParticleSetForTesting(component,
                                                      wrong_partition);
  std::vector<ValueType> wrong_partition_output(electron_count,
                                                ValueType(12));
  require_failure(wrong_partition, wrong_partition_output, nullptr, [&]() {
    component.evaluateRatiosAlltoOne(wrong_partition,
                                     wrong_partition_output);
  });
  testing::TestPsiFormerWF::bindParticleSetForTesting(component, electrons);

  const std::vector<ParticleSet::SingleParticlePos> displacements{
      {0.017, -0.012, 0.009}, {-0.026, 0.018, 0.013}};
  VirtualParticleSet virtual_particles(electrons);
  virtual_particles.makeMoves(electrons, 1, displacements);
  std::vector<ValueType> virtual_ratios(displacements.size(), ValueType(13));

  VirtualParticleSet foreign_virtual(foreign_electrons);
  foreign_virtual.makeMoves(foreign_electrons, 1, displacements);
  require_failure(foreign_electrons, virtual_ratios, &foreign_virtual, [&]() {
    component.evaluateRatios(foreign_virtual, virtual_ratios);
  });

  virtual_particles.refPtcl = -1;
  require_failure(electrons, virtual_ratios, &virtual_particles, [&]() {
    component.evaluateRatios(virtual_particles, virtual_ratios);
  });
  virtual_particles.refPtcl = 1;

  virtual_particles.setSpinor(true);
  require_failure(electrons, virtual_ratios, &virtual_particles, [&]() {
    component.evaluateRatios(virtual_particles, virtual_ratios);
  });
  virtual_particles.setSpinor(false);

  std::vector<ValueType> wrong_virtual_size(displacements.size() - 1,
                                            ValueType(15));
  require_failure(electrons, wrong_virtual_size, &virtual_particles, [&]() {
    component.evaluateRatios(virtual_particles, wrong_virtual_size);
  });

  virtual_particles.R.resize(displacements.size() - 1);
  require_failure(electrons, virtual_ratios, &virtual_particles, [&]() {
    component.evaluateRatios(virtual_particles, virtual_ratios);
  });
  virtual_particles.makeMoves(electrons, 1, displacements);

  const ParticleSet::RealType saved_virtual_coordinate =
      virtual_particles.R[0][0];
  virtual_particles.R[0][0] =
      std::numeric_limits<ParticleSet::RealType>::quiet_NaN();
  virtual_particles.update();
  require_failure(electrons, virtual_ratios, &virtual_particles, [&]() {
    component.evaluateRatios(virtual_particles, virtual_ratios);
  });
  virtual_particles.R[0][0] = saved_virtual_coordinate;
  virtual_particles.update();

  using OutputFault = testing::TestPsiFormerWF::PlannedScalarOutputFault;
  const auto require_output_probe_failure =
      [&](OutputFault fault, const VirtualParticleSet* virtual_input,
          std::vector<ValueType>& output, const char* expected_message) {
        const PlannedScalarTransactionSnapshot before =
            capturePlannedScalarTransaction(component, electrons, output,
                                            virtual_input);
        CHECK_THROWS_WITH(
            testing::TestPsiFormerWF::probePlannedScalarOutput(
                component, electrons, virtual_input, output, fault),
            Catch::Matchers::ContainsSubstring(expected_message));
        checkPlannedScalarTransaction(component, electrons, output, before,
                                      virtual_input, true, true);
      };
  require_output_probe_failure(OutputFault::NULL_STORAGE, nullptr, all_to_one,
                               "planned output storage is null");
  require_output_probe_failure(OutputFault::RANGE_OVERFLOW, nullptr,
                               all_to_one, "output range overflowed");
  require_output_probe_failure(OutputFault::PUBLICATION_ALIAS, nullptr,
                               all_to_one, "aliases prepared scratch");
  require_output_probe_failure(OutputFault::COMPONENT_ALIAS, nullptr,
                               all_to_one, "aliases component state");
  require_output_probe_failure(OutputFault::PARTICLE_ALIAS, nullptr,
                               all_to_one, "aliases ParticleSet state");
  require_output_probe_failure(OutputFault::VIRTUAL_PARTICLE_ALIAS,
                               &virtual_particles, virtual_ratios,
                               "aliases VirtualParticleSet state");

  // Every rejected path leaves the prepared owner immediately reusable.
  std::fill(virtual_ratios.begin(), virtual_ratios.end(), ValueType(19));
  CHECK_NOTHROW(component.evaluateRatios(virtual_particles, virtual_ratios));
  CHECK(std::any_of(virtual_ratios.begin(), virtual_ratios.end(),
                    [](ValueType value) { return value != ValueType(19); }));
}

TEST_CASE("PsiFormer planned scalar VALUE rejects corrupted prepared storage atomically",
          "[wavefunction][psiformer][batch_memory][scalar_value][preflight]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  ParticleSet electrons = makeLiHElectrons(simulation_cell);
  ParticleSet donor_electrons = makeLiHElectrons(simulation_cell);
  std::unique_ptr<ParticleSet> ions = makeLiHIons(simulation_cell);
  const std::size_t electron_count =
      static_cast<std::size_t>(electrons.getTotalNum());

  PsiFormerWF component("pf_scalar_storage_corruption",
                        files.parameters.string(),
                        files.configuration.string());
  PsiFormerWF donor("pf_scalar_storage_donor", files.parameters.string(),
                    files.configuration.string());
  PsiFormerWF empty_owner("pf_scalar_storage_empty", files.parameters.string(),
                          files.configuration.string());
  component.validateSystem(electrons, *ions, "all_electron");
  donor.validateSystem(donor_electrons, *ions, "all_electron");
  const PreparedScalarPlan prepared = prepareScalarValuePlan(
      component, "test/psiformer/scalar-storage-corruption",
      "scalar-storage-corruption-v1");
  const PreparedScalarPlan donor_prepared = prepareScalarValuePlan(
      donor, "test/psiformer/scalar-storage-donor",
      "scalar-storage-donor-v1");
  static_cast<void>(prepared);
  static_cast<void>(donor_prepared);

  electrons.makeVirtualMoves(
      ParticleSet::SingleParticlePos{0.38, -0.24, 0.31});
  std::vector<ValueType> ratios(electron_count, ValueType(79));
  const auto require_failure = [&]() {
    const PlannedScalarTransactionSnapshot before =
        capturePlannedScalarTransaction(component, electrons, ratios);
    CHECK_THROWS(component.evaluateRatiosAlltoOne(electrons, ratios));
    checkPlannedScalarTransaction(component, electrons, ratios, before,
                                  nullptr, true, true);
  };

  // Missing and foreign workspace owners must fail before any numerical work.
  testing::TestPsiFormerWF::swapBatchWorkspaces(component, empty_owner);
  require_failure();
  testing::TestPsiFormerWF::swapBatchWorkspaces(component, empty_owner);

  testing::TestPsiFormerWF::swapBatchWorkspaces(component, donor);
  require_failure();
  testing::TestPsiFormerWF::swapBatchWorkspaces(component, donor);

  // Independently corrupt logical and tile-capacity preparation records while
  // retaining the same workspace allocation, then restore the exact plan.
  using WorkspaceFault =
      testing::TestPsiFormerWF::PreparedScalarWorkspaceFault;
  testing::TestPsiFormerWF::setPreparedScalarWorkspaceFault(
      component, WorkspaceFault::LOGICAL_CAPACITY);
  require_failure();
  testing::TestPsiFormerWF::setPreparedScalarWorkspaceFault(
      component, WorkspaceFault::NONE);

  testing::TestPsiFormerWF::setPreparedScalarWorkspaceFault(
      component, WorkspaceFault::TILE_CAPACITY);
  require_failure();
  testing::TestPsiFormerWF::setPreparedScalarWorkspaceFault(
      component, WorkspaceFault::NONE);

  // Missing, foreign, wrong-extent, and wrong-capacity publication storage are
  // distinct provenance failures even when their element type is compatible.
  auto retained_publication =
      testing::TestPsiFormerWF::takeScalarValuePublication(component);
  const std::vector<ValueType> publication_contents = retained_publication;
  const std::size_t publication_size = retained_publication.size();
  const std::size_t publication_capacity = retained_publication.capacity();
  testing::TestPsiFormerWF::restoreScalarValuePublication(component, {});
  require_failure();
  testing::TestPsiFormerWF::restoreScalarValuePublication(
      component, std::move(retained_publication));

  retained_publication =
      testing::TestPsiFormerWF::takeScalarValuePublication(component);
  testing::TestPsiFormerWF::restoreScalarValuePublication(
      component, std::vector<ValueType>(publication_size, ValueType(83)));
  require_failure();
  auto foreign_publication =
      testing::TestPsiFormerWF::takeScalarValuePublication(component);
  testing::TestPsiFormerWF::restoreScalarValuePublication(
      component, std::move(retained_publication));

  testing::TestPsiFormerWF::resizeScalarValuePublication(
      component, publication_size - 1);
  require_failure();
  retained_publication =
      testing::TestPsiFormerWF::takeScalarValuePublication(component);
  retained_publication.resize(publication_size);
  std::copy(publication_contents.begin(), publication_contents.end(),
            retained_publication.begin());
  testing::TestPsiFormerWF::restoreScalarValuePublication(
      component, std::move(retained_publication));

  retained_publication =
      testing::TestPsiFormerWF::takeScalarValuePublication(component);
  std::vector<ValueType> oversized_publication(publication_size,
                                                ValueType(89));
  oversized_publication.reserve(publication_capacity + 1);
  REQUIRE(oversized_publication.capacity() != publication_capacity);
  testing::TestPsiFormerWF::restoreScalarValuePublication(
      component, std::move(oversized_publication));
  require_failure();
  oversized_publication =
      testing::TestPsiFormerWF::takeScalarValuePublication(component);
  testing::TestPsiFormerWF::restoreScalarValuePublication(
      component, std::move(retained_publication));
  static_cast<void>(foreign_publication);
  static_cast<void>(oversized_publication);

  // All rejected corruptions leave the restored owner immediately usable.
  CHECK_NOTHROW(component.evaluateRatiosAlltoOne(electrons, ratios));
  CHECK(std::any_of(ratios.begin(), ratios.end(), [](ValueType ratio) {
    return ratio != ValueType(79);
  }));
}

TEST_CASE("PsiFormer planned scalar VALUE rejects incompatible plan provenance",
          "[wavefunction][psiformer][batch_memory][scalar_value][preflight]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  std::unique_ptr<ParticleSet> ions = makeLiHIons(simulation_cell);

  const auto require_runtime_rejection =
      [&](const std::string& label,
          BatchExecutionTargetCoordinate target_coordinate,
          const std::string& topology_backend,
          std::optional<std::size_t> device_id,
          const char* expected_message) {
        ParticleSet electrons = makeLiHElectrons(simulation_cell);
        PsiFormerWF component("pf_scalar_plan_" + label,
                              files.parameters.string(),
                              files.configuration.string());
        component.validateSystem(electrons, *ions, "all_electron");
        electrons.G = ValueType(0);
        electrons.L = ValueType(0);
        component.evaluateLog(electrons, electrons.G, electrons.L);

        BatchExecutionRequirements requirements;
        component.contributeBatchExecutionRequirements(requirements);
        requirements.require(BatchExecutionMode::SCALAR_VALUE_COMPATIBILITY);
        const std::string participant_id =
            "test/psiformer/scalar-plan-" + label;
        const auto plan = makeClonePreparationTestPlan(
            component, requirements, participant_id,
            "scalar-plan-" + label + "-v1", 2, 0, target_coordinate,
            topology_backend, device_id);
        const BatchExecutionParticipantPlan participant =
            makeBatchExecutionParticipantPlan(plan, participant_id);
        component.bindBatchExecutionPlan(participant);
        component.prepareBatchExecutionClone(participant);

        electrons.makeVirtualMoves(
            ParticleSet::SingleParticlePos{0.34, -0.27, 0.39});
        std::vector<ValueType> output(
            static_cast<std::size_t>(electrons.getTotalNum()), ValueType(139));
        const PlannedScalarTransactionSnapshot before =
            capturePlannedScalarTransaction(component, electrons, output);
        CHECK_THROWS_WITH(
            component.evaluateRatiosAlltoOne(electrons, output),
            Catch::Matchers::ContainsSubstring(expected_message));
        checkPlannedScalarTransaction(component, electrons, output, before,
                                      nullptr, true, true);
      };

  for (const auto target : {BatchExecutionTargetCoordinate::UNKNOWN,
                            BatchExecutionTargetCoordinate::POS_SPIN})
  {
    CAPTURE(static_cast<int>(target));
    require_runtime_rejection(
        target == BatchExecutionTargetCoordinate::UNKNOWN ? "unknown-target"
                                                          : "spin-target",
        target, "cpu", std::nullopt,
        "requires explicit POS-only target evidence");
  }
  require_runtime_rejection(
      "device-topology", BatchExecutionTargetCoordinate::POS_ONLY,
      "accelerator", std::size_t{0},
      "requires direct nonserialized CPU execution");

  for (const char* backend : {"oracle", "compare"})
  {
    CAPTURE(backend);
    ScopedEnvironmentVariable select_backend("PSIFORMER_VALUE_BACKEND",
                                              backend);
    ParticleSet electrons = makeLiHElectrons(simulation_cell);
    PsiFormerWF component(std::string("pf_scalar_plan_") + backend,
                          files.parameters.string(),
                          files.configuration.string());
    component.validateSystem(electrons, *ions, "all_electron");
    electrons.G = ValueType(0);
    electrons.L = ValueType(0);
    component.evaluateLog(electrons, electrons.G, electrons.L);
    electrons.makeVirtualMoves(
        ParticleSet::SingleParticlePos{0.34, -0.27, 0.39});
    std::vector<ValueType> output(
        static_cast<std::size_t>(electrons.getTotalNum()), ValueType(149));
    const PlannedScalarTransactionSnapshot before =
        capturePlannedScalarTransaction(component, electrons, output);

    BatchExecutionRequirements requirements;
    component.contributeBatchExecutionRequirements(requirements);
    requirements.require(BatchExecutionMode::SCALAR_VALUE_COMPATIBILITY);
    const std::string participant_id =
        std::string("test/psiformer/scalar-plan-") + backend;
    const auto plan = makeClonePreparationTestPlan(
        component, requirements, participant_id,
        std::string("scalar-plan-") + backend + "-v1");
    const BatchExecutionParticipantPlan participant =
        makeBatchExecutionParticipantPlan(plan, participant_id);
    CHECK_THROWS_WITH(
        component.validateBatchExecutionPlanBinding(participant),
        Catch::Matchers::ContainsSubstring(
            "planned VALUE execution requires the direct backend"));
    CHECK_FALSE(testing::TestPsiFormerWF::hasBatchExecutionPlan(component));
    checkPlannedScalarTransaction(component, electrons, output, before,
                                  nullptr, true, true);
  }

  // Serialized topology is rejected transactionally during clone preparation,
  // before any scalar owner or prepared marker can be published.
  ParticleSet serialized_electrons = makeLiHElectrons(simulation_cell);
  PsiFormerWF serialized_component("pf_scalar_plan_serialized",
                                   files.parameters.string(),
                                   files.configuration.string());
  serialized_component.validateSystem(serialized_electrons, *ions,
                                      "all_electron");
  BatchExecutionRequirements serialized_requirements;
  serialized_component.contributeBatchExecutionRequirements(
      serialized_requirements);
  serialized_requirements.require(
      BatchExecutionMode::SCALAR_VALUE_COMPATIBILITY);
  const std::string serialized_participant_id =
      "test/psiformer/scalar-plan-serialized";
  const auto serialized_plan = makeClonePreparationTestPlan(
      serialized_component, serialized_requirements,
      serialized_participant_id, "scalar-plan-serialized-v1", 2, 0,
      BatchExecutionTargetCoordinate::POS_ONLY, "cpu", std::nullopt, true);
  const BatchExecutionParticipantPlan serialized_participant =
      makeBatchExecutionParticipantPlan(serialized_plan,
                                        serialized_participant_id);
  serialized_component.bindBatchExecutionPlan(serialized_participant);
  const auto serialized_before =
      testing::TestPsiFormerWF::directWorkspaceDiagnostics(
          serialized_component);
  CHECK_THROWS_WITH(
      serialized_component.prepareBatchExecutionClone(serialized_participant),
      Catch::Matchers::ContainsSubstring(
          "does not admit serialized-walker execution"));
  checkScalarWorkspaceStorage(
      testing::TestPsiFormerWF::directWorkspaceDiagnostics(
          serialized_component),
      serialized_before);
}

TEST_CASE("PsiFormer planned scalar VALUE Phase-B faults are atomic and retryable",
          "[wavefunction][psiformer][batch_memory][scalar_value][atomic]")
{
  using Fault = testing::TestPsiFormerWF::PlannedScalarValueFault;
  struct FaultCase
  {
    Fault fault;
    const char* label;
    const char* expected_message;
  };
  const std::array<FaultCase, 11> faults{{
      {Fault::RESULT_OWNER, "result owner", "evidence changed during evaluation"},
      {Fault::RESULT_GENERATION, "result generation", "evidence changed during evaluation"},
      {Fault::RESULT_SIZE, "result size", "evidence changed during evaluation"},
      {Fault::RESULT_VERSION, "result version", "result changed before publication"},
      {Fault::RESULT_SIGN, "result sign", "result changed before publication"},
      {Fault::RESULT_LOG_MAGNITUDE, "result log magnitude", "result changed before publication"},
      {Fault::RESULT_RATIO, "result ratio", "PsiFormer batch ratio is non-finite"},
      {Fault::INPUT_FINGERPRINT, "input fingerprint", "evidence changed during evaluation"},
      {Fault::OUTPUT_IDENTITY, "output identity", "planned output storage is null"},
      {Fault::WORKSPACE_EVIDENCE, "workspace evidence",
       "PsiFormer scalar VALUE workspace was not prepared for the bound batch plan"},
      {Fault::PUBLICATION_EVIDENCE, "publication evidence",
       "PsiFormer scalar VALUE workspace was not prepared for the bound batch plan"},
  }};

  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  ParticleSet electrons = makeLiHElectrons(simulation_cell);
  ParticleSet reference_electrons = makeLiHElectrons(simulation_cell);
  std::unique_ptr<ParticleSet> ions = makeLiHIons(simulation_cell);
  const std::size_t electron_count =
      static_cast<std::size_t>(electrons.getTotalNum());
  PsiFormerWF component("pf_scalar_phase_b", files.parameters.string(),
                        files.configuration.string());
  PsiFormerWF reference("pf_scalar_phase_b_reference",
                        files.parameters.string(),
                        files.configuration.string());
  component.validateSystem(electrons, *ions, "all_electron");
  const PreparedScalarPlan prepared = prepareScalarValuePlan(
      component, "test/psiformer/scalar-phase-b", "scalar-phase-b-v1");
  static_cast<void>(prepared);

  const ParticleSet::SingleParticlePos common_position{0.42, -0.29, 0.35};
  electrons.makeVirtualMoves(common_position);
  reference_electrons.makeVirtualMoves(common_position);
  std::vector<ValueType> expected_all_to_one(electron_count);
  reference.evaluateRatiosAlltoOne(reference_electrons,
                                   expected_all_to_one);
  const std::vector<ParticleSet::SingleParticlePos> displacements{
      {0.024, -0.015, 0.009}, {-0.028, 0.019, 0.014}};
  VirtualParticleSet virtual_particles(electrons);
  VirtualParticleSet reference_virtual(reference_electrons);
  virtual_particles.makeMoves(electrons, 1, displacements);
  reference_virtual.makeMoves(reference_electrons, 1, displacements);
  std::vector<ValueType> expected_virtual(displacements.size());
  reference.evaluateRatios(reference_virtual, expected_virtual);

  for (std::size_t index = 0; index < faults.size(); ++index)
  {
    CAPTURE(faults[index].label);
    const bool use_all_to_one = index % 2 == 0;
    std::vector<ValueType> output(
        use_all_to_one ? electron_count : displacements.size(), ValueType(97));
    const PlannedScalarTransactionSnapshot before =
        capturePlannedScalarTransaction(
            component, electrons, output,
            use_all_to_one ? nullptr : &virtual_particles);
    testing::TestPsiFormerWF::setPlannedScalarValueFault(component,
                                                         faults[index].fault);
    std::string exception_message;
    bool unexpected_exception = false;
    try
    {
      if (use_all_to_one)
        component.evaluateRatiosAlltoOne(electrons, output);
      else
        component.evaluateRatios(virtual_particles, output);
    }
    catch (const std::exception& error)
    {
      exception_message = error.what();
    }
    catch (...)
    {
      unexpected_exception = true;
    }
    testing::TestPsiFormerWF::setPlannedScalarValueFault(component,
                                                         Fault::NONE);
    CHECK_FALSE(unexpected_exception);
    CHECK(exception_message.find(faults[index].expected_message) !=
          std::string::npos);
    CHECK(exception_message.find("test fault was not detected") ==
          std::string::npos);
    checkPlannedScalarTransaction(
        component, electrons, output, before,
        use_all_to_one ? nullptr : &virtual_particles);

    if (use_all_to_one)
      component.evaluateRatiosAlltoOne(electrons, output);
    else
      component.evaluateRatios(virtual_particles, output);
    const std::vector<ValueType>& expected =
        use_all_to_one ? expected_all_to_one : expected_virtual;
    REQUIRE(output.size() == expected.size());
    for (std::size_t value = 0; value < output.size(); ++value)
      checkScalarRatio(output[value], expected[value]);
  }
}

TEST_CASE("PsiFormer planned scalar VALUE rejects derivative API bypasses",
          "[wavefunction][psiformer][batch_memory][scalar_value][derivative_guard]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  std::unique_ptr<ParticleSet> ions = makeLiHIons(simulation_cell);

  for (const bool optimize : {false, true})
  {
    DYNAMIC_SECTION("optimization enabled = " << optimize)
    {
      ParticleSet electrons = makeLiHElectrons(simulation_cell);
      const std::vector<std::size_t> selected_indices =
          optimize ? std::vector<std::size_t>{0, 127}
                   : std::vector<std::size_t>{};
      PsiFormerWF component(
          optimize ? "pf_planned_derivative_active"
                   : "pf_planned_derivative_inactive",
          files.parameters.string(), files.configuration.string(), optimize,
          selected_indices);
      component.validateSystem(electrons, *ions, "all_electron");
      electrons.G = ValueType(0);
      electrons.L = ValueType(0);
      component.evaluateLog(electrons, electrons.G, electrons.L);
      const PreparedScalarPlan prepared = prepareScalarValuePlan(
          component,
          optimize ? "test/psiformer/scalar-derivative-active"
                   : "test/psiformer/scalar-derivative-inactive",
          optimize ? "scalar-derivative-active-v1"
                   : "scalar-derivative-inactive-v1");
      static_cast<void>(prepared);

      const std::vector<ParticleSet::SingleParticlePos> displacements{
          {0.021, -0.014, 0.008}, {-0.029, 0.016, 0.025}};
      VirtualParticleSet virtual_particles(electrons);
      virtual_particles.makeMoves(electrons, 1, displacements);
      OptVariables active = optimize ? registerSelectedParameters(component)
                                     : OptVariables{};

      std::vector<ValueType> ratios(displacements.size(), ValueType(41));
      Matrix<ValueType> derivative_ratios(displacements.size(), 3);
      derivative_ratios = ValueType(43);
      const ValueType* const derivative_data =
          std::addressof(*derivative_ratios.begin());
      const std::vector<ValueType> derivative_before(
          derivative_ratios.begin(), derivative_ratios.end());
      const PlannedScalarTransactionSnapshot ratio_before =
          capturePlannedScalarTransaction(component, electrons, ratios,
                                          &virtual_particles);
      CHECK_THROWS_WITH(
          component.evaluateDerivRatios(virtual_particles, active, ratios,
                                        derivative_ratios),
          Catch::Matchers::ContainsSubstring(
              "derivative-ratio evaluation is not admitted as a clone-local operation"));
      checkPlannedScalarTransaction(component, electrons, ratios, ratio_before,
                                    &virtual_particles, true, true);
      CHECK(std::addressof(*derivative_ratios.begin()) == derivative_data);
      REQUIRE(derivative_ratios.size() == derivative_before.size());
      for (std::size_t value = 0; value < derivative_before.size(); ++value)
        CHECK(sameScalarBits(derivative_ratios.begin()[value],
                             derivative_before[value]));

      const std::vector<ValueType> weights{ValueType(0.37), ValueType(-0.19)};
      std::vector<ValueType> weighted_derivatives(3, ValueType(47));
      const ValueType* const weighted_data = weighted_derivatives.data();
      const std::vector<ValueType> weighted_before = weighted_derivatives;
      std::vector<ValueType> unchanged_ratios(displacements.size(),
                                               ValueType(53));
      const PlannedScalarTransactionSnapshot weighted_transaction_before =
          capturePlannedScalarTransaction(component, electrons,
                                          unchanged_ratios,
                                          &virtual_particles);
      CHECK_THROWS_WITH(
          component.evaluateDerivRatiosWeighted(
              virtual_particles, active, weights,
              {weighted_derivatives.data(), weighted_derivatives.size()}),
          Catch::Matchers::ContainsSubstring(
              "weighted derivative-ratio evaluation is not admitted as a clone-local operation"));
      checkPlannedScalarTransaction(component, electrons, unchanged_ratios,
                                    weighted_transaction_before,
                                    &virtual_particles, true, true);
      CHECK(weighted_derivatives.data() == weighted_data);
      REQUIRE(weighted_derivatives.size() == weighted_before.size());
      for (std::size_t value = 0; value < weighted_before.size(); ++value)
        CHECK(sameScalarBits(weighted_derivatives[value],
                             weighted_before[value]));

      Vector<ValueType> score(3);
      Vector<ValueType> kinetic(3);
      score   = ValueType(101);
      kinetic = ValueType(-103);
      const ValueType* const score_data = std::addressof(*score.begin());
      const ValueType* const kinetic_data = std::addressof(*kinetic.begin());
      const std::vector<ValueType> score_before(score.begin(), score.end());
      const std::vector<ValueType> kinetic_before(kinetic.begin(),
                                                  kinetic.end());
      const PlannedScalarTransactionSnapshot score_transaction_before =
          capturePlannedScalarTransaction(component, electrons, ratios,
                                          &virtual_particles);
      CHECK_THROWS_WITH(
          component.evaluateDerivativesWF(electrons, active, score),
          Catch::Matchers::ContainsSubstring(
              "parameter-score evaluation is not admitted as a clone-local operation by the explicit batch plan"));
      checkPlannedScalarTransaction(component, electrons, ratios,
                                    score_transaction_before,
                                    &virtual_particles, true, true);
      CHECK(std::addressof(*score.begin()) == score_data);
      for (std::size_t value = 0; value < score_before.size(); ++value)
        CHECK(sameScalarBits(score[value], score_before[value]));

      const PlannedScalarTransactionSnapshot kinetic_transaction_before =
          capturePlannedScalarTransaction(component, electrons, ratios,
                                          &virtual_particles);
      CHECK_THROWS_WITH(
          component.evaluateDerivatives(electrons, active, score, kinetic),
          Catch::Matchers::ContainsSubstring(
              "kinetic-parameter evaluation is not admitted as a clone-local operation by the explicit batch plan"));
      checkPlannedScalarTransaction(component, electrons, ratios,
                                    kinetic_transaction_before,
                                    &virtual_particles, true, true);
      CHECK(std::addressof(*score.begin()) == score_data);
      CHECK(std::addressof(*kinetic.begin()) == kinetic_data);
      for (std::size_t value = 0; value < score_before.size(); ++value)
      {
        CHECK(sameScalarBits(score[value], score_before[value]));
        CHECK(sameScalarBits(kinetic[value], kinetic_before[value]));
      }
    }
  }
}

TEST_CASE("PsiFormer scalar plan rejects every deferred crowd API before dispatch",
          "[wavefunction][psiformer][batch_memory][scalar_value][derivative_guard]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  std::unique_ptr<ParticleSet> ions = makeLiHIons(simulation_cell);

  for (const bool optimize : {false, true})
  {
    DYNAMIC_SECTION("optimization enabled = " << optimize)
    {
      ParticleSet electrons = makeLiHElectrons(simulation_cell);
      const std::vector<std::size_t> selected_indices =
          optimize ? std::vector<std::size_t>{0, 127}
                   : std::vector<std::size_t>{};
      PsiFormerWF component(
          optimize ? "pf_planned_crowd_guard_active"
                   : "pf_planned_crowd_guard_inactive",
          files.parameters.string(), files.configuration.string(), optimize,
          selected_indices);
      component.validateSystem(electrons, *ions, "all_electron");
      electrons.G = ValueType(0);
      electrons.L = ValueType(0);
      component.evaluateLog(electrons, electrons.G, electrons.L);
      OptVariables active = optimize ? registerSelectedParameters(component)
                                     : OptVariables{};
      const PreparedScalarPlan prepared = prepareScalarValuePlan(
          component,
          optimize ? "test/psiformer/scalar-crowd-guard-active"
                   : "test/psiformer/scalar-crowd-guard-inactive",
          optimize ? "scalar-crowd-guard-active-v1"
                   : "scalar-crowd-guard-inactive-v1");
      static_cast<void>(prepared);

      const std::vector<ParticleSet::SingleParticlePos> displacements{
          {0.016, -0.012, 0.009}, {-0.023, 0.018, 0.011}};
      VirtualParticleSet virtual_particles(electrons);
      virtual_particles.makeMoves(electrons, 1, displacements);
      RefVectorWithLeader<WaveFunctionComponent> components(component,
                                                             {component});
      RefVectorWithLeader<ParticleSet> particles(electrons, {electrons});
      RefVectorWithLeader<const VirtualParticleSet> virtual_list(
          virtual_particles);
      virtual_list.push_back(virtual_particles);
      RefVectorWithLeader<VirtualParticleSet> scratch_list(virtual_particles);
      scratch_list.push_back(virtual_particles);

      const std::vector<std::size_t> offsets{0, displacements.size()};
      const std::vector<VirtualParticleBatch::Segment> segments{{0, 1}};
      const std::vector<ParticleSet::PosType> absolute_positions(
          virtual_particles.R.begin(), virtual_particles.R.end());
      const VirtualParticleBatch batch(1, offsets, segments,
                                       absolute_positions);

      std::vector<std::vector<ValueType>> ragged_ratios{
          std::vector<ValueType>(displacements.size(), ValueType(107))};
      std::vector<ValueType> flat_ratios(displacements.size(),
                                         ValueType(109));
      const std::vector<ValueType> total_weights(displacements.size(),
                                                 ValueType(0.25));
      RefVector<const std::vector<ValueType>> weight_views;
      weight_views.push_back(std::cref(total_weights));
      std::vector<ValueType> weighted_derivatives(3, ValueType(113));
      std::vector<WaveFunctionComponent::ParameterDerivativeView>
          derivative_views{{weighted_derivatives.data(),
                            weighted_derivatives.size()}};
      RecordArray<ValueType> score_rows(1, 3);
      RecordArray<ValueType> kinetic_rows(1, 3);
      std::fill(score_rows.begin(), score_rows.end(), ValueType(127));
      std::fill(kinetic_rows.begin(), kinetic_rows.end(), ValueType(-131));

      const ValueType* const ragged_data = ragged_ratios.front().data();
      const ValueType* const flat_data = flat_ratios.data();
      const ValueType* const weighted_data = weighted_derivatives.data();
      const std::vector<ValueType> ragged_before = ragged_ratios.front();
      const std::vector<ValueType> flat_before = flat_ratios;
      const std::vector<ValueType> weighted_before = weighted_derivatives;
      const std::vector<ValueType> score_before(score_rows.begin(),
                                                score_rows.end());
      const std::vector<ValueType> kinetic_before(kinetic_rows.begin(),
                                                  kinetic_rows.end());
      std::vector<ValueType> scalar_probe(
          static_cast<std::size_t>(electrons.getTotalNum()), ValueType(137));
      const PlannedScalarTransactionSnapshot transaction_before =
          capturePlannedScalarTransaction(component, electrons, scalar_probe,
                                          &virtual_particles);

      const auto require_crowd_guard = [&](const char* operation,
                                           auto&& invocation) {
        const std::string expected = std::string(operation) +
            " is not admitted as a multi-walker operation by the explicit batch plan";
        CHECK_THROWS_WITH(invocation(),
                          Catch::Matchers::ContainsSubstring(expected));
        checkPlannedScalarTransaction(component, electrons, scalar_probe,
                                      transaction_before, &virtual_particles,
                                      true, true);
      };

      require_crowd_guard("ragged virtual-ratio evaluation", [&]() {
        component.mw_evaluateRatios(components, virtual_list, ragged_ratios);
      });
      require_crowd_guard("flattened virtual-ratio evaluation", [&]() {
        component.mw_evaluateVirtualRatios(components, particles, scratch_list,
                                           batch, flat_ratios);
      });
      require_crowd_guard("batched weighted derivative-ratio evaluation", [&]() {
        component.mw_evaluateDerivRatiosWeighted(
            components, virtual_list, active, weight_views, derivative_views);
      });
      require_crowd_guard("flattened weighted derivative-ratio evaluation",
                          [&]() {
                            component.mw_evaluateVirtualDerivRatiosWeighted(
                                components, particles, scratch_list, batch,
                                active, total_weights, derivative_views);
                          });
      require_crowd_guard("batched parameter-score evaluation", [&]() {
        component.mw_evaluateParameterDerivativesWF(components, particles,
                                                    active, score_rows);
      });
      require_crowd_guard("batched kinetic-parameter evaluation", [&]() {
        component.mw_evaluateParameterDerivatives(
            components, particles, active, score_rows, kinetic_rows);
      });

      CHECK(ragged_ratios.front().data() == ragged_data);
      CHECK(flat_ratios.data() == flat_data);
      CHECK(weighted_derivatives.data() == weighted_data);
      for (std::size_t value = 0; value < ragged_before.size(); ++value)
      {
        CHECK(sameScalarBits(ragged_ratios.front()[value],
                             ragged_before[value]));
        CHECK(sameScalarBits(flat_ratios[value], flat_before[value]));
      }
      for (std::size_t value = 0; value < weighted_before.size(); ++value)
      {
        CHECK(sameScalarBits(weighted_derivatives[value],
                             weighted_before[value]));
        CHECK(sameScalarBits(score_rows.begin()[value], score_before[value]));
        CHECK(sameScalarBits(kinetic_rows.begin()[value],
                             kinetic_before[value]));
      }
    }
  }
}

TEST_CASE("PsiFormer no-plan scalar VALUE backends and derivative fallback remain compatible",
          "[wavefunction][psiformer][scalar_value][legacy]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  const ParticleSet::SingleParticlePos common_position{0.36, -0.28, 0.32};
  const std::vector<ParticleSet::SingleParticlePos> displacements{
      {0.018, -0.011, 0.007}, {-0.025, 0.017, 0.012},
      {0.014, 0.022, -0.019}};

  struct LegacyScalarResults
  {
    std::vector<ValueType> all_to_one;
    std::vector<ValueType> virtual_ratios;
  };

  const auto evaluate_backend = [&](const char* backend) {
    ScopedEnvironmentVariable select_backend("PSIFORMER_VALUE_BACKEND",
                                              backend);
    ParticleSet electrons = makeLiHElectrons(simulation_cell);
    PsiFormerWF component(std::string("pf_legacy_scalar_") + backend,
                          files.parameters.string(),
                          files.configuration.string());
    electrons.makeVirtualMoves(common_position);
    const ScalarParticleStateSnapshot before_all_to_one =
        captureScalarParticleState(electrons);
    LegacyScalarResults results;
    results.all_to_one.resize(
        static_cast<std::size_t>(electrons.getTotalNum()), ValueType(59));
    component.evaluateRatiosAlltoOne(electrons, results.all_to_one);
    checkScalarParticleState(electrons, before_all_to_one);

    VirtualParticleSet virtual_particles(electrons);
    virtual_particles.makeMoves(electrons, 1, displacements);
    const ScalarParticleStateSnapshot before_reference =
        captureScalarParticleState(electrons);
    const ScalarVirtualParticleStateSnapshot before_virtual =
        captureScalarVirtualParticleState(virtual_particles);
    results.virtual_ratios.resize(displacements.size(), ValueType(61));
    component.evaluateRatios(virtual_particles, results.virtual_ratios);
    checkScalarParticleState(electrons, before_reference);
    checkScalarVirtualParticleState(virtual_particles, before_virtual);
    return results;
  };

  const LegacyScalarResults direct = evaluate_backend("direct");
  for (const char* backend : {"oracle", "compare"})
  {
    CAPTURE(backend);
    const LegacyScalarResults candidate = evaluate_backend(backend);
    REQUIRE(candidate.all_to_one.size() == direct.all_to_one.size());
    REQUIRE(candidate.virtual_ratios.size() == direct.virtual_ratios.size());
    for (std::size_t electron = 0; electron < direct.all_to_one.size();
         ++electron)
      checkScalarRatio(candidate.all_to_one[electron],
                       direct.all_to_one[electron]);
    for (std::size_t move = 0; move < direct.virtual_ratios.size(); ++move)
      checkScalarRatio(candidate.virtual_ratios[move],
                       direct.virtual_ratios[move]);
  }
  CHECK(std::any_of(direct.all_to_one.begin(), direct.all_to_one.end(),
                    [](ValueType ratio) {
                      return std::abs(ratio - ValueType(1)) > 1.0e-8;
                    }));
  CHECK(std::any_of(direct.virtual_ratios.begin(),
                    direct.virtual_ratios.end(), [](ValueType ratio) {
                      return std::abs(ratio - ValueType(1)) > 1.0e-10;
                    }));

  // A fixed, unplanned component historically treats the derivative-ratio API
  // as an ordinary ratio request while leaving derivative destinations alone.
  ParticleSet electrons = makeLiHElectrons(simulation_cell);
  PsiFormerWF component("pf_legacy_no_active_derivatives",
                        files.parameters.string(),
                        files.configuration.string());
  VirtualParticleSet virtual_particles(electrons);
  virtual_particles.makeMoves(electrons, 1, displacements);
  std::vector<ValueType> expected_ratios(displacements.size());
  component.evaluateRatios(virtual_particles, expected_ratios);

  std::vector<ValueType> ratios(displacements.size(), ValueType(67));
  Matrix<ValueType> derivative_ratios(displacements.size(), 3);
  derivative_ratios = ValueType(71);
  const ValueType* const derivative_data =
      std::addressof(*derivative_ratios.begin());
  const std::vector<ValueType> derivative_before(
      derivative_ratios.begin(), derivative_ratios.end());
  const ScalarParticleStateSnapshot derivative_reference_before =
      captureScalarParticleState(electrons);
  const ScalarVirtualParticleStateSnapshot derivative_virtual_before =
      captureScalarVirtualParticleState(virtual_particles);
  component.evaluateDerivRatios(virtual_particles, OptVariables{}, ratios,
                                derivative_ratios);
  checkScalarParticleState(electrons, derivative_reference_before);
  checkScalarVirtualParticleState(virtual_particles,
                                  derivative_virtual_before);
  CHECK(std::addressof(*derivative_ratios.begin()) == derivative_data);
  for (std::size_t move = 0; move < ratios.size(); ++move)
    checkScalarRatio(ratios[move], expected_ratios[move]);
  for (std::size_t value = 0; value < derivative_before.size(); ++value)
    CHECK(sameScalarBits(derivative_ratios.begin()[value],
                         derivative_before[value]));

  const std::vector<ValueType> weights{ValueType(0.2), ValueType(-0.3),
                                       ValueType(0.4)};
  std::vector<ValueType> weighted_derivatives(3, ValueType(73));
  const ValueType* const weighted_data = weighted_derivatives.data();
  const std::vector<ValueType> weighted_before = weighted_derivatives;
  component.evaluateDerivRatiosWeighted(
      virtual_particles, OptVariables{}, weights,
      {weighted_derivatives.data(), weighted_derivatives.size()});
  CHECK(weighted_derivatives.data() == weighted_data);
  for (std::size_t value = 0; value < weighted_before.size(); ++value)
    CHECK(sameScalarBits(weighted_derivatives[value],
                         weighted_before[value]));
}

TEST_CASE("PsiFormer hard plans reject scalar lifecycle entries before mutation",
          "[wavefunction][psiformer][batch_memory][scalar_guard]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  ParticleSet electrons = makeLiHElectrons(simulation_cell);
  std::unique_ptr<ParticleSet> ions = makeLiHIons(simulation_cell);
  PsiFormerWF component("pf_scalar_guard", files.parameters.string(),
                        files.configuration.string(), true, {0, 1});
  component.validateSystem(electrons, *ions, "all_electron");

  // Keep a small legacy smoke path for each inherited/no-op wrapper added by
  // the hard-plan guard. Spin-independent wrappers must leave the spin output
  // alone while delegating to the ordinary scalar evaluator.
  electrons.G = ValueType(0);
  electrons.L = ValueType(0);
  component.evaluateLog(electrons, electrons.G, electrons.L);
  component.recompute(electrons);
  component.prepareGroup(electrons, 0);
  component.completeUpdates();
  PsiFormerWF::ComplexType legacy_spin_gradient(1.25, -0.75);
  const auto legacy_spin_before = legacy_spin_gradient;
  const PsiFormerWF::GradType legacy_gradient =
      component.evalGradWithSpin(electrons, 0, legacy_spin_gradient);
  CHECK(legacy_spin_gradient == legacy_spin_before);
  for (std::size_t dimension = 0; dimension < OHMMS_DIM; ++dimension)
    CHECK(std::isfinite(std::real(legacy_gradient[dimension])));

  electrons.makeMove(0, ParticleSet::SingleParticlePos{0.013, -0.009, 0.007});
  PsiFormerWF::GradType legacy_ratio_gradient;
  for (std::size_t dimension = 0; dimension < OHMMS_DIM; ++dimension)
    legacy_ratio_gradient[dimension] = ValueType(0);
  PsiFormerWF::ComplexType legacy_ratio_spin(-2.0, 0.625);
  const auto legacy_ratio_spin_before = legacy_ratio_spin;
  const ValueType legacy_ratio = component.ratioGradWithSpin(
      electrons, 0, legacy_ratio_gradient, legacy_ratio_spin);
  CHECK(std::isfinite(std::real(legacy_ratio)));
  CHECK(std::isfinite(std::imag(legacy_ratio)));
  CHECK(legacy_ratio_spin == legacy_ratio_spin_before);
  component.restore(0);
  electrons.rejectMove(0);

  PsiFormerWF::WFBufferType legacy_buffer;
  component.registerData(electrons, legacy_buffer);
  REQUIRE(legacy_buffer.current() > 0);
  REQUIRE(legacy_buffer.current_scalar() > 0);
  legacy_buffer.allocate();
  electrons.G = ValueType(0);
  electrons.L = ValueType(0);
  legacy_buffer.rewind();
  component.updateBuffer(electrons, legacy_buffer, false);
  legacy_buffer.rewind();
  component.copyFromBuffer(electrons, legacy_buffer);

  // Reestablish one unambiguous FULL accepted state before entering the plan.
  electrons.G = ValueType(0);
  electrons.L = ValueType(0);
  component.evaluateLog(electrons, electrons.G, electrons.L);

  BatchExecutionRequirements requirements;
  component.contributeBatchExecutionRequirements(requirements);
  requirements.require(BatchExecutionMode::SCALAR_VALUE_COMPATIBILITY);
  const std::string participant_id = "test/psiformer/scalar-guard";
  const auto plan = makeClonePreparationTestPlan(
      component, requirements, participant_id, "scalar-guard-v1");
  const BatchExecutionParticipantPlan participant_plan =
      makeBatchExecutionParticipantPlan(plan, participant_id);
  component.bindBatchExecutionPlan(participant_plan);
  component.prepareBatchExecutionClone(participant_plan);

  // The pending scalar proposal makes legacy accept/restore observably
  // destructive, so retaining this exact state proves the plan guard ran first.
  testing::TestPsiFormerWF::markScalarProposalPending(component, 0);
  const auto component_before =
      testing::TestPsiFormerWF::scalarStateSnapshot(component);
  const auto workspace_before =
      testing::TestPsiFormerWF::directWorkspaceDiagnostics(component);
  const ParticleSet::ParticleGradient particle_gradient_before = electrons.G;
  const ParticleSet::ParticleLaplacian particle_laplacian_before = electrons.L;

  // Seed both pooled-buffer regions and advance both cursors. A first-entry
  // rejection must neither consume nor rewrite either region.
  PsiFormerWF::WFBufferType guarded_buffer;
  PsiFormerWF::GradType buffer_gradient;
  for (std::size_t dimension = 0; dimension < OHMMS_DIM; ++dimension)
    buffer_gradient[dimension] = ValueType(0.375 * (dimension + 1));
  double buffer_scalar = -4.25;
  guarded_buffer.add(&buffer_gradient, &buffer_gradient + 1);
  guarded_buffer.add(buffer_scalar);
  guarded_buffer.allocate();
  guarded_buffer.rewind();
  guarded_buffer.put(&buffer_gradient, &buffer_gradient + 1);
  guarded_buffer.put(buffer_scalar);
  const auto buffer_bulk_cursor   = guarded_buffer.current();
  const auto buffer_scalar_cursor = guarded_buffer.current_scalar();
  const auto buffer_storage       = guarded_buffer.myData;
  const auto* buffer_scalar_data  = guarded_buffer.Scalar_ptr;

  const auto check_unchanged = [&]() {
    CHECK(testing::TestPsiFormerWF::scalarStateMatches(component,
                                                        component_before));
    REQUIRE(electrons.G.size() == particle_gradient_before.size());
    REQUIRE(electrons.L.size() == particle_laplacian_before.size());
    for (std::size_t electron = 0; electron < electrons.G.size(); ++electron)
    {
      for (std::size_t dimension = 0; dimension < OHMMS_DIM; ++dimension)
        CHECK(electrons.G[electron][dimension] ==
              particle_gradient_before[electron][dimension]);
      CHECK(electrons.L[electron] == particle_laplacian_before[electron]);
    }

    const auto workspace_after =
        testing::TestPsiFormerWF::directWorkspaceDiagnostics(component);
    CHECK(workspace_after.has_prepared_clone_plan ==
          workspace_before.has_prepared_clone_plan);
    CHECK(workspace_after.owns_batch_workspace ==
          workspace_before.owns_batch_workspace);
    CHECK(workspace_after.batch_workspace_identity ==
          workspace_before.batch_workspace_identity);
    CHECK(workspace_after.batch_storage_fingerprint ==
          workspace_before.batch_storage_fingerprint);
    CHECK(workspace_after.accountedBytes() == workspace_before.accountedBytes());
    CHECK(workspace_after.accepted_spatial_bytes ==
          workspace_before.accepted_spatial_bytes);
    CHECK(workspace_after.proposed_spatial_bytes ==
          workspace_before.proposed_spatial_bytes);

    CHECK(guarded_buffer.current() == buffer_bulk_cursor);
    CHECK(guarded_buffer.current_scalar() == buffer_scalar_cursor);
    CHECK(guarded_buffer.Scalar_ptr == buffer_scalar_data);
    REQUIRE(guarded_buffer.myData.size() == buffer_storage.size());
    for (std::size_t byte = 0; byte < buffer_storage.size(); ++byte)
      CHECK(guarded_buffer.myData[byte] == buffer_storage[byte]);
  };
  const auto expect_plan_guard = [&](auto&& operation) {
    CHECK_THROWS_WITH(
        operation(),
        Catch::Matchers::ContainsSubstring(
            "not admitted as a scalar operation by the explicit batch plan"));
    check_unchanged();
  };

  expect_plan_guard([&]() { component.recompute(electrons); });
  expect_plan_guard([&]() { component.acceptMove(electrons, 0); });
  expect_plan_guard([&]() { component.restore(0); });
  CHECK_THROWS_WITH(
      component.prepareGroup(electrons, 0),
      Catch::Matchers::ContainsSubstring("requires absent proposal state"));
  check_unchanged();
  CHECK_THROWS_WITH(
      component.completeUpdates(),
      Catch::Matchers::ContainsSubstring("requires absent proposal state"));
  check_unchanged();

  PsiFormerWF::ComplexType spin_gradient(3.5, -1.75);
  const auto spin_gradient_before = spin_gradient;
  expect_plan_guard(
      [&]() { component.evalGradWithSpin(electrons, 0, spin_gradient); });
  CHECK(spin_gradient == spin_gradient_before);

  PsiFormerWF::GradType ratio_gradient;
  for (std::size_t dimension = 0; dimension < OHMMS_DIM; ++dimension)
    ratio_gradient[dimension] = ValueType(-0.5 * (dimension + 1));
  const PsiFormerWF::GradType ratio_gradient_before = ratio_gradient;
  PsiFormerWF::ComplexType ratio_spin_gradient(-0.875, 2.625);
  const auto ratio_spin_gradient_before = ratio_spin_gradient;
  expect_plan_guard([&]() {
    component.ratioGradWithSpin(electrons, 0, ratio_gradient,
                                ratio_spin_gradient);
  });
  for (std::size_t dimension = 0; dimension < OHMMS_DIM; ++dimension)
    CHECK(ratio_gradient[dimension] == ratio_gradient_before[dimension]);
  CHECK(ratio_spin_gradient == ratio_spin_gradient_before);

  CHECK_THROWS_WITH(
      component.registerData(electrons, guarded_buffer),
      Catch::Matchers::ContainsSubstring("requires absent proposal state"));
  check_unchanged();
  CHECK_THROWS_WITH(
      component.updateBuffer(electrons, guarded_buffer, true),
      Catch::Matchers::ContainsSubstring("does not support from_scratch"));
  check_unchanged();
  CHECK_THROWS_WITH(
      component.copyFromBuffer(electrons, guarded_buffer),
      Catch::Matchers::ContainsSubstring("requires absent proposal state"));
  check_unchanged();
}

TEST_CASE("PsiFormer planned scalar scope excludes unselected and legacy ECP paths",
          "[wavefunction][psiformer][batch_memory]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  ParticleSet electrons = makeLiHElectrons(simulation_cell);
  std::unique_ptr<ParticleSet> ions = makeLiHIons(simulation_cell);

  PsiFormerWF no_scalar("pf_no_scalar", files.parameters.string(),
                        files.configuration.string(), true, {0, 1});
  no_scalar.validateSystem(electrons, *ions, "all_electron");
  BatchExecutionRequirements no_scalar_requirements;
  no_scalar.contributeBatchExecutionRequirements(no_scalar_requirements);
  const std::string no_scalar_id = "test/psiformer/no-scalar";
  const auto no_scalar_plan = makeClonePreparationTestPlan(
      no_scalar, no_scalar_requirements, no_scalar_id, "no-scalar-v1");
  const BatchExecutionParticipantPlan no_scalar_view =
      makeBatchExecutionParticipantPlan(no_scalar_plan, no_scalar_id);
  no_scalar.bindBatchExecutionPlan(no_scalar_view);
  no_scalar.prepareBatchExecutionClone(no_scalar_view);
  electrons.makeVirtualMoves(
      ParticleSet::SingleParticlePos{0.17, -0.12, 0.21});
  std::vector<ValueType> all_to_one(electrons.getTotalNum());
  CHECK_THROWS_WITH(
      no_scalar.evaluateRatiosAlltoOne(electrons, all_to_one),
      Catch::Matchers::ContainsSubstring("not admitted by the explicit batch plan"));
  CHECK_FALSE(testing::TestPsiFormerWF::directWorkspaceDiagnostics(no_scalar)
                  .owns_batch_workspace);

  PsiFormerWF ecp_component("pf_ecp_scalar", files.parameters.string(),
                            files.configuration.string());
  ecp_component.validateSystem(electrons, *ions, "all_electron");
  BatchExecutionRequirements ecp_requirements;
  ecp_component.contributeBatchExecutionRequirements(ecp_requirements);
  ecp_requirements.require(BatchExecutionMode::VALUE);
  ecp_requirements.require(BatchExecutionMode::SCALAR_VALUE_COMPATIBILITY);
  ecp_requirements.require(BatchExecutionMode::ECP_OUTER);
  const std::string ecp_id = "test/psiformer/ecp-scalar";
  const auto ecp_plan = makeClonePreparationTestPlan(
      ecp_component, ecp_requirements, ecp_id, "ecp-scalar-v1", 2, 2);
  const BatchExecutionParticipantPlan ecp_view =
      makeBatchExecutionParticipantPlan(ecp_plan, ecp_id);
  ecp_component.bindBatchExecutionPlan(ecp_view);
  ecp_component.prepareBatchExecutionClone(ecp_view);

  const std::vector<ParticleSet::SingleParticlePos> displacements{
      {0.02, -0.01, 0.03}};
  VirtualParticleSet virtual_particles(electrons);
  virtual_particles.makeMoves(electrons, 1, displacements);
  std::vector<ValueType> virtual_ratios(displacements.size(), ValueType(9));
  CHECK_THROWS_WITH(
      ecp_component.evaluateRatios(virtual_particles, virtual_ratios),
      Catch::Matchers::ContainsSubstring("flattened multiwalker ECP dispatch"));
  CHECK(virtual_ratios[0] == ValueType(9));

  // Scalar compatibility can coexist with flattened ECP for unrelated
  // estimator calls; only the legacy virtual-particle dispatch is excluded.
  ecp_component.evaluateRatiosAlltoOne(electrons, all_to_one);
}

TEST_CASE("PsiFormer kinetic parameter derivatives require unit electron masses",
          "[wavefunction][psiformer][optimizer]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  ParticleSet electrons = makeLiHElectrons(simulation_cell);
  PsiFormerWF component(
      "pf_unit_mass", files.parameters.string(), files.configuration.string(), true, {0});
  OptVariables active = registerSelectedParameters(component);

  SpeciesSet& species = electrons.getSpeciesSet();
  const int mass       = species.getAttribute("mass");
  REQUIRE(mass < species.numAttributes());
  SECTION("equal nonunit masses")
  {
    species(mass, 0) = 2.0;
    species(mass, 1) = 2.0;
  }
  SECTION("unequal masses")
  {
    species(mass, 0) = 1.0;
    species(mass, 1) = 2.0;
  }
  electrons.resetGroups();

  electrons.G = ValueType(0);
  electrons.L = ValueType(0);
  component.evaluateLog(electrons, electrons.G, electrons.L);
  Vector<ValueType> score(active.size());
  Vector<ValueType> kinetic_response(active.size());
  score            = ValueType(0);
  kinetic_response = ValueType(0);

  // A score-only reverse does not use the electron masses and remains valid.
  CHECK_NOTHROW(component.evaluateDerivativesWF(electrons, active, score));
  CHECK_THROWS_WITH(component.evaluateDerivatives(electrons, active, score, kinetic_response),
                    Catch::Matchers::ContainsSubstring("require unit electron masses"));
}

TEST_CASE("PsiFormer nonlocal virtual ratios and parameter derivatives", "[wavefunction][psiformer][ecp]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  ParticleSet electrons = makeLiHElectrons(simulation_cell);
  PsiFormerWF component("pf_virtual", files.parameters.string(), files.configuration.string(), true, {0, 127});
  OptVariables active = registerSelectedParameters(component);

  electrons.G = ValueType(0);
  electrons.L = ValueType(0);
  const PsiFormerWF::LogValue reference_log = component.evaluateLog(electrons, electrons.G, electrons.L);
  const std::vector<ParticleSet::SingleParticlePos> displacements{{0.08, -0.03, 0.02},
                                                                   {-0.04, 0.06, -0.05},
                                                                   {0.03, 0.01, 0.07}};
  VirtualParticleSet virtual_particles(electrons);
  virtual_particles.makeMoves(electrons, 1, displacements);

  std::vector<ValueType> ratios(displacements.size());
  Matrix<ValueType> derivative_ratios(displacements.size(), active.size());
  derivative_ratios = ValueType(0);
  component.evaluateDerivRatios(virtual_particles, active, ratios, derivative_ratios);

  // Full reevaluation supplies an independent high-level ratio check at each
  // quadrature point; no particle-by-particle proposal cache is involved.
  for (std::size_t move = 0; move < displacements.size(); ++move)
  {
    ParticleSet moved = makeLiHElectrons(simulation_cell);
    moved.R[1] += displacements[move];
    moved.update();
    PsiFormerWF fixed("pf_virtual_fixed", files.parameters.string(), files.configuration.string());
    moved.G = ValueType(0);
    moved.L = ValueType(0);
    const PsiFormerWF::LogValue moved_log = fixed.evaluateLog(moved, moved.G, moved.L);
    const auto expected_ratio             = std::exp(moved_log - reference_log);
    CHECK(std::real(ratios[move]) ==
          Catch::Approx(std::real(expected_ratio)).epsilon(2e-9).margin(2e-12));
    CHECK(std::imag(ratios[move]) ==
          Catch::Approx(std::imag(expected_ratio)).epsilon(2e-9).margin(2e-12));
  }

  // The ECP contract is d log(Psi_virtual/Psi_reference)/d theta. Verify
  // both selected columns by centered finite differences of full log values.
  const double parameter_step = 2e-5;
  for (int parameter = 0; parameter < active.size(); ++parameter)
  {
    const double original = active[parameter];
    std::vector<double> log_ratio_plus(displacements.size());
    std::vector<double> log_ratio_minus(displacements.size());
    for (int direction : {-1, 1})
    {
      active[parameter] = original + direction * parameter_step;
      component.resetParametersExclusive(active);
      electrons.G = ValueType(0);
      electrons.L = ValueType(0);
      const double base_log = std::real(component.evaluateLog(electrons, electrons.G, electrons.L));
      for (std::size_t move = 0; move < displacements.size(); ++move)
      {
        ParticleSet moved = makeLiHElectrons(simulation_cell);
        moved.R[1] += displacements[move];
        moved.update();
        moved.G = ValueType(0);
        moved.L = ValueType(0);
        const double moved_log = std::real(component.evaluateLog(moved, moved.G, moved.L));
        (direction > 0 ? log_ratio_plus : log_ratio_minus)[move] = moved_log - base_log;
      }
    }
    active[parameter] = original;
    component.resetParametersExclusive(active);
    for (std::size_t move = 0; move < displacements.size(); ++move)
    {
      const double finite_difference = (log_ratio_plus[move] - log_ratio_minus[move]) / (2 * parameter_step);
      CHECK(std::real(derivative_ratios(move, parameter)) ==
            Catch::Approx(finite_difference).epsilon(8e-5).margin(8e-5));
    }
  }
}

TEST_CASE("PsiFormer full-network derivatives update and restart", "[wavefunction][psiformer]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  ParticleSet full_electrons     = makeLiHElectrons(simulation_cell);
  ParticleSet selected_electrons = makeLiHElectrons(simulation_cell);

  PsiFormerWF full("pf_full", files.parameters.string(), files.configuration.string(), true, {}, true);
  OptVariables full_active = registerSelectedParameters(full);
  REQUIRE(full_active.size() > 2048);
  const std::vector<std::size_t> probes{0, 127, 2047, full_active.size() - 1};

  full_electrons.G = ValueType(0);
  full_electrons.L = ValueType(0);
  const double baseline_log = std::real(full.evaluateLog(full_electrons, full_electrons.G, full_electrons.L));
  Vector<ValueType> full_dlog(full_active.size());
  Vector<ValueType> full_denergy(full_active.size());
  full_dlog    = ValueType(0);
  full_denergy = ValueType(0);
  full.evaluateDerivatives(full_electrons, full_active, full_dlog, full_denergy);

  PsiFormerWF selected(
      "pf_selected_full_check", files.parameters.string(), files.configuration.string(), true, probes);
  OptVariables selected_active = registerSelectedParameters(selected);
  const ComponentSnapshot selected_snapshot = evaluateComponent(selected, selected_electrons, selected_active);
  for (std::size_t probe = 0; probe < probes.size(); ++probe)
  {
    CHECK(std::real(full_dlog[probes[probe]]) ==
          Catch::Approx(selected_snapshot.log_parameter_derivative[probe]).epsilon(2e-10).margin(2e-10));
    CHECK(std::real(full_denergy[probes[probe]]) ==
          Catch::Approx(selected_snapshot.kinetic_parameter_derivative[probe]).epsilon(2e-9).margin(2e-9));
  }

  // Exercise the full-vector reset while perturbing only two entries. The
  // optimizer still supplies the complete active vector on every update.
  full_active[0] -= 1e-5 * std::real(full_dlog[0]);
  full_active[127] -= 1e-5 * std::real(full_dlog[127]);
  full.resetParametersExclusive(full_active);
  full_electrons.G = ValueType(0);
  full_electrons.L = ValueType(0);
  const double updated_log = std::real(full.evaluateLog(full_electrons, full_electrons.G, full_electrons.L));
  CHECK(std::abs(updated_log - baseline_log) > 1e-10);

  const std::filesystem::path state_path = files.directory / "psiformer_full_restart.vp.h5";
  hdf_archive output;
  REQUIRE(output.create(state_path));
  full.writeVariationalParameters(output);
  output.close();

  ParticleSet restored_electrons = makeLiHElectrons(simulation_cell);
  PsiFormerWF restored("pf_full", files.parameters.string(), files.configuration.string(), true, {}, true);
  hdf_archive input;
  REQUIRE(input.open(state_path, H5F_ACC_RDONLY));
  restored.readVariationalParameters(input);
  input.close();
  OptVariables restored_active = registerSelectedParameters(restored);
  restored.resetParametersExclusive(restored_active);
  restored_electrons.G = ValueType(0);
  restored_electrons.L = ValueType(0);
  const double restored_log =
      std::real(restored.evaluateLog(restored_electrons, restored_electrons.G, restored_electrons.L));
  CHECK(restored_log == Catch::Approx(updated_log).epsilon(2e-10).margin(2e-10));
}

TEST_CASE("PsiFormer LiH pair full-network update", "[wavefunction][psiformer]")
{
  GeneratedFiles files = generateFiles("lih_pair");
  const Geometry geometry = makeGeometry("lih_pair");
  const SimulationCell simulation_cell;
  ParticleSet electrons(simulation_cell);
  electrons.setName("e");
  electrons.create({4, 4});
  for (int electron = 0; electron < electrons.getTotalNum(); ++electron)
    for (int dimension = 0; dimension < 3; ++dimension)
      electrons.R[electron][dimension] = geometry.electrons[3 * electron + dimension];
  electrons.update();

  PsiFormerWF full("pf_pair_full", files.parameters.string(), files.configuration.string(), true, {}, true);
  OptVariables active = registerSelectedParameters(full);
  electrons.G = ValueType(0);
  electrons.L = ValueType(0);
  const double initial_log = std::real(full.evaluateLog(electrons, electrons.G, electrons.L));
  active[0] += 1e-4;
  full.resetParametersExclusive(active);
  electrons.G = ValueType(0);
  electrons.L = ValueType(0);
  const double updated_log = std::real(full.evaluateLog(electrons, electrons.G, electrons.L));
  CHECK(std::abs(updated_log - initial_log) > 1e-10);
}

TEST_CASE("PsiFormer selected parameters follow QMCPACK registration reset and derivative paths",
          "[wavefunction][psiformer]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  ParticleSet electrons = makeLiHElectrons(simulation_cell);
  RuntimeOptions runtime_options;
  TrialWaveFunction trial_wavefunction(runtime_options, "psiformer_selected_test");
  trial_wavefunction.addComponent(std::make_unique<PsiFormerWF>(
      "pf", files.parameters.string(), files.configuration.string(), true, std::vector<std::size_t>{127, 0, 1}));

  OptVariables active;
  trial_wavefunction.checkInVariables(active);
  active.resetIndex();
  trial_wavefunction.checkOutVariables(active);
  REQUIRE(active.size() == 3);
  CHECK(active.name(0) == "pf_pf_0000000");
  CHECK(active.name(1) == "pf_pf_0000001");
  CHECK(active.name(2) == "pf_pf_0000127");
  REQUIRE(trial_wavefunction.extractOptimizableObjectRefs().size() == 1);

  const double baseline_log = trial_wavefunction.evaluateLog(electrons);
  addLinearLogGradient(electrons);

  Vector<ValueType> dlogpsi(active.size());
  Vector<ValueType> dhpsioverpsi(active.size());
  Vector<ValueType> dlogpsi_wf(active.size());
  dlogpsi       = ValueType(-0.125);
  dhpsioverpsi  = ValueType(0.625);
  dlogpsi_wf    = ValueType(0.375);
  trial_wavefunction.evaluateDerivatives(electrons, active, dlogpsi, dhpsioverpsi);
  trial_wavefunction.evaluateDerivativesWF(electrons, active, dlogpsi_wf);

  const double log_derivative = std::real(dlogpsi[0]) + 0.125;
  const double kinetic_derivative = std::real(dhpsioverpsi[0]) - 0.625;
  CHECK(std::real(dlogpsi_wf[0]) - 0.375 == Catch::Approx(log_derivative).epsilon(2e-10).margin(2e-10));

  const double original_value = active[0];
  const double parameter_step = 2e-5;
  active[0] = original_value + parameter_step;
  trial_wavefunction.resetParameters(active);
  const double plus_log = trial_wavefunction.evaluateLog(electrons);
  addLinearLogGradient(electrons);
  const double plus_kinetic = kineticEnergy(electrons);

  active[0] = original_value - parameter_step;
  trial_wavefunction.resetParameters(active);
  const double minus_log = trial_wavefunction.evaluateLog(electrons);
  addLinearLogGradient(electrons);
  const double minus_kinetic = kineticEnergy(electrons);

  active[0] = original_value;
  trial_wavefunction.resetParameters(active);
  const double restored_log = trial_wavefunction.evaluateLog(electrons);
  CHECK(restored_log == Catch::Approx(baseline_log).epsilon(2e-10).margin(2e-10));

  const double log_finite_difference = (plus_log - minus_log) / (2 * parameter_step);
  const double kinetic_finite_difference = (plus_kinetic - minus_kinetic) / (2 * parameter_step);
  CHECK(log_finite_difference == Catch::Approx(log_derivative).epsilon(5e-5).margin(5e-5));
  CHECK(kinetic_finite_difference == Catch::Approx(kinetic_derivative).epsilon(2e-4).margin(2e-4));

  // Apply one deterministic first-order update through the public reset path.
  // This is the small vertical slice used before streaming full-network descent.
  active[0] = original_value - 1e-4 * log_derivative;
  trial_wavefunction.resetParameters(active);
  const double updated_log = trial_wavefunction.evaluateLog(electrons);
  CHECK(std::abs(updated_log - baseline_log) > 1e-8);
}

TEST_CASE("PsiFormer component-major kinetic derivatives reuse one crowd tape",
          "[wavefunction][psiformer][multiwalker]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;

  ParticleSet batch_electrons0 = makeLiHElectrons(simulation_cell);
  ParticleSet batch_electrons1 = makeLiHElectrons(simulation_cell);
  batch_electrons1.R[0][0] += 0.11;
  batch_electrons1.update();

  PsiFormerWF leader(
      "pf_kinetic_pool", files.parameters.string(), files.configuration.string(), true, {0, 127});
  OptVariables active = registerSelectedParameters(leader);
  std::unique_ptr<WaveFunctionComponent> clone_storage = leader.makeClone(batch_electrons1);
  auto* clone = dynamic_cast<PsiFormerWF*>(clone_storage.get());
  REQUIRE(clone != nullptr);
  clone->checkOutVariables(active);

  // Build the complete TrialWaveFunction drift independently for each walker.
  // The added factors emulate distinct surrounding wavefunction components.
  auto initialize_total_drift = [](PsiFormerWF& component, ParticleSet& electrons, double scale) {
    electrons.G = ValueType(0);
    electrons.L = ValueType(0);
    component.evaluateLog(electrons, electrons.G, electrons.L);
    for (int electron = 0; electron < electrons.getTotalNum(); ++electron)
      for (int dimension = 0; dimension < 3; ++dimension)
        electrons.G[electron][dimension] +=
            ValueType(scale * (1 + 3 * electron + dimension));
  };
  initialize_total_drift(leader, batch_electrons0, 0.007);
  initialize_total_drift(*clone, batch_electrons1, -0.011);

  RefVectorWithLeader<WaveFunctionComponent> components(leader, {leader, *clone});
  RefVectorWithLeader<ParticleSet> particles(
      batch_electrons0, {batch_electrons0, batch_electrons1});
  RecordArray<ValueType> batch_scores(2, active.size());
  RecordArray<ValueType> batch_kinetic(2, active.size());
  std::fill(batch_scores.begin(), batch_scores.end(), ValueType(0.25));
  std::fill(batch_kinetic.begin(), batch_kinetic.end(), ValueType(-0.5));

  ResourceCollection resource_template("psiformer_kinetic_pool_template");
  leader.createResource(resource_template);
  ResourceCollection crowd_resource(resource_template);
  {
    ResourceCollectionTeamLock<WaveFunctionComponent> lock(crowd_resource, components);
    const std::array<std::size_t, 2> no_kinetic_tapes{0, 0};
    CHECK(testing::TestPsiFormerWF::directKineticWorkspaceOwnership(leader, components) == no_kinetic_tapes);

    // Reject a heterogeneous-mass crowd before allocating the shared tape.
    SpeciesSet& second_species = batch_electrons1.getSpeciesSet();
    const int second_mass      = second_species.getAttribute("mass");
    REQUIRE(second_mass < second_species.numAttributes());
    second_species(second_mass, 1) = 2.0;
    batch_electrons1.resetGroups();
    CHECK_THROWS_WITH(
        leader.mw_evaluateParameterDerivatives(
            components, particles, active, batch_scores, batch_kinetic),
        Catch::Matchers::ContainsSubstring("require unit electron masses"));
    CHECK(testing::TestPsiFormerWF::directKineticWorkspaceOwnership(leader, components) == no_kinetic_tapes);
    second_species(second_mass, 1) = 1.0;
    batch_electrons1.resetGroups();

    leader.mw_evaluateParameterDerivatives(
        components, particles, active, batch_scores, batch_kinetic);

    // Neither component clone owns a kinetic tape; exactly one tape belongs to
    // the acquired crowd resource after the first component-major call.
    const std::array<std::size_t, 2> one_crowd_kinetic_tape{0, 1};
    CHECK(testing::TestPsiFormerWF::directKineticWorkspaceOwnership(leader, components) ==
          one_crowd_kinetic_tape);
  }

  // Independent scalar calls provide the numerical oracle and, because their
  // external drifts differ, catch failure to repack ParticleSet::G per walker.
  ParticleSet scalar_electrons0 = makeLiHElectrons(simulation_cell);
  ParticleSet scalar_electrons1 = makeLiHElectrons(simulation_cell);
  scalar_electrons1.R[0][0] += 0.11;
  scalar_electrons1.update();
  PsiFormerWF scalar0(
      "pf_kinetic_scalar0", files.parameters.string(), files.configuration.string(), true, {0, 127});
  PsiFormerWF scalar1(
      "pf_kinetic_scalar1", files.parameters.string(), files.configuration.string(), true, {0, 127});
  OptVariables scalar_active0 = registerSelectedParameters(scalar0);
  OptVariables scalar_active1 = registerSelectedParameters(scalar1);
  initialize_total_drift(scalar0, scalar_electrons0, 0.007);
  initialize_total_drift(scalar1, scalar_electrons1, -0.011);

  std::array<Vector<ValueType>, 2> scalar_scores{
      Vector<ValueType>(active.size()), Vector<ValueType>(active.size())};
  std::array<Vector<ValueType>, 2> scalar_kinetic{
      Vector<ValueType>(active.size()), Vector<ValueType>(active.size())};
  for (int walker = 0; walker < 2; ++walker)
  {
    scalar_scores[walker]  = ValueType(0.25);
    scalar_kinetic[walker] = ValueType(-0.5);
  }
  scalar0.evaluateDerivatives(
      scalar_electrons0, scalar_active0, scalar_scores[0], scalar_kinetic[0]);
  scalar1.evaluateDerivatives(
      scalar_electrons1, scalar_active1, scalar_scores[1], scalar_kinetic[1]);

  for (int walker = 0; walker < 2; ++walker)
    for (std::size_t parameter = 0; parameter < active.size(); ++parameter)
    {
      CHECK(std::abs(batch_scores[walker][parameter] - scalar_scores[walker][parameter]) <
            2e-10 * (1 + std::abs(scalar_scores[walker][parameter])));
      CHECK(std::abs(batch_kinetic[walker][parameter] - scalar_kinetic[walker][parameter]) <
            2e-9 * (1 + std::abs(scalar_kinetic[walker][parameter])));
    }
  CHECK(std::real(batch_kinetic[0][0]) != Approx(std::real(batch_kinetic[1][0])));
}

TEST_CASE("PsiFormer registration maps through surrounding ordinary parameters",
          "[wavefunction][psiformer]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  ParticleSet electrons = makeLiHElectrons(simulation_cell);
  PsiFormerWF component(
      "pf_block", files.parameters.string(), files.configuration.string(), true, {0, 1, 127});

  OptVariables active;
  active.insert("ordinary_before", -1.0, true, optimize::LINEAR_P);
  component.checkInVariablesExclusive(active);
  active.insert("ordinary_after", 1.0, true, optimize::LOGLINEAR_P);
  active.resetIndex();
  component.checkOutVariables(active);

  REQUIRE(active.size() == 5);
  CHECK(active.name(0) == "ordinary_before");
  CHECK(active.name(1) == "pf_block_pf_0000000");
  CHECK(active.name(2) == "pf_block_pf_0000001");
  CHECK(active.name(3) == "pf_block_pf_0000127");
  CHECK(active.name(4) == "ordinary_after");

  // PsiFormer must scatter its results only into mapped global entries,
  // leaving derivative contributions owned by neighboring objects untouched.
  electrons.G = ValueType(0);
  electrons.L = ValueType(0);
  component.evaluateLog(electrons, electrons.G, electrons.L);
  Vector<ValueType> dlogpsi(active.size());
  Vector<ValueType> dhpsioverpsi(active.size());
  dlogpsi      = ValueType(-91.0);
  dhpsioverpsi = ValueType(37.0);
  component.evaluateDerivatives(electrons, active, dlogpsi, dhpsioverpsi);

  CHECK(std::real(dlogpsi[0]) == Approx(-91.0));
  CHECK(std::real(dhpsioverpsi[0]) == Approx(37.0));
  CHECK(std::real(dlogpsi[4]) == Approx(-91.0));
  CHECK(std::real(dhpsioverpsi[4]) == Approx(37.0));
  for (int global_index = 1; global_index <= 3; ++global_index)
  {
    CHECK(psiformer::determinant::isFiniteReal(std::real(dlogpsi[global_index])));
    CHECK(psiformer::determinant::isFiniteReal(std::real(dhpsioverpsi[global_index])));
    CHECK(std::real(dlogpsi[global_index]) != Approx(-91.0));
    CHECK(std::real(dhpsioverpsi[global_index]) != Approx(37.0));
  }
}

TEST_CASE("PsiFormer clones share versioned parameters and retain local move state", "[wavefunction][psiformer]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  ParticleSet leader_electrons = makeLiHElectrons(simulation_cell);
  ParticleSet clone_electrons  = makeLiHElectrons(simulation_cell);

  PsiFormerWF leader(
      "pf_clone", files.parameters.string(), files.configuration.string(), true, std::vector<std::size_t>{0, 127});
  OptVariables active = registerSelectedParameters(leader);
  std::unique_ptr<WaveFunctionComponent> clone_base = leader.makeClone(clone_electrons);
  auto* clone = dynamic_cast<PsiFormerWF*>(clone_base.get());
  REQUIRE(clone != nullptr);

  const ComponentSnapshot initial_leader = evaluateComponent(leader, leader_electrons, active);
  const ComponentSnapshot initial_clone  = evaluateComponent(*clone, clone_electrons, active);
  checkComponentSnapshot(initial_clone, initial_leader);

  const std::size_t initial_version = leader.parameterVersion();
  active[0] += 2e-4;
  leader.resetParametersExclusive(active);
  CHECK(leader.parameterVersion() == initial_version + 1);
  CHECK(clone->parameterVersion() == initial_version + 1);
  OptVariables clone_active = registerSelectedParameters(*clone);
  CHECK(std::real(clone_active[0]) == Catch::Approx(std::real(active[0])));

  const ComponentSnapshot updated_leader = evaluateComponent(leader, leader_electrons, active);
  const ComponentSnapshot updated_clone  = evaluateComponent(*clone, clone_electrons, active);
  checkComponentSnapshot(updated_clone, updated_leader);
  CHECK(std::abs(updated_leader.log_value - initial_leader.log_value) > 1e-8);

  // Replaying the same global reset through a clone must not create another
  // model version or rewrite the shared parameter leaves.
  clone->resetParametersExclusive(active);
  CHECK(leader.parameterVersion() == initial_version + 1);

  // Cache a proposal in the clone, update through the leader, and verify that
  // accepting the now-stale proposal cannot promote its old log value.
  clone_electrons.makeMove(0, ParticleSet::SingleParticlePos{0.01, -0.02, 0.015});
  clone->ratio(clone_electrons, 0);
  active[0] += 1e-4;
  leader.resetParametersExclusive(active);
  CHECK(leader.parameterVersion() == initial_version + 2);
  clone->acceptMove(clone_electrons, 0);
  CHECK(std::real(clone->get_log_value()) == 0.0);
  clone_electrons.rejectMove(0);

  // Two clone-local evaluations may read the same immutable parameter version
  // concurrently. The shared lock excludes optimizer resets during each
  // native graph traversal.
  std::promise<void> start_promise;
  const std::shared_future<void> start = start_promise.get_future().share();
  auto evaluate_log = [](PsiFormerWF& component, ParticleSet& electrons, std::shared_future<void> gate) {
    gate.wait();
    electrons.G = ValueType(0);
    electrons.L = ValueType(0);
    return std::real(component.evaluateLog(electrons, electrons.G, electrons.L));
  };
  auto leader_future =
      std::async(std::launch::async, evaluate_log, std::ref(leader), std::ref(leader_electrons), start);
  auto clone_future =
      std::async(std::launch::async, evaluate_log, std::ref(*clone), std::ref(clone_electrons), start);
  start_promise.set_value();

  const double concurrent_leader_log = leader_future.get();
  const double concurrent_clone_log  = clone_future.get();
  CHECK(concurrent_clone_log ==
        Catch::Approx(concurrent_leader_log).epsilon(2e-10).margin(2e-10));
}

TEST_CASE("PsiFormer complete model persistence and DeepQMC export round trip", "[wavefunction][psiformer]")
{
  GeneratedFiles files = generateFiles("lih");
  const SimulationCell simulation_cell;
  ParticleSet original_electrons = makeLiHElectrons(simulation_cell);

  const std::vector<std::size_t> selected_indices{0, 127, 2047};
  const std::filesystem::path export_path = files.directory / "parameters_optimized.h5";
  PsiFormerWF original(
      "pf_restart", files.parameters.string(), files.configuration.string(), true, selected_indices, false,
      export_path.string());
  OptVariables original_active = registerSelectedParameters(original);
  original_active[0] += 1.5e-4;
  original_active[1] -= 2.0e-4;
  original_active[2] += 2.5e-4;
  original.resetParametersExclusive(original_active);
  const ComponentSnapshot expected = evaluateComponent(original, original_electrons, original_active);

  const std::filesystem::path vp_path = files.directory / "psiformer_restart.vp.h5";
  hdf_archive output;
  original_active.writeToHDF(vp_path.string(), output);
  original.writeVariationalParameters(output);
  output.close();
  CHECK(std::filesystem::exists(export_path));

  ParticleSet restored_electrons = makeLiHElectrons(simulation_cell);
  PsiFormerWF restored(
      "pf_restart", files.parameters.string(), files.configuration.string(), true, selected_indices);
  OptVariables restored_active = registerSelectedParameters(restored);
  hdf_archive input;
  restored_active.readFromHDF(vp_path.string(), input);
  restored.readVariationalParameters(input);
  input.close();
  restored.resetParametersExclusive(restored_active);

  const ComponentSnapshot restarted = evaluateComponent(restored, restored_electrons, restored_active);
  checkComponentSnapshot(restarted, expected);

  // The explicit export is intentionally separate from optimizer restart: it
  // contains only the DeepQMC flat values and immutable tensor layout.
  ParticleSet exported_electrons = makeLiHElectrons(simulation_cell);
  PsiFormerWF exported(
      "pf_export", export_path.string(), files.configuration.string(), true, selected_indices);
  OptVariables exported_active = registerSelectedParameters(exported);
  const ComponentSnapshot exported_snapshot = evaluateComponent(exported, exported_electrons, exported_active);
  checkComponentSnapshot(exported_snapshot, expected);

  // A restart payload cannot silently bind to a different internal selection.
  PsiFormerWF mismatched_selection(
      "pf_restart", files.parameters.string(), files.configuration.string(), true, {0, 128, 2047});
  hdf_archive mismatch_input;
  REQUIRE(mismatch_input.open(vp_path.string(), H5F_ACC_RDONLY));
  CHECK_THROWS_AS(mismatched_selection.readVariationalParameters(mismatch_input), std::runtime_error);
  mismatch_input.close();

  // The compact generic selected list is duplicated for compatibility. It
  // must agree with the authoritative complete model payload on restart.
  PsiFormerWF inconsistent_generic(
      "pf_restart", files.parameters.string(), files.configuration.string(), true, selected_indices);
  OptVariables inconsistent_active = registerSelectedParameters(inconsistent_generic);
  hdf_archive inconsistent_input;
  inconsistent_active.readFromHDF(vp_path.string(), inconsistent_input);
  inconsistent_generic.readVariationalParameters(inconsistent_input);
  inconsistent_input.close();
  inconsistent_active[0] += 1e-3;
  CHECK_THROWS_AS(inconsistent_generic.resetParametersExclusive(inconsistent_active), std::runtime_error);
}

} // namespace qmcplusplus
