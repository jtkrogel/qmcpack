//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in the QMCPACK source tree for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
/** @file PsiFormerComplexWF.h
 * @brief Native-free declaration of the explicit CPU paired-head inference sibling.
 * Builder admission remains disabled until clone/persistence and rejection gates close.
 */
#ifndef QMCPLUSPLUS_PSIFORMER_COMPLEX_WF_H
#define QMCPLUSPLUS_PSIFORMER_COMPLEX_WF_H
#include "QMCWaveFunctions/WaveFunctionComponent.h"
#include <array>
#include <memory>
#include <string>

namespace qmcplusplus
{
namespace psiformer
{
struct InitializedPsiFormerParameters;
}

#ifdef QMC_COMPLEX
/// CPU full/G-free complex inference; real optimization and device paths are not inherited.
class PsiFormerComplexWF final : public WaveFunctionComponent
{
public:
  /// Consume one explicit paired profile and bind immutable geometry to the target.
  PsiFormerComplexWF(std::string name, psiformer::InitializedPsiFormerParameters initialized,
                     const ParticleSet& electrons, const ParticleSet& ions,
                     std::size_t maximum_clone_numeric_bytes,
                     std::array<double, 3> reduced_twist = {});

  /// Destroy the Native-free PIMPL in its sole native implementation translation unit.
  ~PsiFormerComplexWF() override;

  /// Numerical workspaces and move state cannot be copied implicitly.
  PsiFormerComplexWF(const PsiFormerComplexWF&) = delete;

  /// Preserve independently owned scalar scratch and identity.
  PsiFormerComplexWF& operator=(const PsiFormerComplexWF&) = delete;

  /// Report the explicitly qualified component rather than impersonating the real adapter.
  std::string getClassName() const override { return "PsiFormerComplexWF"; }

  /// This component supplies the complete antisymmetric fermionic ansatz.
  bool isFermionic() const override { return true; }

  /// Report the prepared determinant expansion's actual extent.
  bool isMultiDet() const override;

  /// Add full physical logarithmic G/L and publish the accepted scalar amplitude.
  LogValue evaluateLog(const ParticleSet&, ParticleSet::ParticleGradient&, ParticleSet::ParticleLaplacian&) override;

  /// Evaluate the value-only active replacement without committing it.
  PsiValue ratio(ParticleSet&, int) override;

  /// Return one accepted physical logarithmic gradient.
  GradType evalGrad(ParticleSet&, int) override;

  /// Add a proposed physical gradient and cache its finite ratio atomically.
  PsiValue ratioGrad(ParticleSet&, int, GradType&) override;

  /// Commit only a matching cached single-electron proposal.
  void acceptMove(ParticleSet&, int, bool safe_to_delay = false) override;

  /// Discard only the matching pending proposal.
  void restore(int) override;

  /// Share immutable parameters, but prepare independent scratch and no accepted/proposal state.
  std::unique_ptr<WaveFunctionComponent> makeClone(ParticleSet&) const override;

  /// Per-clone numeric buffers only: excludes shared model, allocator overhead, ParticleSet and WFBuffer.
  struct NumericStorage
  {
    std::size_t required_bytes, allocated_bytes, limit_bytes, fingerprint;
  };

  /// Report exact preparation; this is not an aggregate budget or adaptive large-system qualification.
  NumericStorage numericStorage() const;

  /// Fail closed until the explicit complex walker record is qualified.
  void registerData(ParticleSet&, WFBufferType&) override;

  /// Do not reuse the legacy real-sign persistence record.
  LogValue updateBuffer(ParticleSet&, WFBufferType&, bool from_scratch = false) override;

  /// Reject restoration of unqualified real or complex buffer data.
  void copyFromBuffer(ParticleSet&, WFBufferType&) override;

  /// Complex score availability alone does not qualify public kinetic optimization.
  void evaluateDerivatives(ParticleSet&, const OptVariables&, Vector<ValueType>&, Vector<ValueType>&) override;

private:
  struct Impl;
  /// Internal clone entry point retains immutable model ownership and the explicit numeric cap.
  PsiFormerComplexWF(std::string name, std::unique_ptr<Impl>);
  std::unique_ptr<Impl> impl_;
};
#endif
} // namespace qmcplusplus
#endif
