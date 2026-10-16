//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_MODEL_FILE_H
#define QMCPLUSPLUS_PSIFORMER_MODEL_FILE_H

#include "QMCWaveFunctions/PsiFormer/PsiFormerInitialization.h"
#include <array>
#include <string>

namespace qmcplusplus
{
class ParticleSet;
namespace psiformer
{
/// Standalone model import: only weights and physical metadata, never sampler/optimizer state.
struct PsiFormerModelFile
{
  InitializedPsiFormerParameters parameters;
  bool paired_complex = false;
  bool periodic = false;
  std::string system;
  std::array<double, 3> twist{};
};
PsiFormerModelFile readPsiFormerModelFile(const std::string& path);
void validatePsiFormerModelFile(const PsiFormerModelFile& model, const std::string& path,
                               const ParticleSet& electrons, const ParticleSet& ions,
                               const std::string& system, const std::array<double, 3>& twist);
}
}
#endif
