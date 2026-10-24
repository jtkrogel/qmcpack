//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//////////////////////////////////////////////////////////////////////////////////////

/** @file KFACDenseSolver.h
 * @brief Optional vendor FP64 Cholesky/application resources for existing KFAC blocks.
 *
 * Canonical HOST factors and checkpoint semantics remain independent of this
 * execution resource. No provider, curvature, damping or EMA policy is owned here.
 */
#ifndef QMCPLUSPLUS_KFAC_DENSE_SOLVER_H
#define QMCPLUSPLUS_KFAC_DENSE_SOLVER_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace qmcplusplus::wftrain
{
class KFACBlockRegistry;
class KFACDenseGeneration;

/// AUTO selects admitted residency; explicit alternatives are focused-test controls only.
enum class KFACDenseCapacity { AUTO, RESIDENT, STREAMED };

/** Optimizer-local queue, vendor handles and fixed factor/RHS/workspace capacity. */
class KFACDenseSolver : public std::enable_shared_from_this<KFACDenseSolver>
{
public:
  /// Prepare before collection; zero DEVICE uses remaining existing aggregate admission.
  static std::shared_ptr<KFACDenseSolver> prepare(const KFACBlockRegistry& registry,
      std::size_t maximum_device_bytes, std::size_t maximum_host_bytes,
      KFACDenseCapacity preference = KFACDenseCapacity::AUTO);
  ~KFACDenseSolver();
  KFACDenseSolver(const KFACDenseSolver&) = delete;
  KFACDenseSolver& operator=(const KFACDenseSolver&) = delete;

  /// Reserve one of two immutable accepted/pending generation slots.
  std::shared_ptr<KFACDenseGeneration> beginGeneration();
  KFACDenseCapacity capacity() const noexcept;
  std::size_t deviceBytes() const noexcept;
  std::size_t hostBytes() const noexcept;
  std::size_t workspaceBytes() const noexcept;
  const std::string& registryFingerprint() const noexcept;

private:
  struct Impl;
  explicit KFACDenseSolver(std::unique_ptr<Impl> implementation);
  friend class KFACDenseGeneration;
  std::unique_ptr<Impl> impl_;
};

/** Immutable complete Cholesky generation; destruction releases its slot only. */
class KFACDenseGeneration
{
public:
  ~KFACDenseGeneration();
  KFACDenseGeneration(const KFACDenseGeneration&) = delete;
  KFACDenseGeneration& operator=(const KFACDenseGeneration&) = delete;

  /// Factor fresh damped matrices and return canonical row-major lower factors.
  void factorize(std::size_t block, std::vector<double>& activation, std::vector<double>& sensitivity);

  /// Import checkpoint factors exactly, without refactorization.
  void import(std::size_t block, const std::vector<double>& activation, const std::vector<double>& sensitivity);

  /// Admit only a complete generation; failed/incomplete generations cannot apply.
  void finish();

  /// Verify a borrowed inverse-period generation still represents exact HOST factors.
  bool matches(std::size_t block, const std::vector<double>& activation,
      const std::vector<double>& sensitivity) const;

  /// Apply A^{-1} R G^{-1} to one existing row-major augmented REAL matrix.
  void apply(std::size_t block, const std::vector<double>& activation,
      const std::vector<double>& sensitivity, double* matrix, std::size_t count) const;

  const std::shared_ptr<KFACDenseSolver>& resource() const noexcept { return resource_; }

private:
  friend class KFACDenseSolver;
  KFACDenseGeneration(std::shared_ptr<KFACDenseSolver> resource, std::size_t slot, std::size_t blocks);
  std::shared_ptr<KFACDenseSolver> resource_;
  std::size_t slot_;
  std::vector<unsigned char> prepared_;
  std::vector<std::uint64_t> fingerprints_;
  bool complete_ = false;
  bool failed_ = false;
};
}
#endif
