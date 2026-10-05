//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerWorkspace.h
 * @brief Reusable aligned storage and non-owning C++17 views for PsiFormer kernels.
 *
 * Workspace growth is explicit and occurs before evaluator kernels run.  Once a
 * requested capacity has been prepared, repeated evaluations construct only
 * lightweight views and perform no workspace allocation.  Separate allocation
 * and layout generations make stale-view detection available to tests and
 * diagnostic builds.
 */

#ifndef QMCPLUSPLUS_PSIFORMER_WORKSPACE_H
#define QMCPLUSPLUS_PSIFORMER_WORKSPACE_H

#include "CPU/SIMD/aligned_allocator.hpp"
#include "QMCWaveFunctions/PsiFormer/PsiFormerExecutionPlan.h"

#include <array>
#include <cstddef>
#include <type_traits>

namespace qmcplusplus::psiformer
{

/// Non-owning contiguous view used instead of C++20 std::span.
template<class T>
class ArrayView
{
public:
  using value_type = std::remove_cv_t<T>;

  /// Construct an empty view.
  ArrayView() = default;

  /// Construct a view over `size` elements beginning at `data`.
  ArrayView(T* data, std::size_t size) : data_(data), size_(size) {}

  /// Convert a mutable view to its const-qualified counterpart.
  template<class U, std::enable_if_t<std::is_const_v<T> && std::is_same_v<std::remove_const_t<T>, U>, int> = 0>
  ArrayView(const ArrayView<U>& other) : data_(other.data()), size_(other.size())
  {}

  /// Return the first element address, or nullptr for an empty view.
  T* data() const { return data_; }

  /// Return the number of elements in the view.
  std::size_t size() const { return size_; }

  /// Report whether the view contains no elements.
  bool empty() const { return size_ == 0; }

  /// Access one element without bounds checking, matching std::span semantics.
  T& operator[](std::size_t index) const { return data_[index]; }

  /// Return an iterator to the first element.
  T* begin() const { return data_; }

  /// Return an iterator one past the last element.
  T* end() const { return size_ == 0 ? data_ : data_ + size_; }

private:
  T* data_          = nullptr;
  std::size_t size_ = 0;
};

/// Identify independently sized scratch regions used by the direct evaluator.
enum class WorkspaceRegion : std::size_t
{
  GEOMETRY,
  FEATURES,
  ATTENTION,
  ORBITALS,
  DETERMINANTS,
  FIRST_DERIVATIVES,
  SECOND_DERIVATIVES,
  REVERSE_ADJOINTS,
  PARAMETER_OUTPUT,
  REDUCTIONS,
  COUNT
};

inline constexpr std::size_t WORKSPACE_REGION_COUNT = static_cast<std::size_t>(WorkspaceRegion::COUNT);

/// Select the high-level output family that determines scratch capacity.
enum class EvaluationMode
{
  VALUE_ONLY,
  ACTIVE_ELECTRON_GRADIENT,
  FULL_VGL,
  PARAMETER_SCORE,
  PARAMETER_SCORE_AND_KINETIC,
  VIRTUAL_RATIOS,
  VIRTUAL_WEIGHTED_PARAMETER_VJP
};

/** Describe maximum work submitted to one reusable workspace. */
struct WorkspaceWorkload
{
  EvaluationMode mode             = EvaluationMode::VALUE_ONLY;
  std::size_t walkers             = 1;
  std::size_t virtual_positions   = 0;
  std::size_t derivative_lanes    = 0;
};

/** Hold exact per-region element requirements for one workload. */
struct WorkspaceRequirements
{
  WorkspaceWorkload workload;
  std::array<std::size_t, WORKSPACE_REGION_COUNT> elements{};

  /// Return the required element count for one scratch region.
  std::size_t operator[](WorkspaceRegion region) const { return elements[static_cast<std::size_t>(region)]; }
};

/** A set of non-owning region views invalidated by workspace layout growth. */
template<class T>
struct BasicWorkspaceView
{
  std::array<ArrayView<T>, WORKSPACE_REGION_COUNT> regions;
  std::size_t allocation_generation = 0;
  std::size_t layout_generation     = 0;

  /// Return the non-owning view for one named scratch region.
  ArrayView<T> operator[](WorkspaceRegion region) const
  {
    return regions[static_cast<std::size_t>(region)];
  }
};

/**
 * Own aligned scratch memory while exposing only non-owning region views.
 *
 * Region capacities grow monotonically.  `prepare` allocates only when some
 * requested region exceeds its high-water mark; callers should invoke it at a
 * crowd/setup boundary rather than inside per-walker evaluation calls.
 */
template<class Scalar>
class BasicPsiFormerWorkspace
{
public:
  using WorkspaceView      = BasicWorkspaceView<Scalar>;
  using ConstWorkspaceView = BasicWorkspaceView<const Scalar>;

  /// Ensure every region can hold the requested workload and return whether growth occurred.
  bool prepare(const WorkspaceRequirements& requirements);

  /// Return mutable views into the current region layout without allocating.
  WorkspaceView view();

  /// Return const views into the current region layout without allocating.
  ConstWorkspaceView view() const;

  /// Return the high-water capacity of one region in scalar elements.
  std::size_t capacity(WorkspaceRegion region) const
  {
    return capacities_[static_cast<std::size_t>(region)];
  }

  /// Return the total number of aligned scalar slots owned by the workspace.
  std::size_t storageSize() const { return storage_.size(); }

  /// Count underlying storage replacements caused by capacity growth.
  std::size_t allocationGeneration() const { return allocation_generation_; }

  /// Count offset/layout changes that invalidate previously returned views.
  std::size_t layoutGeneration() const { return layout_generation_; }

  /// Test whether a previously returned mutable or const view remains current.
  template<class T>
  bool isCurrent(const BasicWorkspaceView<T>& candidate) const
  {
    return candidate.allocation_generation == allocation_generation_ &&
        candidate.layout_generation == layout_generation_;
  }

private:
  std::array<std::size_t, WORKSPACE_REGION_COUNT> capacities_{};
  std::array<std::size_t, WORKSPACE_REGION_COUNT> offsets_{};
  aligned_vector<Scalar> storage_;
  std::size_t allocation_generation_ = 0;
  std::size_t layout_generation_     = 0;
};

/// Real-scalar workspace instantiated by the current molecular evaluator.
using PsiFormerWorkspace = BasicPsiFormerWorkspace<double>;

/// Avoid duplicate real-workspace instantiations across translation units.
extern template class BasicPsiFormerWorkspace<double>;

/// Compute mode-specific region sizes for one plan and maximum workload.
WorkspaceRequirements makeWorkspaceRequirements(const PsiFormerExecutionPlan& plan,
                                                WorkspaceWorkload workload);

/// Return a stable diagnostic name for a workspace region.
const char* workspaceRegionName(WorkspaceRegion region);

} // namespace qmcplusplus::psiformer

#endif
