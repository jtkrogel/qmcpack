//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2021 QMCPACK developers.
//
// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//////////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_RESOURCECOLLECTION_H
#define QMCPLUSPLUS_RESOURCECOLLECTION_H

#include <string>
#include <memory>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>
#include "Resource.h"
#include "ResourceHandle.h"
#include "type_traits/RefVectorWithLeader.h"

namespace qmcplusplus
{
struct BatchResourcePreparationContext;
class BatchExecutionPlan;
struct DriverWalkerResourceCollection;
template<class CONSUMER>
class ResourceCollectionTeamLock;

/** Describes whether collection storage may accept a nonnull batch plan. */
enum class BatchResourcePreparationState
{
  UNPREPARED,
  PREPARED,
  DERIVED_REQUIRES_CLEAR
};

/** Records the plan identity carried by prepared or derived collection storage. */
struct BatchResourcePreparationProvenance
{
  BatchResourcePreparationState state = BatchResourcePreparationState::UNPREPARED;
  std::shared_ptr<const BatchExecutionPlan> plan;
  std::size_t crowd_index = 0;
};

/** Owns the ordered resource clones lent to one multi-walker consumer family. */
class ResourceCollection
{
public:
  ResourceCollection(const std::string& name);
  ResourceCollection(const ResourceCollection&);
  ResourceCollection(ResourceCollection&&);

  const std::string& getName() const { return name_; }

  size_t size() const { return collection_.size(); }
  bool empty() const { return collection_.size() == 0; }

  /** Return immutable preparation provenance for acquisition-time validation. */
  const BatchResourcePreparationProvenance& getBatchResourcePreparationProvenance() const noexcept
  {
    return batch_preparation_;
  }

  size_t addResource(std::unique_ptr<Resource>&& res, bool noprint = false);
  void printResources(std::ostream& os) const;

  /**
   * Prepare every resource transactionally while the collection is idle.
   *
   * A nonnull plan may be applied only to unprepared template storage.  A
   * null context rebuilds no-policy resources and clears prior provenance.
   */
  void prepareBatchResources(const BatchResourcePreparationContext& context);

  /**
   * Validate collection and resource preconditions without cloning or mutation.
   *
   * Compound owners use this preflight to validate all related collections
   * before allocating the first replacement collection.
   */
  void validateBatchResourcePreparation(const BatchResourcePreparationContext& context) const;

  template<class RS>
  ResourceHandle<RS> lendResource()
  {
    const size_t cursor_begin = cursor_index_;
    try
    {
      RS& resource = dynamic_cast<RS&>(lendResourceImpl());
      ResourceHandle<RS> handle(resource);
      if (outstanding_loans_ == std::numeric_limits<size_t>::max())
        throw std::overflow_error("ResourceCollection outstanding loan count overflow");
      ++outstanding_loans_;
      return handle;
    }
    catch (...)
    {
      cursor_index_ = cursor_begin;
      throw;
    }
  }

  template<class RS>
  void takebackResource(ResourceHandle<RS>& res_handle)
  {
    // Do not clear the caller's handle until every ownership and ordering
    // check succeeds.  A failed return remains retryable after a rewind.
    RS& resource = res_handle.getResource();
    if (outstanding_loans_ == 0)
      throw std::logic_error("ResourceCollection has no outstanding resource loan to take back");
    takebackResourceImpl(resource);
    res_handle.release();
    --outstanding_loans_;
  }

  /// Return the next collection slot, for transactional acquisition rollback.
  size_t getCursor() const noexcept { return cursor_index_; }

  /// Return the number of lent resources whose handles have not been returned.
  size_t getOutstandingLoanCount() const noexcept { return outstanding_loans_; }

  void rewind(size_t cursor = 0) { cursor_index_ = cursor; }

private:
  /** Build a fully prepared clone without changing this collection. */
  ResourceCollection makePreparedBatchResources(const BatchResourcePreparationContext& context) const;

  /** Build a prepared clone after the caller has completed collection preflight. */
  ResourceCollection makePreparedBatchResourcesAfterValidation(
      const BatchResourcePreparationContext& context) const;

  /** Publish already-prepared storage without an allocation or exception. */
  void swapResourceStorage(ResourceCollection& other) noexcept;

  /** Restore the ownership checkpoint after a failed team-lock acquisition. */
  void restoreOutstandingLoanCount(size_t count) noexcept { outstanding_loans_ = count; }

  Resource& lendResourceImpl();
  void takebackResourceImpl(Resource& res);

  const std::string name_;
  size_t cursor_index_;
  size_t outstanding_loans_;
  std::vector<std::unique_ptr<Resource>> collection_;
  BatchResourcePreparationProvenance batch_preparation_;

  friend struct DriverWalkerResourceCollection;
  template<class CONSUMER>
  friend class ResourceCollectionTeamLock;
};

/** handles acquire/release resource by the consumer (RefVectorWithLeader type).
 *  A failed construction restores the collection cursor. The consumer remains
 *  responsible for any handles it published before throwing.
 */
template<class CONSUMER>
class ResourceCollectionTeamLock
{
public:
  ResourceCollectionTeamLock(ResourceCollection& res_ref,
                             const RefVectorWithLeader<CONSUMER>& consumer_ref,
                             size_t cursor = 0)
      : resource(res_ref),
        consumer(consumer_ref),
        cursor_begin_(cursor),
        active(!res_ref.empty()),
        outstanding_loans_begin_(res_ref.getOutstandingLoanCount())
  {
    if (active)
    {
      resource.rewind(cursor_begin_);
      try
      {
        consumer.getLeader().acquireResource(resource, consumer);
      }
      catch (...)
      {
        // A consumer owns unwinding any handles it published before throwing.
        // Restore collection traversal and ownership bookkeeping as the final
        // construction-failure guarantee.
        resource.rewind(cursor_begin_);
        resource.restoreOutstandingLoanCount(outstanding_loans_begin_);
        throw;
      }
    }
  }

  ~ResourceCollectionTeamLock()
  {
    if (active)
    {
      resource.rewind(cursor_begin_);
      consumer.getLeader().releaseResource(resource, consumer);
    }
  }

  ResourceCollectionTeamLock(const ResourceCollectionTeamLock&) = delete;
  ResourceCollectionTeamLock(ResourceCollectionTeamLock&&)      = delete;

private:
  ResourceCollection& resource;
  const RefVectorWithLeader<CONSUMER>& consumer;
  const size_t cursor_begin_;
  const bool active;
  const size_t outstanding_loans_begin_;
};

} // namespace qmcplusplus
#endif
