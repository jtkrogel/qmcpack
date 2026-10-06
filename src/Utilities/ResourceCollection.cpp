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

#include "ResourceCollection.h"
#include "BatchResourcePreparation.h"

#include <iostream>
#include <stdexcept>
#include <utility>

#include <Host/OutputManager.h>

namespace qmcplusplus
{
ResourceCollection::ResourceCollection(const std::string& name) : name_(name), cursor_index_(0), outstanding_loans_(0)
{}

ResourceCollection::ResourceCollection(const ResourceCollection& ref)
    : name_(ref.getName()), cursor_index_(0), outstanding_loans_(0)
{
  for (auto& res : ref.collection_)
    addResource(std::unique_ptr<Resource>(res->makeClone()), true);

  batch_preparation_ = ref.batch_preparation_;
  if (batch_preparation_.state != BatchResourcePreparationState::UNPREPARED)
    batch_preparation_.state = BatchResourcePreparationState::DERIVED_REQUIRES_CLEAR;
}

ResourceCollection::ResourceCollection(ResourceCollection&& ref)
    : name_(ref.getName()), cursor_index_(0), outstanding_loans_(0)
{
  if (ref.outstanding_loans_ != 0)
    throw std::logic_error("Cannot move a ResourceCollection while resources are acquired");

  cursor_index_      = ref.cursor_index_;
  collection_        = std::move(ref.collection_);
  batch_preparation_ = std::move(ref.batch_preparation_);
  ref.cursor_index_ = 0;
  ref.batch_preparation_ = {};
}

void ResourceCollection::printResources(std::ostream& os) const
{
  os << "list resources in " << getName() << std::endl;
  os << "-------------------------------" << std::endl;
  for (int i = 0; i < collection_.size(); i++)
    os << "resource " << i << "    name: " << collection_[i]->getName() << "    address: " << collection_[i].get()
       << std::endl;
  os << "-------------------------------" << std::endl << std::endl;
}

void ResourceCollection::validateBatchResourcePreparation(const BatchResourcePreparationContext& context) const
{
  context.validate();
  if (outstanding_loans_ != 0)
    throw std::logic_error("Cannot prepare a ResourceCollection while resources are acquired");
  if (context.plan && batch_preparation_.state != BatchResourcePreparationState::UNPREPARED)
    throw std::logic_error(
        "Cannot apply a nonnull batch plan to prepared or prepared-derived ResourceCollection storage; "
        "clear it with a null preparation context first");

  for (const std::unique_ptr<Resource>& resource : collection_)
    resource->validateBatchResourcePreparation(context);
}

ResourceCollection ResourceCollection::makePreparedBatchResourcesAfterValidation(
    const BatchResourcePreparationContext& context) const
{
  ResourceCollection prepared(name_);
  prepared.collection_.reserve(collection_.size());
  for (const std::unique_ptr<Resource>& resource : collection_)
  {
    std::unique_ptr<Resource> clone = resource->makeClone();
    if (!clone)
      throw std::logic_error("Resource::makeClone returned null during batch resource preparation");

    clone->index_in_collection_ = static_cast<int>(prepared.collection_.size());
    clone->prepareBatchResource(context);
    prepared.collection_.emplace_back(std::move(clone));
  }

  if (context.plan)
    prepared.batch_preparation_ =
        {BatchResourcePreparationState::PREPARED, context.plan, context.crowd_index};
  return prepared;
}

ResourceCollection ResourceCollection::makePreparedBatchResources(
    const BatchResourcePreparationContext& context) const
{
  // Cursor position is traversal state, not loan ownership: normal release
  // leaves it at the end and callers may rewind it at any time.
  validateBatchResourcePreparation(context);
  return makePreparedBatchResourcesAfterValidation(context);
}

void ResourceCollection::swapResourceStorage(ResourceCollection& other) noexcept
{
  using std::swap;
  swap(cursor_index_, other.cursor_index_);
  swap(outstanding_loans_, other.outstanding_loans_);
  collection_.swap(other.collection_);
  swap(batch_preparation_, other.batch_preparation_);
}

void ResourceCollection::prepareBatchResources(const BatchResourcePreparationContext& context)
{
  ResourceCollection prepared = makePreparedBatchResources(context);
  swapResourceStorage(prepared);
}

size_t ResourceCollection::addResource(std::unique_ptr<Resource>&& res, bool noprint)
{
  if (batch_preparation_.state != BatchResourcePreparationState::UNPREPARED)
    throw std::logic_error("Cannot add a resource to prepared or prepared-derived ResourceCollection storage");

  size_t index              = collection_.size();
  res->index_in_collection_ = index;
  if (!noprint)
    app_debug_stream() << "Multi walker shared resource \"" << res->getName() << "\" created in resource collection \""
                       << name_ << "\" index " << index << std::endl;
  collection_.emplace_back(std::move(res));
  return index;
}

Resource& ResourceCollection::lendResourceImpl()
{
  if (cursor_index_ >= collection_.size())
    throw std::runtime_error("ResourceCollection::lendResource BUG no more resource to lend.");
  if (cursor_index_ != collection_[cursor_index_]->index_in_collection_)
    throw std::runtime_error(
        "ResourceCollection::lendResource BUG mismatched cursor index and recorded index in the resource.");
  return *collection_[cursor_index_++];
}

void ResourceCollection::takebackResourceImpl(Resource& res)
{
  if (cursor_index_ >= collection_.size())
    throw std::runtime_error("ResourceCollection::takebackResource BUG cannot take back resources more than owned.");
  if (cursor_index_ != res.index_in_collection_)
    throw std::runtime_error(
        "ResourceCollection::takebackResource BUG mismatched cursor index and recorded index in the resource.");
  if (&res != collection_[cursor_index_].get())
    throw std::runtime_error(
        "ResourceCollection::takebackResource BUG the resource taken back mismatches the one lent.");
  ++cursor_index_;
}

} // namespace qmcplusplus
