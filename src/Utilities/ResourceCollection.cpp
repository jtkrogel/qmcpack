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
}

ResourceCollection::ResourceCollection(ResourceCollection&& ref)
    : name_(ref.getName()), cursor_index_(0), outstanding_loans_(0)
{
  if (ref.outstanding_loans_ != 0)
    throw std::logic_error("Cannot move a ResourceCollection while resources are acquired");

  cursor_index_     = ref.cursor_index_;
  collection_       = std::move(ref.collection_);
  ref.cursor_index_ = 0;
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

ResourceCollection ResourceCollection::makePreparedBatchResources(
    const BatchResourcePreparationContext& context) const
{
  // Validate all collection-wide preconditions before cloning or invoking a
  // resource hook.  Cursor position is traversal state, not loan ownership:
  // normal release leaves it at the end and callers may rewind it at any time.
  context.validate();
  if (outstanding_loans_ != 0)
    throw std::logic_error("Cannot prepare a ResourceCollection while resources are acquired");

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
  return prepared;
}

void ResourceCollection::swapResourceStorage(ResourceCollection& other) noexcept
{
  using std::swap;
  swap(cursor_index_, other.cursor_index_);
  swap(outstanding_loans_, other.outstanding_loans_);
  collection_.swap(other.collection_);
}

void ResourceCollection::prepareBatchResources(const BatchResourcePreparationContext& context)
{
  ResourceCollection prepared = makePreparedBatchResources(context);
  swapResourceStorage(prepared);
}

size_t ResourceCollection::addResource(std::unique_ptr<Resource>&& res, bool noprint)
{
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
