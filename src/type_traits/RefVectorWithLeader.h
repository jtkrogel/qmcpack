//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2021 QMCPACK developers
//
// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
//////////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_REFVECTORWITHLEADER_H
#define QMCPLUSPLUS_REFVECTORWITHLEADER_H

#include <cassert>
#include <functional>
#include <memory>
#include <type_traits>
#include <vector>

namespace qmcplusplus
{
template<typename T>
class RefVectorWithLeader : public std::vector<std::reference_wrapper<T>>
{
public:
  using BaseVec = std::vector<std::reference_wrapper<T>>;

  RefVectorWithLeader(T& leader) : leader_(leader) {}

  RefVectorWithLeader(T& leader, const BaseVec& vec) : leader_(leader)
  {
    for (T& element : vec)
      this->push_back(element);
  }

  RefVectorWithLeader(T& leader, BaseVec&& vec) : BaseVec(std::move(vec)), leader_(leader) {}

  T& getLeader() const { return leader_; }

  /** Rebind the leader wrapper without assigning to the previously referred object. */
  void rebindLeader(T& leader) noexcept
  {
    static_assert(std::is_nothrow_copy_assignable_v<std::reference_wrapper<T>>,
                  "Rebinding a reference wrapper must not throw.");
    leader_ = std::ref(leader);
  }

  /** Rebind an existing element slot without changing the vector's storage or size. */
  void rebindElement(size_t i, T& element) noexcept
  {
    static_assert(std::is_nothrow_copy_assignable_v<std::reference_wrapper<T>>,
                  "Rebinding a reference wrapper must not throw.");
    assert(i < BaseVec::size());
    // operator[] below returns T&, so qualify the base operation to assign the wrapper itself.
    BaseVec::operator[](i) = std::ref(element);
  }

  T& operator[](size_t i) const { return BaseVec::operator[](i).get(); }

  template<typename CASTTYPE>
  CASTTYPE& getCastedLeader() const
  {
    static_assert(std::is_const<T>::value == std::is_const<CASTTYPE>::value, "Unmatched const type qualifier!");
    assert(dynamic_cast<CASTTYPE*>(&leader_.get()) != nullptr);
    return static_cast<CASTTYPE&>(leader_.get());
  }

  template<typename CASTTYPE>
  CASTTYPE& getCastedElement(size_t i) const
  {
    static_assert(std::is_const<T>::value == std::is_const<CASTTYPE>::value, "Unmatched const type qualifier!");
    assert(dynamic_cast<CASTTYPE*>(&(*this)[i]) != nullptr);
    return static_cast<CASTTYPE&>((*this)[i]);
  }

private:
  std::reference_wrapper<T> leader_;
};
} // namespace qmcplusplus

#endif
