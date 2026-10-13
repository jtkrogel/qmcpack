//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Mark Dewing, markdewing@gmail.com, University of Illinois at Urbana-Champaign
//
// File created by: Mark Dewing, markdewing@gmail.com, University of Illinois at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////


#include "FakeRandom.h"

#include <exception>
#include <type_traits>
#include <utility>

namespace qmcplusplus
{
template<class T>
FakeRandom<T>::FakeRandom() = default;

template<class T>
void FakeRandom<T>::set_value(double val)
{
  m_val = val;
}

template<class T>
T FakeRandom<T>::operator()()
{
  return m_val;
}

template<typename T>
void FakeRandom<T>::swapState(RandomBase<T>& other) noexcept
{
  static_assert(std::is_nothrow_swappable_v<T>);
  auto* typed_other = dynamic_cast<FakeRandom<T>*>(&other);
  if (!typed_other)
    std::terminate();
  using std::swap;
  swap(m_val, typed_other->m_val);
}

template<typename T>
RandomBase<T>& FakeRandom<T>::operator=(const RandomBase<T>& other)
{
  return *this = dynamic_cast<const FakeRandom<T>&>(other);
}

template class FakeRandom<float>;
template class FakeRandom<double>;

} // namespace qmcplusplus
