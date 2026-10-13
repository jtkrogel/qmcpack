//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2023 QMCPACK developers.
//
// File developed by: Peter Doak, doakpw@ornl.gov, Oak Ridge National Laboratory
//                    Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//                    Steven Hahn, hahnse@ornl.gov, Oak Ridge National Laboratory
//
//////////////////////////////////////////////////////////////////////////////////////

#include "StdRandom.h"

#include <exception>
#include <type_traits>
#include <utility>
namespace qmcplusplus
{
template<typename T>
StdRandom<T>::StdRandom(uint_type iseed) : engine(iseed)
{
  static_assert(std::is_same_v<Engine::result_type, uint_type>);
  // Although MT19937 needs only 624 numbers to hold the state, C++ standard libraries may choose different
  // ways to represent the state. libc++ uses 624 numbers but libstdc++ uses 625 numbers. The additional
  // number is an index to the 624 numbers. So we will just count and store the number.
  std::vector<uint_type> state;
  state.reserve(625); // the magic number is chosen based on libstdc++ using 625 numbers while libc++ uses 624
  std::stringstream otemp;
  otemp << engine;
  std::copy(std::istream_iterator<uint_type>(otemp), std::istream_iterator<uint_type>(), std::back_inserter(state));
  stream_state_size = state.size();
}

template<typename T>
void StdRandom<T>::load(const std::vector<uint_type>& newstate)
{
  std::stringstream otemp;
  std::copy(newstate.begin(), newstate.end(), std::ostream_iterator<uint_type>(otemp, " "));
  otemp >> engine;
}

template<typename T>
void StdRandom<T>::save(std::vector<uint_type>& curstate) const
{
  curstate.clear();
  std::stringstream otemp;
  otemp << engine;
  std::copy(std::istream_iterator<uint_type>(otemp), std::istream_iterator<uint_type>(), std::back_inserter(curstate));
}

template<typename T>
typename StdRandom<T>::result_type StdRandom<T>::operator()()
{
  return distribution(engine);
}

template<typename T>
void StdRandom<T>::swapState(RandomBase<T>& other) noexcept
{
  static_assert(std::is_nothrow_swappable_v<Engine>);
  static_assert(std::is_nothrow_swappable_v<uniform_real_distribution_as_boost<T>>);
  static_assert(std::is_nothrow_swappable_v<std::size_t>);
  auto* typed_other = dynamic_cast<StdRandom<T>*>(&other);
  if (!typed_other)
    std::terminate();
  using std::swap;
  swap(engine, typed_other->engine);
  swap(distribution, typed_other->distribution);
  swap(stream_state_size, typed_other->stream_state_size);
}

template<typename T>
RandomBase<T>& StdRandom<T>::operator=(const RandomBase<T>& other)
{
  return *this = dynamic_cast<const StdRandom<T>&>(other);
}

template class StdRandom<double>;
template class StdRandom<float>;
} // namespace qmcplusplus
