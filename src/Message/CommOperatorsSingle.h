//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//                    Ken Esler, kpesler@gmail.com, University of Illinois at Urbana-Champaign
//                    Jeremy McMinnis, jmcminis@gmail.com, University of Illinois at Urbana-Champaign
//
// File created by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////


#ifndef OHMMS_COMMUNICATION_OPERATORS_SINGLE_H
#define OHMMS_COMMUNICATION_OPERATORS_SINGLE_H

#include "container_proxy.h"

#include <limits>
#include <stdexcept>

///dummy declarations to be specialized


template<typename T>
inline void Communicate::allreduce(T&)
{}

template<typename T>
inline void Communicate::allreduce_in_place(T* restrict values, std::size_t count)
{
  constexpr std::size_t scalar_dimension = qmcplusplus::scalar_traits<T>::DIM;
  if (count == 0)
    return;
  if (values == nullptr)
    throw std::invalid_argument("Communicate::allreduce_in_place requires storage for a nonzero count");
  if (count > static_cast<std::size_t>(std::numeric_limits<int>::max()) / scalar_dimension)
    throw std::overflow_error("Communicate::allreduce_in_place count exceeds the MPI int count domain");
}

template<typename T>
inline void Communicate::reduce(T&)
{}


template<typename T>
inline void Communicate::reduce_in_place(T* restrict res, int n)
{}

template<typename T>
inline void Communicate::bcast(T&)
{}

template<typename T>
inline void Communicate::bcast(T* restrict, int n)
{}


template<typename T>
inline void Communicate::gather(T& sb, T& rb, int dest)
{ rb = sb; }

template<typename T>
inline void Communicate::allgather(T& sb, T& rb)
{ rb = sb; }

template<typename T>
inline void Communicate::scatter(T& sb, T& rb, int dest)
{ rb = sb; }


template<typename T, typename IT>
inline void Communicate::gatherv(T& sb, T& rb, IT&, IT&, int dest)
{ rb = sb; }

template<typename T, typename IT>
inline void Communicate::scatterv(T& sb, T& rb, IT&, IT&, int source)
{ rb = sb; }

template<typename T, typename IT>
inline void Communicate::gatherv(T* sb, T* rb, int n, IT& counts, IT& displ, int dest)
{
  for (int i = 0; i < n; ++i)
    rb[i] = sb[i];
}

template<typename T, typename TMPI, typename IT>
inline void Communicate::gatherv_in_place(T* buf, const TMPI& datatype, IT& counts, IT& displ, int dest)
{}

template<typename T>
inline void Communicate::allgather(T* sb, T* rb, int count)
{
  for (int i = 0; i < count; ++i)
    rb[i] = sb[i];
}


#endif
