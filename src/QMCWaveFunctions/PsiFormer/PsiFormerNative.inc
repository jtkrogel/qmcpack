//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerNative.inc
 * @brief Native implementation of the DeepQMC PsiFormer architecture.
 *
 * This implementation evaluates a fixed, HDF5-exported PsiFormer model without
 * a JAX runtime. Spatial derivatives are propagated as analytic first- and
 * diagonal-second- derivative jets. Parameter derivatives, used by the
 * standalone validation driver, are accumulated with explicit vector-Jacobian
 * products (VJPs).
 *
 * Define PSIFORMER_LIBRARY before including this file to omit the standalone
 * comparison driver. The QMCPACK WaveFunctionComponent does this in
 * PsiFormerWF.cpp.
 */
#include <hdf5.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace pf
{

// Tensor storage and NumPy-style broadcasting utilities.
//
// Tensors use row-major storage. Keeping these operations local makes the
// imported model independent of a third-party tensor or
// automatic-differentiation library.
using Shape = std::vector<size_t>;
size_t product(const Shape& s) { return std::accumulate(s.begin(), s.end(), size_t{1}, std::multiplies<>()); }

/** Lightweight, owning, row-major tensor used throughout the native evaluator.
 */
struct Tensor
{
  Shape shape;
  std::vector<double> x;
  Tensor() = default;
  explicit Tensor(Shape s, double v = 0) : shape(std::move(s)), x(product(shape), v) {}
  Tensor(Shape s, std::vector<double> v) : shape(std::move(s)), x(std::move(v))
  {
    if (x.size() != product(shape))
      throw std::runtime_error("Tensor size mismatch");
  }
  bool empty() const { return shape.empty() && x.empty(); }
  size_t size() const { return x.size(); }
};

Shape strides(const Shape& s)
{
  Shape z(s.size(), 1);
  for (int i = int(s.size()) - 2; i >= 0; --i)
    z[i] = z[i + 1] * s[i + 1];
  return z;
}
Shape unravel(size_t flat, const Shape& s)
{
  Shape idx(s.size());
  auto st = strides(s);
  for (size_t i = 0; i < s.size(); ++i)
  {
    idx[i] = flat / st[i];
    flat %= st[i];
  }
  return idx;
}
size_t ravel(const Shape& idx, const Shape& s)
{
  auto st  = strides(s);
  size_t z = 0;
  for (size_t i = 0; i < s.size(); ++i)
    z += idx[i] * st[i];
  return z;
}
Shape broadcast_shape(const Shape& a, const Shape& b)
{
  size_t n = std::max(a.size(), b.size());
  Shape o(n, 1);
  for (size_t k = 0; k < n; ++k)
  {
    size_t ai = k < n - a.size() ? 1 : a[k - (n - a.size())];
    size_t bi = k < n - b.size() ? 1 : b[k - (n - b.size())];
    if (ai != bi && ai != 1 && bi != 1)
    {
      throw std::runtime_error("incompatible broadcast");
    }
    o[k] = std::max(ai, bi);
  }
  return o;
}
size_t broadcast_offset(const Shape& out_idx, const Shape& in_shape)
{
  size_t lead = out_idx.size() - in_shape.size();
  Shape idx(in_shape.size());
  for (size_t i = 0; i < in_shape.size(); ++i)
  {
    idx[i] = in_shape[i] == 1 ? 0 : out_idx[lead + i];
  }
  return ravel(idx, in_shape);
}
Tensor unbroadcast(const Tensor& g, const Shape& target)
{
  Tensor z(target);
  for (size_t i = 0; i < g.size(); ++i)
  {
    auto oi = unravel(i, g.shape);
    z.x[broadcast_offset(oi, target)] += g.x[i];
  }
  return z;
}
Tensor elem_binary(const Tensor& a, const Tensor& b, const std::function<double(double, double)>& f)
{
  Shape os = broadcast_shape(a.shape, b.shape);
  Tensor o(os);
  for (size_t i = 0; i < o.size(); ++i)
  {
    auto idx = unravel(i, os);
    o.x[i]   = f(a.x[broadcast_offset(idx, a.shape)], b.x[broadcast_offset(idx, b.shape)]);
  }
  return o;
}
Tensor elem_unary(const Tensor& a, const std::function<double(double)>& f)
{
  Tensor o(a.shape);
  for (size_t i = 0; i < a.size(); ++i)
    o.x[i] = f(a.x[i]);
  return o;
}

// Differentiation graph and coordinate jets.
//
// A coordinate-dependent node stores d1[c,...] = d(value)/dR_c and the diagonal
// d2[c,...] = d^2(value)/dR_c^2. Mixed second derivatives are not needed for
// the electronic Laplacian. Each parent edge stores a VJP for reverse parameter
// derivatives.
struct Node;
using NP = std::shared_ptr<Node>;
struct Edge
{
  NP parent;
  std::function<Tensor(const Tensor&)> vjp;
};
/** A tensor value together with spatial jets and reverse-mode graph edges. */
struct Node
{
  Tensor value, d1, d2;
  std::vector<Edge> parents;
  std::string parameter_key;
};
NP node(Tensor v, Tensor d1 = {}, Tensor d2 = {}, std::vector<Edge> p = {})
{
  auto n     = std::make_shared<Node>();
  n->value   = std::move(v);
  n->d1      = std::move(d1);
  n->d2      = std::move(d2);
  n->parents = std::move(p);
  return n;
}
NP constant(Tensor v) { return node(std::move(v)); }
NP scalar(double v) { return constant(Tensor({}, std::vector<double>{v})); }
/** Create the coordinate leaf and seed its first derivative with the identity
 * matrix. */
NP coordinates(const Tensor& v)
{
  Tensor d1(Shape{v.size()});
  d1.shape.insert(d1.shape.end(), v.shape.begin(), v.shape.end());
  d1.x.assign(v.size() * v.size(), 0);
  for (size_t i = 0; i < v.size(); ++i)
    d1.x[i * v.size() + i] = 1;
  Tensor d2(d1.shape);
  return node(v, d1, d2);
}

Tensor broadcast_jet(const Tensor& j, const Shape& source, const Shape& out, size_t C)
{
  Shape os{C};
  os.insert(os.end(), out.begin(), out.end());
  Tensor z(os);
  if (j.empty())
    return z;
  for (size_t c = 0; c < C; ++c)
  {
    for (size_t i = 0; i < product(out); ++i)
    {
      auto oi                   = unravel(i, out);
      z.x[c * product(out) + i] = j.x[c * product(source) + broadcast_offset(oi, source)];
    }
  }
  return z;
}
// Primitive differentiable operations. Each operation propagates the coordinate
// jets and records the VJP needed to send a parameter adjoint back to its
// parents.
NP unary(const NP& a,
         const std::function<double(double)>& f,
         const std::function<double(double)>& fp,
         const std::function<double(double)>& fpp)
{
  Tensor v = elem_unary(a->value, f), q1, q2, first = elem_unary(a->value, fp), second = elem_unary(a->value, fpp);
  if (!a->d1.empty())
  {
    q1       = Tensor(a->d1.shape);
    q2       = Tensor(a->d2.shape);
    size_t n = a->value.size(), C = a->d1.size() / n;
    for (size_t c = 0; c < C; ++c)
      for (size_t i = 0; i < n; ++i)
      {
        size_t k = c * n + i;
        q1.x[k]  = first.x[i] * a->d1.x[k];
        q2.x[k]  = first.x[i] * a->d2.x[k] + second.x[i] * a->d1.x[k] * a->d1.x[k];
      }
  }
  return node(v, q1, q2, {{a, [first, sh = a->value.shape](const Tensor& g) {
                             return unbroadcast(elem_binary(g, first, [](double x, double y) { return x * y; }), sh);
                           }}});
}
NP add(const NP& a, const NP& b)
{
  Tensor v = elem_binary(a->value, b->value, [](double x, double y) { return x + y; }), d1, d2;
  size_t C = 0;
  if (!a->d1.empty())
    C = a->d1.shape[0];
  else if (!b->d1.empty())
    C = b->d1.shape[0];
  if (C)
  {
    auto aa = broadcast_jet(a->d1, a->value.shape, v.shape, C), bb = broadcast_jet(b->d1, b->value.shape, v.shape, C);
    auto aaa = broadcast_jet(a->d2, a->value.shape, v.shape, C), bbb = broadcast_jet(b->d2, b->value.shape, v.shape, C);
    d1 = elem_binary(aa, bb, [](double x, double y) { return x + y; });
    d2 = elem_binary(aaa, bbb, [](double x, double y) { return x + y; });
  }
  return node(v, d1, d2,
              {{a, [sh = a->value.shape](const Tensor& g) { return unbroadcast(g, sh); }},
               {b, [sh = b->value.shape](const Tensor& g) { return unbroadcast(g, sh); }}});
}
NP neg(const NP& a)
{
  return unary(a, [](double x) { return -x; }, [](double) { return -1.; }, [](double) { return 0.; });
}
NP mul(const NP& a, const NP& b)
{
  Tensor v = elem_binary(a->value, b->value, [](double x, double y) { return x * y; }), d1, d2;
  size_t C = 0;
  if (!a->d1.empty())
    C = a->d1.shape[0];
  else if (!b->d1.empty())
    C = b->d1.shape[0];
  if (C)
  {
    auto ad1 = broadcast_jet(a->d1, a->value.shape, v.shape, C), bd1 = broadcast_jet(b->d1, b->value.shape, v.shape, C),
         ad2 = broadcast_jet(a->d2, a->value.shape, v.shape, C), bd2 = broadcast_jet(b->d2, b->value.shape, v.shape, C);
    Shape js{C};
    js.insert(js.end(), v.shape.begin(), v.shape.end());
    d1 = Tensor(js);
    d2 = Tensor(js);
    for (size_t c = 0; c < C; ++c)
      for (size_t i = 0; i < v.size(); ++i)
      {
        auto oi   = unravel(i, v.shape);
        double av = a->value.x[broadcast_offset(oi, a->value.shape)],
               bv = b->value.x[broadcast_offset(oi, b->value.shape)];
        size_t k  = c * v.size() + i;
        d1.x[k]   = ad1.x[k] * bv + av * bd1.x[k];
        d2.x[k]   = ad2.x[k] * bv + 2 * ad1.x[k] * bd1.x[k] + av * bd2.x[k];
      }
  }
  return node(v, d1, d2,
              {{a,
                [bv = b->value, sh = a->value.shape](const Tensor& g) {
                  return unbroadcast(elem_binary(g, bv, [](double x, double y) { return x * y; }), sh);
                }},
               {b, [av = a->value, sh = b->value.shape](const Tensor& g) {
                  return unbroadcast(elem_binary(g, av, [](double x, double y) { return x * y; }), sh);
                }}});
}
NP recip(const NP& a)
{
  return unary(
      a, [](double x) { return 1 / x; }, [](double x) { return -1 / (x * x); },
      [](double x) { return 2 / (x * x * x); });
}
NP divn(const NP& a, const NP& b) { return mul(a, recip(b)); }
NP ex(const NP& a)
{
  return unary(
      a, [](double x) { return std::exp(x); }, [](double x) { return std::exp(x); },
      [](double x) { return std::exp(x); });
}
NP lg(const NP& a)
{
  return unary(
      a, [](double x) { return std::log(x); }, [](double x) { return 1 / x; }, [](double x) { return -1 / (x * x); });
}
NP log1p_n(const NP& a)
{
  return unary(
      a, [](double x) { return std::log1p(x); }, [](double x) { return 1 / (1 + x); },
      [](double x) { return -1 / ((1 + x) * (1 + x)); });
}
NP sqrtn(const NP& a)
{
  return unary(
      a, [](double x) { return std::sqrt(x); }, [](double x) { return .5 / std::sqrt(x); },
      [](double x) { return -.25 / std::pow(x, 1.5); });
}
NP tanhn(const NP& a)
{
  return unary(
      a, [](double x) { return std::tanh(x); },
      [](double x) {
        double t = std::tanh(x);
        return 1 - t * t;
      },
      [](double x) {
        double t = std::tanh(x);
        return -2 * t * (1 - t * t);
      });
}
NP absn(const NP& a)
{
  return unary(
      a, [](double x) { return std::abs(x); }, [](double x) { return x > 0 ? 1. : (x < 0 ? -1. : 0.); },
      [](double) { return 0.; });
}

// Tensor transformations and reductions.
NP reshape(const NP& a, Shape sh)
{
  if (product(sh) != a->value.size())
    throw std::runtime_error("reshape");
  Tensor v(sh, a->value.x), d1, d2;
  if (!a->d1.empty())
  {
    Shape js{a->d1.shape[0]};
    js.insert(js.end(), sh.begin(), sh.end());
    d1 = Tensor(js, a->d1.x);
    d2 = Tensor(js, a->d2.x);
  }
  return node(v, d1, d2, {{a, [sh0 = a->value.shape](const Tensor& g) { return Tensor(sh0, g.x); }}});
}
Tensor transpose_t(const Tensor& t, const std::vector<size_t>& axes)
{
  Shape sh;
  for (auto a : axes)
    sh.push_back(t.shape[a]);
  Tensor o(sh);
  for (size_t i = 0; i < o.size(); ++i)
  {
    auto ni = unravel(i, sh);
    Shape oi(t.shape.size());
    for (size_t k = 0; k < axes.size(); ++k)
      oi[axes[k]] = ni[k];
    o.x[i] = t.x[ravel(oi, t.shape)];
  }
  return o;
}
NP transpose(const NP& a, const std::vector<size_t>& axes)
{
  Tensor v = transpose_t(a->value, axes), d1, d2;
  if (!a->d1.empty())
  {
    std::vector<size_t> ja{0};
    for (auto x : axes)
      ja.push_back(x + 1);
    d1 = transpose_t(a->d1, ja);
    d2 = transpose_t(a->d2, ja);
  }
  std::vector<size_t> inv(axes.size());
  for (size_t i = 0; i < axes.size(); ++i)
    inv[axes[i]] = i;
  return node(v, d1, d2, {{a, [inv](const Tensor& g) { return transpose_t(g, inv); }}});
}
Tensor reduce_sum_t(const Tensor& t, const std::vector<size_t>& axes)
{
  std::vector<bool> red(t.shape.size());
  for (auto a : axes)
    red[a] = true;
  Shape sh;
  for (size_t i = 0; i < t.shape.size(); ++i)
    if (!red[i])
      sh.push_back(t.shape[i]);
  Tensor o(sh);
  for (size_t i = 0; i < t.size(); ++i)
  {
    auto ix = unravel(i, t.shape);
    Shape ox;
    for (size_t k = 0; k < ix.size(); ++k)
      if (!red[k])
        ox.push_back(ix[k]);
    o.x[ravel(ox, sh)] += t.x[i];
  }
  return o;
}
NP sum_axes(const NP& a, std::vector<size_t> axes)
{
  Tensor v = reduce_sum_t(a->value, axes), d1, d2;
  if (!a->d1.empty())
  {
    std::vector<size_t> ja;
    for (auto x : axes)
      ja.push_back(x + 1);
    d1 = reduce_sum_t(a->d1, ja);
    d2 = reduce_sum_t(a->d2, ja);
  }
  return node(v, d1, d2, {{a, [axes, sh = a->value.shape](const Tensor& g) {
                             Tensor z(sh);
                             std::vector<bool> red(sh.size());
                             for (auto x : axes)
                               red[x] = true;
                             for (size_t i = 0; i < z.size(); ++i)
                             {
                               auto ix = unravel(i, sh);
                               Shape gi;
                               for (size_t k = 0; k < ix.size(); ++k)
                                 if (!red[k])
                                   gi.push_back(ix[k]);
                               z.x[i] = g.x[ravel(gi, g.shape)];
                             }
                             return z;
                           }}});
}
NP sum_all(const NP& a)
{
  std::vector<size_t> ax(a->value.shape.size());
  std::iota(ax.begin(), ax.end(), 0);
  return sum_axes(a, ax);
}
NP slice0(const NP& a, size_t begin, size_t end)
{
  Shape sh     = a->value.shape;
  size_t block = product(Shape(sh.begin() + 1, sh.end()));
  sh[0]        = end - begin;
  Tensor v(sh);
  std::copy(a->value.x.begin() + begin * block, a->value.x.begin() + end * block, v.x.begin());
  Tensor d1, d2;
  if (!a->d1.empty())
  {
    Shape js = a->d1.shape;
    js[1]    = end - begin;
    d1       = Tensor(js);
    d2       = Tensor(js);
    for (size_t c = 0; c < js[0]; ++c)
    {
      size_t oldn = a->value.size(), newn = v.size();
      std::copy(a->d1.x.begin() + c * oldn + begin * block, a->d1.x.begin() + c * oldn + end * block,
                d1.x.begin() + c * newn);
      std::copy(a->d2.x.begin() + c * oldn + begin * block, a->d2.x.begin() + c * oldn + end * block,
                d2.x.begin() + c * newn);
    }
  }
  return node(v, d1, d2, {{a, [begin, end, sh0 = a->value.shape](const Tensor& g) {
                             Tensor z(sh0);
                             size_t block = product(Shape(sh0.begin() + 1, sh0.end()));
                             std::copy(g.x.begin(), g.x.end(), z.x.begin() + begin * block);
                             return z;
                           }}});
}
NP concat(const std::vector<NP>& a, size_t axis)
{
  if (a.empty())
    throw std::runtime_error("empty concat");
  if (a[0]->value.shape.empty())
  {
    std::vector<NP> b;
    for (auto& n : a)
      b.push_back(reshape(n, {1}));
    return concat(b, 0);
  }
  Shape sh     = a[0]->value.shape;
  size_t total = 0;
  for (auto& n : a)
    total += n->value.shape[axis];
  sh[axis] = total;
  Tensor v(sh);
  size_t outer = 1, inner = 1;
  for (size_t i = 0; i < axis; ++i)
    outer *= sh[i];
  for (size_t i = axis + 1; i < sh.size(); ++i)
    inner *= sh[i];
  for (size_t o = 0; o < outer; ++o)
  {
    size_t cur = 0;
    for (auto& n : a)
    {
      size_t len = n->value.shape[axis] * inner;
      std::copy(n->value.x.begin() + o * len, n->value.x.begin() + (o + 1) * len,
                v.x.begin() + o * total * inner + cur);
      cur += len;
    }
  }
  Tensor d1, d2;
  size_t C = 0;
  for (auto& n : a)
    if (!n->d1.empty())
    {
      C = n->d1.shape[0];
      break;
    }
  if (C)
  {
    Shape js{C};
    js.insert(js.end(), sh.begin(), sh.end());
    d1 = Tensor(js);
    d2 = Tensor(js);
    for (size_t c = 0; c < C; ++c)
    {
      size_t base = c * v.size();
      for (size_t o = 0; o < outer; ++o)
      {
        size_t cur = 0;
        for (auto& n : a)
        {
          size_t len = n->value.shape[axis] * inner, nn = n->value.size();
          if (!n->d1.empty())
          {
            std::copy(n->d1.x.begin() + c * nn + o * len, n->d1.x.begin() + c * nn + (o + 1) * len,
                      d1.x.begin() + base + o * total * inner + cur);
            std::copy(n->d2.x.begin() + c * nn + o * len, n->d2.x.begin() + c * nn + (o + 1) * len,
                      d2.x.begin() + base + o * total * inner + cur);
          }
          cur += len;
        }
      }
    }
  }
  std::vector<Edge> edges;
  size_t offset = 0;
  for (auto& n : a)
  {
    size_t begin = offset, lenaxis = n->value.shape[axis];
    offset += lenaxis;
    edges.push_back({n, [begin, lenaxis, axis, shn = n->value.shape](const Tensor& g) {
                       Tensor z(shn);
                       size_t outer = 1, inner = 1, total = g.shape[axis];
                       for (size_t i = 0; i < axis; ++i)
                         outer *= g.shape[i];
                       for (size_t i = axis + 1; i < g.shape.size(); ++i)
                         inner *= g.shape[i];
                       size_t len = lenaxis * inner;
                       for (size_t o = 0; o < outer; ++o)
                         std::copy(g.x.begin() + o * total * inner + begin * inner,
                                   g.x.begin() + o * total * inner + begin * inner + len, z.x.begin() + o * len);
                       return z;
                     }});
  }
  return node(v, d1, d2, std::move(edges));
}

// Dense and self-attention kernels used by the PsiFormer feature layers.
NP linear(const NP& a, const NP& w, const NP& b = nullptr)
{
  size_t I = a->value.shape.back(), O = w->value.shape[1], rows = a->value.size() / I;
  Shape sh  = a->value.shape;
  sh.back() = O;
  Tensor v(sh);
  for (size_t r = 0; r < rows; ++r)
    for (size_t o = 0; o < O; ++o)
      for (size_t i = 0; i < I; ++i)
        v.x[r * O + o] += a->value.x[r * I + i] * w->value.x[i * O + o];
  Tensor d1, d2;
  if (!a->d1.empty())
  {
    size_t C = a->d1.shape[0];
    Shape js{C};
    js.insert(js.end(), sh.begin(), sh.end());
    d1 = Tensor(js);
    d2 = Tensor(js);
    for (size_t c = 0; c < C; ++c)
      for (size_t r = 0; r < rows; ++r)
        for (size_t o = 0; o < O; ++o)
          for (size_t i = 0; i < I; ++i)
          {
            d1.x[c * v.size() + r * O + o] += a->d1.x[c * a->value.size() + r * I + i] * w->value.x[i * O + o];
            d2.x[c * v.size() + r * O + o] += a->d2.x[c * a->value.size() + r * I + i] * w->value.x[i * O + o];
          }
  }
  auto out = node(v, d1, d2,
                  {{a,
                    [wv = w->value, I, O, sha = a->value.shape](const Tensor& g) {
                      size_t rows = g.size() / O;
                      Tensor z(sha);
                      for (size_t r = 0; r < rows; ++r)
                        for (size_t i = 0; i < I; ++i)
                          for (size_t o = 0; o < O; ++o)
                            z.x[r * I + i] += g.x[r * O + o] * wv.x[i * O + o];
                      return z;
                    }},
                   {w, [av = a->value, I, O](const Tensor& g) {
                      size_t rows = g.size() / O;
                      Tensor z({I, O});
                      for (size_t i = 0; i < I; ++i)
                        for (size_t o = 0; o < O; ++o)
                          for (size_t r = 0; r < rows; ++r)
                            z.x[i * O + o] += av.x[r * I + i] * g.x[r * O + o];
                      return z;
                    }}});
  return b ? add(out, b) : out;
}

NP attention_logits(const NP& q, const NP& k)
{
  size_t N = q->value.shape[0], H = q->value.shape[1], D = q->value.shape[2];
  Tensor v({H, N, N});
  auto Q = [=](size_t i, size_t h, size_t d) { return (i * H + h) * D + d; };
  auto V = [=](size_t h, size_t i, size_t j) { return (h * N + i) * N + j; };
  for (size_t h = 0; h < H; ++h)
    for (size_t i = 0; i < N; ++i)
      for (size_t j = 0; j < N; ++j)
        for (size_t d = 0; d < D; ++d)
          v.x[V(h, i, j)] += q->value.x[Q(i, h, d)] * k->value.x[Q(j, h, d)];
  Tensor d1, d2;
  if (!q->d1.empty())
  {
    size_t C = q->d1.shape[0];
    d1       = Tensor({C, H, N, N});
    d2       = Tensor({C, H, N, N});
    for (size_t c = 0; c < C; ++c)
      for (size_t h = 0; h < H; ++h)
        for (size_t i = 0; i < N; ++i)
          for (size_t j = 0; j < N; ++j)
            for (size_t d = 0; d < D; ++d)
            {
              size_t o = c * v.size() + V(h, i, j), qi = c * q->value.size() + Q(i, h, d),
                     kj = c * k->value.size() + Q(j, h, d);
              d1.x[o] += q->d1.x[qi] * k->value.x[Q(j, h, d)] + q->value.x[Q(i, h, d)] * k->d1.x[kj];
              d2.x[o] += q->d2.x[qi] * k->value.x[Q(j, h, d)] + 2 * q->d1.x[qi] * k->d1.x[kj] +
                  q->value.x[Q(i, h, d)] * k->d2.x[kj];
            }
  }
  return node(v, d1, d2,
              {{q,
                [kv = k->value, N, H, D, Q, V](const Tensor& g) {
                  Tensor z({N, H, D});
                  for (size_t i = 0; i < N; ++i)
                    for (size_t h = 0; h < H; ++h)
                      for (size_t d = 0; d < D; ++d)
                        for (size_t j = 0; j < N; ++j)
                          z.x[Q(i, h, d)] += g.x[V(h, i, j)] * kv.x[Q(j, h, d)];
                  return z;
                }},
               {k, [qv = q->value, N, H, D, Q, V](const Tensor& g) {
                  Tensor z({N, H, D});
                  for (size_t j = 0; j < N; ++j)
                    for (size_t h = 0; h < H; ++h)
                      for (size_t d = 0; d < D; ++d)
                        for (size_t i = 0; i < N; ++i)
                          z.x[Q(j, h, d)] += g.x[V(h, i, j)] * qv.x[Q(i, h, d)];
                  return z;
                }}});
}

NP attention_context(const NP& w, const NP& val)
{
  size_t H = w->value.shape[0], N = w->value.shape[1], D = val->value.shape[2];
  auto W = [=](size_t h, size_t i, size_t j) { return (h * N + i) * N + j; };
  auto V = [=](size_t i, size_t h, size_t d) { return (i * H + h) * D + d; };
  Tensor out({N, H, D});
  for (size_t i = 0; i < N; ++i)
    for (size_t h = 0; h < H; ++h)
      for (size_t d = 0; d < D; ++d)
        for (size_t j = 0; j < N; ++j)
          out.x[V(i, h, d)] += w->value.x[W(h, i, j)] * val->value.x[V(j, h, d)];
  Tensor d1, d2;
  if (!w->d1.empty())
  {
    size_t C = w->d1.shape[0];
    d1       = Tensor({C, N, H, D});
    d2       = Tensor({C, N, H, D});
    for (size_t c = 0; c < C; ++c)
      for (size_t i = 0; i < N; ++i)
        for (size_t h = 0; h < H; ++h)
          for (size_t d = 0; d < D; ++d)
            for (size_t j = 0; j < N; ++j)
            {
              size_t o = c * out.size() + V(i, h, d), wi = c * w->value.size() + W(h, i, j),
                     vi = c * val->value.size() + V(j, h, d);
              d1.x[o] += w->d1.x[wi] * val->value.x[V(j, h, d)] + w->value.x[W(h, i, j)] * val->d1.x[vi];
              d2.x[o] += w->d2.x[wi] * val->value.x[V(j, h, d)] + 2 * w->d1.x[wi] * val->d1.x[vi] +
                  w->value.x[W(h, i, j)] * val->d2.x[vi];
            }
  }
  return node(out, d1, d2,
              {{w,
                [vv = val->value, H, N, D, W, V](const Tensor& g) {
                  Tensor z({H, N, N});
                  for (size_t h = 0; h < H; ++h)
                    for (size_t i = 0; i < N; ++i)
                      for (size_t j = 0; j < N; ++j)
                        for (size_t d = 0; d < D; ++d)
                          z.x[W(h, i, j)] += g.x[V(i, h, d)] * vv.x[V(j, h, d)];
                  return z;
                }},
               {val, [wv = w->value, H, N, D, W, V](const Tensor& g) {
                  Tensor z({N, H, D});
                  for (size_t j = 0; j < N; ++j)
                    for (size_t h = 0; h < H; ++h)
                      for (size_t d = 0; d < D; ++d)
                        for (size_t i = 0; i < N; ++i)
                          z.x[V(j, h, d)] += wv.x[W(h, i, j)] * g.x[V(i, h, d)];
                  return z;
                }}});
}

NP softmax(const NP& a)
{
  // Subtracting the row maximum avoids overflow without changing the normalized
  // weights.
  size_t K = a->value.shape.back(), rows = a->value.size() / K;
  Tensor v(a->value.shape);
  for (size_t r = 0; r < rows; ++r)
  {
    double m = *std::max_element(a->value.x.begin() + r * K, a->value.x.begin() + (r + 1) * K), s = 0;
    for (size_t k = 0; k < K; ++k)
      s += (v.x[r * K + k] = std::exp(a->value.x[r * K + k] - m));
    for (size_t k = 0; k < K; ++k)
      v.x[r * K + k] /= s;
  }
  Tensor d1, d2;
  if (!a->d1.empty())
  {
    size_t C = a->d1.shape[0];
    d1       = Tensor(a->d1.shape);
    d2       = Tensor(a->d2.shape);
    for (size_t c = 0; c < C; ++c)
      for (size_t r = 0; r < rows; ++r)
      {
        double m1 = 0, m2 = 0;
        for (size_t k = 0; k < K; ++k)
        {
          size_t z = c * a->value.size() + r * K + k;
          m1 += v.x[r * K + k] * a->d1.x[z];
          m2 += v.x[r * K + k] * (a->d2.x[z] + a->d1.x[z] * a->d1.x[z]);
        }
        m2 -= m1 * m1;
        for (size_t k = 0; k < K; ++k)
        {
          size_t z    = c * a->value.size() + r * K + k;
          double cent = a->d1.x[z] - m1;
          d1.x[z]     = v.x[r * K + k] * cent;
          d2.x[z]     = v.x[r * K + k] * (cent * cent + a->d2.x[z] - m2);
        }
      }
  }
  return node(v, d1, d2, {{a, [v, K, rows](const Tensor& g) {
                             Tensor z(g.shape);
                             for (size_t r = 0; r < rows; ++r)
                             {
                               double dot = 0;
                               for (size_t k = 0; k < K; ++k)
                                 dot += g.x[r * K + k] * v.x[r * K + k];
                               for (size_t k = 0; k < K; ++k)
                                 z.x[r * K + k] = v.x[r * K + k] * (g.x[r * K + k] - dot);
                             }
                             return z;
                           }}});
}

std::pair<double, std::vector<double>> det_inv(const double* A, size_t n)
{
  std::vector<double> m(A, A + n * n), inv(n * n);
  for (size_t i = 0; i < n; ++i)
    inv[i * n + i] = 1;
  double det = 1;
  int sign   = 1;
  for (size_t c = 0; c < n; ++c)
  {
    size_t p = c;
    for (size_t r = c + 1; r < n; ++r)
      if (std::abs(m[r * n + c]) > std::abs(m[p * n + c]))
        p = r;
    if (std::abs(m[p * n + c]) < 1e-14)
      throw std::runtime_error("singular determinant");
    if (p != c)
    {
      for (size_t j = 0; j < n; ++j)
      {
        std::swap(m[c * n + j], m[p * n + j]);
        std::swap(inv[c * n + j], inv[p * n + j]);
      }
      sign = -sign;
    }
    double pivot = m[c * n + c];
    det *= pivot;
    for (size_t j = 0; j < n; ++j)
    {
      m[c * n + j] /= pivot;
      inv[c * n + j] /= pivot;
    }
    for (size_t r = 0; r < n; ++r)
      if (r != c)
      {
        double f = m[r * n + c];
        for (size_t j = 0; j < n; ++j)
        {
          m[r * n + j] -= f * m[c * n + j];
          inv[r * n + j] -= f * inv[c * n + j];
        }
      }
  }
  return {det * sign, inv};
}
double trace_product(const double* A, const double* B, size_t n)
{
  double z = 0;
  for (size_t i = 0; i < n; ++i)
    for (size_t j = 0; j < n; ++j)
      z += A[i * n + j] * B[j * n + i];
  return z;
}
// Determinant primitive used to combine the spin-resolved orbital matrices.
NP determinants(const NP& a)
{
  size_t M = a->value.shape[0], N = a->value.shape[1], nn = N * N;
  Tensor v({M});
  std::vector<std::vector<double>> inv(M);
  for (size_t m = 0; m < M; ++m)
  {
    auto di = det_inv(a->value.x.data() + m * nn, N);
    v.x[m]  = di.first;
    inv[m]  = std::move(di.second);
  }
  Tensor d1, d2;
  if (!a->d1.empty())
  {
    size_t C = a->d1.shape[0];
    d1       = Tensor({C, M});
    d2       = Tensor({C, M});
    std::vector<double> tmp(nn);
    for (size_t c = 0; c < C; ++c)
      for (size_t m = 0; m < M; ++m)
      {
        auto A1 = a->d1.x.data() + c * a->value.size() + m * nn, A2 = a->d2.x.data() + c * a->value.size() + m * nn;
        // Apply d(det A) = det(A) tr(A^-1 dA) and its diagonal
        // second-derivative counterpart to every coordinate jet.
        double t1 = trace_product(inv[m].data(), A1, N), t2 = trace_product(inv[m].data(), A2, N);
        for (size_t i = 0; i < N; ++i)
          for (size_t j = 0; j < N; ++j)
          {
            tmp[i * N + j] = 0;
            for (size_t k = 0; k < N; ++k)
              tmp[i * N + j] += inv[m][i * N + k] * A1[k * N + j];
          }
        double cur      = trace_product(tmp.data(), tmp.data(), N);
        d1.x[c * M + m] = v.x[m] * t1;
        d2.x[c * M + m] = v.x[m] * (t1 * t1 + t2 - cur);
      }
  }
  return node(v, d1, d2, {{a, [v, inv, M, N, nn](const Tensor& g) {
                             Tensor z({M, N, N});
                             for (size_t m = 0; m < M; ++m)
                               for (size_t i = 0; i < N; ++i)
                                 for (size_t j = 0; j < N; ++j)
                                   z.x[m * nn + i * N + j] = g.x[m] * v.x[m] * inv[m][j * N + i];
                             return z;
                           }}});
}

/** Reverse-accumulate adjoints from a scalar output through the recorded VJP
 * graph. */
std::unordered_map<const Node*, Tensor> backward(const NP& root)
{
  std::vector<NP> topo;
  std::unordered_map<const Node*, bool> seen;
  std::function<void(const NP&)> visit = [&](const NP& n) {
    if (seen[n.get()])
      return;
    seen[n.get()] = true;
    for (auto& e : n->parents)
      visit(e.parent);
    topo.push_back(n);
  };
  visit(root);
  std::unordered_map<const Node*, Tensor> adj;
  adj[root.get()] = Tensor(root->value.shape, 1);
  for (auto it = topo.rbegin(); it != topo.rend(); ++it)
  {
    auto f = adj.find(it->get());
    if (f == adj.end())
      continue;
    for (auto& e : (*it)->parents)
    {
      Tensor c = e.vjp(f->second);
      auto p   = adj.find(e.parent.get());
      if (p == adj.end())
        adj[e.parent.get()] = std::move(c);
      else
        for (size_t i = 0; i < c.size(); ++i)
          p->second.x[i] += c.x[i];
    }
  }
  return adj;
}

// HDF5 import helpers for the flattened DeepQMC parameter export and
// configuration data.
std::vector<double> read_double(hid_t file, const std::string& path, Shape* shape = nullptr)
{
  hid_t d = H5Dopen2(file, path.c_str(), H5P_DEFAULT), s = H5Dget_space(d);
  int n = H5Sget_simple_extent_ndims(s);
  std::vector<hsize_t> dims(n);
  H5Sget_simple_extent_dims(s, dims.data(), nullptr);
  Shape sh(dims.begin(), dims.end());
  std::vector<double> v(product(sh));
  H5Dread(d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, v.data());
  H5Sclose(s);
  H5Dclose(d);
  if (shape)
    *shape = sh;
  return v;
}
std::vector<int64_t> read_i64(hid_t file, const std::string& path, Shape* shape = nullptr)
{
  hid_t d = H5Dopen2(file, path.c_str(), H5P_DEFAULT), s = H5Dget_space(d);
  int n = H5Sget_simple_extent_ndims(s);
  std::vector<hsize_t> dims(n);
  H5Sget_simple_extent_dims(s, dims.data(), nullptr);
  Shape sh(dims.begin(), dims.end());
  std::vector<int64_t> v(product(sh));
  H5Dread(d, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, v.data());
  H5Sclose(s);
  H5Dclose(d);
  if (shape)
    *shape = sh;
  return v;
}
std::vector<std::string> read_strings(hid_t file, const std::string& path)
{
  hid_t d = H5Dopen2(file, path.c_str(), H5P_DEFAULT), s = H5Dget_space(d);
  hsize_t n;
  H5Sget_simple_extent_dims(s, &n, nullptr);
  hid_t t = H5Dget_type(d);
  std::vector<char*> p(n);
  H5Dread(d, t, H5S_ALL, H5S_ALL, H5P_DEFAULT, p.data());
  std::vector<std::string> v;
  for (auto x : p)
    v.emplace_back(x);
  H5Dvlen_reclaim(t, s, H5P_DEFAULT, p.data());
  H5Tclose(t);
  H5Sclose(s);
  H5Dclose(d);
  return v;
}
int64_t read_attr_i64(hid_t file, const std::string& name)
{
  hid_t a = H5Aopen(file, name.c_str(), H5P_DEFAULT);
  int64_t v;
  H5Aread(a, H5T_NATIVE_LLONG, &v);
  H5Aclose(a);
  return v;
}

struct Layout
{
  std::string module, name;
  Shape shape;
  size_t begin, end;
};
/** Owns the flat exported parameters and exposes named graph leaves for model
 * assembly. */
struct Parameters
{
  std::vector<double> values;
  std::vector<Layout> layouts;
  std::map<std::pair<std::string, std::string>, NP> nodes;
  explicit Parameters(const std::string& path)
  {
    hid_t f   = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    values    = read_double(f, "/values");
    auto mods = read_strings(f, "/layout/modules"), names = read_strings(f, "/layout/names");
    Shape ss;
    auto ranks = read_i64(f, "/layout/ranks"), shapes = read_i64(f, "/layout/shapes", &ss),
         off = read_i64(f, "/layout/offsets");
    for (size_t i = 0; i < mods.size(); ++i)
    {
      Shape sh;
      for (int r = 0; r < ranks[i]; ++r)
        sh.push_back(shapes[i * ss[1] + r]);
      Layout l{mods[i], names[i], sh, size_t(off[i]), size_t(off[i + 1])};
      std::vector<double> v(values.begin() + l.begin, values.begin() + l.end);
      auto n           = node(Tensor(sh, std::move(v)));
      n->parameter_key = mods[i] + "/" + names[i];
      layouts.push_back(l);
      nodes[{mods[i], names[i]}] = n;
    }
    H5Fclose(f);
  }
  NP find(const std::string& suffix, const std::string& name)
  {
    // DeepQMC/Haiku module paths contain generated prefixes. A unique suffix
    // plus the parameter name identifies the corresponding weight or bias
    // robustly.
    NP hit;
    int count = 0;
    for (auto& [k, n] : nodes)
      if (k.second == name && k.first.size() >= suffix.size() &&
          k.first.compare(k.first.size() - suffix.size(), suffix.size(), suffix) == 0)
      {
        hit = n;
        ++count;
      }
    if (count != 1)
      throw std::runtime_error("parameter lookup " + suffix + "/" + name + " count=" + std::to_string(count));
    return hit;
  }
  std::vector<double> flat_gradient(const std::unordered_map<const Node*, Tensor>& adj)
  {
    std::vector<double> g(values.size());
    for (auto& l : layouts)
    {
      auto n  = nodes[{l.module, l.name}];
      auto it = adj.find(n.get());
      if (it != adj.end())
        std::copy(it->second.x.begin(), it->second.x.end(), g.begin() + l.begin);
    }
    return g;
  }
};

/** System metadata and reference configurations read from the configuration
 * HDF5 file. */
struct ConfigData
{
  Tensor electrons, nuclei, charges;
  size_t nup, ndown, nconfig;
  explicit ConfigData(const std::string& path)
  {
    hid_t f = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    Shape s;
    electrons   = Tensor((read_double(f, "/electron_positions", &s), s));
    electrons.x = read_double(f, "/electron_positions");
    nconfig     = s[0];
    nuclei.x    = read_double(f, "/nuclear_positions", &nuclei.shape);
    charges.x   = read_double(f, "/nuclear_charges", &charges.shape);
    nup         = read_attr_i64(f, "n_up");
    ndown       = read_attr_i64(f, "n_down");
    H5Fclose(f);
  }
  Tensor configuration(size_t c) const
  {
    size_t n = (nup + ndown) * 3;
    return Tensor({nup + ndown, 3},
                  std::vector<double>(electrons.x.begin() + c * n, electrons.x.begin() + (c + 1) * n));
  }
};

/** Quantities produced by one all-electron PsiFormer evaluation. */
struct Result
{
  double sign, logabs, value, local_energy;
  std::vector<double> gradient, lap_log, lap_ratio, potential, param_gradient;
};
/** Builds and evaluates the four-block PsiFormer wavefunction from imported
 * parameters. */
struct PsiFormer
{
  Parameters p;
  ConfigData cfg;
  size_t ne, ndet = 16, dim = 256, heads = 4;
  PsiFormer(const std::string& pp, const std::string& cp) : p(pp), cfg(cp), ne(cfg.nup + cfg.ndown) {}
  NP embedding(const NP& pos)
  {
    // Per electron-nucleus pair, construct the DeepQMC feature tuple
    // [log(1+r), (r_vec/r) log(1+r)], then append the electron spin indicator.
    std::vector<NP> feat;
    for (size_t i = 0; i < ne; ++i)
      for (size_t a = 0; a < cfg.nuclei.shape[0]; ++a)
      {
        std::vector<NP> d;
        for (size_t k = 0; k < 3; ++k)
          d.push_back(add(slice_scalar(pos, i, k), scalar(-cfg.nuclei.x[a * 3 + k])));
        auto r   = sqrtn(sum_all(mul(concat(d, 0), concat(d, 0))));
        auto rad = log1p_n(r);
        feat.push_back(rad);
        for (auto& x : d)
          feat.push_back(mul(x, divn(rad, r)));
      }
    std::vector<NP> rows;
    size_t per = cfg.nuclei.shape[0] * 4;
    for (size_t i = 0; i < ne; ++i)
    {
      std::vector<NP> f(feat.begin() + i * per, feat.begin() + (i + 1) * per);
      f.push_back(scalar(i < cfg.nup ? 1 : -1));
      rows.push_back(reshape(concat(f, 0), {1, per + 1}));
    }
    return linear(concat(rows, 0), p.find("electron_embedding/linear", "w"));
  }
  NP slice_scalar(const NP& a, size_t i, size_t j)
  {
    auto r    = slice0(a, i, i + 1);
    auto flat = reshape(r, {3});
    return reshape(slice0(flat, j, j + 1), {});
  }
  NP attention_block(const NP& h, int layer)
  {
    // Multi-head self-attention is followed by a residual update, a two-layer
    // tanh MLP, and a second residual update. Parameter suffixes mirror the
    // exported Haiku names.
    std::string s = layer == 0 ? "electron_gnn_layer" : "electron_gnn_layer_" + std::to_string(layer);
    std::string b = s + "/~/node_attention_electron_update_feature";
    auto q        = reshape(linear(h, p.find(b + "/multi_head_attention/query", "w")), {ne, heads, dim / heads});
    auto k        = reshape(linear(h, p.find(b + "/multi_head_attention/key", "w")), {ne, heads, dim / heads});
    auto v        = reshape(linear(h, p.find(b + "/multi_head_attention/value", "w")), {ne, heads, dim / heads});
    auto logits   = mul(attention_logits(q, k), scalar(1 / std::sqrt(double(dim / heads))));
    auto att      = reshape(attention_context(softmax(logits), v), {ne, dim});
    att           = add(h, linear(att, p.find(b + "/multi_head_attention/linear", "w")));
    auto hid      = tanhn(linear(att, p.find(b + "/mlp/linear_0", "w"), p.find(b + "/mlp/linear_0", "b")));
    auto upd      = tanhn(linear(hid, p.find(b + "/mlp/linear_1", "w"), p.find(b + "/mlp/linear_1", "b")));
    return add(att, upd);
  }
  NP backflow(const NP& h, bool up)
  {
    // Map the final electron features to determinant-specific spin-orbital
    // matrices.
    size_t count = up ? cfg.nup : cfg.ndown;
    auto e       = slice0(h, up ? 0 : cfg.nup, up ? cfg.nup : ne);
    auto o       = linear(e, p.find(up ? "Backflow/~/mlp/linear_0" : "Backflow_1/~/mlp/linear_0", "w"));
    return transpose(reshape(o, {count, ndet, ne}), {1, 0, 2});
  }
  NP envelope(const NP& pos, bool up)
  {
    // Form the learned atom-centred exponential envelope for each spin orbital.
    size_t begin = up ? 0 : cfg.nup, count = up ? cfg.nup : cfg.ndown;
    auto pi   = p.find("exponential_envelopes", up ? "pi_up" : "pi_down"),
         zeta = p.find("exponential_envelopes", up ? "zetas_up" : "zetas_down");
    std::vector<NP> all;
    for (size_t i = 0; i < count; ++i)
      for (size_t d = 0; d < ndet; ++d)
        for (size_t o = 0; o < ne; ++o)
        {
          NP z = scalar(0);
          for (size_t a = 0; a < cfg.nuclei.shape[0]; ++a)
          {
            std::vector<NP> dv;
            for (size_t k = 0; k < 3; ++k)
              dv.push_back(add(slice_scalar(pos, begin + i, k), scalar(-cfg.nuclei.x[a * 3 + k])));
            auto r  = sqrtn(sum_all(mul(concat(dv, 0), concat(dv, 0))));
            auto zz = reshape(slice0(reshape(zeta, {zeta->value.size()}),
                                     d * ne * cfg.nuclei.shape[0] + o * cfg.nuclei.shape[0] + a,
                                     d * ne * cfg.nuclei.shape[0] + o * cfg.nuclei.shape[0] + a + 1),
                              {});
            auto pp = reshape(slice0(reshape(pi, {pi->value.size()}),
                                     d * ne * cfg.nuclei.shape[0] + o * cfg.nuclei.shape[0] + a,
                                     d * ne * cfg.nuclei.shape[0] + o * cfg.nuclei.shape[0] + a + 1),
                              {});
            z       = add(z, mul(pp, ex(neg(absn(mul(zz, r))))));
          }
          all.push_back(z);
        }
    auto raw = reshape(concat(all, 0), {count, ndet, ne});
    return transpose(raw, {1, 0, 2});
  }
  NP cusp(const NP& pos)
  {
    // Same-spin and opposite-spin electron pairs use the physical 1/4 and 1/2
    // cusp factors, respectively, with learned asymptotic length scales.
    auto sa   = p.find("electronic_cusp_asymptotic", "same_alpha"),
         aa   = p.find("electronic_cusp_asymptotic", "anti_alpha");
    NP out    = scalar(0);
    auto pair = [&](size_t i, size_t j, const NP& alpha, double factor) {
      std::vector<NP> d;
      for (size_t k = 0; k < 3; ++k)
        d.push_back(add(slice_scalar(pos, i, k), neg(slice_scalar(pos, j, k))));
      auto r = sqrtn(sum_all(mul(concat(d, 0), concat(d, 0))));
      out    = add(out, neg(divn(mul(scalar(factor), mul(alpha, alpha)), add(alpha, r))));
    };
    for (size_t b = 0; b < 2; ++b)
    {
      size_t lo = b ? cfg.nup : 0, hi = b ? ne : cfg.nup;
      for (size_t i = lo; i < hi; ++i)
        for (size_t j = i + 1; j < hi; ++j)
          pair(i, j, sa, .25);
    }
    for (size_t i = 0; i < cfg.nup; ++i)
      for (size_t j = cfg.nup; j < ne; ++j)
        pair(i, j, aa, .5);
    return out;
  }
  Result evaluate(const Tensor& ep, bool with_param = true)
  {
    // Feature layers: electron-nucleus embedding followed by four attention
    // blocks.
    auto pos = coordinates(ep);
    auto h   = embedding(pos);
    for (int l = 0; l < 4; ++l)
      h = attention_block(h, l);
    // Remaining wavefunction layers: spin-resolved backflow orbitals,
    // exponential envelopes, determinant sum, and the additive
    // electron-electron cusp correction.
    auto mats = concat({mul(envelope(pos, true), backflow(h, true)), mul(envelope(pos, false), backflow(h, false))}, 1);
    auto ds   = sum_all(determinants(mats));
    auto la   = add(lg(absn(ds)), cusp(pos));
    Result r;
    r.sign     = ds->value.x[0] > 0 ? 1 : -1;
    r.logabs   = la->value.x[0];
    r.value    = r.sign * std::exp(r.logabs);
    r.gradient = la->d1.x;
    r.lap_log.resize(ne);
    r.lap_ratio.resize(ne);
    // For psi = sign * exp(logabs), (nabla^2 psi)/psi equals the sum of
    // nabla^2 logabs and the squared norm of nabla logabs for each electron.
    for (size_t i = 0; i < ne; ++i)
    {
      double g2 = 0;
      for (size_t k = 0; k < 3; ++k)
      {
        r.lap_log[i] += la->d2.x[(i * 3 + k)];
        g2 += r.gradient[i * 3 + k] * r.gradient[i * 3 + k];
      }
      r.lap_ratio[i] = r.lap_log[i] + g2;
    }
    // Accumulate electron-electron, electron-nucleus, and nucleus-nucleus
    // Coulomb terms separately so they remain available to validation callers.
    double ee = 0, en = 0, nn = 0;
    auto dist = [](const double* a, const double* b) {
      double s = 0;
      for (int k = 0; k < 3; ++k)
        s += (a[k] - b[k]) * (a[k] - b[k]);
      return std::sqrt(s);
    };
    for (size_t i = 0; i < ne; ++i)
      for (size_t j = i + 1; j < ne; ++j)
        ee += 1 / dist(&ep.x[i * 3], &ep.x[j * 3]);
    for (size_t i = 0; i < ne; ++i)
      for (size_t a = 0; a < cfg.nuclei.shape[0]; ++a)
        en -= cfg.charges.x[a] / dist(&ep.x[i * 3], &cfg.nuclei.x[a * 3]);
    for (size_t a = 0; a < cfg.nuclei.shape[0]; ++a)
      for (size_t b = a + 1; b < cfg.nuclei.shape[0]; ++b)
        nn += cfg.charges.x[a] * cfg.charges.x[b] / dist(&cfg.nuclei.x[a * 3], &cfg.nuclei.x[b * 3]);
    r.potential = {ee, en, nn};
    // Straight-Coulomb local energy: -1/2 (nabla^2 psi)/psi + V_ee + V_en +
    // V_nn.
    r.local_energy = -.5 * std::accumulate(r.lap_ratio.begin(), r.lap_ratio.end(), 0.) + ee + en + nn;
    if (with_param)
      r.param_gradient = p.flat_gradient(backward(la));
    return r;
  }
};

// Standalone validation support. This code is excluded when included by
// QMCPACK.
std::vector<double> read_reference(hid_t f, const std::string& name, Shape* sh = nullptr)
{
  return read_double(f, "/" + name, sh);
}
void write_dataset(hid_t f, const std::string& name, const Shape& sh, const std::vector<double>& v)
{
  std::vector<hsize_t> d(sh.begin(), sh.end());
  hid_t s  = H5Screate_simple(d.size(), d.data(), nullptr),
        ds = H5Dcreate2(f, name.c_str(), H5T_IEEE_F64LE, s, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, v.data());
  H5Dclose(ds);
  H5Sclose(s);
}
int run(const std::string& param, const std::string& config, const std::string& reference, const std::string& output)
{
  PsiFormer model(param, config);
  std::vector<Result> rs;
  for (size_t c = 0; c < model.cfg.nconfig; ++c)
  {
    auto r = model.evaluate(model.cfg.configuration(c));
    std::cout << "config " << c + 1 << " sign=" << r.sign << " logabs=" << std::setprecision(12) << r.logabs
              << " E_L=" << r.local_energy << "\n";
    rs.push_back(std::move(r));
  }
  hid_t rf = H5Fopen(reference.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  struct Field
  {
    std::string name;
    Shape tail;
    std::function<std::vector<double>(const Result&)> get;
  };
  std::vector<Field> fields = {{"sign", {}, [](auto& r) { return std::vector<double>{r.sign}; }},
                               {"logabs", {}, [](auto& r) { return std::vector<double>{r.logabs}; }},
                               {"value", {}, [](auto& r) { return std::vector<double>{r.value}; }},
                               {"gradient_logabs", {model.ne, 3}, [](auto& r) { return r.gradient; }},
                               {"laplacian_logabs_electron", {model.ne}, [](auto& r) { return r.lap_log; }},
                               {"laplacian_ratio_electron", {model.ne}, [](auto& r) { return r.lap_ratio; }},
                               {"potential_ee_en_nn", {3}, [](auto& r) { return r.potential; }},
                               {"local_energy", {}, [](auto& r) { return std::vector<double>{r.local_energy}; }},
                               {"parameter_gradient_logabs", {model.p.values.size()}, [](auto& r) {
                                  return r.param_gradient;
                                }}};
  hid_t of                  = H5Fcreate(output.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  bool pass                 = true;
  for (auto& field : fields)
  {
    std::vector<double> got;
    for (auto& r : rs)
    {
      auto v = field.get(r);
      got.insert(got.end(), v.begin(), v.end());
    }
    Shape refsh;
    auto ref  = read_reference(rf, field.name, &refsh);
    double ma = 0, mr = 0;
    for (size_t i = 0; i < got.size(); ++i)
    {
      ma = std::max(ma, std::abs(got[i] - ref[i]));
      mr = std::max(mr, std::abs(got[i] - ref[i]) / (1e-12 + std::abs(ref[i])));
    }
    double tol = field.name == "parameter_gradient_logabs" ? 2e-9 : 2e-9;
    bool ok    = ma <= tol || mr <= tol;
    pass &= ok;
    std::cout << std::setw(32) << field.name << " max_abs=" << std::scientific << ma << " max_rel=" << mr << " "
              << (ok ? "PASS" : "FAIL") << "\n";
    Shape sh{rs.size()};
    sh.insert(sh.end(), field.tail.begin(), field.tail.end());
    write_dataset(of, field.name, sh, got);
  }
  H5Fclose(of);
  H5Fclose(rf);
  return pass ? 0 : 2;
}
} // namespace pf

#ifndef PSIFORMER_LIBRARY
int main(int argc, char** argv)
{
  if (argc != 5)
  {
    std::cerr << "usage: psiformer_cpp PARAMETERS.h5 CONFIGS.h5 REFERENCE.h5 "
                 "OUTPUT.h5\n";
    return 1;
  }
  try
  {
    return pf::run(argv[1], argv[2], argv[3], argv[4]);
  }
  catch (const std::exception& e)
  {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
#endif
