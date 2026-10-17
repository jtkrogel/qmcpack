//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#include "QMCDrivers/WFTrain/AcceleratedLinearMethod.h"
#include "QMCDrivers/WFTrain/TrainingNumerics.h"
#include "Platforms/CPU/Blasf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

namespace qmcplusplus::wftrain
{
namespace
{
using Vector = std::vector<DerivativeValue>;
using Clock = std::chrono::steady_clock;
/// Contract real augmented vectors and reject overflow under fast-math builds.
double dot(const Vector& a, const Vector& b)
{
  double value = 0;
  for (std::size_t i = 0; i < a.size(); ++i)
    value += a[i].real() * b[i].real();
  if (!isFiniteTrainingReal(value))
    throw MatrixFreeNumericalError("Accelerated LM contraction is non-finite");
  return value;
}
/// Return the Euclidean norm of a real augmented vector.
double norm(const Vector& a) { return std::sqrt(dot(a, a)); }
/// Add a scaled augmented vector without allocating a temporary.
void axpy(double alpha, const Vector& x, Vector& y)
{
  for (std::size_t i = 0; i < y.size(); ++i)
    y[i] += alpha * x[i];
}
/// Scale one owning augmented vector in place.
void scale(double alpha, Vector& x)
{
  for (auto& value : x)
    value *= alpha;
}
/// Retain overlap-orthonormal search vectors and their two operator images.
struct Basis
{
  std::vector<Vector> v, h, s;
};
/// Carry a trusted real Ritz root and its already computed operator images.
struct Ritz
{
  double energy;
  Vector x, h, s;
};
/// Unwind nested correction work with one shared solver stop reason.
struct Stop
{
  LinearMethodStopReason reason;
};

/// Own one bounded outer solve and its nested projected GMRES corrections.
class Solver
{
public:
  /// Bind immutable problem identities and start the shared wall-time clock.
  Solver(const AugmentedLinearMethodOperator& h, const AugmentedLinearMethodOperator& s,
         const MatrixFreePreconditioner& p, const AcceleratedLinearMethodControl& c)
      : h_(h), s_(s), p_(p), c_(c), n_(h.descriptor().tangent_count + 1), start_(Clock::now())
  {
    result.provider_id = h.descriptor().provider_id;
    result.schema_fingerprint = h.descriptor().schema_fingerprint;
    result.parameter_version = h.descriptor().parameter_version;
  }
  MatrixFreeLinearMethodResult result;
  /// Enforce wall time without requiring further operator work.
  void checkTime() const
  {
    if (c_.outer.maximum_seconds > 0 && elapsed() >= c_.outer.maximum_seconds)
      throw Stop{LinearMethodStopReason::TIME_LIMIT};
  }
  /// Enforce both work bounds immediately before an operator application.
  void check() const
  {
    checkTime();
    if (result.operator_applications >= c_.outer.maximum_operator_applications)
      throw Stop{LinearMethodStopReason::OPERATOR_APPLICATION_LIMIT};
  }
  /// Return monotonic elapsed time for diagnostics and the shared deadline.
  double elapsed() const { return std::chrono::duration<double>(Clock::now() - start_).count(); }
  /// Charge every H/S application, including all inner correction work.
  void apply(const AugmentedLinearMethodOperator& op, const Vector& v, Vector& out)
  {
    check();
    ++result.operator_applications;
    op.apply({v.data(), v.size()}, {out.data(), out.size()});
    checkTime();
  }
  /// Track a conservative bound including retained Ritz and inner Arnoldi vectors.
  void memory(const Basis& b, std::size_t extra = 0)
  {
    result.peak_subspace_dimension = std::max(result.peak_subspace_dimension, b.v.size());
    // Includes owning Ritz vectors, residual/correction and restart temporaries.
    result.peak_parameter_vectors = std::max(result.peak_parameter_vectors,
                                             9 * b.v.size() + 18 + extra);
  }
  /// Normalize an independent correction in S and cache both checked images.
  void add(Basis& b, Vector v)
  {
    Vector sv(n_), hv(n_);
    apply(s_, v, sv);
    // Twice modified Gram--Schmidt in the overlap metric.
    for (int pass = 0; pass < 2; ++pass)
      for (std::size_t j = 0; j < b.v.size(); ++j)
      {
        double alpha = dot(b.v[j], sv);
        axpy(-alpha, b.v[j], v);
        axpy(-alpha, b.s[j], sv);
      }
    const double squared = dot(v, sv);
    if (!isFiniteTrainingReal(squared))
      throw Stop{LinearMethodStopReason::NONFINITE_RESULT};
    if (squared < 0)
      throw Stop{LinearMethodStopReason::NON_POSITIVE_OVERLAP};
    if (squared <= c_.outer.minimum_overlap_norm * c_.outer.minimum_overlap_norm)
      throw Stop{LinearMethodStopReason::LINEAR_DEPENDENCE};
    scale(1 / std::sqrt(squared), v);
    scale(1 / std::sqrt(squared), sv);
    apply(h_, v, hv);
    b.v.push_back(std::move(v));
    b.s.push_back(std::move(sv));
    b.h.push_back(std::move(hv));
    memory(b);
  }
  /// Solve the small nonsymmetric pencil and rank trusted real roots by E0 distance.
  std::vector<Ritz> ritz(const Basis& b, double target)
  {
    checkTime();
    int m = static_cast<int>(b.v.size()), one = 1, info = 0;
    // LAPACK dimensions are int; matrix indexing must remain size_t to avoid m*m overflow.
    const std::size_t stride = b.v.size();
    std::vector<double> hp(stride * stride), sp(stride * stride), ar(m), ai(m), beta(m), vr(stride * stride);
    for (int j = 0; j < m; ++j)
      for (int i = 0; i < m; ++i)
      {
        hp[i + stride * j] = dot(b.v[i], b.h[j]);
        sp[i + stride * j] = dot(b.v[i], b.s[j]);
      }
    // Do not silently symmetrize a provider that violates its S promise.
    for (int j = 0; j < m; ++j)
      for (int i = 0; i < j; ++i)
        if (std::abs(sp[i + stride * j] - sp[j + stride * i]) >
            c_.outer.hermitian_tolerance * std::numeric_limits<double>::epsilon() * n_ *
                std::max({1.0, std::abs(sp[i + stride * j]), std::abs(sp[j + stride * i])}))
          throw Stop{LinearMethodStopReason::NON_HERMITIAN_ACTION};
    char no = 'N', yes = 'V';
    int lwork = std::max(1, 8 * m);
    std::vector<double> work(lwork);
    double unused;
    if (m == 2)
      std::fprintf(stderr, "ALM TEMP pencil H %.17g %.17g %.17g %.17g S %.17g %.17g %.17g %.17g\n",
          hp[0], hp[2], hp[1], hp[3], sp[0], sp[2], sp[1], sp[3]);
    dggev(&no, &yes, &m, hp.data(), &m, sp.data(), &m, ar.data(), ai.data(), beta.data(),
          &unused, &one, vr.data(), &m, work.data(), &lwork, &info);
    checkTime();
    if (info != 0)
      throw Stop{LinearMethodStopReason::PROJECTED_SOLVE_FAILURE};
    std::vector<Ritz> candidates;
    for (int root = 0; root < m; ++root)
    {
      if (m == 2)
        std::fprintf(stderr, "ALM TEMP root %d ar %.17g ai %.17g beta %.17g\n",
            root, ar[root], ai[root], beta[root]);
      // DGGEV stores a complex pair in adjacent columns; neither is a real vector.
      if (!isFiniteTrainingReal(ai[root]) || !isFiniteTrainingReal(ar[root]) ||
          !isFiniteTrainingReal(beta[root]) || ai[root] != 0 || beta[root] == 0 ||
          !isFiniteTrainingReal(ar[root] / beta[root]))
        continue;
      Ritz pair{ar[root] / beta[root], Vector(n_), Vector(n_), Vector(n_)};
      for (int j = 0; j < m; ++j)
      {
        axpy(vr[j + stride * root], b.v[j], pair.x);
        axpy(vr[j + stride * root], b.h[j], pair.h);
        axpy(vr[j + stride * root], b.s[j], pair.s);
      }
      const double snorm2 = dot(pair.x, pair.s);
      if (m == 2)
        std::fprintf(stderr, "ALM TEMP admission snorm2 %.17g x0 %.17g xnorm %.17g minS %.17g minRef %.17g\n",
            snorm2, pair.x[0].real(), norm(pair.x), c_.outer.minimum_overlap_norm,
            c_.outer.minimum_reference_overlap);
      if (!isFiniteTrainingReal(snorm2) || snorm2 <= c_.outer.minimum_overlap_norm * c_.outer.minimum_overlap_norm)
        continue;
      const double snorm = std::sqrt(snorm2);
      if (std::abs(pair.x[0].real()) / snorm < c_.outer.minimum_reference_overlap ||
          std::abs(pair.x[0].real()) <= std::numeric_limits<double>::epsilon() * norm(pair.x))
        continue;
      scale(1 / snorm, pair.x);
      scale(1 / snorm, pair.h);
      scale(1 / snorm, pair.s);
      candidates.push_back(std::move(pair));
    }
    std::stable_sort(candidates.begin(), candidates.end(), [target](const Ritz& a, const Ritz& b) {
      return std::abs(a.energy - target) < std::abs(b.energy - target);
    });
    if (candidates.empty())
      throw Stop{LinearMethodStopReason::UNTRUSTED_ROOT};
    return candidates;
  }
  /// Approximately solve the generalized JD equation with right-preconditioned GMRES.
  Vector correction(const Ritz& u, const Vector& residual, const Basis& b)
  {
    // Right and left JD projections have different orientations for generalized S.
    auto right = [&](Vector& v) { axpy(-dot(u.s, v), u.x, v); };
    auto left = [&](Vector& v) { axpy(-dot(u.x, v), u.s, v); };
    auto precondition = [&](const Vector& v) {
      checkTime();
      Vector z(n_);
      z[0] = v[0];
      StructuredParameterVectorConstView view(p_.parameterSchema(), p_.parameterVersion(),
                                              {v.data() + 1, n_ - 1});
      p_.apply(view, {z.data() + 1, n_ - 1});
      checkTime();
      right(z);
      return z;
    };
    auto action = [&](const Vector& z) {
      Vector hz(n_), sz(n_);
      apply(h_, z, hz);
      apply(s_, z, sz);
      axpy(-u.energy, sz, hz);
      left(hz);
      return hz;
    };
    Vector rhs = residual, t(n_);
    scale(-1, rhs);
    left(rhs);
    const double initial_norm = norm(rhs);
    std::size_t iterations = 0;
    while (iterations < c_.inner_iterations && norm(rhs) > c_.inner_relative_tolerance * initial_norm)
    {
      const std::size_t k = std::min(c_.inner_restart_length, c_.inner_iterations - iterations);
      std::vector<Vector> v, z;
      std::vector<double> a((k + 1) * k), cs(k), sn(k), g(k + 1);
      g[0] = norm(rhs);
      v.push_back(rhs);
      scale(1 / g[0], v.back());
      std::size_t used = 0;
      for (std::size_t j = 0; j < k; ++j)
      {
        z.push_back(precondition(v[j]));
        Vector w = action(z.back());
        for (int pass = 0; pass < 2; ++pass)
          for (std::size_t i = 0; i <= j; ++i)
          {
            double alpha = dot(v[i], w);
            a[i + (k + 1) * j] += alpha;
            axpy(-alpha, v[i], w);
          }
        const double next_norm = norm(w);
        a[j + 1 + (k + 1) * j] = next_norm;
        if (next_norm > std::numeric_limits<double>::epsilon())
        {
          scale(1 / next_norm, w);
          v.push_back(std::move(w));
        }
        memory(b, v.size() + z.size() + 3);
        // Apply old and new Givens rotations to maintain a triangular least-squares problem.
        for (std::size_t i = 0; i < j; ++i)
        {
          double& upper = a[i + (k + 1) * j];
          double& lower = a[i + 1 + (k + 1) * j];
          double value = cs[i] * upper + sn[i] * lower;
          lower = -sn[i] * upper + cs[i] * lower;
          upper = value;
        }
        double& diagonal = a[j + (k + 1) * j];
        double& lower = a[j + 1 + (k + 1) * j];
        double magnitude = std::hypot(diagonal, lower);
        if (!isFiniteTrainingReal(magnitude) || magnitude == 0)
          break;
        cs[j] = diagonal / magnitude;
        sn[j] = lower / magnitude;
        diagonal = magnitude;
        lower = 0;
        g[j + 1] = -sn[j] * g[j];
        g[j] *= cs[j];
        ++iterations;
        used = j + 1;
        if (std::abs(g[j + 1]) <= c_.inner_relative_tolerance * initial_norm || v.size() <= j + 1)
          break;
      }
      if (used == 0)
        break;
      std::vector<double> y(used);
      // Back substitution combines the right-preconditioned Arnoldi directions.
      for (std::size_t ii = used; ii-- > 0;)
      {
        double value = g[ii];
        for (std::size_t j = ii + 1; j < used; ++j)
          value -= a[ii + (k + 1) * j] * y[j];
        y[ii] = value / a[ii + (k + 1) * ii];
        axpy(y[ii], z[ii], t);
      }
      rhs = residual;
      scale(-1, rhs);
      left(rhs);
      Vector image = action(t);
      axpy(-1, image, rhs);
    }
    // A finite inexact JD step is useful even if the inner residual did not converge.
    if (norm(t) <= c_.outer.minimum_overlap_norm)
      t = precondition(residual);
    return t;
  }
  /// Iterate real Ritz extraction, residual acceptance, JD expansion and thick restart.
  void run(Vector initial)
  {
    Basis b;
    add(b, std::move(initial));
    const double target = dot(b.v[0], b.h[0]);
    if (!isFiniteTrainingReal(target))
      throw Stop{LinearMethodStopReason::INVALID_REFERENCE_OVERLAP};
    for (std::size_t iteration = 0; iteration < c_.outer.maximum_iterations; ++iteration)
    {
      checkTime();
      auto roots = ritz(b, target);
      Ritz& u = roots.front();
      Vector residual = u.h;
      axpy(-u.energy, u.s, residual);
      result.iterations = iteration + 1;
      result.eigenvalue = u.energy;
      result.eigenvector = u.x;
      result.residual_norm = norm(residual);
      const double residual_scale = norm(u.h) + std::abs(u.energy) * norm(u.s);
      if (!isFiniteTrainingReal(residual_scale))
        throw Stop{LinearMethodStopReason::NONFINITE_RESULT};
      result.relative_residual_norm = residual_scale > 0 ? result.residual_norm / residual_scale : result.residual_norm;
      if (result.residual_norm <= c_.outer.absolute_residual_tolerance +
          c_.outer.relative_residual_tolerance * residual_scale)
      {
        result.converged = true;
        throw Stop{LinearMethodStopReason::CONVERGED};
      }
      if (iteration + 1 == c_.outer.maximum_iterations)
        break;
      Vector step = correction(u, residual, b);
      if (b.v.size() == c_.outer.maximum_subspace_dimension)
      {
        // Retain several trusted Ritz vectors, reorthogonalizing their stored images.
        Basis retained;
        const std::size_t keep = std::min({c_.retained_ritz_vectors, roots.size(),
                                           c_.outer.maximum_subspace_dimension - 1});
        for (std::size_t i = 0; i < keep; ++i)
        {
          Ritz pair = std::move(roots[i]);
          for (int pass = 0; pass < 2; ++pass)
            for (std::size_t j = 0; j < retained.v.size(); ++j)
            {
              double alpha = dot(retained.v[j], pair.s);
              axpy(-alpha, retained.v[j], pair.x);
              axpy(-alpha, retained.h[j], pair.h);
              axpy(-alpha, retained.s[j], pair.s);
            }
          double squared = dot(pair.x, pair.s);
          if (squared <= c_.outer.minimum_overlap_norm * c_.outer.minimum_overlap_norm)
            continue;
          scale(1 / std::sqrt(squared), pair.x);
          scale(1 / std::sqrt(squared), pair.h);
          scale(1 / std::sqrt(squared), pair.s);
          retained.v.push_back(std::move(pair.x));
          retained.h.push_back(std::move(pair.h));
          retained.s.push_back(std::move(pair.s));
        }
        b = std::move(retained);
        ++result.restart_count;
      }
      add(b, std::move(step));
    }
    throw Stop{LinearMethodStopReason::ITERATION_LIMIT};
  }
private:
  const AugmentedLinearMethodOperator &h_, &s_;
  const MatrixFreePreconditioner& p_;
  const AcceleratedLinearMethodControl& c_;
  std::size_t n_;
  Clock::time_point start_;
};
} // namespace

/// Validate contracts before allocating bounded work and translate numerical stop conditions.
MatrixFreeLinearMethodResult solveAcceleratedMatrixFreeLinearMethod(
    const AugmentedLinearMethodOperator& h, const AugmentedLinearMethodOperator& s,
    const MatrixFreePreconditioner& p, DerivativeArrayView<const DerivativeValue> initial,
    const AcceleratedLinearMethodControl& c)
{
  const auto& hd = h.descriptor();
  const auto& sd = s.descriptor();
  if (hd.reduction_domain != ReductionDomain::GLOBAL || sd.reduction_domain != ReductionDomain::GLOBAL ||
      !sd.hermitian || hd.provider_id != sd.provider_id || hd.schema_fingerprint != sd.schema_fingerprint ||
      hd.parameter_version != sd.parameter_version || hd.tangent_count != sd.tangent_count ||
      p.parameterSchema().fingerprint() != hd.schema_fingerprint ||
      p.parameterSchema().providerId() != hd.provider_id || p.parameterVersion() != hd.parameter_version)
    throw std::invalid_argument("Accelerated LM requires matching global H, Hermitian S, and preconditioner identities");
  if (initial.size() != hd.tangent_count + 1 || initial.data() == nullptr)
    throw std::invalid_argument("Accelerated LM initial extent must be P+1");
  for (std::size_t i = 0; i < initial.size(); ++i)
    if (initial[i] != DerivativeValue(i == 0 ? 1.0 : 0.0, 0.0))
      throw std::invalid_argument("Accelerated LM requires the canonical reference initial vector");
  const auto& o = c.outer;
  // In particular, 8*m must fit the DGGEV int workspace argument.
  const std::size_t maximum_lapack_dimension = std::numeric_limits<int>::max() / 8;
  const std::size_t maximum_inner_dimension = std::min(c.inner_iterations, c.inner_restart_length);
  if (!isFiniteTrainingReal(o.relative_residual_tolerance) || o.relative_residual_tolerance < 0 ||
      !isFiniteTrainingReal(o.absolute_residual_tolerance) || o.absolute_residual_tolerance < 0 ||
      o.relative_residual_tolerance + o.absolute_residual_tolerance <= 0 ||
      !isFiniteTrainingReal(o.maximum_seconds) || o.maximum_seconds < 0 ||
      !isFiniteTrainingReal(o.minimum_overlap_norm) || o.minimum_overlap_norm <= 0 ||
      !isFiniteTrainingReal(o.minimum_reference_overlap) || o.minimum_reference_overlap < 0 ||
      o.minimum_reference_overlap > 1 || !isFiniteTrainingReal(o.hermitian_tolerance) || o.hermitian_tolerance <= 0 ||
      o.maximum_subspace_dimension < 2 || o.maximum_subspace_dimension > maximum_lapack_dimension ||
      o.maximum_subspace_dimension > std::vector<double>().max_size() / o.maximum_subspace_dimension ||
      c.inner_iterations == 0 || c.inner_restart_length == 0 || c.retained_ritz_vectors == 0 ||
      maximum_inner_dimension >= std::vector<double>().max_size() ||
      maximum_inner_dimension > std::vector<double>().max_size() / (maximum_inner_dimension + 1) ||
      !isFiniteTrainingReal(c.inner_relative_tolerance) || c.inner_relative_tolerance <= 0 || c.inner_relative_tolerance >= 1)
    throw std::invalid_argument("Invalid accelerated LM work bounds or numerical thresholds");
  Solver solver(h, s, p, c);
  try
  {
    solver.run(Vector(initial.begin(), initial.end()));
  }
  catch (const Stop& stop) { solver.result.stop_reason = stop.reason; }
  catch (const MatrixFreeNumericalError&) { solver.result.stop_reason = LinearMethodStopReason::NONFINITE_RESULT; }
  solver.result.elapsed_seconds = solver.elapsed();
  return std::move(solver.result);
}
} // namespace qmcplusplus::wftrain
