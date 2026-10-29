//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
/** Focused small paired owner/strict-epoch oracle; ordinary execution requires a GPU. */
#define PSIFORMER_LIBRARY
#include "PsiFormerDeviceMixedPeriodicGeometryValidation.h"
#include "PsiFormerDeviceMixedComplexSpatialValidation.h"

namespace
{
using namespace qmcplusplus::psiformer;
using namespace qmcplusplus::psiformer::device;

/// Reduce production canonical metadata to a small test-only architecture.
/// This does not change or weaken the versioned production initializer profile.
pf::Parameters smallParameters(const ExecutionEnvironment& environment)
{
  const ModelShape production{2, 2, 2, 16, 256, 4, 4};
  auto profile = initializePsiFormerParameters(production, 1729).profile;
  if (environment.geometry_feature_policy == GeometryFeaturePolicy::PERIODIC_TWO_CELL_V1)
    profile = PSIFORMER_PERIODIC_PAIR_V1;
  else if (environment.head_representation == HeadRepresentation::PAIRED_REAL_COMPLEX)
    profile = PSIFORMER_COMPLEX_PERIODIC_V1;
  const auto initialized = initializePsiFormerParameters(production, 1729, profile, environment);
  const PsiFormerExecutionPlan production_plan(production, initialized.layouts, environment);
  std::vector<pf::Layout> layouts;
  std::vector<double> values;
  for (const auto& tensor : production_plan.parameterTensors())
  {
    if (tensor.attention_block != NO_ATTENTION_BLOCK && tensor.attention_block >= 2)
      continue;
    auto shape = tensor.shape;
    if (tensor.role == ParameterRole::PAIR_ATTENTION_WEIGHT || tensor.role == ParameterRole::PAIR_ATTENTION_BIAS)
      shape.back() = 2;
    std::size_t count = 1;
    for (auto& extent : shape)
    {
      if (extent == 256 || extent == 64)
        extent = 16; // width 16, and four determinants times four electrons.
      count *= extent;
    }
    const auto begin = values.size();
    const bool rescale = tensor.role != ParameterRole::ELECTRON_EMBEDDING_WEIGHT &&
        (tensor.shape.size() == 1 || (tensor.shape.size() == 2 && tensor.shape[0] == 256));
    for (std::size_t index = 0; index < count; ++index)
      values.push_back(tensor.role == ParameterRole::PAIR_ATTENTION_WEIGHT || tensor.role == ParameterRole::PAIR_ATTENTION_BIAS
          ? .03 * (static_cast<int>((index + 3 * tensor.attention_block) % 11) - 5)
          : initialized.values[tensor.begin + index] * (rescale ? 4.0 : 1.0));
    layouts.push_back({tensor.module, tensor.name, shape, begin, values.size()});
  }
  return pf::Parameters(std::move(values), std::move(layouts));
}

/// Native, file-free configurations include an independent perturbation and image.
pf::ConfigData configurations(const ExecutionEnvironment& environment)
{
  std::vector<double> positions{0.4, 0.7, 0.9, -0.2, 1.3, -0.7,
                                1.1, -0.6, 0.3, -0.8, -0.4, 1.2};
  const auto base = positions;
  positions.insert(positions.end(), base.begin(), base.end());
  positions[12] += 0.031;
  positions[17] -= 0.017;
  positions.insert(positions.end(), base.begin(), base.end());
  for (std::size_t dimension = 0; dimension < 3; ++dimension)
    positions[24 + dimension] += environment.lattice_vectors[0][dimension] -
        2 * environment.lattice_vectors[2][dimension];
  return pf::ConfigData(pf::Tensor({3, 4, 3}, std::move(positions)),
      pf::Tensor({2, 3}, {0.1, 0.2, -0.1, 0.8, -0.3, 0.7}),
      pf::Tensor({2}, {3.0, 1.0}), 2, 2, 4);
}

} // namespace

/// Exercise only the existing file-free Ne4 paired fixture, without native XML/performance admission.
int main()
{
  using namespace qmcplusplus::psiformer;
  using namespace qmcplusplus::psiformer::device;
  try
  {
    int devices = 0;
#ifdef QMC_CUDA2HIP
    checkRuntime(hipGetDeviceCount(&devices), "paired owner visible devices");
    const char* backend = "HIP";
#else
    checkRuntime(cudaGetDeviceCount(&devices), "paired owner visible devices");
    const char* backend = "CUDA";
#endif
    if (!devices) throw std::runtime_error("Paired owner oracle requires a visible GPU");
    ExecutionEnvironment environment;
    environment.boundary = BoundaryCondition::PERIODIC;
    environment.geometry_feature_policy = GeometryFeaturePolicy::PERIODIC_TORUS_V1;
    environment.lattice_vectors = {{{4., .4, -.2}, {.7, 5., .3}, {-.1, .8, 6.}}};
    environment.periodic_axes = {true, true, true};
    environment.head_representation = HeadRepresentation::PAIRED_REAL_COMPLEX;
    environment.amplitude_scalar_domain = ScalarDomain::COMPLEX;
    pf::PsiFormer model(smallParameters(environment), configurations(environment), 4, 16, 2, 2);
    const auto plan = PsiFormerExecutionPlan::fromParameters(model.p, {2, 2, 2, 4, 16, 2, 2}, environment);
    auto values = model.p.flat_values();
    for (auto role : {ParameterRole::BACKFLOW_UP_WEIGHT, ParameterRole::BACKFLOW_DOWN_WEIGHT})
    {
      const auto& imaginary = plan.parameter(role, NO_ATTENTION_BLOCK, HeadTensorComponent::IMAG);
      for (std::size_t i = imaginary.begin; i < imaginary.end; ++i)
        values[i] = .09 * std::sin(double(i - imaginary.begin + 1));
    }
    model.p.set_flat_values(values);
    auto image = PreparedDeviceModel::preparePairedComplexValue(plan, values, model.cfg.nuclei.x);
    qmcplusplus::compute::Queue<qmcplusplus::PlatformKind::CUDA> queue;
    AcceleratorBlasHandle blas(queue);
    std::cout.precision(17);
    std::cout << "Actual-device paired owner oracle backend=" << backend
              << " Ne=4 D=4 F=16 component_only=1 native_admission=held\n";
    validateMixedComplexNativeOwners(image,
        std::vector<double>(model.cfg.electrons.x.begin(), model.cfg.electrons.x.begin() + 12), values, queue, blas);
    return 0;
  }
  catch (const std::exception& error)
  {
    std::cerr << "FAIL paired owner oracle: " << error.what() << '\n';
    return 1;
  }
}
