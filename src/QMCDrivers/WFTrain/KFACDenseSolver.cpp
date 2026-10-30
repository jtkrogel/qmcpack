//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//////////////////////////////////////////////////////////////////////////////////////
#include "KFACDenseSolver.h"
#include "KFACPreconditioner.h"
#include "TrainingNumerics.h"
#include "OutputManager.h"
#include "config.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unistd.h>

#if defined(ENABLE_CUDA)
#include "QMCWaveFunctions/detail/CUDA/PsiFormerDeviceResources.h"
#include "QMCWaveFunctions/detail/CUDA/PsiFormerDeviceDerivativeKernels.h"
#ifdef QMC_CUDA2HIP
#include <rocsolver/rocsolver.h>
#else
#include <cusolverDn.h>
#include <cublas_v2.h>
#endif
#endif

namespace qmcplusplus::wftrain
{
namespace
{
std::uint64_t nextKfacEventGenerationId() noexcept
{
  static std::atomic<std::uint64_t> next{1};
  return next.fetch_add(1, std::memory_order_relaxed);
}

bool kfacGpuEventTraceRequested() noexcept
{
  static const bool enabled = [] {
    const char* setting = std::getenv("PSIFORMER_KFAC_GPU_EVENT_TRACE");
    return setting && *setting && std::string_view(setting) != "0";
  }();
  return enabled;
}

void warnKfacEventTrace(const char* reason) noexcept
{
  static std::atomic<bool> warned{false};
  if (!warned.exchange(true))
    std::fprintf(stderr, "KFAC GPU event timing disabled: %s; scientific execution continues\n", reason);
}

class KFACGpuEventTraceSink
{
public:
  KFACGpuEventTraceSink() noexcept
  {
    try
    {
      const char* prefix = std::getenv("PSIFORMER_KFAC_GPU_EVENT_TRACE");
      if (!prefix || !*prefix)
        return;
      std::string rank = "unknown";
      for (const char* name : {"OMPI_COMM_WORLD_RANK", "PMI_RANK", "SLURM_PROCID"})
        if (const char* value = std::getenv(name))
        {
          if (*value && std::string_view(value).find_first_not_of("0123456789") == std::string_view::npos)
            rank = value;
          break;
        }
      rank_ = rank;
      pid_ = getpid();
      const std::string path = std::string(prefix) + ".rank" + rank_ + ".pid" +
          std::to_string(pid_) + ".jsonl";
      file_.open(path, std::ios::out | std::ios::app);
      if (!file_)
        warnKfacEventTrace("cannot open sidecar path");
    }
    catch (...)
    {
      warnKfacEventTrace("sidecar initialization failed");
    }
  }

  void record(int device, std::uint64_t generation, std::size_t slot, std::size_t block,
              double milliseconds, const char* status) noexcept
  {
    try
    {
      std::lock_guard<std::mutex> guard(mutex_);
      if (!file_)
        return;
#ifdef QMC_CUDA2HIP
      constexpr const char* backend = "hip";
#else
      constexpr const char* backend = "cuda";
#endif
      file_ << "{\"schema\":1,\"kind\":\"gpu_event\",\"phase\":\"kfac_factor_stream\","
               "\"scope\":\"factor stream including uploads, vendor factorization, validation kernels, and downloads\","
               "\"completion\":\"stop event read after existing queue sync\",\"source\":\"KFACDenseSolver.cpp\","
               "\"generation_scope\":\"factor generation; not QMC iteration or model version\",\"rank\":\""
            << rank_ << "\",\"pid\":" << pid_ << ",\"backend\":\""
            << backend << "\",\"device\":" << device << ",\"generation_id\":" << generation
            << ",\"slot\":" << slot << ",\"block\":" << block << ",\"mono_ns\":"
            << std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now().time_since_epoch()).count()
            << ",\"duration_ms\":"
            << std::setprecision(9) << milliseconds << ",\"status\":\"" << status << "\"}\n";
      file_.flush();
      if (!file_)
        warnKfacEventTrace("sidecar write failed");
    }
    catch (...)
    {
      warnKfacEventTrace("sidecar write failed");
    }
  }

  static KFACGpuEventTraceSink& instance() noexcept
  {
    static auto* sink = new KFACGpuEventTraceSink();
    return *sink;
  }

private:
  std::mutex mutex_;
  std::ofstream file_;
  std::string rank_ = "unknown";
  long long pid_ = -1;
};

/// Keep all fixed resource extents in the checked size_t domain.
std::size_t add(std::size_t a, std::size_t b)
{
  if (b > std::numeric_limits<std::size_t>::max() - a)
    throw std::overflow_error("KFAC dense capacity overflow");
  return a + b;
}

/// Reject capacity products before any allocation or vendor integer conversion.
std::size_t mul(std::size_t a, std::size_t b)
{
  if (a && b > std::numeric_limits<std::size_t>::max() / a)
    throw std::overflow_error("KFAC dense capacity overflow");
  return a * b;
}

/// One augmented activation/response block and its resident-bank offset.
struct Layout { std::size_t a, g, offset; };

/// Bind a borrowed generation to the exact canonical HOST factor bytes.
std::uint64_t fingerprint(const std::vector<double>& a, const std::vector<double>& g)
{
  std::uint64_t hash = UINT64_C(14695981039346656037);
  for (const auto* values : {&a, &g})
    for (double value : *values)
    {
      std::array<unsigned char, sizeof(double)> bytes{};
      std::memcpy(bytes.data(), &value, sizeof(value));
      for (auto byte : bytes) { hash ^= byte; hash *= UINT64_C(1099511628211); }
    }
  return hash;
}

#if defined(ENABLE_CUDA)
using psiformer::device::checkRuntime;
using psiformer::device::DeviceBuffer;
using psiformer::device::Stream;
using psiformer::device::success;

/// Separate rocBLAS solver handle: a hipBLAS handle is never reinterpreted.
class VendorQueue
{
public:
  VendorQueue()
  {
    try
    {
#ifdef QMC_CUDA2HIP
      checkRuntime(hipGetDevice(&device_), "KFAC hipGetDevice");
      checkRuntime(hipStreamCreate(&stream_), "KFAC hipStreamCreate");
      check(rocblas_create_handle(&solver_), "rocblas_create_handle");
      check(rocblas_set_stream(solver_, stream_), "rocblas_set_stream");
      check(rocblas_set_pointer_mode(solver_, rocblas_pointer_mode_host), "rocblas_set_pointer_mode");
#else
      checkRuntime(cudaGetDevice(&device_), "KFAC cudaGetDevice");
      checkRuntime(cudaStreamCreate(&stream_), "KFAC cudaStreamCreate");
      check(cusolverDnCreate(&solver_), "cusolverDnCreate");
      check(cusolverDnSetStream(solver_, stream_), "cusolverDnSetStream");
      check(cublasCreate(&blas_), "cublasCreate");
      check(cublasSetStream(blas_, stream_), "cublasSetStream");
      check(cublasSetPointerMode(blas_, CUBLAS_POINTER_MODE_HOST), "cublasSetPointerMode");
#endif
      createEventTimer();
    }
    catch (...) { release(); throw; }
  }

  ~VendorQueue() { release(); }

  /// Close handles explicitly while their caller-owned workspace is still alive.
  void close() noexcept { release(); }

  VendorQueue(const VendorQueue&) = delete;
  VendorQueue& operator=(const VendorQueue&) = delete;

  /// Translate vendor API failure without substituting another numerical route.
  template<class Status> static void check(Status status, const char* operation)
  {
    if (static_cast<int>(status) != 0)
      throw std::runtime_error(std::string("KFAC ") + operation + " status=" + std::to_string(static_cast<int>(status)));
  }

  /// Every operation must remain on the device captured by this optimizer queue.
  void current() const
  {
    int device = -1;
#ifdef QMC_CUDA2HIP
    checkRuntime(hipGetDevice(&device), "KFAC hipGetDevice");
#else
    checkRuntime(cudaGetDevice(&device), "KFAC cudaGetDevice");
#endif
    if (device != device_) throw std::logic_error("KFAC solver owning device changed");
  }

  /// Complete queued work at the existing HOST validation/publication boundary.
  void sync() const
  {
#ifdef QMC_CUDA2HIP
    checkRuntime(hipStreamSynchronize(stream_), "KFAC hipStreamSynchronize");
#else
    checkRuntime(cudaStreamSynchronize(stream_), "KFAC cudaStreamSynchronize");
#endif
  }

  /// Copy only a strided diagonal; no quadratic factor materialization.
  void downloadDiagonal(double* target, const double* source, std::size_t n) const
  {
#ifdef QMC_CUDA2HIP
    checkRuntime(hipMemcpy2DAsync(target, sizeof(double), source, mul(add(n, 1), sizeof(double)),
        sizeof(double), n, hipMemcpyDeviceToHost, stream_), "KFAC diagonal D2H");
#else
    checkRuntime(cudaMemcpy2DAsync(target, sizeof(double), source, mul(add(n, 1), sizeof(double)),
        sizeof(double), n, cudaMemcpyDeviceToHost, stream_), "KFAC diagonal D2H");
#endif
    psiformer::device::deviceMemoryMetrics(device_).downloaded(mul(n, sizeof(double)));
  }
  Stream stream() const noexcept { return stream_; }
  int device() const noexcept { return device_; }

  /// Record a selected factor stream interval only when explicitly requested.
  bool beginEventInterval() noexcept
  {
    if (!event_timing_enabled_ || event_interval_active_)
      return false;
#ifdef QMC_CUDA2HIP
    const auto result = hipEventRecord(event_start_, stream_);
#else
    const auto result = cudaEventRecord(event_start_, stream_);
#endif
    if (result != success)
    {
      disableEventTimer("start event record failed");
      return false;
    }
    event_interval_active_ = true;
    event_interval_stopped_ = false;
    return true;
  }

  /// Place the interval endpoint before the caller's already-required sync.
  bool stopEventInterval() noexcept
  {
    if (!event_timing_enabled_ || !event_interval_active_ || event_interval_stopped_)
      return false;
#ifdef QMC_CUDA2HIP
    const auto result = hipEventRecord(event_stop_, stream_);
#else
    const auto result = cudaEventRecord(event_stop_, stream_);
#endif
    if (result != success)
    {
      disableEventTimer("stop event record failed");
      return false;
    }
    event_interval_stopped_ = true;
    return true;
  }

  /// Query only after the caller's normal stream sync has completed.
  bool readEventIntervalMilliseconds(float& milliseconds) noexcept
  {
    if (!event_timing_enabled_ || !event_interval_active_ || !event_interval_stopped_)
      return false;
#ifdef QMC_CUDA2HIP
    const auto result = hipEventElapsedTime(&milliseconds, event_start_, event_stop_);
#else
    const auto result = cudaEventElapsedTime(&milliseconds, event_start_, event_stop_);
#endif
    event_interval_active_ = false;
    event_interval_stopped_ = false;
    if (result != success)
    {
      disableEventTimer("elapsed-time query failed after queue sync");
      return false;
    }
    return true;
  }

  /// Upload deliberate staging traffic and charge existing transfer diagnostics.
  template<class T> void upload(T* target, const T* source, std::size_t count) const
  {
#ifdef QMC_CUDA2HIP
    checkRuntime(hipMemcpyAsync(target, source, mul(count, sizeof(T)), hipMemcpyHostToDevice, stream_), "KFAC H2D");
#else
    checkRuntime(cudaMemcpyAsync(target, source, mul(count, sizeof(T)), cudaMemcpyHostToDevice, stream_), "KFAC H2D");
#endif
    psiformer::device::deviceMemoryMetrics(device_).uploaded(mul(count, sizeof(T)));
  }

  /// Download canonical factors/results before HOST validation and checkpointing.
  template<class T> void download(T* target, const T* source, std::size_t count) const
  {
#ifdef QMC_CUDA2HIP
    checkRuntime(hipMemcpyAsync(target, source, mul(count, sizeof(T)), hipMemcpyDeviceToHost, stream_), "KFAC D2H");
#else
    checkRuntime(cudaMemcpyAsync(target, source, mul(count, sizeof(T)), cudaMemcpyDeviceToHost, stream_), "KFAC D2H");
#endif
    psiformer::device::deviceMemoryMetrics(device_).downloaded(mul(count, sizeof(T)));
  }

  /// Row-major lower storage is column-major U=L^T; no transpose owner is needed.
  void factor(int n, double* matrix, unsigned char* workspace, std::size_t bytes, int* info)
  {
#ifdef QMC_CUDA2HIP
    check(rocsolver_dpotrf(solver_, rocblas_fill_upper, n, matrix, n, info), "rocsolver_dpotrf");
#else
    check(cusolverDnDpotrf(solver_, CUBLAS_FILL_MODE_UPPER, n, matrix, n,
        reinterpret_cast<double*>(workspace), static_cast<int>(bytes / sizeof(double)), info), "cusolverDnDpotrf");
#endif
  }

  /// M=R^T is already column-major: right A solve, then left G solve.
  void apply(int a, int g, const double* A, const double* G, double* matrix)
  {
    const double one = 1.0;
#ifdef QMC_CUDA2HIP
    check(rocblas_dtrsm(solver_, rocblas_side_right, rocblas_fill_upper, rocblas_operation_none,
        rocblas_diagonal_non_unit, g, a, &one, A, a, matrix, g), "rocblas_dtrsm A/U");
    check(rocblas_dtrsm(solver_, rocblas_side_right, rocblas_fill_upper, rocblas_operation_transpose,
        rocblas_diagonal_non_unit, g, a, &one, A, a, matrix, g), "rocblas_dtrsm A/Ut");
    check(rocblas_dtrsm(solver_, rocblas_side_left, rocblas_fill_upper, rocblas_operation_transpose,
        rocblas_diagonal_non_unit, g, a, &one, G, g, matrix, g), "rocblas_dtrsm G/Ut");
    check(rocblas_dtrsm(solver_, rocblas_side_left, rocblas_fill_upper, rocblas_operation_none,
        rocblas_diagonal_non_unit, g, a, &one, G, g, matrix, g), "rocblas_dtrsm G/U");
#else
    check(cublasDtrsm(blas_, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
        CUBLAS_DIAG_NON_UNIT, g, a, &one, A, a, matrix, g), "cublasDtrsm A/U");
    check(cublasDtrsm(blas_, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
        CUBLAS_DIAG_NON_UNIT, g, a, &one, A, a, matrix, g), "cublasDtrsm A/Ut");
    check(cublasDtrsm(blas_, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
        CUBLAS_DIAG_NON_UNIT, g, a, &one, G, g, matrix, g), "cublasDtrsm G/Ut");
    check(cublasDtrsm(blas_, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
        CUBLAS_DIAG_NON_UNIT, g, a, &one, G, g, matrix, g), "cublasDtrsm G/U");
#endif
  }

  /// Query the maximum sequential vendor workspace during cold preparation.
  std::size_t query(const std::vector<Layout>& blocks, double* factor, double* rhs, int* info)
  {
#ifdef QMC_CUDA2HIP
    check(rocblas_start_device_memory_size_query(solver_), "rocblas_start_device_memory_size_query");
    const auto query_status = [](rocblas_status status) {
      if (status != rocblas_status_size_increased && status != rocblas_status_size_unchanged)
        check(status, "rocBLAS workspace query");
    };
    const double one = 1.0;
    for (const auto& block : blocks)
    {
      for (auto n : {block.a, block.g})
        query_status(rocsolver_dpotrf(solver_, rocblas_fill_upper, static_cast<int>(n), factor, static_cast<int>(n), info));
      for (auto side : {rocblas_side_left, rocblas_side_right})
        for (auto trans : {rocblas_operation_none, rocblas_operation_transpose})
        {
          const int n = static_cast<int>(side == rocblas_side_left ? block.g : block.a);
          query_status(rocblas_dtrsm(solver_, side, rocblas_fill_upper, trans, rocblas_diagonal_non_unit,
              static_cast<int>(block.g), static_cast<int>(block.a), &one, factor, n, rhs, static_cast<int>(block.g)));
        }
    }
    std::size_t bytes = 0;
    check(rocblas_stop_device_memory_size_query(solver_, &bytes), "rocblas_stop_device_memory_size_query");
    return bytes;
#else
    std::size_t bytes = 0;
    for (const auto& block : blocks)
      for (auto n : {block.a, block.g})
      {
        int count = 0;
        check(cusolverDnDpotrf_bufferSize(solver_, CUBLAS_FILL_MODE_UPPER, static_cast<int>(n), factor,
            static_cast<int>(n), &count), "cusolverDnDpotrf_bufferSize");
        if (count < 0) throw std::overflow_error("Negative KFAC vendor workspace");
        bytes = std::max(bytes, mul(static_cast<std::size_t>(count), sizeof(double)));
      }
    // Explicit BLAS workspace shares the sequential solver workspace owner.
    return std::max(bytes, std::size_t{32768});
#endif
  }

  /// Attach the fixed caller-owned workspace admitted with the factor banks.
  void attach(unsigned char* workspace, std::size_t bytes)
  {
#ifdef QMC_CUDA2HIP
    check(rocblas_set_workspace(solver_, workspace, bytes), "rocblas_set_workspace");
#else
    check(cublasSetWorkspace(blas_, workspace, bytes), "cublasSetWorkspace");
#endif
  }

private:
  void createEventTimer() noexcept
  {
    if (!kfacGpuEventTraceRequested())
      return;
#ifdef QMC_CUDA2HIP
    if (hipEventCreate(&event_start_) != success || hipEventCreate(&event_stop_) != success)
    {
      if (event_start_) (void)hipEventDestroy(event_start_);
      if (event_stop_) (void)hipEventDestroy(event_stop_);
      event_start_ = nullptr;
      event_stop_ = nullptr;
      warnKfacEventTrace("event creation failed");
      return;
    }
#else
    if (cudaEventCreate(&event_start_) != success || cudaEventCreate(&event_stop_) != success)
    {
      if (event_start_) cudaEventDestroy(event_start_);
      if (event_stop_) cudaEventDestroy(event_stop_);
      event_start_ = nullptr;
      event_stop_ = nullptr;
      warnKfacEventTrace("event creation failed");
      return;
    }
#endif
    event_timing_enabled_ = true;
  }

  void disableEventTimer(const char* reason) noexcept
  {
    event_timing_enabled_ = false;
    event_interval_active_ = false;
    event_interval_stopped_ = false;
    warnKfacEventTrace(reason);
  }

  /// Nonthrowing cleanup also handles partially constructed vendor resources.
  void release() noexcept
  {
#ifdef QMC_CUDA2HIP
    if (event_start_) { (void)hipEventDestroy(event_start_); event_start_ = nullptr; }
    if (event_stop_) { (void)hipEventDestroy(event_stop_); event_stop_ = nullptr; }
    if (solver_) { rocblas_destroy_handle(solver_); solver_ = nullptr; }
    if (stream_) { (void)hipStreamDestroy(stream_); stream_ = nullptr; }
#else
    if (event_start_) { cudaEventDestroy(event_start_); event_start_ = nullptr; }
    if (event_stop_) { cudaEventDestroy(event_stop_); event_stop_ = nullptr; }
    if (blas_) { cublasDestroy(blas_); blas_ = nullptr; }
    if (solver_) { cusolverDnDestroy(solver_); solver_ = nullptr; }
    if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
#endif
    event_timing_enabled_ = false;
    event_interval_active_ = false;
    event_interval_stopped_ = false;
  }

  int device_ = -1;
  Stream stream_ = nullptr;
  bool event_timing_enabled_ = false;
  bool event_interval_active_ = false;
  bool event_interval_stopped_ = false;
#ifdef QMC_CUDA2HIP
  hipEvent_t event_start_ = nullptr;
  hipEvent_t event_stop_ = nullptr;
  rocblas_handle solver_ = nullptr;
#else
  cudaEvent_t event_start_ = nullptr;
  cudaEvent_t event_stop_ = nullptr;
  cusolverDnHandle_t solver_ = nullptr;
  cublasHandle_t blas_ = nullptr;
#endif
};
#endif
}

/// Optimizer-local capacity, shared scratch and accepted/pending slot ownership.
struct KFACDenseSolver::Impl
{
  std::string fingerprint;
  std::vector<Layout> blocks;
  KFACDenseCapacity capacity = KFACDenseCapacity::STREAMED;
  bool device_authority = false;
  std::size_t factor_elements = 0, block_elements = 0, rhs_elements = 0;
  std::size_t device_bytes = 0, host_bytes = 0, workspace_bytes = 0;
  std::array<bool, 2> occupied{};
  psiformer::memory::HostCapacity host_trace;
#if defined(ENABLE_CUDA)
  VendorQueue queue;
  std::unique_ptr<DeviceBuffer<double>> factors;
  DeviceBuffer<double> rhs;
  DeviceBuffer<unsigned char> workspace;
  DeviceBuffer<int> info;
  DeviceBuffer<psiformer::device::DerivativeFiniteStatus> finite_status;
  std::vector<psiformer::device::DerivativeFiniteStatus> host_status;
  std::vector<double> host_diagonal;

  /// Close the queue before its attached buffers are destroyed.
  ~Impl()
  {
    try { queue.sync(); } catch (...) {}
    // Destroy handles while their explicitly attached workspace is still owned.
    queue.close();
  }

  /// Select a resident generation/block region or the single streaming scratch.
  double* blockPointer(std::size_t slot, std::size_t block)
  {
    return factors->data() + (capacity == KFACDenseCapacity::RESIDENT ? slot * factor_elements + blocks.at(block).offset : 0);
  }
#endif
};

KFACDenseSolver::KFACDenseSolver(std::unique_ptr<Impl> implementation) : impl_(std::move(implementation)) {}

KFACDenseSolver::~KFACDenseSolver() = default;

KFACDenseCapacity KFACDenseSolver::capacity() const noexcept { return impl_->capacity; }

std::size_t KFACDenseSolver::deviceBytes() const noexcept { return impl_->device_bytes; }

std::size_t KFACDenseSolver::hostBytes() const noexcept { return impl_->host_bytes; }

std::size_t KFACDenseSolver::workspaceBytes() const noexcept { return impl_->workspace_bytes; }

const std::string& KFACDenseSolver::registryFingerprint() const noexcept { return impl_->fingerprint; }
bool KFACDenseSolver::deviceAuthority() const noexcept { return impl_->device_authority; }

std::shared_ptr<KFACDenseSolver> KFACDenseSolver::prepare(const KFACBlockRegistry& registry,
    std::size_t maximum_device_bytes, std::size_t maximum_host_bytes, KFACDenseCapacity preference,
    bool device_authority)
{
#if defined(ENABLE_CUDA)
  if (!maximum_device_bytes)
  {
    const auto& admission = psiformer::device::configuredDeviceMemoryAdmission();
    if (!admission.limit_bytes)
      throw std::length_error("KFAC device backend requires explicit aggregate DEVICE admission");
    int device = -1;
#ifdef QMC_CUDA2HIP
    checkRuntime(hipGetDevice(&device), "KFAC admission hipGetDevice");
#else
    checkRuntime(cudaGetDevice(&device), "KFAC admission cudaGetDevice");
#endif
    const auto live = psiformer::device::deviceMemoryMetrics(device).snapshot().live_bytes;
    const auto charged = add(live, admission.external_reserve_bytes);
    if (charged >= *admission.limit_bytes)
      throw std::length_error("KFAC device aggregate admission is exhausted");
    maximum_device_bytes = *admission.limit_bytes - charged;
  }
  if (!maximum_device_bytes || !maximum_host_bytes)
    throw std::length_error("KFAC device solver requires existing DEVICE and HOST numeric admission");
  auto p = std::make_unique<Impl>();
  p->device_authority = device_authority;
  p->fingerprint = registry.fingerprint();
  p->blocks.reserve(registry.affineBlocks().size());
  for (const auto& block : registry.affineBlocks())
  {
    const auto a = add(block.input_width, block.bias_block == NO_KFAC_BIAS_BLOCK ? 0 : 1), g = block.output_width;
    if (!a || !g || a > INT_MAX || g > INT_MAX)
      throw std::length_error("KFAC vendor factor dimensions exceed the integer domain");
    const auto count = add(mul(a, a), mul(g, g));
    p->blocks.push_back({a, g, p->factor_elements});
    p->factor_elements = add(p->factor_elements, count);
    p->block_elements = std::max(p->block_elements, count);
    p->rhs_elements = std::max(p->rhs_elements, mul(a, g));
  }
  if (p->blocks.empty()) throw std::invalid_argument("KFAC device solver requires affine blocks");
  p->host_bytes = add(mul(p->blocks.capacity(), sizeof(Layout)),
      add(mul(p->blocks.size(), 2 * (sizeof(unsigned char) + sizeof(std::uint64_t))), sizeof(int) * 2));
  std::size_t status_count = 0, diagonal_count = 0;
  if (device_authority)
  {
    status_count = p->block_elements / 4096 + (p->block_elements % 4096 != 0);
    for (const auto& block : p->blocks) diagonal_count = std::max(diagonal_count, add(block.a, block.g));
    p->host_bytes = add(p->host_bytes, add(mul(status_count, sizeof(psiformer::device::DerivativeFiniteStatus)),
        mul(diagonal_count, sizeof(double))));
    p->host_bytes = add(p->host_bytes, mul(mul(p->blocks.size(), 2), sizeof(KFACDenseGeneration::CanonicalBlock)));
    // Reserve admission for two future HOST checkpoint images, without allocating them now.
    if (add(p->host_bytes, mul(mul(p->factor_elements, 2), sizeof(double))) > maximum_host_bytes)
      throw std::length_error("KFAC authority future HOST materialization exceeds admission");
  }
  if (p->host_bytes > maximum_host_bytes) throw std::length_error("KFAC solver HOST metadata exceeds admission");
  const auto base = add(add(mul(p->rhs_elements, sizeof(double)), sizeof(int) * 2),
      mul(status_count, sizeof(psiformer::device::DerivativeFiniteStatus)));
  if (add(base, mul(p->block_elements, sizeof(double))) > maximum_device_bytes)
    throw std::length_error("KFAC one-block device solver exceeds admission");
  p->rhs.reserveNamed(p->rhs_elements, "KFAC dense RHS", __FILE__, __LINE__);
  p->info.reserveNamed(2, "KFAC solver status", __FILE__, __LINE__);
  auto query = std::make_unique<DeviceBuffer<double>>(p->block_elements, "KFAC factor query", __FILE__, __LINE__);
  p->workspace_bytes = p->queue.query(p->blocks, query->data(), p->rhs.data(), p->info.data());
  const auto fixed = add(base, p->workspace_bytes);
  const auto resident = add(fixed, mul(mul(p->factor_elements, 2), sizeof(double)));
  const auto streamed = add(fixed, mul(p->block_elements, sizeof(double)));
  if (streamed > maximum_device_bytes) throw std::length_error("KFAC queried workspace exceeds device admission");
  p->capacity = preference == KFACDenseCapacity::STREAMED ? KFACDenseCapacity::STREAMED :
      (resident <= maximum_device_bytes ? KFACDenseCapacity::RESIDENT : KFACDenseCapacity::STREAMED);
  if (preference == KFACDenseCapacity::RESIDENT && p->capacity != preference)
    throw std::length_error("KFAC resident test capacity exceeds device admission");
  if (device_authority && p->capacity != KFACDenseCapacity::RESIDENT)
    throw std::length_error("KFAC device authority requires admitted resident generations");
  if (p->capacity == KFACDenseCapacity::RESIDENT)
  {
    query.reset();
    p->factors = std::make_unique<DeviceBuffer<double>>(mul(p->factor_elements, 2), "KFAC accepted/pending factors", __FILE__, __LINE__);
    p->device_bytes = resident;
  }
  else { p->factors = std::move(query); p->device_bytes = streamed; }
  p->workspace.reserveNamed(p->workspace_bytes, "KFAC vendor workspace", __FILE__, __LINE__);
  if (device_authority)
  {
    p->finite_status.reserveNamed(status_count, "KFAC authority finite statuses", __FILE__, __LINE__);
    p->host_status.resize(status_count);
    p->host_diagonal.resize(diagonal_count);
  }
  p->queue.attach(p->workspace.data(), p->workspace_bytes);
  if (psiformer::memory::enabled())
    p->host_trace.snapshotAt(__FILE__, __LINE__, "KFACDenseSolver::HOST_numeric_metadata_bound", p->host_bytes);
  return std::shared_ptr<KFACDenseSolver>(new KFACDenseSolver(std::move(p)));
#else
  throw std::invalid_argument("KFAC device dense backend is unavailable in this build");
#endif
}

std::shared_ptr<KFACDenseGeneration> KFACDenseSolver::beginGeneration()
{
  for (std::size_t slot = 0; slot < impl_->occupied.size(); ++slot)
    if (!impl_->occupied[slot])
    {
      auto result = std::shared_ptr<KFACDenseGeneration>(new KFACDenseGeneration(shared_from_this(), slot, impl_->blocks.size()));
      impl_->occupied[slot] = true;
      return result;
    }
  throw std::logic_error("KFAC accepted/pending generation slots exhausted");
}

KFACDenseGeneration::KFACDenseGeneration(std::shared_ptr<KFACDenseSolver> resource, std::size_t slot, std::size_t blocks)
    : resource_(std::move(resource)), slot_(slot),
      event_generation_id_(kfacGpuEventTraceRequested() ? nextKfacEventGenerationId() : 0),
      prepared_(blocks, 0), fingerprints_(blocks, 0),
      host_cache_(resource_->deviceAuthority() ? blocks : 0) {}

KFACDenseGeneration::~KFACDenseGeneration() { resource_->impl_->occupied[slot_] = false; }

void KFACDenseGeneration::factorize(std::size_t block, std::vector<double>& activation, std::vector<double>& sensitivity)
{
#if defined(ENABLE_CUDA)
  auto& p = *resource_->impl_;
  if (p.device_authority && imported_)
    throw std::logic_error("KFAC generation cannot mix imported HOST and resident factor authority");
  if (complete_ || failed_ || block >= prepared_.size() || prepared_[block])
    throw std::logic_error("KFAC factor generation is not writable");
  const auto& layout = p.blocks.at(block);
  if (activation.size() != mul(layout.a, layout.a) || sensitivity.size() != mul(layout.g, layout.g))
    throw std::invalid_argument("KFAC damped matrix extent mismatch");
  bool event_interval = false;
  bool event_elapsed_available = false;
  float event_duration_ms = 0.0f;
  try
  {
    p.queue.current();
    event_interval = p.queue.beginEventInterval();
    auto* A = p.blockPointer(slot_, block); auto* G = A + activation.size();
    p.queue.upload(A, activation.data(), activation.size());
    p.queue.upload(G, sensitivity.data(), sensitivity.size());
    p.queue.factor(static_cast<int>(layout.a), A, p.workspace.data(), p.workspace_bytes, p.info.data());
    p.queue.factor(static_cast<int>(layout.g), G, p.workspace.data(), p.workspace_bytes, p.info.data() + 1);
    std::array<int, 2> status{};
    p.queue.download(status.data(), p.info.data(), status.size());
    if (p.device_authority)
    {
      const auto cells = add(activation.size(), sensitivity.size());
      const auto pieces = cells / 4096 + (cells % 4096 != 0);
      checkRuntime(psiformer::device::launchDerivativeFiniteScan(p.queue.stream(),
          {cells, 4096, pieces, 0, 1}, A, p.finite_status.data()), "KFAC authority finite scan");
      p.queue.download(p.host_status.data(), p.finite_status.data(), pieces);
      p.queue.downloadDiagonal(p.host_diagonal.data(), A, layout.a);
      p.queue.downloadDiagonal(p.host_diagonal.data() + layout.a, G, layout.g);
    }
    else
    {
      p.queue.download(activation.data(), A, activation.size());
      p.queue.download(sensitivity.data(), G, sensitivity.size());
    }
    if (event_interval)
      p.queue.stopEventInterval();
    p.queue.sync();
    if (event_interval)
    {
      event_elapsed_available = p.queue.readEventIntervalMilliseconds(event_duration_ms);
      event_interval = false;
    }
    if (status[0] || status[1]) throw MatrixFreeNumericalError("KFAC vendor Cholesky rejected non-SPD factor");
    if (p.device_authority)
    {
      const auto cells = add(activation.size(), sensitivity.size());
      const auto pieces = cells / 4096 + (cells % 4096 != 0);
      for (std::size_t i = 0; i < pieces; ++i)
        if (p.host_status[i].nonfinite_count) throw MatrixFreeNumericalError("KFAC resident factor is nonfinite");
      for (std::size_t i = 0; i < add(layout.a, layout.g); ++i)
        if (!isFiniteTrainingReal(p.host_diagonal[i]) || p.host_diagonal[i] <= 0.)
          throw MatrixFreeNumericalError("KFAC resident factor diagonal is invalid");
      std::vector<double>().swap(activation);
      std::vector<double>().swap(sensitivity);
      factorized_ = true;
      prepared_[block] = 1;
      if (event_elapsed_available)
        KFACGpuEventTraceSink::instance().record(p.queue.device(), event_generation_id_, slot_, block,
            event_duration_ms, "complete");
      return;
    }
    for (auto item : {std::make_pair(&activation, layout.a), std::make_pair(&sensitivity, layout.g)})
      for (std::size_t row = 0; row < item.second; ++row)
      {
        for (std::size_t col = 0; col <= row; ++col)
          if (!isFiniteTrainingReal((*item.first)[row * item.second + col]) ||
              (row == col && (*item.first)[row * item.second + col] <= 0.0))
            throw MatrixFreeNumericalError("KFAC vendor Cholesky factor is nonfinite");
        std::fill(item.first->begin() + row * item.second + row + 1, item.first->begin() + (row + 1) * item.second, 0.0);
      }
    fingerprints_[block] = fingerprint(activation, sensitivity);
    prepared_[block] = 1;
    if (event_elapsed_available)
      KFACGpuEventTraceSink::instance().record(p.queue.device(), event_generation_id_, slot_, block,
          event_duration_ms, "complete");
  }
  catch (...)
  {
    failed_ = true;
    if (event_interval)
      p.queue.stopEventInterval();
    try { p.queue.sync(); } catch (...) {}
    if (event_interval && p.queue.readEventIntervalMilliseconds(event_duration_ms))
      KFACGpuEventTraceSink::instance().record(p.queue.device(), event_generation_id_, slot_, block,
          event_duration_ms, "failed");
    else if (event_elapsed_available)
      KFACGpuEventTraceSink::instance().record(p.queue.device(), event_generation_id_, slot_, block,
          event_duration_ms, "validation_failed");
    throw;
  }
#else
  throw std::logic_error("KFAC device backend unavailable");
#endif
}

void KFACDenseGeneration::import(std::size_t block, const std::vector<double>& activation,
    const std::vector<double>& sensitivity)
{
#if defined(ENABLE_CUDA)
  auto& p = *resource_->impl_;
  if (p.device_authority && factorized_)
    throw std::logic_error("KFAC generation cannot mix resident factors and imported HOST authority");
  if (complete_ || failed_ || block >= prepared_.size() || prepared_[block])
    throw std::logic_error("KFAC factor generation is not writable");
  const auto& layout = p.blocks.at(block);
  if (activation.size() != mul(layout.a, layout.a) || sensitivity.size() != mul(layout.g, layout.g))
    throw std::invalid_argument("KFAC canonical factor extent mismatch");
  try
  {
    p.queue.current();
    if (p.capacity == KFACDenseCapacity::RESIDENT)
    {
      auto* A = p.blockPointer(slot_, block);
      p.queue.upload(A, activation.data(), activation.size());
      p.queue.upload(A + activation.size(), sensitivity.data(), sensitivity.size());
      p.queue.sync();
    }
    fingerprints_[block] = fingerprint(activation, sensitivity);
    imported_ = true;
    prepared_[block] = 1;
  }
  catch (...) { failed_ = true; try { p.queue.sync(); } catch (...) {} throw; }
#else
  throw std::logic_error("KFAC device backend unavailable");
#endif
}

void KFACDenseGeneration::finish()
{
  if (complete_ || failed_ || !std::all_of(prepared_.begin(), prepared_.end(), [](auto value) { return value == 1; }))
    throw std::logic_error("KFAC factor generation is incomplete");
  complete_ = true;
}

bool KFACDenseGeneration::matches(std::size_t block, const std::vector<double>& activation,
    const std::vector<double>& sensitivity) const
{
  if (deviceAuthority())
  {
    if (!complete() || block >= prepared_.size()) return false;
    // Explicit legacy HOST matching is a cold materialization request. Internal
    // resident snapshots bypass this API and bind immutable shared identity.
    const auto& canonical = materialize(block);
    if (activation.size() != canonical.activation.size() || sensitivity.size() != canonical.sensitivity.size())
      return false;
    return fingerprint(canonical.activation, canonical.sensitivity) == fingerprint(activation, sensitivity);
  }
  return complete_ && !failed_ && block < fingerprints_.size() &&
      fingerprints_[block] == fingerprint(activation, sensitivity);
}

bool KFACDenseGeneration::deviceAuthority() const noexcept { return resource_->deviceAuthority() && !imported_; }

std::size_t KFACDenseGeneration::materializedBytes() const noexcept
{
  return host_cache_bytes_.load(std::memory_order_relaxed);
}

// Cache only complete, validated HOST images; a failed copy leaves device authority intact.
const KFACDenseGeneration::CanonicalBlock& KFACDenseGeneration::materialize(std::size_t block) const
{
#if defined(ENABLE_CUDA)
  auto& p = *resource_->impl_;
  if (!complete() || !deviceAuthority() || p.capacity != KFACDenseCapacity::RESIDENT || block >= prepared_.size())
    throw std::logic_error("KFAC HOST materialization requires a complete resident authority");
  std::lock_guard<std::mutex> lock(host_mutex_);
  auto& cached = host_cache_.at(block);
  if (!cached.activation.empty()) return cached;
  const auto& layout = p.blocks.at(block);
  CanonicalBlock candidate;
  const auto a_elements = mul(layout.a, layout.a), g_elements = mul(layout.g, layout.g);
  const auto planned_bytes = mul(add(a_elements, g_elements), sizeof(double));
  app_log() << "KFAC allocation before owner=canonical_HOST_materialization block=" << block
            << " activation_dimension=" << layout.a << " sensitivity_dimension=" << layout.g
            << " planned_numeric_bytes=" << planned_bytes
            << " authority=immutable_resident_generation lifetime=generation_cache"
            << " slot=" << slot_ << " (numeric_intent_not_RSS_or_checkpoint_peak)" << std::endl;
  candidate.activation.resize(a_elements);
  candidate.sensitivity.resize(g_elements);
  p.queue.current();
  auto* A = p.blockPointer(slot_, block);
  try
  {
    p.queue.download(candidate.activation.data(), A, candidate.activation.size());
    p.queue.download(candidate.sensitivity.data(), A + candidate.activation.size(), candidate.sensitivity.size());
    p.queue.sync();
  }
  catch (...) { try { p.queue.sync(); } catch (...) {} throw; }
  for (auto item : {std::make_pair(&candidate.activation, layout.a), std::make_pair(&candidate.sensitivity, layout.g)})
    for (std::size_t row = 0; row < item.second; ++row)
    {
      for (std::size_t col = 0; col <= row; ++col)
        if (!isFiniteTrainingReal((*item.first)[row * item.second + col]) ||
            (row == col && (*item.first)[row * item.second + col] <= 0.))
          throw MatrixFreeNumericalError("KFAC materialized factor is invalid");
      std::fill(item.first->begin() + row * item.second + row + 1,
          item.first->begin() + (row + 1) * item.second, 0.);
    }
  const auto bytes = mul(add(candidate.activation.capacity(), candidate.sensitivity.capacity()), sizeof(double));
  cached = std::move(candidate);
  host_cache_bytes_.fetch_add(bytes, std::memory_order_relaxed);
  if (psiformer::memory::enabled())
    host_cache_trace_.snapshotAt(__FILE__, __LINE__, "KFACDenseGeneration::canonical_HOST_materialization",
        materializedBytes(), reinterpret_cast<std::uintptr_t>(this), -1, slot_);
  return cached;
#else
  throw std::logic_error("KFAC device backend unavailable");
#endif
}

// Applying this authority neither reads nor hashes a HOST factor vector.
void KFACDenseGeneration::applyResident(std::size_t block, double* matrix, std::size_t count) const
{
#if defined(ENABLE_CUDA)
  auto& p = *resource_->impl_;
  if (!complete() || !deviceAuthority() || p.capacity != KFACDenseCapacity::RESIDENT)
    throw std::logic_error("KFAC resident authority cannot apply");
  const auto& layout = p.blocks.at(block);
  if (!matrix || count != mul(layout.a, layout.g)) throw std::invalid_argument("KFAC resident RHS extent mismatch");
  try
  {
    p.queue.current();
    auto* A = p.blockPointer(slot_, block);
    auto* G = A + mul(layout.a, layout.a);
    p.queue.upload(p.rhs.data(), matrix, count);
    p.queue.apply(static_cast<int>(layout.a), static_cast<int>(layout.g), A, G, p.rhs.data());
    p.queue.download(matrix, p.rhs.data(), count);
    p.queue.sync();
    if (!std::all_of(matrix, matrix + count, isFiniteTrainingReal))
      throw MatrixFreeNumericalError("KFAC resident solve produced nonfinite output");
  }
  catch (...) { try { p.queue.sync(); } catch (...) {} throw; }
#else
  throw std::logic_error("KFAC device backend unavailable");
#endif
}

void KFACDenseGeneration::apply(std::size_t block, const std::vector<double>& activation,
    const std::vector<double>& sensitivity, double* matrix, std::size_t count) const
{
  if (deviceAuthority())
  {
    if (!matches(block, activation, sensitivity))
      throw std::invalid_argument("KFAC supplied HOST factors differ from resident authority");
    return applyResident(block, matrix, count);
  }
#if defined(ENABLE_CUDA)
  auto& p = *resource_->impl_;
  if (!complete_ || failed_) throw std::logic_error("KFAC factor generation cannot apply");
  const auto& layout = p.blocks.at(block);
  if (!matrix || count != mul(layout.a, layout.g) || activation.size() != mul(layout.a, layout.a) ||
      sensitivity.size() != mul(layout.g, layout.g)) throw std::invalid_argument("KFAC RHS extent mismatch");
  try
  {
    p.queue.current();
    auto* A = p.blockPointer(slot_, block); auto* G = A + activation.size();
    if (p.capacity == KFACDenseCapacity::STREAMED)
    {
      p.queue.upload(A, activation.data(), activation.size());
      p.queue.upload(G, sensitivity.data(), sensitivity.size());
    }
    p.queue.upload(p.rhs.data(), matrix, count);
    p.queue.apply(static_cast<int>(layout.a), static_cast<int>(layout.g), A, G, p.rhs.data());
    p.queue.download(matrix, p.rhs.data(), count);
    p.queue.sync();
    if (!std::all_of(matrix, matrix + count, isFiniteTrainingReal))
      throw MatrixFreeNumericalError("KFAC vendor solve produced nonfinite output");
  }
  catch (...) { try { p.queue.sync(); } catch (...) {} throw; }
#else
  throw std::logic_error("KFAC device backend unavailable");
#endif
}
}
