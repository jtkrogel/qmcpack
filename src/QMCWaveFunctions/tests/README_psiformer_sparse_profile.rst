Sparse CPU memory/request profiling
===================================

These developer benchmarks are not performance CTests. Build only the existing
``benchmark_psiformer_native_memory`` target with micro-benchmarks enabled.
The standalone executable does not require qmcpack or the component benchmark.

Generate deterministic fixtures in a separate process before freezing inputs:

.. code-block:: text

   benchmark_psiformer_native_memory --generate-fixture hydrogen20 NEW_DIRECTORY
   benchmark_psiformer_native_memory --generate-fixture hydrogen50 NEW_DIRECTORY
   benchmark_psiformer_native_memory --generate-fixture hydrogen100 NEW_DIRECTORY
   benchmark_psiformer_native_memory --generate-fixture hydrogen200 NEW_DIRECTORY
   benchmark_psiformer_native_memory --generate-fixture hydrogen300 NEW_DIRECTORY

These are neutral open hydrogen clusters with balanced spins and the existing
D16/width256/head4/depth4 test layout. Random parameters and non-equilibrated
positions provide shape smoke tests, not production physics or scaling claims.
Existing LiH/OH recipes and seeds are unchanged. Hash all generated files.
The H50/H100 recipes and RNG consumption are unchanged. The five exact sizes
share the same trunk but not the same parameter count. Check every determinant
channel's conditioning and finite complete products before profiling; generation
alone does not qualify the random model or authorize larger resource caps.

Complete-request examples, using identical immutable inputs for each pair:

.. code-block:: text

   benchmark_psiformer_native_memory --scenario batch_value --parameters PARAMETERS \
     --configuration CONFIGURATION --walkers 4 --configuration-tile 2 \
     --warm-calls 0 --output value.json
   benchmark_psiformer_native_memory --scenario complete_vgl --parameters PARAMETERS \
     --configuration CONFIGURATION --walkers 1 --target-tile 0 \
     --warm-calls 0 --open-coulomb 1 --output full.json
   benchmark_psiformer_native_memory --scenario complete_vgl --parameters PARAMETERS \
     --configuration CONFIGURATION --walkers 1 --target-tile 1 \
     --warm-calls 0 --open-coulomb 1 --output target.json
   python3 run_psiformer_sparse_profile.py --compare full.json.values.json target.json.values.json

VALUE tile B is independent of logical W. FULL/target VGL process configurations
serially with one shared workspace; target mode visits EVERY electron. W may
exceed the supplied configuration count; configurations are cycled with an
explicit deterministic coordinate shift of 0.0007*w*(dimension+1). All members
of a comparison receive identical coordinates. Results retain O(W*N), never
sample-by-P or P-squared storage. Each VGL configuration records sign/logabs,
all gradient/lap_log/lap_ratio lanes, kinetic energy and, only when explicitly
requested, open Coulomb total energy. NiH PP comparisons must omit open Coulomb:
only the separately pinned native application anchor measures actual ECP energy.

``elapsed_request_seconds`` includes workspace preparation and complete evaluation,
not model loading or JSON writing. ``owner_preparation_seconds`` isolates cold
owner admission/allocation; ``numerical_evaluation_seconds`` sums only executor
calls, excluding preparation, packing, result copying and Coulomb arithmetic.
Whole-process wall time includes loading and
I/O and is reported independently by the controller. Snapshot categories name
only listed owners, not a complete resident-memory model; outputs, allocator,
libraries and vendor storage remain visible in RSS/cgroup evidence.

The diagnostic follow-up (integrated as ``8e9241ccd``) persists the current
JSON report after each complete-request snapshot and logs attempted owner
bytes before allocation. A later timeout can therefore retain prepared-owner
categories. Earlier frozen profiles are unchanged and are not rerun; a partial
prepared report is not evidence that the full numerical request completed.

The controller is inert without ``--execute-case ID`` and root-approved manifest
status ``ready_to_execute``. Root must provide immutable benchmark/application
source and SHA256 fields, all model paths/hashes, allocation/node verification,
the exact runtime root, and an explicit ``approved_case_ids`` launch allowlist.
The first approval can allow only one case while retaining the proposed sparse
manifest. Generated files are prepared and hashed BEFORE that
approval, not silently regenerated in a measured case. Its one-case invocation
uses only an existing allocation, checks a 14 GiB step-user cgroup fence before
launch, stops at 80 percent charged usage or 90 seconds, and prevents repeats.
At most ten attempts and 900 aggregate seconds are permitted. No auto-sweep,
retry, new allocation or scientific convergence recommendation is implemented.

The optional unchanged native NiH anchor requires ``original_native_xml`` plus
the original root/inputs layout and auxiliary PP hashes. Its application source
is recorded separately from the benchmark source. A timeout, singular model,
nonfinite result, allocation refusal or failed numeric pair is retained as a
partial/failure observation, not a complete equal-work timing comparison.

Run the cheap controller tests without Slurm or model data:

.. code-block:: text

   python3 -m unittest discover -s src/QMCWaveFunctions/tests -p test_psiformer_sparse_profile.py

Optional native device ownership summary
---------------------------------------

Set ``PSIFORMER_GPU_MEMORY_SUMMARY=1`` to print a native ownership summary after
each successfully completed QMC section. Loop sections use the same boundary;
there is no extra device synchronization or MPI collective. Other values leave
this reporting disabled without a device-runtime query. GPU-less ranks skip it.

The summary names the communicator rank and vendor-local device ordinal and
reports existing ``DeviceBuffer`` live/peak bytes, allocation/free counts and
successful explicit upload/download/device-copy counts and bytes. Counters are
cumulative since their last explicit reset; reporting itself never resets them.
The normal application log may emit only selected ranks, so these lines are not
a whole-job sum or a maximum across ranks/devices.

Tracked ownership is not runtime HBM consumption or a sampled HBM peak. HOST,
vendor/runtime allocations, unwrapped QMCPACK/offload buffers, other processes,
external allocations and kernel memory traffic are excluded. Keep aggregate
admission caps, external reserves, observed counters and actual HBM/RSS evidence
as separate fields; a byte cap is not a measured peak.

Experimental large VALUE-only cooperative LU
-------------------------------------------

``PSIFORMER_GPU_VALUE_COOPERATIVE_LU=1`` opts into one-block-per-matrix LU for
REAL OPEN FP64 VALUE-only graphs with at least 64 electrons and no retained
score tape.  Unset or ``0`` preserves scalar dispatch; malformed values reject
graph preparation.  FULL spatial, retained score/inverse, complex, periodic,
and mixed requests (including their FP64 recovery) remain scalar.  A training
workflow may still opt in its separate non-retained VALUE proposal graph.

This adds no global numeric owner; static block control is at most 64 bytes.
Existing memory/work limits and numerical acceptance requirements still apply.
The preparation messages are process-scoped diagnostic output, not a job-wide
sum; ``usesCooperativeValueLu()`` exposes actual graph eligibility/selection.
This experimental switch does not change defaults or advertise a new product
profile.  ``psiformer_device_runtime_validation --value-lu-only`` exercises the
raw LU, metadata, permutation, edge and scalar-fallback contracts on hardware.

Experimental large OPEN spatial feature map
------------------------------------------

``PSIFORMER_GPU_OPEN_FEATURE_COOPERATIVE=1`` opts into one 128-thread block
per configuration for non-retained REAL OPEN FP64 FULL spatial feature jets
with at least 64 electrons. Unset or ``0`` preserves scalar dispatch; malformed
values reject preparation. VALUE, retained tapes/targets, kinetic-trunk-only,
complex, periodic and mixed requests (including FP64 recovery) remain scalar.
The switch is independent of the VALUE-only cooperative LU switch.

The existing output layout, target ranges and status contract are unchanged.
The map adds no global numeric owner and uses four bytes of static block control.
Failed work replays the canonical scalar device helper, preserving its first
status and unpublished partial scratch. Existing memory/work limits still apply.
``usesCooperativeOpenFeatures()`` reports actual eligibility/selection; messages
are process-scoped, not a whole-job sum. Defaults and admitted products are
unchanged. ``psiformer_device_runtime_validation --open-feature-only`` checks
the bounded raw layout, padding, failure/recovery and small-size fallback cases.

Bounded small real determinant LU
--------------------------------

``PSIFORMER_GPU_SMALL_MATRIX_COOPERATIVE_LU=1`` additionally admits sizes32--63
through the existing real cooperative VALUE LU and independent-column inverse.
The VALUE graph still requires ``PSIFORMER_GPU_VALUE_COOPERATIVE_LU=1`` and its
existing real/domain/mode eligibility; preparation logs the effective selection.
Inverse preparation independently requires ``PSIFORMER_GPU_COOPERATIVE_LU_INVERSE=1``.
The new selector accepts only0 or1; unset/0 preserves previous size admission.
In-place input/LU aliasing retains the scalar route. No new global owner is added.

``psiformer_device_runtime_validation --small-lu-only`` compares N44 scalar and
cooperative VALUE/inverse results for eight configurations with D16 and D64,
checking all LU/permutation/metadata/inverse fields, the native CPU oracle,
inverse identity products, singular/nonfinite/scaled/near-singular fixtures,
failed refresh/recovery and in-place fallback. Run on an actual CUDA/HIP device;
no usable device exits77. This focused qualification does not cover complex or
mixed-precision execution.

``PSIFORMER_GPU_SMALL_MATRIX_SPATIAL_TARGETS=1`` independently selects one block
per(configuration,electron) for real spatial traces at sizes32--63 with at most
N*N targets. It uses N*N doubles of dynamic shared memory per target block
(15488bytes at N44), reusing the existing global matrix scratch for independent
failure flags. Stream-ordered admission/finite scans and completion preserve
configuration-wide status and clearing without status atomics or new owners.
Unset/0 and unsupported dimensions preserve the previous spatial selection.
The selector accepts only0 or1. The existing cooperative-spatial flag can remain
set; the bounded target selection takes priority when admitted.

``psiformer_device_spatial_trace_validation --small-targets-only`` checks N44,
two configurations, D16/D64 and3/44 targets against both previous GPU routes
bitwise and the native CPU tolerance. It also checks zero-weight derivative
scans, singular/unavailable factors, nonfinite inputs, arithmetic overflow,
configuration-wide clearing, failed refresh/recovery and zero hidden allocations.

Dedicated ECP VALUE batch capacity
----------------------------------

With ``PSIFORMER_GPU_ECP_VALUE_TILE16=1``, the optional strict-positive-decimal
``PSIFORMER_GPU_ECP_VALUE_TILE_CAPACITY`` selects the dedicated virtual VALUE
capacity. Unset retains16. It does not change ordinary sampler/FULL capacity,
ECP quadrature, logical virtual-point counts, or existing domain/reuse admission.
The outer Hamiltonian virtual-point tile remains independently bounded at256;
requested capacities above its effective request size do not imply that occupancy.
Malformed, zero or overflowing enabled capacity selections fail explicitly.
Compare the completed maximum occupancy and physical memory, not only the option.

Grouped ECP orbital projection
------------------------------

``PSIFORMER_GPU_ECP_VALUE_PROJECTED_BACKFLOW=1`` admits ordinary, lazy,
single-crowd real OPEN FP64 virtual VALUE requests to a grouped BLAS orbital
projection. Unset/0 preserves the previous route; malformed selections and
unsupported explicit backend configurations fail. This is independent of the
periodic envelope-anchor policy. Retained derivatives, mixed precision and
periodic/complex requests are outside this selector's supported domain.

Independent configurations, jet planes and determinants share one pointer-batched
BLAS call per occupied spin. Pointer arrays are refreshed from the current model
on every request; no host numerical pointer bank is uploaded. Projection output
reuses the existing VALUE matrix bank before determinant extraction. At N44,
D64 and capacity264 the three device pointer banks occupy811008 bytes. Their
capacity is included in device workspace accounting. Logical tails, current
parameter addresses and request/error completion retain their own boundaries.

The projection validation compares grouped and previous projections against CPU
results, including padded layouts and empty-spin cases. Different FP64 reduction
association can change roundoff; stochastic optimization trajectories need not
remain identical over aggressive updates. Compare fixed-input evaluations and
controlled nonzero updates as well as complete-unit work and timings.

Small-matrix VALUE LU storage
-----------------------------

``PSIFORMER_GPU_SMALL_MATRIX_LU_SHARED=1`` selects a block-local shared-memory
LU working matrix for dimensions32 through63. Unset/0 and other dimensions retain
the global working matrix. The existing global LU owner still receives the full
or partially factored matrix before status publication, including early failure
paths. The N44 shared matrix occupies15488 bytes per block; this changes transient
on-chip storage, not the persistent global allocation count. Invalid selections
fail. Pivot order, per-cell arithmetic and permutation semantics are preserved.

This selector is independent of ``PSIFORMER_GPU_VALUE_LU_PARALLEL_SCALE=1``;
their interaction should be measured with complete units rather than inferred
from isolated kernel timings. The small-LU validator checks scalar GPU and CPU
results, inverse on/off, scaled/singular/nonfinite matrices, failure recovery and
allocation ownership. Explicit selectors remain off by default.
