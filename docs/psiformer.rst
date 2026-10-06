PsiFormer wavefunction and optimization
=======================================

.. _psiformer_wavefunction:

Scope
-----

QMCPACK can evaluate and optimize a PsiFormer model imported from the compact
DeepQMC HDF5 export used by the native C++ evaluator. The implementation
supports fixed evaluation, selected canonical parameters, or the complete
network. Spatial derivatives, score derivatives, kinetic-energy derivatives,
and nonlocal pseudopotential derivative ratios use graph-free direct kernels by
default. The original native graph/autodifferentiation evaluator remains an
oracle for validation and migration comparisons.

The architecture is unchanged between all-electron and pseudopotential
calculations. The configuration must nevertheless describe the same physical
system. A pseudopotential export contains the explicitly represented
electrons and effective ionic charges used during training; an all-electron
export cannot be silently reused.

Wavefunction input
------------------

A fixed component needs the parameter and configuration exports:

.. code-block:: xml

  <wavefunction name="psi0" target="e">
    <psiformer name="pf"
               parameters="parameters.h5"
               configuration="configuration.h5"/>
  </wavefunction>

Optimization requires an explicit system declaration and source ion set:

.. code-block:: xml

  <wavefunction name="psi0" target="e">
    <psiformer name="pf"
               source="ion0"
               system="pseudopotential"
               parameters="parameters.h5"
               configuration="configuration.h5"
               optimize="yes"
               optimize_scope="all"
               export_parameters="parameters_optimized.h5"/>
  </wavefunction>

``name``
  Unique component and optimization-object name. Distinct optimizable objects
  with the same name are rejected.

``parameters`` and ``configuration``
  DeepQMC-compatible HDF5 files containing the flat parameter vector and
  layout, and the nuclei, effective charges, electron counts, spin split, and
  reference configurations.

``source``
  Ion particle set used for runtime metadata validation; default ``ion0``.

``system``
  ``all_electron`` or ``pseudopotential``. Optimization rejects the default
  ``auto`` value so the physical system must be validated.

``optimize``
  ``yes`` exposes parameters to QMCPACK. The default is ``no``.

``optimize_scope``
  ``indices`` or ``all``. For ``indices``, provide canonical flat indices with
  ``optimize_indices``, separated by spaces or commas. ``all`` registers the
  complete export and cannot be combined with ``optimize_indices``.

``export_parameters``
  Optional final DeepQMC-format destination. QMCPACK writes it while reporting
  final variational parameters, using a sibling temporary file and rename.

The target must have exactly two discrete spin groups matching ``n_up`` and
``n_down``. Source ion positions and charges must match the export. For a
pseudopotential calculation, these are effective charges, not necessarily
atomic numbers.

Optimizer choice and memory
---------------------------

Use streaming descent for a complete PsiFormer. RMSProp, ADAM, and AMSGrad are
the intended first-order choices:

.. code-block:: xml

  <loop max="100">
    <qmc method="linear" move="pbyp" checkpoint="0">
      <parameter name="blocks">100</parameter>
      <parameter name="steps">1</parameter>
      <parameter name="samples">2048</parameter>
      <parameter name="warmupsteps">100</parameter>
      <parameter name="timestep">0.02</parameter>
      <parameter name="MinMethod">descent</parameter>
      <parameter name="flavor">RMSprop</parameter>
      <parameter name="Neural_eta">0.001</parameter>
      <parameter name="Ramp_eta">no</parameter>
      <parameter name="descent_state_file">psiformer.descent.h5</parameter>
    </qmc>
  </loop>

``Neural_eta`` controls parameters whose stable names contain ``_pf_``. Start
conservatively and qualify the learning rate for the system and sample count.
The descent handle consumes each crowd batch online and retains only
parameter-sized accumulators.

Dense linear-method storage and the current stored-score SR path are not
production choices for a complete network. Before allocating persistent
sample-by-parameter records, QMCPACK estimates their per-rank size. A
PsiFormer allocation above the current 2 GiB safety limit is rejected with
the estimate and a streaming-descent recommendation.

Checkpoints, export, and restart
--------------------------------

Three files have different roles:

* ``<root>.vp.h5`` contains the complete model, layout and system fingerprints,
  selected-index mapping, and ordinary QMCPACK variational parameters.
* ``descent_state_file`` contains optimizer iteration, parameter identity,
  learning-rate state, moments, and bounded gradient history.
* ``export_parameters`` contains current values and immutable DeepQMC layout
  for later inference or interchange.

An exact continuation needs the matching VP and descent-state files. Supplying
only the exported parameter file is a warm start. Restart rejects stale format
versions, different layouts or systems, changed selections or identities,
inconsistent generic/complete values, and non-finite checkpoint data.

At startup PsiFormer reports its model fingerprint, total and active counts,
selected tensors, parameter version, derivative mode, validated system, and
export destination. The cost function reports estimated persistent derivative
bytes and whether the mode is streaming or stored.

Walker buffers are a separate, short-lived cache used for walker branching and
migration inside a run. They carry a schema tag, model/configuration identity,
parameter version, electron count, accepted coordinates, value validity, and,
when available, full gradient/Laplacian data. ``copyFromBuffer`` validates this
metadata before adding cached component derivatives to the target walker.
Parameter changes, foreign models, corrupt headers, coordinate changes, and an
insufficient cached evaluation requirement invalidate or reject the cache rather
than reusing stale observables. Accepting a particle move retains only value
validity until full spatial data are refreshed; rejecting a proposal preserves
the accepted full cache.

This walker buffer is not a substitute for the VP and descent-state files and
does not promise an independent cross-process disk checkpoint format. A new
clone can restore a copied, validated buffer in process; durable optimization
restart still uses the files described above.

Pseudopotentials and current limits
-----------------------------------

Scalar-relativistic local and semilocal nonlocal pseudopotentials are supported.
The local channel uses the ordinary coordinate-only local-ECP operator. With the
batched virtual-particle nonlocal operator, the following paths are supported:

* locality-approximation and determinant-localization-approximation (DLA)
  energies;
* ordinary T-move candidate generation and the fermionic/nonfermionic split
  used by TMDLA;
* the V1 T-move electron sweep, whose candidate ratios are flattened across
  walkers before per-walker selection and acceptance; and
* energy parameter derivatives, reduced directly into the selected optimizer
  destinations.

V0 and V3 T-move selection remain supported by their established per-walker
fallback; only the V1 selection sweep uses the new flattened candidate path.
For every virtual quadrature move, PsiFormer evaluates

.. math::

   R_q = \frac{\Psi(\mathbf R_q)}{\Psi(\mathbf R)}

and contributes the logarithmic score change

.. math::

   O_{q,\theta} - O_\theta =
   \frac{\partial \log|\Psi(\mathbf R_q)|}{\partial\theta}
   -
   \frac{\partial \log|\Psi(\mathbf R)|}{\partial\theta}.

The nonlocal operator multiplies this by its quadrature weight and ``R_q``
exactly once. It enumerates jobs in electron-group/walker/job order and packs
their quadrature knots into bounded outer tiles. A job larger than the tile is
split only at knot boundaries, and a tile does not cross an electron-group
boundary. The current conservative outer capacity is 256 knots and is not a
user tuning input in this revision.

Within one outer tile, the PsiFormer descriptor is sparse: it contains one
reference configuration for each walker represented in that tile and only the
electron-replacement configurations actually requested. This is distinct from
the inner direct-evaluator tile (capacity four by default) used to bound dense
projection scratch. Value ratios use its prepared direct batch workspace. The
weighted derivative path reduces directly into the caller's selected-parameter
destination and does not materialize a knot-by-parameter matrix.

Flattening the weighted dispatch does **not** make parameter reverse mode a
batched reverse kernel. Virtual reference and replacement scores are still
reversed serially through one resource-owned tape. The compact reduction staging
is active-walker by active-parameter and independent of the number of knots,
while the surrounding nonlocal orchestration retains small per-call vectors.

Spinor electrons and spin-orbit pseudopotential ratios are unsupported and
fail during validation. Source-ion gradients, Pulay terms, and force evaluation
with a PsiFormer component are also unsupported; successful scalar-relativistic
energy evaluation does not imply force support. The model is real-valued; it
can be represented in a complex QMCPACK build with phase zero or pi, but is not
a general complex neural ansatz.

ECP resource, version, and memory ownership
-------------------------------------------

Each active crowd acquires one nonlocal-ECP resource whose immutable identity
and shape must match the operator leader and all its clones. That resource owns
the outer tile, job and neighbor-list staging, optional final T-move candidates,
and whole-request energy or derivative staging. The corresponding PsiFormer
crowd resource owns the sparse direct batch workspace and, when derivatives are
requested, one lazily allocated score tape. Component clones share the immutable
model and its published parameter version but do not share mutable accepted or
proposal state.

Value and weighted-score calls return opaque component stamps. All selected
components must agree within a tile, and every outer tile in the request must
retain the first tile's stamps. A parameter publication during the transaction
therefore fails before Hamiltonian-owned results are committed. Energies, jobs,
neighbor lists, T-move candidates, and derivative rows are staged privately and
published only after the complete internal request succeeds. This guarantee
does not rewind consumed random numbers or rotated quadrature grids after a
failure. Listener callbacks form a later external commit phase and cannot be
rolled back if a callback throws after an earlier callback has returned.

Memory reported for this path has several deliberately separate classes:

* shared persistent model storage, held once by the clone family;
* clone-local accepted/proposal state and scalar workspaces;
* bounded crowd scratch proportional to the ECP outer capacity and the
  PsiFormer inner tile capacity;
* logical job metadata proportional to the number of ion-electron jobs;
* optional T-move candidate output proportional to the total knot count;
* whole-request derivative staging proportional to walkers times caller-row
  extent; and
* one full canonical PsiFormer score tape per active crowd, plus compact
  active-walker by active-parameter reduction staging.

Capacity accounting covers owned numeric/vector storage, not allocator
metadata, libraries, HDF5 state, or process RSS. Zero-allocation guarantees
apply only to warmed native kernels and prepared workspaces, not to the complete
Hamiltonian adapter transaction.

Execution backends and workspaces
---------------------------------

The production molecular path uses graph-free ``direct`` evaluators for
value, full value/gradient/Laplacian (VGL), active-electron gradient,
parameter score, and score-plus-kinetic response. Value and spatial workspaces
have fixed clone-local capacity. The much larger score and kinetic tapes are
allocated lazily for scalar calls; normal component-major crowd calls reuse one
lazy tape owned by the crowd ``ResourceCollection`` rather than one per walker.
Scratch is never shared by two simultaneously active crowds. Capacity growth
occurs at explicit preparation boundaries, and warmed kernel calls retain their
backing addresses.

The value/full-VGL/active-gradient batch boundary owns contiguous
configuration-major logical inputs and outputs. It partitions a logical batch
``B`` into bounded tiles ``T`` and stacks all configuration/electron rows (and,
for spatial modes, all value/gradient/Laplacian jet planes) into each shared
dense projection. Stable attention, orbital envelopes, determinant reduction,
and cusp terms remain configuration-local. Warming a prepared capacity makes
subsequent calls allocation-free. The default tile capacity is four and callers
can prepare a different positive capacity explicitly.

Expensive scratch is bounded by ``T`` rather than ``B``. Value execution owns
one contiguous tile arena. Full-VGL and active-gradient execution retain
separate, fixed-shape pools of ``T`` spatial slots because their derivative jet
layouts differ; both pools reuse one packed dense source/target arena. Switching
among modes therefore retains an additive high-water mark, exposed separately
from logical storage by ``logicalStorageBytes()`` and ``tileScratchBytes()``.
This is bounded execution storage, not a scalar-executor loop or one workspace
per logical configuration. Score, kinetic-response, and nonlocal virtual-move
reverse passes remain serialized through the resource-owned tape.

Zero warmed-call allocation is a contract measured inside the direct native
executors and their prepared batch workspaces, not across every public adapter
or Hamiltonian orchestration layer. In particular, current scalar spatial
adapters materialize owning result vectors at the public boundary, and the
nonlocal total-weight path retains the small vectors described above.

Selecting a subset of optimizable parameters limits registration and output
scatter, but the production score and kinetic executors still perform the full
canonical reverse pass and retain full-size reverse output. Tensor- or
block-pruned selected reverse execution remains deferred work.

The following environment variables are developer migration controls rather
than normal input parameters:

``PSIFORMER_VALUE_BACKEND``
  Select ``direct``, ``oracle``, or ``compare`` for value-only requests.

``PSIFORMER_SPATIAL_BACKEND``
  Select the backend for full VGL and active-electron gradients.

``PSIFORMER_SCORE_BACKEND``
  Select the backend for logarithmic parameter derivatives.

``PSIFORMER_KINETIC_BACKEND``
  Select the backend for the combined score and kinetic-energy parameter
  response.

The default for each variable is ``direct``. ``oracle`` retains the original
native graph/autodifferentiation evaluator for diagnosis. ``compare`` evaluates
both implementations and checks their outputs; it is intentionally much more
expensive and should not be used for production timing.

Boundary, scalar, and mass capability guards
--------------------------------------------

This implementation revision accepts fixed-ion, open-boundary molecular models
with real parameters, real internal arithmetic, and a real sign/log-magnitude
amplitude. A periodic target is rejected by the builder before the model is
evaluated, and explicit system validation rejects periodic electron or source-ion
particle sets. Source-ion gradient interfaces fail explicitly because force
evaluation and moving nuclei are not implemented. The execution-plan boundary
and the parameter, compute, and amplitude scalar domains are separate capability
axes so a later periodic or genuinely complex implementation can add kernels
without changing the public request contract. At present, requesting any
periodic or complex axis fails explicitly.

A complex QMCPACK build is supported only as an embedding of this real ansatz:
the returned phase is zero or pi and parameter/spatial derivatives have no
independent imaginary part. This does not constitute complex network weights,
Hermitian reverse products, twisted periodic geometry, or complex determinant
support. Score and fixed-inference calls remain available for that embedding;
kinetic-parameter response rejects a nonzero imaginary total wavefunction drift
instead of silently discarding it.

Wavefunction values, spatial derivatives, and log-parameter scores are mass
independent and may be used for fixed inference. PsiFormer kinetic-parameter
derivatives currently require every electron group to have mass one. The
``WaveFunctionComponent`` interface contracts electron contributions before
the legacy kinetic operator applies mass scaling, so unequal or nonunit masses
cannot be reconstructed correctly afterward. Optimization rejects such a
particle set instead of silently returning a mis-scaled response.

An exact same-spin coalescence is represented as sign zero, negative-infinite
log magnitude, and value zero in the value-only executor. Public particle-move
ratios therefore return zero at that node. Spatial, score, and kinetic
derivatives continue to reject nodes, where logarithmic derivatives are
undefined.

CPU crowd and BLAS threading
----------------------------

PsiFormer dense projections use the QMCPACK CPU BLAS backend, while crowd-level
parallelism is owned by QMCPACK. Avoid multiplying both sources of parallelism.
For the usual many-walker run, use one BLAS thread per crowd worker, for example:

.. code-block:: bash

  export OPENBLAS_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export BLIS_NUM_THREADS=1
  export VECLIB_MAXIMUM_THREADS=1
  export OMP_NUM_THREADS=<QMCPACK crowd thread count>
  export OMP_MAX_ACTIVE_LEVELS=1
  export OMP_PROC_BIND=spread
  export OMP_PLACES=cores

The 256-wide projections are generally too small to repay nested BLAS threading.
A serial, very small-crowd calculation may benchmark a threaded BLAS setting,
but it should record CPU affinity and all thread variables and confirm that no
outer parallel region is active. Component clones and crowd resources are the
unit of mutable ownership; invoking one clone concurrently from multiple host
threads is unsupported.

Developer timing manifests
--------------------------

When ``BUILD_MICRO_BENCHMARKS=ON``, ``benchmark_psiformer_modes`` loads a
developer-supplied export and writes one JSON manifest containing paired direct
and oracle timings for value, full VGL, active gradient, score, and
score-plus-kinetic modes. Value/VGL/active measurements sweep logical batch
size and tile capacity independently and report prepacked-execution and
pack-plus-execution scopes. Each batch record includes scalar-loop speedup,
logical and tile-scratch bytes, allocated tile capacity, tile occupancy, grouped
dense calls and rows, and scalar-executor call count. The manifest also records
output sizes, affinity, peak resident memory, and thread environment. The
executable is deliberately not a CTest and applies no fragile absolute timing
threshold. It records one untimed warmup result and all raw measured repeats for
each mode.

.. code-block:: bash

  taskset -c 0 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    ./src/QMCWaveFunctions/tests/benchmark_psiformer_modes \
    parameters.h5 electron_configurations.h5 \
    --repeats 10 --configuration-limit 10 --kinetic-configurations 2 \
    --batch-sizes 1,2,4,8,16 --tile-sizes 1,2,4,8 \
    --output psiformer_modes.json

The supplied HDF5 files are timing inputs only. Deterministic correctness tests
generate compact random-but-reproducible models at runtime and do not depend on
external checkpoints.

``benchmark_psiformer_ecp`` measures the production nonlocal-ECP crowd boundary
with a generated pseudo-LiH fixture. It times locality energy, T-move candidate
generation, and weighted parameter derivatives while sweeping diagnostic outer
tile capacities. The sweep uses a test-only seam and does not advertise a
production tile-size input. The JSON report separates logical job, knot, and
candidate counts; tile occupancy and split-job counters; derivative and bounded
weight capacities; and PsiFormer crowd-resource byte counts. It also records
parameter-version and sparse-work counters and labels reverse execution as
serialized through the crowd-owned score tape. Each requested mode is run with
an independent crowd so retained scratch from one mode cannot contaminate the
next mode's memory report. Two warmup calls are the default because the staged
operator/public job-list swap needs two calls before both sides are warm.

Build and run it from the Hamiltonian test binary directory so the configured
``Na.BFD.xml`` link is available:

.. code-block:: bash

  cmake --build build --target benchmark_psiformer_ecp
  cd build/src/QMCHamiltonians/tests
  taskset -c 0 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    ./benchmark_psiformer_ecp \
    --walkers 4 --outer-tile-sizes 3,12,256 \
    --modes energy,tmove,derivative --warmup-calls 2 --repeats 5 \
    --output psiformer_ecp.json

This benchmark uses ordinary scalar-relativistic locality/T-move ratios; DLA,
TMDLA, stochastic V1 move acceptance, spin-orbit ECPs, and force estimators are
outside its timing workload. It is deliberately not registered with CTest and
has no absolute timing threshold.

Validation coverage
-------------------

Deterministic tests generate their HDF5 fixtures at runtime and require no
external trained parameter file.  Fixture recipe version 1 uses fixed SplitMix64
parameter and geometry seeds; temporary file names are unique per process and
fixture instance.  Changing that recipe requires deliberately regenerating and
reviewing the independent JAX references rather than silently changing the
meaning of an existing test.

The permanent test matrix has the following tiers:

.. list-table:: PsiFormer validation tiers
   :header-rows: 1
   :widths: 22 48 30

   * - Tier
     - Required coverage
     - Data and gating policy
   * - Primitive and execution-plan
     - Geometry, determinant signed-log arithmetic, typed parameter layouts,
       workspace bounds, malformed input, and node behavior.
     - Generated values; ordinary deterministic CI.
   * - Evaluator contracts
     - Value, full VGL, active-electron gradient, score, and kinetic/local-energy
       parameter response against native, JAX, and finite-difference references.
     - Generated LiH and separated LiH pair; ordinary deterministic CI.
   * - Public component lifecycle
     - Builder, registration, reset, moves, buffers, cloning, persistence,
       allocation stability, crowd equivalence, and concurrent resources.
     - Generated models; ordinary deterministic CI in real and complex-adapter
       builds.
   * - Hamiltonian
     - Assembled all-electron kinetic and Coulomb local energy plus selected
       derivatives; pseudo-LiH local/nonlocal ECP values and weighted derivative
       reductions through scalar and multiwalker operators.
     - Generated models plus the small repository pseudopotential test asset;
       ordinary deterministic CI.
   * - Performance
     - Value, spatial, score, kinetic response, crowd batches, workspace bytes,
       allocation behavior, affinity, and raw warmed timings.
     - Non-gating developer/nightly runs; trained exports are optional timing
       inputs and are never required by correctness CI.

The named matrix can be selected without knowing which broad executable owns a
case:

.. code-block:: bash

  ctest --test-dir build -R psiformer --output-on-failure

For a source-build validation, build at least ``test_wavefunction_trialwf``,
``test_psiformer_native``, all direct-executor test targets,
``test_psiformer_determinant``, ``test_psiformer_adapter_sinks``,
``test_hamiltonian_ham``, and ``test_hamiltonian_coulomb``.  Repeat the focused
matrix in real and complex builds.  The complex build currently validates only
the documented real/open embedding and must not be reported as native complex
PsiFormer support.

Numerical, shape, version, allocation, and workspace invariants may gate CI.
Absolute wall-clock thresholds may not: performance manifests must preserve
warmup policy, every raw sample, build/compiler revision, precision, affinity,
thread environment, workload dimensions, workspace bytes, and peak resident
memory.  Performance comparisons are valid only between otherwise matched
runs on idle resources.
