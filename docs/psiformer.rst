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

Native streamed optimization
----------------------------

The batched ``wf_train`` driver accepts ``optimizer="sgd"``, ``momentum_sgd``,
``rmsprop``, ``adam``, ``amsgrad``, ``sr``, ``minsr``, ``spring``, ``pii``,
``minpii``, ``lm``, ``alm``, and ``kfac`` in its ``training`` child. Parameters
remain real, even in a complex QMCPACK build. This route is distinct from the
legacy optimization drivers. Initialize PsiFormer internally or load a compatible
parameter export, set ``optimize="yes"``, and choose the active parameter scope.

For example, two bounded accelerated linear-method iterations can be requested with:

.. code-block:: xml

  <qmc method="wf_train" move="pbyp" checkpoint="-1">
    <training objective="energy" optimizer="alm">
      <parameter name="iterations">2</parameter>
      <parameter name="warmupsteps">0</parameter>
      <parameter name="learning_rate">1</parameter>
      <parameter name="damping">0.01</parameter>
      <parameter name="lm_overlap_shift">0.0001</parameter>
      <parameter name="lm_maximum_subspace_dimension">8</parameter>
      <parameter name="lm_inner_iterations">8</parameter>
      <parameter name="lm_inner_restart_length">4</parameter>
      <parameter name="lm_inner_relative_tolerance">0.1</parameter>
      <parameter name="krylov_relative_tolerance">0.001</parameter>
      <parameter name="krylov_maximum_iterations">30</parameter>
      <parameter name="krylov_maximum_operator_applications">100</parameter>
      <parameter name="maximum_update_norm">0.01</parameter>
    </training>
    <parameter name="walkers_per_rank">16</parameter>
    <parameter name="crowds">1</parameter>
    <parameter name="blocks">1</parameter>
    <parameter name="steps">1</parameter>
    <parameter name="substeps">1</parameter>
    <parameter name="timestep">0.01</parameter>
    <parameter name="usedrift">no</parameter>
  </qmc>

Only ``objective`` and ``optimizer`` are training attributes; all other training
controls are named ``parameter`` children. Obsolete scalar attributes are rejected
with migration guidance rather than silently changing an existing run's meaning.
``iterations`` defaults to one and counts optimizer-update attempts independently
of statistical blocks. At fixed parameters, each iteration collects
``walkers * blocks * steps`` configurations, then computes streamed derivative
reductions and applies one update. ``steps`` defaults to one when omitted.
``substeps`` counts MC advances between retained configurations: an all-particle
move or a particle-by-particle sweep, according to the move mode. Intermediate
substeps are not retained optimization samples.

The ``qmc`` warmup is initial equilibration. The separate ``training``
``warmupsteps`` defaults to zero and discards additional advances before every
iteration, including the first; both warmups honor ``substeps``. Population
positions continue across iterations. Statistical scalar blocks describe sampling
at the current parameter version, not optimizer-update boundaries. Matrix-free
solvers revisit the retained samples without storing a samples-by-parameters
matrix. Line-search endpoint warmup and sampling remain separate controls.
Repeated configurations from the same walker share a chain identity in the
line-search uncertainty calculation; importance-weight ESS alone does not
measure serial independence.

Retained snapshots keep distinct configurations, energies and ECP replay data,
but share the large score/kinetic workspaces and parameter-channel buffers.
An exclusive lease protects the current single execution lane per model; this
is memory sharing, not an assertion of threaded derivative scaling. Numeric
diagnostics count shared scratch once. Sample-local ECP responses and optimizer
vectors remain separately owned. Increasing blocks/steps still increases sample
storage and derivative work. ``samples`` is rejected: specify walkers, blocks
and steps instead.

``maximum_retained_bytes`` limits accounted numeric storage, including copied
position/gradient/Laplacian arrays. It is not a process-RSS ceiling. The optional
training parameter ``maximum_resident_bytes`` (default zero, disabled) instead
checks measured per-rank resident memory at initialization, sample collection,
derivative preparation and update boundaries. On Linux it includes opaque
ParticleSet/teacher caches, model storage and allocator overhead. It is a
boundary check, not an allocator or transient-peak limit; a requested guard
fails explicitly if RSS cannot be measured. MPI ranks each apply their own limit.

Every training section writes ``<project>.sNNN.training_info.csv`` with one row per
optimizer attempt, independently of scalar statistical blocks. It records the
sampled/published parameter versions, sample count, energy/variance/gradient
norm or overlap loss, selected alpha, committed flag, sampling/full-iteration
wall time, retained numeric bytes, rank-zero RSS and solver diagnostics.
Full-iteration timing excludes the subsequent checkpoint write. Restart appends
to an existing history for the same output root, or writes a fresh header when
the output root changes. Iteration indices start at zero and count complete outer
optimization attempts, including rejected updates. The training ``iterations``
parameter is the total requested count, including attempts restored from a
checkpoint. ``sample_version`` and ``parameter_version`` identify process-local
provider generations; restart republishes the saved source generation into the
new process and may rebase those version numbers. Outer iteration counts and
optimizer update/bias-correction counts retain their cumulative saved values.
Scalar statistics use one file per iteration in
``R.training/R.i0000000.scalar.dat``, where ``R`` is the series-qualified project
root. Each contains one row per sampling block at the starting parameter version.
Training writes no top-level scalar file or statistical HDF5 file. Supplemental
global/local scalar and operator estimator requests are ignored with a warning
before estimator construction. Mandatory energy/variance and basic sampling
diagnostics remain active, with all physical Hamiltonian terms included.
Restarting an older checkpoint under a root with conflicting future output is
rejected before sampling; select a new project root to branch that checkpoint.

To suppress the potentially enormous startup parameter table while keeping
other diagnostics, use ``<wavefunction ... print_parameters="no">``. The default
is ``yes`` for compatibility; ``no`` prints the parameter count only.

Optional interval inference evaluates the accepted parameters after a complete
optimizer attempt using the native VMC sampler in a separate population. Add a
sibling of ``training`` inside ``qmc``::

  <inference interval="1000" last="no" move="pbyp">
    <parameter name="walkers_per_rank">8</parameter>
    <parameter name="crowds">1</parameter>
    <parameter name="warmupsteps">10</parameter>
    <parameter name="blocks">4</parameter>
    <parameter name="steps">10</parameter>
    <parameter name="substeps">1</parameter>
    <parameter name="timestep">0.01</parameter>
    <parameter name="usedrift">yes</parameter>
  </inference>

Absent ``inference`` disables this work and its output. ``interval`` is a positive
outer-attempt count (default 1000): interval 2 runs after indices 1, 3, and so on,
including attempts whose update was rejected. ``last="yes"`` additionally runs on
the final completed attempt unless already due; its default is ``no``. This
cadence is independent of checkpoint cadence. Sampling parameters and move modes
have the same meanings and defaults as native VMC; inference uses its own walker
and crowd controls. Fresh ParticleSet, wavefunction/Hamiltonian clones and RNGs
isolate inference from subsequent training walkers, optimizer state and random
draws. Seeds derive deterministically from a read-only copy of the initial run
RNG state, cumulative attempt index, rank and an inference domain.

Inference writes only ``R.inference/R.i0000001.scalar.dat``, with regular VMC
blocks and a comment recording the attempt index, accepted ``parameter_version``,
completed/requested blocks and completion status. Supplemental estimators are
ignored with a warning; every physical Hamiltonian term remains present.
No training-history row or iteration increment is added for an inference block.
Inference timing is logged separately and recorded by the medium-level native
``WFTrain::inference`` timer; the existing history's iteration timing excludes
subsequent inference and checkpoint work, while the runtime budget includes both.
A runtime/STOP request can skip inference or leave an explicitly incomplete
scalar file; the completed training attempt is still checkpointed. Exceptions
prevent checkpoint/latest publication. Inference files are not restart inputs,
and existing future inference files follow the training collision policy.

Exact training checkpoints are taken only after a complete iteration. The training
parameter ``checkpoint_interval`` (positive integer, default 1000) counts outer
attempts independently of the qmc block checkpoint period. Thus interval 1000
writes after indices 999, 1999, and so on. Completion and supported graceful
runtime stops also write a checkpoint. Each export contains
``R.training/R.i0000999.parameters.h5`` for direct PsiFormer input and a restart
bundle ``R.training/R.i0000999.checkpoint/`` with manifest, global state, and rank
sampler files.
``R.last_checkpoint.h5`` selects the latest completely published export. Restart
accepts an archived bundle directory, its manifest, or this latest pointer; it
does not require the separate parameter export.
Ordinary-locality ECP training now persists each walker's accepted rotated
quadrature grids alongside walker and RNG state. Restart validates the grid
layout and finite unit-sphere coordinates, restores them transactionally, then
reconstructs caches deterministically without consuming new random rotations.
The sampler contributor uses payload version two for ECP state; legacy
all-electron version-one payloads remain supported. Envelope and contributor
format versions are separate. Physical-system/Hamiltonian inputs, rank/crowd
topology and sampling controls must still match. A short uninterrupted-versus-
restarted regression compares parameters, optimizer, sampler and grids exactly.

To migrate the earlier final-population behavior, move the old ``blocks`` count
to training ``iterations``, set sampling ``blocks=1`` and ``steps=1``, and multiply
the old ``substeps`` by the old ``steps``. This preserves production chain
advancement and update sample counts as closely as possible, but changes scalar
statistics by omitting intermediate energy measurements. Initial warmup may
also advance farther if ``substeps`` is increased.

The numerical choices above illustrate input syntax, not recommended production
hyperparameters. Optimizer-specific parameters supplied to another method are
rejected. All routes use the common publication, cache refresh, proposal scaling,
typed weight decay, and optional line-search machinery; failed eigensolves do not
publish parameters.

First-order methods
~~~~~~~~~~~~~~~~~~~

``learning_rate`` defaults to 0.001. Momentum SGD adds ``momentum_decay`` (0.9);
RMSprop adds ``rms_decay`` (0.9) and ``epsilon`` (1e-8). Adam and AMSGrad accept
``adam_beta1`` (0.9), ``adam_beta2`` (0.999), and ``epsilon``. AMSGrad keeps the
ordinary second-moment EMA and its elementwise running maximum separately, using
the bias-corrected maximum in the denominator. Its maximum vector is checkpointed
alongside the moments and accepted update count. Rejected proposals do not advance
these recurrence states. All first-order methods can also optimize the existing
``determinant_overlap`` objective.

Score residual policy
~~~~~~~~~~~~~~~~~~~~~

``gradient_estimator=score_covariance`` selects the physical energy gradient from
score products without local-energy parameter responses. Its first-order default
is the raw covariance residual. ``score_residual_policy=raw`` makes that choice
explicit; ``score_residual_policy=soft_squeeze`` opts into the existing exact
median/linear-quantile transform with logarithmic tails. The soft policy accepts
``score_residual_width`` (positive, default 1) and ``score_residual_quantile``
(in ``(0,1]``, default 0.95). Those controls require the soft policy declaration.

The transform is prepared once from the full retained energy population across
crowds and MPI ranks. Its scalar statistics are computed on the host; device
score products consume the same coefficients. The transform changes the real
energy component and preserves the imaginary component. Transformed energies
are centered by their population mean for the score covariance; reported energies
and variances remain raw. The current soft policy requires unit sample weights.

These controls require an ENERGY score estimator, a first-order optimizer or KFAC,
and ``line_search=off``. They do not select a derivative family implicitly. KFAC
may retain its explicit legacy score-gradient convention and update scaling.
Compatible KFAC clipping aliases resolve to the same scalar policy; contradictory
objectives, hard windows, exclusion, widths or quantiles are rejected. Curvature,
damping, optimizer state and inverse cadence retain their separate controls.
Raw first-order checkpoint identities remain compatible; selecting a soft residual
changes the checkpoint configuration identity.

For the real/open ECP GPU profile, explicit ``gradient_estimator=score_covariance``
plus ``PSIFORMER_GPU_ADAM_BYTES`` and ``PSIFORMER_GPU_RESIDENT_ENERGY_ADAM_BYTES``
selects an optional ordered EO/O Adam update bridge. It requires one rank, crowd,
block and step, at least seven walkers, FP64, and the fixed unmodified Adam step.
Raw and soft residuals share the coefficients above. The second budget admits only
the two score-contraction arrays and norm scratch; model, score tapes, sampling,
coefficients, vendor storage and checkpoints require separate accounting.

This bridge retains HOST-controlled sampling, physical local energy, scalar
statistics and parameter publication. Requested DEVICE parameter authority is
rejected until physical PP observations are implemented. Periodic and complex
model heads require separate implementations. The legacy full-response T/EO/O
bridge retains its existing contract and checkpoint identity.

Sampled linear methods
~~~~~~~~~~~~~~~~~~~~~~

``lm`` preserves the existing symmetrized generalized eigensolver. ``alm`` is a
separate nonsymmetric generalized Jacobi--Davidson route: it retains the raw
finite-sample Hamiltonian orientation, chooses a real reference-connected Ritz
root near the initial energy, and uses bounded restarted GMRES corrections.
Both stream score and local-energy derivative products, avoiding dense
sample-by-parameter and parameter-squared matrices. ECP products use the accepted
quadrature replay shared by Hamiltonian JVP and VJP actions.

Both default to ``learning_rate=1``, ``krylov_relative_tolerance=0.001``,
``lm_overlap_shift=0.0001``, ``lm_maximum_subspace_dimension=8``, and
``lm_normalization_xi=0.5``. ``damping`` shifts only the Hamiltonian tangent block;
``lm_overlap_shift`` shifts only the overlap tangent block. Both must be positive.
All active PsiFormer parameters are treated as nonlinear in the sampled-metric
normalization; ``lm_normalization_xi=1`` disables that correction. ALM alone accepts
``lm_inner_iterations=8``, ``lm_inner_restart_length=4``, and
``lm_inner_relative_tolerance=0.1``. Inner and outer actions share the configured
operator-application and wall-time bounds. Small projected matrices scale with
the configured subspace bounds, not the number of parameters.

Live KFAC
~~~~~~~~~

``kfac`` observes affine activations and score sensitivities through the selected
provider, reduces their factors across ranks, and applies damped Kronecker
inverses to the sampled energy gradient. Trainable parameters not belonging to
an observed affine block use a diagonal fallback, controlled by
``kfac_fallback_damping`` (0.001). ``damping`` (0.001) stabilizes the affine factors.
By default the live route rebuilds factors each training iteration and retains
no factor EMA or momentum history. Providers without affine observations fail
explicitly. No Krylov controls are accepted because this update uses direct
factor solves rather than an iterative solve.

The following optional training parameters select the objective and curvature
conventions independently. Their defaults retain the original live update:

* ``kfac_objective=raw_covariance`` uses the weighted raw-energy score covariance
  with its factor of two. ``clipped_covariance`` instead contracts raw scores with
  ``mask * w * (T(E)-mean(w*T(E))) / count(mask)``, where weights have mean one.
  The transformed mean includes all samples before masking, and this gradient has
  no factor of two. Physical energy and variance reporting always use raw energies.
* ``kfac_clip_mode=soft_median`` (default) uses the median energy center and a
  logarithmic tail transform. ``kfac_clip_quantile=0.95`` selects a linearly
  interpolated quantile of absolute residuals, and ``kfac_clip_width=1`` multiplies
  that scale. ``hard_median`` and ``hard_mean`` use their named center and clamp
  energies to center plus/minus width times the mean absolute deviation.
* ``kfac_exclude_width=none`` disables masking. A positive value keeps samples
  whose absolute centered residual divided by the unmultiplied quantile scale is
  strictly below that value in soft mode. In hard mode it is a strict absolute
  energy-residual bound. Clipping controls require ``clipped_covariance``;
  ``kfac_clip_quantile`` is accepted only in soft mode. Nonfinite transforms,
  zero soft scales and all-masked populations fail before parameter publication.
* ``kfac_curvature=native_score`` retains samplewise score factors.
  ``registered_gaussian`` selects unweighted normal-predictive curvature with
  variance 0.5: affine sensitivities and envelope diagonals gain a factor of two,
  while typed generic cusp entries use ``2*(sum(score))^2/N``. Objective weights
  and clipping do not enter this curvature. MPI ranks combine signed generic
  sums before squaring, corresponding to one global reference device.
* ``kfac_damping_policy=equal_split`` retains equal square-root shifts.
  ``trace_balanced`` balances two nonscalar factors by their normalized traces;
  scalar factors are absorbed before adding damping and a zero factor yields
  the inverse of the damped identity.
* ``kfac_constraint=none`` retains the unconstrained preconditioned update.
  ``damped_kfac`` requires a positive ``kfac_norm_constraint`` and scales the
  inverse direction by ``min(1,sqrt(bound/(learning_rate^2*g.dot(P(g)))))``.
  This damped approximate-curvature surrogate is distinct from
  ``maximum_update_norm``, which independently caps Euclidean displacement.

All these controls are KFAC-only. Resolved controls, transformed population
statistics and constraint metric/scale are logged explicitly. A changed profile
is bound into the stage checkpoint configuration identity; it cannot silently
replace the profile of an exact restart.

``kfac_persist_state=yes`` opts into carrying optimizer state both across updates
and through native checkpoints; it is not an output-cadence setting. Its default
``no`` preserves stateless checkpoint compatibility. The following recurrence
controls require the explicit opt-in:

* ``kfac_ema_decay=0`` retains only the current factors. A value in ``[0,1)``
  maintains normalized averages with ``W'=decay*W+1`` and
  ``F'=(decay*W*F+F_batch)/W'``. The first batch is used without shrinkage.
* ``kfac_inverse_period=1`` refreshes affine inverses at pre-update optimizer
  steps divisible by that positive period, including step zero. Fallback diagonal
  inverses use the current averaged diagonal and damping on every update.
* ``kfac_learning_rate_decay=0`` keeps ``learning_rate`` constant. A positive
  value ``d`` selects ``learning_rate/(1+t/d)`` using accepted optimizer step
  ``t`` before the update. ``kfac_schedule_precision=float64`` is the default;
  ``float32`` rounds the inverse schedule and its squared-rate constraint factor
  in that dtype before promotion into the parameter/factor calculations.
* The admitted profile has constant ``damping``/``kfac_fallback_damping``,
  ``kfac_momentum=0``, ``kfac_burn_in=0`` and ``kfac_curvature_period=1``.
  Other values of these last three controls are rejected explicitly.

Native checkpoints preserve normalized factors and their weight, exact prepared
affine inverse caches, fallback inverses, accepted-step/sample counters and cache
step. Failed publication advances none of this state. Restore validates the
configuration, shapes, finite values and state fingerprint before publication;
the native sampler/RNG topology contract continues to apply. A parameter-only
import starts a fresh optimizer at step zero even when its historical filename
contains a nonzero training label. Exact continued-state restarts require a
checkpoint containing the matching optimizer contributor; missing state is not
silently initialized.

Qualification includes deterministic algebra/recurrence/input tests and very
short internally initialized H2 execution and AMSGrad restart tests. These checks
establish execution and numerical contracts, not optimization convergence or
production performance. Trained LiH optimizer comparisons remain a separate
qualification step.

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
