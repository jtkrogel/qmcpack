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

Scalar-relativistic nonlocal pseudopotentials are supported. For every virtual
quadrature move, PsiFormer evaluates

.. math::

   R_q = \frac{\Psi(\mathbf R_q)}{\Psi(\mathbf R)}

and contributes the logarithmic score change

.. math::

   O_{q,\theta} - O_\theta =
   \frac{\partial \log|\Psi(\mathbf R_q)|}{\partial\theta}
   -
   \frac{\partial \log|\Psi(\mathbf R)|}{\partial\theta}.

The nonlocal operator multiplies this by its quadrature weight and ``R_q``
exactly once. Value ratios use a prepared, allocation-free direct configuration
batch workspace. The weighted derivative path reduces directly into the
caller's selected-parameter destination and does not materialize a
knot-by-parameter matrix. The surrounding nonlocal total-weight orchestration
still creates small per-call vectors. Virtual score reverses are currently
serialized through one resource-owned tape; this bounds crowd memory but is not
a grouped reverse kernel.

Spinor electrons and spin-orbit pseudopotential ratios are unsupported and
fail during validation. The model is real-valued; it can be represented in a
complex QMCPACK build with phase zero or pi, but is not a general complex
neural ansatz.

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

The current value/full-VGL/active-gradient batch boundary owns contiguous
configuration-major outputs and performs allocation-free warmed evaluations,
but its inner implementation loops over scalar direct workspaces. It must not be
interpreted as a grouped or strided-batched GEMM throughput claim. Score,
kinetic-response, and nonlocal virtual-move reverse passes are likewise
serialized through the resource-owned tape at present.

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

Developer timing manifest
-------------------------

When ``BUILD_MICRO_BENCHMARKS=ON``, ``benchmark_psiformer_modes`` loads a
developer-supplied export and writes one JSON manifest containing paired direct
and oracle timings for value, full VGL, active gradient, score, and
score-plus-kinetic modes. It also records supported value/VGL/active crowd batch
sizes, fixed workspace bytes, output sizes, affinity, peak resident memory, and
thread environment. The executable is deliberately not a CTest and applies no
fragile absolute timing threshold. It records one untimed warmup result and all
raw measured repeats for each mode.

.. code-block:: bash

  taskset -c 0 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    ./src/QMCWaveFunctions/tests/benchmark_psiformer_modes \
    parameters.h5 electron_configurations.h5 \
    --repeats 10 --configuration-limit 10 --kinetic-configurations 2 \
    --batch-sizes 1,2,4 --output psiformer_modes.json

The supplied HDF5 files are timing inputs only. Deterministic correctness tests
generate compact random-but-reproducible models at runtime and do not depend on
external checkpoints.

Validation coverage
-------------------

Deterministic tests generate their HDF5 fixtures at runtime and require no
external trained parameter file. They cover JAX and finite-difference native
observables; selected/full registration, reset, clone sharing and restart;
virtual ratios and score differences; a generated pseudo-LiH nonlocal energy
derivative and selected update; system mismatch rejection; streaming/stored
allocation behavior; and duplicate names or stale/inconsistent state. The same
public real/open component tests compile and run in real and complex QMCPACK
builds. Additional guards cover periodic targets, nonunit/unequal optimization
masses, both source-gradient force interfaces, exact-node ratios, complex total
drift, spinors, and spin-orbit calls. Crowd tests run with one- and two-thread
outer layouts while forcing single-threaded BLAS, and low-level executor tests
check warmed storage stability without imposing wall-clock pass/fail limits.
