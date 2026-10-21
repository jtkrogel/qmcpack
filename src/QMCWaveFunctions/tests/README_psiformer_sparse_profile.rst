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
