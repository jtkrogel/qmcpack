Source-specific PsiFormer release recipes
================================================

This is a small consumer of Task35/36/45 evidence, not a new workflow system or
a release of every current-HEAD profile. ``qualified_profiles.json`` names the
actual source, binary, inputs and evidence for each observed positive. Historical
Task44 NOT_STARTED notes predate this recipe delivery; production calibration
is still pending. These recipes are derived inputs, not newly executed timing
or convergence studies. Existing defaults are unchanged.
The GPU XML/launcher additionally passed one finite weight-two VMC block on
the recorded binary; the CPU launcher is dry-run checked, without repeating
the qualified distributed experiment. Evidence paths refer to the separately
maintained ``06_intensive_impl_tasks`` archive, not a bundled binary release.

* SINGLE_GPU: actual Fiji real/open FP64, four-electron LiH, one crowd and true
  B2 inference with explicit K1/caps; scalar/parallel parity and exact native
  split restart passed. This is neural GPU, not conventional GPU.
* CPU_MULTINODE: actual BESMS CPU-complex R4 orbital-MSE, four WORLD ranks,
  two independent groups spanning two nodes, matched-slot isolation and exact
  state passed. It is not GPU MPI or a general complex ENERGY claim.
* HIP_COMPILE: prior-source compile/link only. Neither AMD execution nor a HIP
  compile of this new batched-parallel snapshot is claimed.
* PRODUCTION_PENDING: AMD runtime, multiple GPUs/device-aware MPI, 40--80+ GiB
  calibration and combined CUDA/OpenMP-target toolchain remain separate gates.

Validate without a device or scheduler::

  python3 launch_recipe.py --check

The default recipe mode hashes and validates, then prints JSON with the exact
command, controlled environment and QUALIFIED_PIN or NOT_QUALIFIED. It neither
creates output directories nor launches anything. Execution requires the exact
qualified binary and protected inputs; mismatches are refused before outputs.
Do not replace a pin merely to make an unrelated binary/model pass.

GPU recipe (dry-run)::

  python3 launch_recipe.py --profile SINGLE_GPU --app /path/to/pinned/qmcpack \
    --model /path/to/tiny-real-lih.h5 --output /new/output/directory

The protected P666 model is not distributed. It must have the recorded digest,
real/open AE metadata, up/down=2/2, D2/width8/heads2/blocks1, Li charge3 at the
origin and H charge1 at (3.0154361724853516,0,0). The XML is a complete ordinary
VMC step with two walkers, one crowd, no drift and tau=1e-8. Its small timestep
is a correctness control, not an efficiency recommendation. The 2 GiB device
cap includes a 256 MiB external reserve; it is not a whole-process memory bound.

CPU recipe (dry-run)::

  python3 launch_recipe.py --profile CPU_MULTINODE --app /path/to/pinned/qmcpack \
    --input-list /path/to/prepared/mixed/inputs.txt --allocation EXISTING_ID \
    --runtime-path-file /path/to/qualified-library-path.txt --output /new/output/directory

Reuse the existing prepared A/B tree from
``tests/molecules/H2_ae_psiformer_wftrain/check_complex_cpu_ensemble.py`` or its
archived matching inputs. The list and both XML files must match the snapshot
hashes; no teacher/model generator is duplicated here. The existing allocation
must contain BESMS c555/c556. The recipe only issues a bounded ``srun --jobid``;
it never allocates, builds or changes jobs. Verify allocation state and runtime
dependencies separately before explicit execution.

Append ``--execute`` only when authorized to run that exact profile. A new
output directory receives the input and ``launch.json`` before the launcher
becomes the bounded timeout command. All inherited PSIFORMER controls are
cleared; the CPU runtime library path is assigned exactly, not appended. Record
the actual exit, native CSV/scalars and existing checker results afterward:
QUALIFIED_PIN means input provenance, not that a new run passed. No sweep,
autotuning, speedup or universal memory/default policy is supplied.
