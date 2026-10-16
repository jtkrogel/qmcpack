Explicit paired GPU target VGL (initialized inference only)
==========================================================

Keep the existing paired all-electron periodic XML, ``gpu="cuda"``/``"hip"``,
``optimize="no"``, ``fp64_reference``, positive finite clone/shared GPU caps.
With no developer target controls, the lane remains the unchanged B1 FULL path.

Opt in with all three existing controls:

* ``PSIFORMER_GPU_VGL_TARGETS=K``,1<=K<=Ne;
* ``PSIFORMER_GPU_TARGET_WORKSPACE_BYTES`` bounds the COMPLETE spatial owner;
* ``PSIFORMER_GPU_TARGET_SCALAR_WORK_BUDGET`` bounds its COMPLETE physical sweep.

All three values must be positive finite decimal integers (not SIZE_MAX). The
shared ``max_gpu_numeric_bytes`` additionally admits exact VALUE+spatial workspace
sum before image/owner allocation. Image, vendor reserve, other crowds/ranks and
host allocations are NOT covered by this workspace cap. Canonical model/parameter
host initialization occurs outside device-owner admission; rejection tests count
DEVICE allocations, not every host allocation. Host numeric accounting
still includes both owners and fixed conversion staging, including the target
owner's additional canonical amplitude. Clones share one image/lane mutex and
do not allocate CPU neural tapes. Even one-electron gradient requests currently
run the complete physical target sweep; no cheap single-target cost is claimed.

Automatic selection is NOT supported here (unset/off only). Kinetic/lifted/
retained-coordinate/Metropolis/drift/Adam/resident-energy-Adam controls are
rejected, not silently ignored; they belong to separately qualified real paths.
The paired lane remains initialized-only inference: no trained checkpoint import,
GPU structured training, new optimizer, complex hidden trunk or large-Ne runtime
qualification. CPU components ignore these GPU-only controls.

The two tiny Gamma/nonzero XML checks reuse existing LiH paired fixtures and
matched CPU/CUDA sampling, comparing named nonclock scalar columns. They test
execution/parity, not convergence or throughput. Detailed target-tail/status/math
qualification remains the numerical owner's separate evidence.
CUDA runtime gates are distinct from HIP compile/link; no AMD runtime evidence
is implied by running these checks on a CUDA device.
