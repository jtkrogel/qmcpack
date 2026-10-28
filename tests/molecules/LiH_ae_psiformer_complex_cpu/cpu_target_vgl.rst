Explicit shared CPU complex target VGL
======================================

The existing periodic paired-complex FP64 CPU component can assemble a complete
spatial gradient and logarithmic/ratio Laplacian using one bounded target lane.
This changes derivative storage and execution; it preserves the physical model,
parameter conventions and all-particle or n-electron Monte Carlo moves.

Opt in on the existing complex CPU psiformer element::

  gpu="no"
  max_clone_numeric_bytes="16777216"
  max_cpu_shared_numeric_bytes="137438953472"
  cpu_vgl_targets="8"

The target count must be a positive decimal integer at most Ne. Zero, malformed
counts, SIZE_MAX, insufficient shared admission and an incompatible GPU request
are rejected. The shared CPU cap is mandatory for this selector. Omitting
cpu_vgl_targets keeps the existing FULL mode; this is an explicit choice, not an
automatic large-system policy. Derivative tiles do not define MC move groups.

One selected lane retains all physical rows and matrices but only K target
derivative planes, plus complete O(Ne) result arrays. A smaller final tile reuses
the same allocation. Each configured crowd has its existing exclusive reusable
CPU lane; scalar fallback and dormant lanes remain charged whenever they retain
buffers. Model isolation preserves the selected target capacity.

Memory accounting
-----------------

Startup reports the FULL oracle requirement and selected streamed requirement
before workspace allocation. The shared family cap admits actual lazy VALUE,
FULL, ACTIVE, STREAMED and collective proposal storage. A streamed requirement
already includes its private target owner and complete result arrays. Borrowed
views allocate no storage. Mode capacities and physical retained mode bytes are
separate: a cold lane can exist without having allocated its numerical bank.

The cap does not cover shared parameters/model metadata, ordinary ParticleSet or
walker buffers, derivative tapes, retained samples, optimizer state or allocator
and object overhead. Those owners retain their existing independent admission.
The clone cap covers component-owned G/L in shared mode; it is not a process RSS
limit. Existing memory tracing reports allocation and release of retained banks,
including dormant ones. Compare complete-workload RSS and time as well as the
spatial owner; reducing a spatial graph need not dominate total training memory.

The first CPU implementation repeats forward values for each derivative target
tile. It does not claim new factor caching, cross-request reuse or a universal
speedup. FULL remains the small-system oracle and an available execution choice.

Manual component probe
----------------------

Build the optional diagnostic target::

  cmake --build BUILD --target psiformer_complex_cpu_probe
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    BUILD/src/QMCWaveFunctions/tests/psiformer_complex_cpu_probe \
    fixture.h5 8 2 compare 137438953472

The positional arguments are fixture, targets, repetitions, operation and physical
owner byte cap. Operation compare evaluates FULL and streamed complete products,
reports absolute/scaled errors and timings, checks finite complex gradients and
both Laplacian forms, and verifies allocation identity. It reports kinetic energy
with the supplied outer twist. The same immutable model and coordinates are used
for each owner. Numerical comparison reference arrays are O(Ne) and reported;
FULL is released before allocating the streamed bank. Consequently the reported
process peak includes the FULL run and is not a streamed-only process peak.
Per-owner RSS snapshots and exact capacity bytes remain separate fields.

Operation memory constructs and releases each owner, checking actual capacities
against its estimate, without evaluating spatial products. Operation plan reports
storage estimates without allocating either derivative owner. Neither establishes
large-system numerical or runtime qualification. The probe has no statistical
trajectory and is not registered as an automatic timing test.

The fixture uses the existing native HDF5 configuration fields:

* electron_positions: [1,Ne,3], nuclear_positions: [Nion,3],
  nuclear_charges: [Nion];
* integer attributes n_up, n_down and n_determinants=16;
* lattice: [3,3] with row lattice vectors; cartesian_twist: [3].

The probe uses the existing paired periodic initializer with seed17, D16, width256,
4heads and4blocks. Input coordinates and effective ionic charges are explicit.
It reports persistent model double-array bytes separately from derivative owners;
RSS additionally includes fixed metadata, libraries and allocator overhead.
The physical kinetic value is a component observation, not a complete ECP energy.
Complete periodic ECP observations require the ordinary native Hamiltonian path.

Qualification bounds
--------------------

The CPU component tests compare full complex values/G/log-Laplacians and kinetic
products, including unequal3/2spin populations with tile2 crossing the spin
boundary and a final tail. They cover strict XML admission, caps, clones,
publication/isolation, all/n-electron proposals, occupied phases, Fourier
envelopes, unchanged score conventions and failure invalidation.

Native periodic Li/H valence1 ECP qualification used N20/W8/C1, nonzero twist,
all-particle moves and one score-only Adam update. FULL and target8 produced
bitwise-identical numeric model, optimizer and sampler state. Single-core N20
and N50 component controls additionally compared complete VGL exactly. Sparse
N100, N200 and N300 actual owner-allocation checks establish only their reported
memory inventory; no large-system determinants or derivatives were evaluated. No larger-system convergence or universal performance claim follows.

CPU mixed precision, GPU changes, forces, Hessian outputs, spinors, T-moves and
new optimizer methods are outside this CPU execution slice.
