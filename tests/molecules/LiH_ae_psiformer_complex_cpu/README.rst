Explicit CPU periodic complex inference smoke
============================================

These three inputs retain fixed starting electron coordinates, physical Li3/H1
all-electron metadata and the actual periodic Coulomb operators. They exercise
legacy real Gamma, paired-complex Gamma and paired-complex nonzero outer twist.
The paired profile requires a 16 MiB numeric cap PER CLONE, not a process or
aggregate budget. Shared model, allocator/object overhead, ParticleSet and walker
records are excluded. Startup reports exact required/live bytes. No adaptive
100-300e, GPU, optimizer, ECP or convergence qualification is inferred.

The smoke checker only requires two finite production VMC blocks. It deliberately
does not compare independently evolved trajectories. The Native-free component
integration test compares ordinary Hamiltonian energy and complete complex
energy using one exact configuration and that traversal's actual Coulomb V.

Two negative inputs exercise actual driver selection, not just Hamiltonian
factory association. A multiple-wavefunction pool leaves one incomplete H
unassociated; another H explicitly names the other real wavefunction and omits
bare kinetic energy. Both explicitly select the complex wavefunction/H pair in
the driver and must fail its stateless selected-system check before sampling.

CPU learned-phase first-order gate
---------------------------------

The separate training checker generates short actual XML from these fixed-position
inputs: Gamma Adam and nonzero-twist plainSGD, with scalar optimize=no unchanged.
Both use the same-sample complete complex Coulomb energy and PATHWISE gradient;
no SR/JVP/metric/line-search/ECP/GPU training support is inferred. A mandatory
positive finite maximum_retained_bytes admits exact named derivative/config/batch/
optimizer storage plus a documented conservative first-order scratch allowance
BEFORE those owners allocate. It is not a process-peak/RSS or adaptive large-Ne cap.

The Python checker requires numpy+h5py, verifies actual imaginary-head canonical
updates, and restores checkpoints. Adam additionally compares exact continuation
with an uninterrupted two-update run and rejects wrong twist, budgetT-1 and an
unsupported optimizer. Stable physical checkpoint identity excludes parameter
values/version and pointer addresses. The Native-free complex_training unit case
checks the complete normalized fixed-sample reweighted objective by finite
differences, with actual periodic Coulomb H, unequal masses and nonzero outer k.
