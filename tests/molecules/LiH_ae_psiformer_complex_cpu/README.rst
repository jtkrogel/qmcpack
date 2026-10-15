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
