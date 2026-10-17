Diamond fixed-spline conventional CPU consumer
==============================================

The qualified shared-checker ``--periodic-spline`` mode derives a task-owned
training XML from qmc_short_vmcbatch.in.xml; it does not duplicate or edit the
tracked ES-HDF or C.BFD.xml. It preserves Ne8 (4/4), C2, lattice, Gamma twist,
shared four-MO double CPU spline and all 24 original J1/J2 coefficients.
It explicitly uses Ewald30 with unchanged LR_tol, ordinary rule4/twelve-point
locality ECP/local1/noDLA, and removes only auxiliary nonphysical MPC/Flux.
The original diamond input instead uses default Esler15; these accuracy
policies must not be conflated.

The existing tiny H2 qmc controls supply Adam LR1e-4/two updates, W2/crowd1/
blocks2/steps1/substeps1/warmup0/tau1e-4/no drift and scheduled inference.
The seed is 1931 and the original random initializer is retained without search.
The existing generic four-sample workflow checks actual source-image/effective
table identity, all three built Ewald30 records, active J1/J2 changes, exported
physical VMC reuse, fresh import/re-export and strict exact initialized split
restart. One tightened cutoff32 import/restart negative preserves accuracy.

Independent direct-adapter physical/cache proof passed 335 assertions at
source ``232a533ae``, separately from Binding/native admission. Native round117
passed at application source ``2b8f88a99`` with checker-only correction
``86b8347d9``: all positive workflows, strict initialized restart and identity
negatives passed. CTest ``short-diamondC-conventional-wftrain-spline`` registers
this one profile with a 300-second deadline. CPU real/full
precision only; no complex, offload, orbital rotation, MD or other loader claim.
