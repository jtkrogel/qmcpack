Fixed-CI MD orbital ECP native fixture
====================================

This tiny real CPU, open-boundary fixture composes two distinct inline
four-MO rotation owners, fixed two-DETS coefficients and four J2 variables
with the tracked carbon pseudopotential. The actual active layout is P16:
six local orbital pairs per spin and four J2 coefficients. It does not
qualify active CI, CSF, external orbitals, complex/GPU/periodic models,
arbitrary determinant sizes, DLA or a general nodal admission policy.

The shared ``check_sj_training.py --md-orbital-ecp`` route prepares one
bounded Adam round trip, scheduled inference, complete two-owner frames,
direct parameter_file VMC/model reuse and strict initialized split restart.
No moment-only ULP exception applies. Its runtime qualification is pending;
the independent physical numerical gate passed separately. No convergence
or timing claim is intended, and the original neutral-C single-Slater
strict-reproducibility failure remains separate and open.
