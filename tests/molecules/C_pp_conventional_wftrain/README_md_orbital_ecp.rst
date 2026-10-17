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
No moment-only ULP exception applies. Native round 93 passed at source
``9fcb356b9`` with app SHA256
``4ee23548d5a4218094d07621f8c8846f27bc4d6613a68e705b956e692e2e9ff7``:
full two-update training/inference/export, direct parameter_file VMC,
fresh import/re-export and strict exact initialized split restart. The
independent physical numerical gate passed separately (2,370 assertions).
Only this narrow profile is registered, on real full-precision CPU builds
without offload and with Python h5py/numpy. No convergence or timing claim
is intended. The original neutral-C single-Slater strict-reproducibility
failure remains separate and open.

The optional ``--joint-ci-orbital-ecp`` checker mode derives P17 from the
same XML by changing only CI optimization to yes. It requires active CI1
to change while frozen C0, original XML provenance, both original orbital
references and strict restart contracts remain intact. Its native runtime
qualification and registration are pending; the independent joint physical
gate passed separately (3,836 assertions). The fixed-P16 route is unchanged.
