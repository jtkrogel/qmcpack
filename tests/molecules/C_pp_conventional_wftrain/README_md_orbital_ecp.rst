Fixed-CI MD orbital ECP native fixture
====================================

This tiny real CPU, open-boundary fixture composes two distinct inline
four-MO rotation owners, fixed two-DETS coefficients and four J2 variables
with the tracked carbon pseudopotential. The actual active layout is P16:
six local orbital pairs per spin and four J2 coefficients. It does not
qualify other CI layouts, CSF, external orbitals, complex/GPU/periodic models,
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
references and strict restart contracts remain intact. Native round 96
passed on its first attempt with app source ``91460fd03`` and SHA256
``f5110041e5f3d059d83a10870c8d26239f62636b8292839107f0594ce0977eac``;
the helper-only source is ``c33d9e566`` (checker SHA256
``691611cce59367ff4c1c8420029cf4d5e09814688f1682723440db087511893c``).
The independent joint physical gate passed separately (3,836 assertions).
One joint mode is registered beside the unchanged fixed-P16 route, with
the same real CPU guard and strict comparisons, not an optimizer ULP exception.
