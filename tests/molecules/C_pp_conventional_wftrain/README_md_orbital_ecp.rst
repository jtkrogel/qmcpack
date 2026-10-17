Fixed-CI MD orbital ECP native fixture
====================================

This tiny real CPU, open-boundary fixture composes two distinct inline
four-MO rotation owners, fixed two-DETS coefficients and four J2 variables
with the tracked carbon pseudopotential. The actual active layout is P16:
six local orbital pairs per spin and four J2 coefficients. It does not
qualify other CI/CSF layouts, external orbitals, complex/GPU/periodic models,
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

The ``--joint-ci-orbital-ecp-minpii`` mode selects the same equal-spin
DETS P17 recipe with minPII only. It retains two updates, four samples per
update, both frames/CI1/J2 changes, exported inference and strict initialized
split restart. Existing bounded family controls are used (damping 0.01,
Krylov 64 iterations/256 applications/15 seconds/relative tolerance 1e-6,
GMRES restart 16, zero damping retries, energy undershoot 1.0). One mixed LM
early rejection remains required. Native round 107 passed on its first attempt
with application source ``d256fed54a750f29bb319bd5fbab3ded47c5258c`` and SHA256
``4f2e439f62fa53d4a65b32c36e1d080b775fdebb1ce539f330872f82ab0b5355``;
the executed checker SHA256 is
``c712be7c4316ed4a77f3f54bedb73c3ad53affde995668d511a203eb8061c777``.
Full/export/import/inference and initialized split restart passed unchanged
strict comparisons, as did mixed-LM early rejection. One minPII-only CTest
uses the same qualified CPU guard and 300-second limit. PII has separate
numerical proof only; no native PII, LM/ALM, CSF or unequal-spin optimizer
qualification is implied.

The optional ``--unequal-spin-orbital-ecp`` mode derives one charged C^-
3-up/2-down case from the same XML. It adds the numerical fixture's third
up coordinate and uses up occupations 1110/1101, giving five/six orbital
pairs and joint P16. Native round 98 passed on its first attempt with app
source ``38729b657`` and SHA256
``8661f27b0705af800303736617c66693154eed69eaa35444378f832c2fb78033``;
the newer helper-only source is ``2cd6d4fe0`` (checker SHA256
``d2ec1d8d332ed0174a2a12773f534f3e68d65811fe112849e806d99e23996c28``).
One guarded 3/2 native case is registered with the same strict comparisons.
The swapped 2/3 orientation is numerical-only: both passed the separate
6,024-assertion physical gate. No arbitrary polarization/size claim is made.

The optional ``--csf-orbital-ecp`` mode derives the exact signed-child CSF
recipe: frozen outer CSF0=0.6 with HF child weight 0.8 (expanded C0=0.48),
active outer CI1=0.1 with ordered opposite-sign sqrt(0.5) cross children,
and both six-pair frames/J2 (P17). It preserves immutable child attributes
and original gauge. Native round 100 passed on its first attempt with app
source ``a4fcaead3`` and SHA256
``51819c248cf6c3151d06ed03e5daf782b37bf89fa7c4df05a2ebe3ecd0962f28``;
the newer helper-only source is ``7dcbd12fa`` (checker SHA256
``3d2c09476dae9689bde8910f7e0ba1dca50e6b958b52062da9888b47cea28153``).
One guarded exact-recipe CSF mode is registered with unchanged strict
model-reuse/restart checks and no ULP exception. The separate independent
physical gate passed 7,133 assertions; no other CSF layout is qualified here.

The optional ``--joint-ci-orbital-ecp-alm`` prepares one ALM consumer of the
same qualified joint-DETS P17 recipe. It retains the original seed, physics,
frozen C0 and both complete frames; training W4/blocks8 gives 32 samples.
Existing limits are damping .01, Krylov64/256/15s/rtol1e-6, augmented18,
ALM inner32/restart10. The full/import/inference/initialized-split comparisons
remain exact. A genuine initial untrusted-root refusal is negative only, with
no update/export/restart claim; no seed, shift or tolerance tuning is allowed.
A same-recipe fixed-CI optimize=no negative must reach the capability guard.
This mode awaits separate driver admission and native qualification; physical
round123 supplies dense actions/local mixed retraction, not a converged ALM root.
