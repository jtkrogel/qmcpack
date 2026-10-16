Conventional SJ native training fixture
======================================

``wftrain_sj.xml`` is a tiny real-valued, two-spin, all-electron CPU fixture.
It fixes one bonding Gaussian SPO shared by the spin determinants and optimizes
four J1 plus four J2 B-spline coefficients. Two walkers, two blocks, and one step
give four samples per iteration; there are two updates, each followed by one
independent inference block. These are correctness settings, not a convergence
recipe. No external HDF orbital data or trained model is required.

Provenance at baseline8cdc7a2d0:

* H2 geometry and initial positions: ``../H2_ae/h2.ptcl.xml``, SHA-256
  ``35d87ef6ea16cb5ec79ca3295b5bf3bb70579034a04164475a922a5f351c2449``.
* Hydrogen STO-3G contractions: ``../H4_ae/H4.wfs.xml``, SHA-256
  ``5413985d750478e83e04cb35f1285ed6d1bbb12a3c8ae798268999e98cea6585``.
* The bonding coefficients (0.55,0.55) and eight small Jastrow coefficients
  are explicit test initializations, not a historical optimized H2 model.

Run ``python3 check_sj_training.py /absolute/path/to/qmcpack --restart --preflight``.
The checker runs Adam and plain SGD, checks committed updates and inference
version labels, compares inference-on/off model/optimizer/sampler state exactly,
checks exported bonding coefficients remain fixed, and checks split continuation.
Preflight cases require rejection of KFAC, zero legacy limits, and active SPOs
in the initial J1/J2-only binding. Evidence directories and complete logs are
retained in the system temporary directory, including on failure.

The application tests require a real double-precision CPU build plus Python
h5py/numpy. The checker defaults to one OpenMP thread; ``--threads`` can exercise
a bounded OpenMP profile. ``--ranks 2 --threads 2 --profile-only`` runs a single
Adam/inference case under mpirun and checks the declared global sample weights,
accepted versions, changed active parameters, and frozen Slater export. It does
not replace the full inference-isolation/restart suite or establish arbitrary
MPI layouts. Later MDJ/orbital/ECP qualification remains separate.

``wftrain_sj_j3.xml`` appends the standard PolynomialFunctor3D/JeeIOrbitalSoA
three-body Jastrow: isize=esize=2 gives8independent coefficients in one default
functor shared by all spin pairs. J1/J2/J3 jointly expose16active coefficients.
The half-cutoff is3bohr so both electrons initially contribute around both ions.
Use the same checker with that XML and ``--companion-optimizer sr --restart``.
The bridge's mandatory scratch limit covers its two derivative rows and three
channel vectors. J3 also retains internal P*Ne gradient/laplacian and functor
derivative caches per private replay worker; those opaque caches are explicitly
outside that bridge-only limit. Their worker count is bounded independently of
sampled blocks; the fixture does not claim a complete process RSS ceiling.

``--optimizer-screen`` selects six additional nonlinear-Jastrow methods:
momentum_sgd, rmsprop, amsgrad, sr, minsr and spring. Each uses two updates and
four samples/update, with interval inference. Matrix-free methods explicitly
bound Krylov iterations32, operator applications128 and solve seconds10, with
damping0.01; SR admits one damping attempt. The screen checks finite committed
updates and exported physical state, and does not establish convergence or
performance. It reuses the same single tiny real CPU fixture.

``--family-screen`` selects lm, alm, pii and minpii with32samples/update
(4walkers x8blocks), P=8 and2iterations, using the same learning rate0.0001.
Each iterative solve declares64iterations,256operator applications,15seconds,
damping0.01 and relative tolerance1e-6. LM/ALM use maximum subspace9; ALM also
uses32inner iterations/restart9. PII/minPII use GMRES restart16 and no damping retries.
The checker records every committed/rejected decision in family_results.csv,
checks accepted/rejected exported state and version consistency, and retains
failures rather than altering the numerical algorithm or forcing acceptance.
