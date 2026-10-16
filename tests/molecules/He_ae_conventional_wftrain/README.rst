Conventional CPU orbital rotation fixture
========================================

``wftrain_orbital.xml`` matches the production orbital-product unit model:
one helium ion, one electron per spin, three analytic STO radial functions,
one occupied orbital and two noncommuting occupied-virtual rotations. Both
spin determinants reference the same named rotation wrapper. The explicit
initial coefficient matrix is numerically identity but is not LCAO's implicit
``Identity`` mode. No external model files are used.

After the bounded product/live-publication gates admit this native profile::

  python3 ../H2_ae_conventional_wftrain/check_sj_training.py /absolute/qmcpack wftrain_orbital.xml --companion-optimizer sr --restart --preflight
  python3 ../H2_ae_conventional_wftrain/check_sj_training.py /absolute/qmcpack wftrain_orbital.xml --orbital-jastrow --companion-optimizer sr --restart

The first command optimizes only the two rotation coordinates. The second
adds the same four-coordinate J2 as the product test. Each runs two Adam/SR
updates with interval inference, exact inference-on/off isolation and exact
split continuation. The checker explicitly identifies the expected rotation
names from the exported registration rather than accepting arbitrary variables.
It requires nonparallel successive rotation increments and a nonzero accumulated
virtual-virtual coordinate. This is a tiny correctness profile, not convergence.

Both checkpoint and parameter export must contain the same complete orbital
frame: source version, aliases, full accumulated coordinates, accepted C and
unchanged original reference C_copy. The original XML coefficients stay fixed;
the accepted physical C must change. Active coordinates alone are not a model
import. Full-history restore retains exact accepted coefficients instead of
silently replacing them with a rounded exp/log reconstruction.

The legacy derivative scratch cap covers adapter rows/channels only. It excludes
component-owned derivative caches, private orbital coefficient banks and retained
configurations. One private replay worker is shared across retained sample blocks;
this fixture does not claim a complete process-memory ceiling.

The implicit-Identity LCAO wrapper construction path has a preexisting null-C
assumption in storeParamsBeforeRotation and is not exercised as a crashing native
negative. Direct component validation tests reject unqualified banks; native
preflight instead checks unsupported KFAC, zero limits and neural weight decay.
MD-orbital, ECP, complex and accelerator profiles remain separate.
