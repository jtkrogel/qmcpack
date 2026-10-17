Conventional CPU ordinary-locality ECP fixture
============================================

``wftrain_ecp.xml`` matches the fixed-Slater/J2 ECP product gate: one Be ion,
two valence electrons, a fixed analytic STO of exponent1.2 and four active J2
coefficients. It references the existing tracked
``../../pseudopotentials_for_tests/Be.BFD.xml``; no radial tables are duplicated.
The ordinary-locality VP traversal uses rule2/four points and local channel1.
Stochastic quadrature remains enabled.

After the numerical and physical-identity gates admit this native profile::

  python3 ../H2_ae_conventional_wftrain/check_sj_training.py /absolute/qmcpack wftrain_ecp.xml --companion-optimizer sr --restart --preflight

The checker resolves the asset relative to the source fixture before copying
each recipe to its isolated evidence directory. It checks two Adam/SR updates,
interval inference, immutable original orbitals, exact inference-on/off model,
optimizer and sampler state, and exact split continuation. Exported physical
identity must include the actual parsed PP content and immutable rule/channel/
grid descriptor. Filenames are retained only for reports.

Preflight rejects DLA and non-VP evaluation. A separate temporary PP copy gets
one harmless XML comment appended; continuation must reject its changed content
even though the radial physics is unchanged. The tracked source asset is checked
unchanged. Existing KFAC, zero-cap and neural-decay negatives remain included.

Complete local energies and accepted nonlocal replay grids come from the SAME
streaming traversal, not old collection energies paired with fresh grids.
Products replay captured positions/weights/ratios without new RNG draws. Replay
storage is O(Q+B*Ne), adapter rows/channels O(P), and never Q*P. The bridge cap
does not cover opaque component scratch/private clones/configurations. PP source
content is immutable shared model metadata, not copied per walker.

The fixed-Slater fixture qualifies only real CPU single-Slater J1/J2 ECP.
The separately qualified small orbital profile is described below. No MD/J3,
DLA/TMDLA, spin-orbit, periodic, complex or accelerator ECP qualification follows.

One-occupied orbital/J2 profile (CPU ONLY)
-----------------------------------------
``wftrain_orbital_ecp.xml`` uses the same Be.BFD file and ordinary-locality VP
rule2. Each spin has one occupied orbital. One Cartesian p shell plus s gives
four AOs; explicit full-rank 3x4 coefficients select only s,px,py (zero pz).
Two active (0,q) rotation tangents plus four J2 coefficients give six parameters.
No many-occupied, MD, external-orbital, cusp, complex or GPU profile is admitted.

The numerical gate separately covers actual accepted Be grids, full-energy
and ECP-only finite differences, score/JVP/VJP duality, noncommuting history,
source/sibling isolation and RNG invariance. Its exact-node variant uses pure
px and the supported rule3 octahedral grid. It requires an exactly zero
captured ratio with a finite nonzero derivative, never a synthetic node record
or a tolerance-zero. Ratio derivatives invert only the nonnodal reference.
The existing two P rows and accepted complete ratios/bare weights are reused;
no Q*P derivative storage or full nodal virtual TWF evaluation is introduced.

The one-Adam native workflow is::

  python3 ../H2_ae_conventional_wftrain/check_sj_training.py /absolute/qmcpack wftrain_orbital_ecp.xml --orbital-ecp

It checks two updates, scheduled inference, complete accepted C/C_copy/full
rotation frames, model import/fresh training/re-export/re-import, exact
initialized restart, explicit-reference physical VMC and changed-PP rejection.
Only that negative accepts either the precise identity-content error or the
earlier trusted model_identity HDF string-width error. Production field
bounds are not relaxed. Native completion is not a separate gradient oracle;
the mathematical derivative claim is supported by the numerical unit gate.
The workflow has a 300-second whole and 180-second process ceiling, with no
optimizer/backend matrix, convergence or performance claim.
