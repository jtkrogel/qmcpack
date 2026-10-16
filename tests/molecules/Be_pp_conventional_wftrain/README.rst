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

This is only the real CPU fixed-single-Slater J1/J2 profile. No MD/orbital/J3 ECP,
DLA/TMDLA, spin-orbit, periodic, complex, or accelerator qualification follows.
