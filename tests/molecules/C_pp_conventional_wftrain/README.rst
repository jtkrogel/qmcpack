Neutral-carbon orbital/ECP consumer (CPU ONLY)
================================================

This small integration fixture is not a converged carbon scientific model.
It uses the existing C.BFD.xml, four valence electrons (two per spin), one
inline STO s shell and one complete Cartesian p shell. Four MOs start from
identity coefficients. Both determinants share the named rotation owner.
The four complete occupied/virtual tangents (0,2), (0,3), (1,2), (1,3) plus
four J2 coefficients give P8. The ordinary-locality VP rule is 2, local
channel 1. No pseudopotential content is copied or modified.

After the independent numerical gate and native admission pass::

  python3 ../H2_ae_conventional_wftrain/check_sj_training.py /absolute/qmcpack wftrain_orbital_ecp.xml --many-occupied-orbital-ecp

The shared checker requires two committed Adam updates, finite complete
energies and gradients, changed orbital/J2 blocks, scheduled inference,
four-MO accepted/reference/full-history frames, physical VMC export/import
agreement and exact initialized split restart. Changed PP content must be
rejected. The whole workflow is capped at 300 seconds. It does not claim
optimizer convergence or production-quality energies.

Qualification is conditional until an actual native positive is recorded.
This fixture does not admit MD, external orbitals, cusp corrections,
periodic/complex wavefunctions, DLA/TMDLA or conventional GPU execution.
