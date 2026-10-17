Captured external LCAO and ECP consumer (CPU ONLY)
==================================================

wftrain_external_ecp.xml references the existing 244008-byte LiH.orbs.h5 and
the exact tracked Li.BFD.xml and H.BFD.xml files. Their source headers, not
the filenames, remain authoritative. Li has local channel 1 and a nonlocal
component; H is local-only with channel 0. Both explicit rules are 2. The
fixed shared one-MO SPO serves two valence electrons (one per spin); only
four ud J2 coefficients are active. No asset content is duplicated or changed.

The explicit file_image capture is capped at 1048576 bytes and uses the
qualified fixed real/open CPU LCAO loading controls. After the numerical gate
and application admission pass::

  python3 ../H2_ae_conventional_wftrain/check_sj_training.py /absolute/qmcpack wftrain_external_ecp.xml --external-lcao

The existing bounded checker performs two Adam updates, scheduled inference,
direct parameter_file VMC versus explicit accepted-parameter reference,
fresh training/re-export/re-import and exact initialized split restart.
It verifies captured-image relocation, basis/coefficient mutation rejection,
both PP source identities and separate changed-PP-copy rejection. All tracked
assets must remain unchanged. The total limit is 300 seconds. No convergence,
performance or general external-SPO qualification is claimed.

Native qualification is pending until an actual positive is recorded. Active
orbitals, MD, J3, cusp, periodic, complex, conventional GPU/offload and
DLA/TMDLA remain outside this fixture's profile.
