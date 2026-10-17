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

Actual native qualification passed on source8cd5467d2 with the corrected
checker6641753a (round85), after the independent numerical gate passed 180
assertions. Compact evidence is retained in task46 validation under
external_lcao_ecp_8cd5467d2. The original numerical setup failure and the
first native checker-label failure are preserved separately. This one
real/full-precision/nonoffload CTest uses the same existing 300-second route;
it adds no extra optimizer matrix or scientific rerun.

Active orbitals, MD, J3, cusp, periodic, complex, conventional GPU/offload and
DLA/TMDLA remain outside this fixture's profile.

The optional ``--external-orbital-ecp`` mode derives a shared spo-ud
rotation wrapper over the first three HDF MOs (base, 3x72 reference C),
giving P6 with J2. Its independent ordinary-VMC reference uses a tiny
legacy VP file with full history, not ineffective inline coefficient text.
Only this exp(log-history) scientific VMC comparison uses rtol=atol=3e-12;
weights/counts and all accepted-C/artifact/optimizer/sampler restart payloads
remain exact. The VP is never a restart initializer. One protected HDF copy
is reused/restored for identity negatives; tracked assets stay untouched.
Native round 106 passed with application source ``1d4c6e758`` and SHA256
``bec4f57553bdb671174a93bff2197bd463fd43a70779c320f8f209add8592e61``;
the separately corrected helper is ``918ce130b`` (checker SHA256
``6e8486c182163a7c209bfdb56db8e54b2675eac152f64e0b25a9b56c620b03ff``).
Both VP scientific comparisons had maximum absolute difference zero, and
the initialized split restart was bitwise exact. One active-mode CTest is
registered under the same real/full-precision/nonoffload/Python guard and
300-second limit. The original round 103 VP charset failure and round 105
negative-label failure remain preserved; neither changed scientific tolerances.
This adds only shared three-MO single-Slater orbital/J2 locality ECP coverage,
not MD, larger determinants, other external loaders or non-CPU profiles.
