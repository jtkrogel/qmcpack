Bounded conventional external LCAO regression (CPU ONLY)
=======================================================

wftrain_external_lcao.xml reuses LiH-pyscf.h5 and the existing LiH geometry.
Two walkers and eight J2 parameters exercise two Adam updates, scheduled
inference, model export/import, fresh training/re-export/re-import, exact
initialized restart and loader-versus-explicit physical VMC comparison.
The shared checker also tests identical-byte relocation, changed actual basis
and coefficient rejection, and unchanged tracked input. No optimizer matrix,
convergence, speed or peak-memory claim is made.

Only explicit source_identity="file_image" with a positive
source_identity_max_bytes <=1048576 enables provenance capture. It is disabled
by default. Before and after each actual basis/coefficient handle is read,
the builder checks a read-only SEC2 image and rejects non-hard links,
external raw storage, virtual datasets, oversize images and changed bytes.
Builder, selected SPOs and clones share one immutable record. This asset has
401464 bytes; its provenance storage is separate from derivative scratch.
Only the recognized collection href is routing-only in scientific identity;
the actual load controls and exact file bytes remain authoritative.

Qualification is real CPU, open fixed single Slater and active J1/J2 only.
Cusp correction, orbital rotations, ECP, periodic, complex, offload/GPU,
external B-spline and other unqualified external assets remain unsupported.
Ordinary LCAO inputs without the explicit opt-in retain their prior behavior.
