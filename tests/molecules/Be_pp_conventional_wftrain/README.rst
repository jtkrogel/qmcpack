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
The separately qualified orbital and fixed-DETS profiles are described below. No J3,
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

Fixed-DETS/J2 profile (CPU ONLY)
-------------------------------
``wftrain_mdj_ecp.xml`` uses fixed inline s/px orbitals, frozen C0=.6, active
C1=.05 and four J2 coordinates with the same ordinary-locality Be BFD operator.
Two Adam/SGD updates, scheduled inference, exact inference isolation and complete
ROOT/optimizer/rank-shard/non-clock-history split restart passed at source
87e982fc07eaa260d7c9885799bf9281f65406dd::

  python3 ../H2_ae_conventional_wftrain/check_sj_training.py /absolute/qmcpack wftrain_mdj_ecp.xml --restart --preflight

The separate mathematical gate compares all five physical derivatives on an
actual Be grid with an independent signed occupied-matrix oracle. It includes
an exact zero captured total ratio with nonzero dCI. A separately labeled
synthetic one-knot replay covers an exact virtual reference determinant node
with nonzero total ratio; that knot is not claimed to belong to the Be grid.
Both energy VP ratios and CI derivatives use the same accepted-reference
bordered determinant algebra. Only private evaluators and training-owned walker
clones enable this path; ordinary VMC/pool prototypes retain their default mode.

Accepted reference determinants must be invertible. A separate real CPU fixed-SPO
full-VGL repair handles exact zero derivative-reference gradients/Laplacians;
its dense-oracle gate and two-MPI native restart passed. Complex, optimized-SPO,
spinor and offload derivative paths are unchanged. CSF, orbital rotations, J3, complex,
offload and DLA are excluded from this fixed-DETS ECP profile. No timing or
overall memory benchmark is claimed.

Mixed fixed-orbital DETS/J2 LM/ALM (CPU ONLY)
------------------------------------------
The bounded mixed-normalization consumer reuses ``wftrain_mdj_ecp.xml``:
real CPU/open, fixed inline orbitals, frozen C0=.6, active C1 and four J2
coordinates, ordinary-locality Be BFD. CSF, orbital, J3, offload and DLA
compositions remain excluded from this LM/ALM extension.

The guarded real-FP64/non-offload CTest registers one ALM workflow::

  python3 ../H2_ae_conventional_wftrain/check_sj_training.py /absolute/qmcpack wftrain_mdj_ecp.xml --mdj-ecp-alm

It uses 32 samples, fixed damping .01, the full six-dimensional augmented
space, existing finite work/time caps, and strict inference/import/re-export/
initialized split restart. No seed, source asset, numerical tolerance or root
selection is changed. A genuine first-iteration ``untrusted_root`` is reported
as EXPECTED REFUSAL only with no history rows, HDF output or publication;
that outcome does not qualify positive updates, inference or restart.

The independent numerical foundation120 passed 1,986 assertions, including
same-grid all-P physical finite differences, dense 6x6 S/symmetric/raw H
actions and mixed normalization. Its synthetic Ritz-vector checks establish
the coordinate map, not convergence of an actual ALM eigensolve.

Actual native122 separately passed two committed 32-sample ALM updates,
scheduled inference, export/direct parameter-file VMC/import/re-export and
strict initialized full/split restart. Frozen C0/SPO and the intended mixed-CSF
driver rejection were checked. The unchanged app was built from
``c2d41dbf40f211f087178fb831c3f563078418ce`` (SHA256 ``b2de0edcaa133a58631829b474945b640e868833e8bc235ab29b7a803b3cf781``);
the executed helper-only correction was ``d50592398`` (SHA256 ``5f5a884300701a332b11b90b13478c7c3b9670f790027e353b5e52f9d67d1a24``).
The first native121 whole-checker failure remains preserved: its last CSF-LM
negative incorrectly inherited ALM-only inner controls and stopped at parser
validation. The correction gates those controls on the actual optimizer;
native122 reached the intended driver guard. Neither seeds/tolerances nor the
app were changed between these attempts. This does not qualify CSF/orbital
mixed ECP LM/ALM, a second optimizer trajectory, or converged optimization.
Compact proof is recorded under Task46
``validation/mixed_ecp_native_c2d41dbf/qualification_C.rst``; no HDF payload is
duplicated in that archive.

Fixed-orbital signed-CSF/J2 full LM (CPU ONLY)
--------------------------------------------

``--csf-j2-ecp-lm`` derives the same Be Ne2/two-MO model used by numerical130:
frozen outer CSF0=.6 with child .8 (expanded C0=.48), and active CSF1=.05
with signed sqrt(.5) s/px spin-interchanged children, plus four J2 coordinates.
It preserves the tracked fixture/PP/seed81357, requests full LM only, and reuses
32 samples/update, damping.01, Krylov64/256/15s, relative tolerance1e-6 and
maximum augmented subspace6. ALM-only inner controls are absent. One rank/OMP1,
180s/application and300s/helper bounds apply. The existing two-update inference,
model import/re-export/direct VMC and strict initialized split restart checks
remain exact; immutable CSF children/outer gauge and fixed SPO are checked.
Numerical130 passed2,742 assertions including the reused same-grid P5 dense/FD
foundation. Inverse routes are not negative capabilities here and receive no new
native claim. Obsolete CSF-ECP mixed-LM guard negatives are retired; genuine
model/identity/unsupported-import-driver failures remain checked.

Actual native131 separately passed two committed 32-sample full-LM updates,
scheduled inference, export/direct VMC/import/re-export and strict initialized
split restart, with frozen outer .6/expanded .48 gauge, immutable signed children
and fixed SPO preserved. Source ``91269943829be4e44b4abd439d9fc6c55115b456``,
app SHA256 ``6ac1b912971492c9a202b161ddb87b49ad2f49d8dff0a45143b9dd1437688a13``
and helper SHA256 ``1911f017c41c522609f3073af68b6da07c40577e9c0dd9975f124a576cdcef16``
identify that one workflow, not converged optimization or a repair/retry of the
distinct joint-orbital ALM refusal128/129. No ALM/inverse-method positive claim
is added. Compact proof: Task46
``validation/fixed_csf_ecp_lm_native131/qualification_C.rst``.

One guarded real-FP64/non-offload CTest registers that SAME full-LM command as
``short-Be-pp-conventional-wftrain-csf-lm``. CMake discovery/help checks for the
registration are not an additional scientific run or an optimizer matrix.
