Tiny fixed-orbital MDJ native fixture
====================================

``wftrain_mdj.xml`` reuses the exact tiny H4-derived DETS/J2 model from
``test_LegacyWaveFunctionAdapter.cpp``. The hydrogen STO-3G contractions and
fixed four-orbital coefficient matrix originate in ``../H4_ae/H4.wfs.xml``
(SHA-256 ``5413985d750478e83e04cb35f1285ed6d1bbb12a3c8ae798268999e98cea6585``).
Linear test geometry, electron positions and small four-coefficient J2 functions
are explicit correctness initializations. The first CI coefficient0.6 is fixed;
the second starts exactly0. Eight J2 coefficients are active jointly with CI1.

The shared checker derives the active manifest from the XML instead of assuming
the H2 parameter count. It checks canonical updates, fixed SPO/CI0 export,
inference accepted-version labels, exact inference isolation, and split restart.
Example from this directory:

.. code-block:: text

   python3 ../H2_ae_conventional_wftrain/check_sj_training.py /absolute/qmcpack \
     wftrain_mdj.xml --companion-optimizer sr --restart

This runs representative Adam and bounded SR,2updates with4samples/update,
plus Adam inference isolation and split continuation. SR has damping0.01 and
Krylov caps32iterations,128operator applications,10seconds, one damping attempt.
``--ci-representation csf`` expresses the same two terms as single-determinant
CSFs; ``--profile-only`` can select one bounded Adam case for that representation.
No external orbital data or trained model is used, and no convergence or orbital
rotation/advanced-linear-optimizer qualification is implied.

Pure-CI PII/minPII regression
---------------------------

``--ci-pii`` generates private copies of this two-DETS fixture with both J2
arrays frozen, leaving only CI1 active (P1). The real CPU regression runs
two updates for each method, with 32 samples per update, and requires finite
committed CI changes. Fixed CI0, SPO and J2 are checked, followed by exact
seeded VMC comparisons of each final exported model against explicit accepted
CI coefficients. Existing Krylov limits and a 300 second whole-checker cap
apply. No controls are tuned to force acceptance.

This profile passed on source ``5f0460dc8``, alongside a separate 65-assertion
dense centered CI oracle. It qualifies pure DETS CI only. Mixed active
CI/Jastrow, CI/orbital and CSF are not qualified by that DETS result; no convergence-quality,
equal-update or GPU claim is made.

Pure-DETS CI LM/ALM regression
-----------------------------

``--ci-lm`` freezes J2 and uses the same fixed-orbital model, with only CI1
active. It requires two committed 32-sample updates for each existing LM/ALM
solver, unchanged frozen C0/SPO/J2, exact exported-model physical inference and
exact split model/state/sampler/scientific history. Mixed active CI/J2 is rejected.
The centered Ritz vector uses A=v0-mean(Jv), deltaCI=learning_rate*vi/A, preserving
fixed C0. Nonlinear xi normalization is not applied to these linear coordinates;
Hamiltonian/overlap stabilization remains in the existing solved H/S actions.

This bounded real CPU profile passed on source ``dd9ee31a8``, following a
574-assertion numerical gate including physical zero/nonzero-CI ratio checks for
both solvers. It does not qualify mixed CI/Jastrow/orbital, CSF, ECP or GPU CI.
Use the shared checker with ``--ci-lm`` instead of ``--ci-pii`` to reproduce it.

Nontrivial pure-CSF LM positive and ALM refusal
---------------------------------------------

``--csf-lm`` privately generates two open-shell CSFs, each containing two fixed
determinants with expansion coefficients 1/sqrt(2), and freezes J2. The actual
reference CSF0=.6 is preserved, rather than treating flattened C0 as CSF0.
The mathematical oracle validates both LM/ALM physical projective maps and
score/local-energy derivatives at zero/nonzero CSF1.

The native regression passed on source ``79392c99c`` with checker ``46851f887``:
LM commits two 32-sample updates, preserves CSF0/expansion/SPO/J2, and passes
exact exported inference and full/split restart. The same protected sample's
raw nonsymmetric ALM pencil has eigenvalues 2.662880510452576 +/-
1.503921983474190 i. ALM must refuse with ``untrusted_root`` before any history
row, HDF checkpoint/model output or publication. This correct refusal is not an
ALM convergence positive; seeds, controls and tolerances are not tuned to force
a real root. Mixed active CI/J2 remains rejected. This is a real CPU,
all-electron fixed-orbital profile, not ECP, orbital or GPU qualification.

Pure-CSF PII/minPII regression
-----------------------------

``--csf-pii`` uses the same nontrivial two-DETs-per-CSF model with frozen J2.
The actual CSF reference is preserved; both existing methods apply additive
coefficient updates rather than LM's Ritz-vector retraction. The physical
oracle compares both methods to the dense centered scalar system and checks
actual private wavefunction ratios and every fixed expansion term.

The native regression passed on full built source ``27a46961f`` (conventional
guard ``f2544a843``): both methods commit two
32-sample updates, preserve CSF0/expansion/SPO/J2, and pass exact exported-model
inference and full/split model/state/sampler/scientific history. Mixed CSF/J2
is explicitly rejected by this profile. There is no mixed-coordinate, orbital,
ECP, conventional GPU or convergence-quality claim.
