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
a real root. Mixed active CI/J2 LM/ALM remains rejected. This is a real CPU,
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
inference and full/split model/state/sampler/scientific history. Mixed CSF/J2 LM
is explicitly rejected by this profile. There is no mixed-coordinate, orbital,
ECP, conventional GPU or convergence-quality claim.

Mixed DETS/J2 PII/minPII regression
---------------------------------

``--mdj-pii`` retains all nine active coordinates: CI1 and the eight J2 values.
Both existing methods use the local additive tangent update, not an LM
Ritz-vector normalization. A separate physical oracle compares both solves to
an explicit dense 9x9 system with unequal sample weights and checks joint
log-wavefunction/kinetic directional derivatives and quadratic convergence.

Native full source ``afe1d724d6ab704994157d8a729fd8d6bf5f650a`` passed two
committed 32-sample updates per method, fixed C0/SPO, exact exported inference
and full/split model/state/sampler/scientific history. Mixed LM remains
explicitly rejected; mixed CSF/J2 is separately qualified below. Only real CPU
all-electron fixed-orbital DETS plus
TwoBodyJastrow is qualified here; J1/J3, orbital and ECP extensions are not
implied. No convergence-quality or timing claim is made.

Nontrivial mixed CSF/J2 PII/minPII regression
-------------------------------------------

``--csf-j2-pii`` retains the same P9 layout using two nontrivial CSFs, each with
two fixed determinant terms, and active J2. The shared physical oracle checks
both dense solves, all score/local-energy finite differences, joint directional
convergence, true frozen CSF0 and expanded reference terms, and source isolation.

Native full source ``0acbc6b456dde9614b6a6b1b25761d2f9b51b6bc`` passed both
methods' two 32-sample committed updates, fixed CSF0/SPO/expansion and exact
exported inference/full-split restart. Mixed CSF/J2 LM remains rejected.
The component whitelist is still only fixed-orbital MSD plus TwoBodyJastrow,
real CPU and all-electron. No mixed LM/ALM, orbital, ECP or conventional GPU
qualification follows from this result.

The ``--mdj-alm`` mode reuses this exact DETS P9 fixture with
the reviewed mixed fixed-reference normalization. It selects one ALM-only
full import/re-export/inference/strict initialized-split workflow, two committed
32-sample updates and explicit CI1/J2 changes while C0=.6 and the SPO stay fixed.
The bounded controls are damping .01, Krylov 64 iterations/256 applications/
15 seconds/relative tolerance 1e-6, augmented dimension 10 and ALM inner
iterations 32/restart 10. Native round 110 passed on its first attempt with
application source ``74faae29447276004e0cd03ea69ab4ad80f82ee1`` and SHA256
``51cd57f38ff4545d97d415442a1d754f93c873a92a7dfa8e1ee414f2b44cfe6c``;
executed checker SHA256 is
``d8a194011ff154e4776b7286ccffe4f9ddb99f95322ca0a61cc398ae1e4d1491``.
Both updates committed with 32 samples, and strict initialized restart passed.
One guarded real/full-precision/nonoffload/Python CTest has a 300-second limit.
CSF has mathematical coverage only here, not native ALM qualification. No
native LM, CSF, orbital or ECP extension is implied by this ALM result.
Historical AE mixed-LM guard-rejection statements above describe their original
sources; the new numerical policy is separately reviewed, while ECP/orbital
mixed LM rejection remains in force. No old default or strict comparison changes.
