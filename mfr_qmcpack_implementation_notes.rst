QMCPACK mean-field-residual implementation: consumer and driver record
======================================================================


2026-11-04 native QE functional hand-off and consumer checkpoint
---------------------------------------------------------------

The current Libxc/HDF5 producer checkpoint and immutable binary/source hashes
are recorded in functional_extension_validation/sol/qe_meta_libxc_20261104/
(review.rst and review_final_manifest.json) in the Hubbard-U project.
Actual completed native r2SCAN, r2SCAN+rVV10 and TPSS jobs were checked through
the local Nexus status/get_result paths: seven paired hand-offs passed;
see native_nexus_handoff_results.json. Existing generic import machinery
accepted the new functional provenance without a new QMCPACK algorithm.

Eight Ti complex-twist standalone cases covered full/mixed precision,
legacy/batched drivers and bare-core/two-field providers, each with short
VMC and DMC. Validation checked 2056 walker observations and 16 scalar series
against the selected estimator arithmetic using the pre-existing precision
bounds. Runtime Hamiltonians contained electron-electron and the selected
MFR provider, without ion-ion or pseudopotential evaluation. Legacy restart
cases required a native legacy checkpoint sidecar; initial attempts using a
batched-only checkpoint were preserved as failed controls. The eight-case
success is execution/arithmetic evidence, not physical acceptance or a claim
of cross-driver restart identity. See ti_standalone_validation.json.

The economical two-field Ti meta-core RMS remains about 90--100 mHa per
primitive cell. An independent direct-plane-wave/cubic-field evaluation
reproduced 89.485412 mHa versus 89.530594 mHa from the meshfactor-2 consumer;
the difference between those difference traces was 0.184834 mHa RMS.
See ti_independent_PW_cubic_diagnostic.json. Thus consumer orbital splines
explain little of this particular discrepancy. No physical tolerance has
been adopted and neither this meta route nor the full Goal is accepted.

Audit date: 2026-10-29. Repository:
``/home/j1k/repositories/qmcpack_dev_new5``. Snapshot:
``72ab2e5277da627c23c71f12d155232a4b1c93f8`` (branch ``qp_mfr``).
Line anchors refer to that snapshot. This retrospective describes implemented
behavior, separates historical validation from new observations, and identifies
limitations relevant to additional DFT codes/functionals.

The producer contract is documented in
``/home/j1k/repositories/qe_dev/mfr_qe_implementation_notes.rst``. Existing
planning/progress files are ``mfr_impl_plan.rst`` and
``mfr_impl_progress.rst``. Those chronological records are useful evidence but
do not supersede the actual source or schema-v2 definitions below.

Mathematical and operational contract
-------------------------------------

For a determinant D made from orbitals of the exported mean-field Hamiltonian,
and trial wavefunction Psi=D*exp(J), the implemented estimator is

.. math::

   E_L^{MFR}(R)=E_{tot}^{MF}+\Delta T_J(R)+V_{ee}(R)
                         -\sum_i[v_H(r_i)+v_{xc,\sigma_i}(r_i)],

where ``Delta T_J = T[Psi]/Psi - T[D]/D`` and the constant is the occupied KS
eigenvalue sum plus ion-ion energy. In Hartree, for real J and unit electron mass,
the residual includes ``-1/2 sum_i [laplacian J + (grad J)^2
+ 2 grad(log D).grad J]``, with the appropriate real kinetic value for complex
orbitals. The code uses existing kinetic helpers and species masses.

The sign of the stored/reported ``MFRPotential`` is the original Hartree+XC
field sum; its subtraction happens in ``qmca`` or the standalone Hamiltonian.
"Positive" in class comments does not require positive-valued samples.

For nonlocal ionic pseudopotentials, this form adopts a determinant-localization-
like approximation: the Jastrow-dependent nonlocal difference is omitted.
Consequently it need not equal the conventional Slater-Jastrow nonlocal local
energy. For no Jastrow and exactly matched KS operators/orbitals, the determinant
identity should close. Numerical residuals must still be checked in practice.

Modes are resolved separately for each QMC section:

.. list-table::
   :header-rows: 1

   * - MFR field / request
     - Resolved mode
     - Walker energy
   * - No field; auto or no
     - STANDARD
     - Conventional LocalEnergy.
   * - Field; explicit no
     - STANDARD_WITH_MFR
     - Conventional LocalEnergy; MFR quantities are auxiliary observables.
   * - Field; auto or yes
     - MFR
     - LocalEnergy is replaced by the MFR expression.
   * - No field; yes
     - Error
     - Inconsistent input is rejected.

In comparison DMC, the population branches on the conventional energy.
``qmca``'s MFR result is an estimator on that population, not a separate MFR-DMC
projection. Standalone DMC uses the substituted MFR walker energy for branching.
The implementation does not introduce a separate DMC algorithm.

Change inventory and source map
-------------------------------

The source delta is the three commits ``49388c2cb`` (initial implementation),
``6a6496b21`` (standalone pathways), and ``72ab2e527`` (tests/precision), relative
to baseline ``86e852d4f``. The following covers all 19 changed/added files.

Hamiltonian and field implementation:

* ``src/QMCHamiltonians/MFRPotential.h:19``: new particle-set-only potential
  class, spline ownership, metadata/state, scalar output and clone APIs.
  Lines 28, 49 and 56 use ``FullPrecRealType`` for the mean-field constant and
  single-particle return, preserving field/constant precision in mixed builds.
* ``src/QMCHamiltonians/MFRPotential.cpp:22``: new low-level HDF5 helpers;
  ``put`` at line 152; spline setup at 233; observables at 289; evaluation at
  317/328; cloning at 343; conditional particle trace support at 349.
* ``src/QMCHamiltonians/KineticJastrowResidual.h:23``: new wavefunction-dependent
  auxiliary kinetic operator and clone API.
* ``src/QMCHamiltonians/KineticJastrowResidual.cpp:16``: mass setup;
  ``evaluate`` at line 34; ``makeClone`` at line 89.
* ``src/QMCHamiltonians/HamiltonianFactory.cpp:39``: includes for the two new
  operators. Lines 160-166 recognize ``extpot`` type ``mfr``/``MFR``, load the
  field, and add both operators as auxiliary, not ordinary physical terms.
* ``src/QMCHamiltonians/QMCHamiltonian.h:54``: resolved-mode enum;
  lines 89-91 expose configuration/accessors; line 440 stores mode;
  line 462 declares the energy substitution helper.
* ``src/QMCHamiltonians/QMCHamiltonian.cpp:176``: mode resolution and validation;
  line 237 implements ``applyMFREnergy``. Calls at 641, 659, 731, 909 and 957
  cover scalar, deterministic, multiwalker, and T-operator evaluations.
  Line 371 rejects standalone per-particle energy listeners; line 1124
  preserves the resolved mode when cloning.
* ``src/QMCHamiltonians/CMakeLists.txt:44``: add both new C++ source files.

Driver integration:

* ``src/QMCDrivers/MFRQMCMode.h:1``: new request enum, AUTO/YES/NO.
* ``src/QMCDrivers/QMCDriverInput.h:125``: parsed request getter;
  line 133 initializes AUTO.
* ``src/QMCDrivers/QMCDriverInput.cpp:41``: temporary string; line 74 registers
  ``mfr_qmc`` with allowed values auto/yes/no; lines 136-141 resolve the enum;
  lines 153-156 print explicit requests.
* ``src/QMCDrivers/QMCDriverFactory.cpp:56``: common XML request parser.
  Lines 211-225 configure the primary Hamiltonian before driver creation,
  validate supported VMC/DMC run types, and reject explicit standalone traces.
  Lines 393-395 disable conventional energy traces in standalone mode.

Nexus and scalar analysis:

* ``nexus/nexus/qmcpack_input.py:2519``: Hamiltonian ``extpot`` children;
  line 2526 adds ``mfrpotential`` and its type factory. Class/type/plural
  registries near 3165, 3189 and 3202 register the new element.
  VMC/DMC XML parameter definitions at 2998, 3021, 3052 and 3064 accept
  ``mfr_qmc`` and write booleans as yes/no.
  ``generate_hamiltonian`` arguments at 6937-6938, ionic-term condition at
  7021, and field construction at 7150 implement mode-dependent content.
  Basic-input defaults at 9412-9413 and validation/path/mode handling at
  9836-9862 expose the public generation arguments. Lines 9890-9891 pass
  them to Hamiltonian generation; lines 9935-9942 propagate explicit mode
  requests into generated VMC/DMC sections, including loop-wrapped sections.
* ``nexus/nexus/bin/qmca:181``: aliases ``emfr``, ``mfr``, ``emf``, and ``tj``;
  line 234 adjusts preferred quantity ordering. Lines 386-391 form the MFR
  sample array and analyze its statistics if all four inputs are present.

Tests/build registration:

* ``src/QMCHamiltonians/tests/test_MFRPotential.cpp:88``: temporary HDF5 fixture;
  line 180 supplies analytic wavefunction components; ten ``[mfr]`` test cases
  begin at lines 252, 272, 284, 297, 308, 318, 327, 362, 439 and 457.
* ``src/QMCHamiltonians/tests/CMakeLists.txt:32``: register the new test source
  in the existing Hamiltonian unit executable.
* ``src/QMCDrivers/tests/test_QMCDriverInput.cpp:23``: request-parser coverage.
* ``nexus/nexus/tests/test_qmcpack_input.py:2344``:
  ``test_generate_mfr_potential`` validates XML round-trip/generation/modes.
* ``nexus/nexus/tests/test_qmca.py:325``: ``test_mfr_local_energy`` validates
  the derived energy and its scalar-analysis behavior.

Reader and spline implementation details
----------------------------------------

The new field XML is

.. code-block:: xml

   <extpot type="mfr" name="MFRPotential" href="path/prefix.mfr.h5" scale="1.0"/>

``put`` also accepts ``file_name`` as an alias for ``href``. It requires a path,
the exact observable name, and positive scale. Use scale=1 for the matching KS
identity: changing scale changes only the field, not the stored mean-field
constant or orbitals, so it is not a self-consistent physical rescaling.

The consumer opens HDF5 directly and reads root attributes through helpers at
lines 22-125. String trimming handles Fortran space padding. These helpers
assume fixed-length strings; variable-length HDF5 strings from another producer
need explicit reader support or compatible fixed-length encoding.

At lines 176-213 the reader requires schema 2, exact quantity/grid-order/endpoint/
ion-exclusion/energy-definition strings, potential and total-energy units
``rydberg``, a nonempty functional label, one/two channels, and at least four
points per grid dimension. Two-channel data require exactly two electron groups.
It does not use the KS tables to verify the actual determinant band occupations,
nor does it enforce a functional whitelist beyond a nonempty label. Producer
correctness is essential; a relabeled unsupported functional can bypass the
producer's intended guard through a manually constructed file.

The file's potential datasets must have C-visible shape ``(nr3,nr2,nr1)``.
Lines 247-259 read in that order and reorder to ``spline_values(i,j,k)`` through
``qe_index=(k*nr2+j)*nr1+i``. Each sample is multiplied by ``0.5*scale`` to obtain
Hartree. The mean-field constant is separately multiplied by 0.5 at line 207.
The input lattice is read as C-visible rows of vectors, reflecting the Fortran
writer's original columns. See the QE record for all file fields and attributes.

Only the lattice and potential datasets plus required root attributes are used
in QMC evaluation. The KS arrays, ``dft_total_energy``, and auxiliary energy
provenance are for auditing; the reader does not recompute the stored constant.
Likewise ``hartree_g0`` is not checked by the current reader.
Dataset-level unit/spin/convention labels are not independently validated;
the implementation relies on root units and the prescribed dataset layout,
assumes lattice values are bohr, and maps spin channels by dataset/group index.

Lines 222-230 require fully periodic boundaries and simulation lattice vectors
that are integer combinations of the field lattice vectors (tolerance 1e-8 in
reduced coordinates). Splines use fractional coordinates on [0,1), with PERIODIC
boundaries in all directions, through ``create_UBspline_3d_d``. The coefficients
are real double precision even when orbital builds are complex or mixed.

``evaluateOne`` wraps positions with ``field_lattice_.toUnit_floor`` and calls
``eval_UBspline_3d_d``. One channel is used for all electrons; two channels map
group ID 0 to spin_1/up and ID 1 to spin_2/down. The code does not map channels
by species names. ``evaluate`` sums over electrons. No forces or potential
derivatives are exported by this operator.

``makeClone`` shares immutable spline coefficients through ``shared_ptr`` with
``destroy_Bspline`` deleters; clones retain their particle set and mean-field
constant. There is no dedicated GPU interpolation kernel or specialized batched
field kernel in this addition. Multiwalker integration uses existing operator
dispatch and per-walker evaluation, so build compatibility should not be confused
with a claim of GPU acceleration.

Kinetic residual and Hamiltonian substitution
---------------------------------------------

``KineticJastrowResidual::evaluate`` inspects ``isFermionic()`` for each
wavefunction component. It rejects a wavefunction with no fermionic component
and returns exactly zero if no nonfermionic factors are present. Otherwise it
allocates separate gradient/Laplacian buffers, calls component ``evaluateLog``
into fermionic/nonfermionic accumulators, then subtracts fermionic-only kinetic
contributions from the full ones using ``BareKineticHelper``. This includes the
determinant-Jastrow gradient cross term and uses group masses. It does not require
changes to the existing Slater or Jastrow implementations.

The intended decomposition is a multiplicative Slater-Jastrow product with a
fixed mean-field determinant. Backflow, orbital optimization, multideterminant
states, spinors, and more general correlated fermionic factors need explicit
analysis of the determinant eigenstate identity, not just successful component
dispatch. Repeated derivative evaluation also carries a cost; no cached analytic
Jastrow-residual optimization was introduced.

In standalone mode, ``configureMFRQMCMode`` requires the field and residual in
``auxH``. Physical operators in ``H`` must be named ``Kinetic`` and
``ElecElec``; the latter must have class ``CoulombPBCAA`` or ``CoulombPotential``.
An electronic Coulomb operator must exist. Pseudopotential, electron-ion,
ion-ion, MPC, and any other physical potential produce a setup error.
Particle sets for ions may still be present to define the structure/orbitals;
the prohibition is on physical Hamiltonian operators, not on ion coordinates.

``applyMFREnergy`` reevaluates the two auxiliaries, records their observables,
uses the evaluated electronic Coulomb term, and replaces ``LocalEnergy`` by the
MFR expression before the existing kinetic/property update. The ordinary kinetic
operator is still evaluated; standalone mode does not remove all kinetic work.
Ionic pseudopotential evaluation is absent because those physical operators are
rejected. In comparison mode the helper returns immediately, leaving conventional
energies untouched; auxiliaries follow the existing observable evaluation path.

Scalar files, traces, and walker logging
----------------------------------------

``MFRPotential::addObservables`` adds ``MFRPotential`` and ``TotalEnergyMF``;
``registerObservables`` also exposes them in scalar/statistical HDF5 output.
The residual operator adds ``KineticJastrowResidual`` through the standard
operator machinery. ``LocalEnergyMFR`` itself is a **qmca-derived** quantity,
not an additional scalar.dat energy column.

In comparison mode, standard ``LocalEnergy`` is preserved and all three
additional columns are available. In standalone mode, ``LocalEnergy`` itself
is the MFR energy; recomputing ``LocalEnergyMFR`` should reproduce it to output
precision. Existing ``Kinetic``/``LocalPotential`` decomposition must not be
mistaken for the eliminated determinant kinetic term: the ordinary kinetic
value is still present, and scalar component labels do not provide a unique
physical MFR per-particle decomposition.

The legacy TraceManager's conventional energy traces are disabled in standalone
mode; explicit ``trace='yes'`` requests error. Per-particle LocalEnergy listeners
also error because their decomposition cannot safely represent the constant and
residual. ``MFRPotential`` retains its own conditional particle trace hooks for
supported contexts. Walker logging is a separate facility: no walker-logging
source was changed by this effort. Interpretation follows the walker properties
written by the selected mode; new full logging/decomposition claims need an
explicit test rather than extrapolation from scalar output.

``qmca`` forms ``TotalEnergyMF + KineticJastrowResidual + ElecElec - MFRPotential``
as a blockwise array, then applies its usual equilibration exclusion/statistics.
This preserves same-run covariance. The analysis uses the already converted
input-energy units; it does not apply an extra Ry-to-Ha conversion to QMC data.
The original estimator is retained. No separate MFR second-moment or variance
estimator was added: ``LocalEnergy_sq``/reported conventional variance in
comparison mode is not an MFR variance.

Nexus usage and input-generation behavior
-----------------------------------------

Public basic generation arguments are ``mfr_potential`` (path string) and
``mfr_qmc`` (bool or None). True without a field errors. With a field, None/True
select standalone generation, omit ionic physical terms, and remove MPC from
corrections/estimator requests. False selects comparison generation. The driver
parameter is written explicitly when the caller supplies a bool; otherwise the
runtime AUTO behavior selects the mode. The XML supports legacy and batched
VMC/DMC forms.

By default the field path must exist. ``check_paths=False`` allows future QE
outputs to be referenced when constructing a dependency workflow. With
``run_path`` available, Nexus converts the path to a relative path from the
simulation directory. The implementation did not add a new Nexus dependency
result named "mfr_potential"; workflows use the SCF/orbital conversion dependency
and a known MFR filename. Ensure export really completed rather than relying on
a stale file from a previous SCF.

For comparison workflows, pass ``mfr_qmc=False`` explicitly. For standalone
inputs, remove ionic and MPC terms; attaching the field to an unchanged
conventional Hamiltonian without the explicit false request now errors.
Optimization in the demonstrated workflows uses the conventional Hamiltonian
without the field, then transfers the optimized Jastrow to VMC/DMC.

The hBN script is external to this repository, at
``/home/j1k/projects/mstd/34_mfrqmc/work/04_hbn_bulk/hbn_bulk_mfr.py``. It runs
separate Gamma/complex off-Gamma SCFs, optimization, comparison VMC, and added
comparison DMC sections. Executable paths are resolved absolutely because Nexus
launches jobs from simulation directories. This workflow detail is independent
of the MFR source changes.

Tests, evidence, and limits of coverage
--------------------------------------

``test_MFRPotential.cpp`` builds small 4x4x4 HDF5 fixtures. Its ten deterministic
cases cover constant interpolation/unit conversion and cloning; separate spin
channels; nonconstant periodic wrapping; rejection of an invalid quantity label,
incompatible groups, and a missing path; analytic zero and nonzero kinetic
residuals including the cross term; mode resolution and ionic/pseudo/MPC/missing-
component rejection; unchanged comparison energy; and exact standalone scalar/
multiwalker arithmetic with clone retention. Not every metadata validation branch
has an individual negative test; the invalid-metadata fixture specifically
exercises the quantity label. These tests do not by themselves verify QE export,
full DMC branching trajectories, or nonlocal localization equivalence.

Driver tests cover parsing AUTO/YES/NO. Nexus tests cover field XML round-trip,
standalone default, explicit comparison, and true-without-field rejection.
The qmca test checks the derived quantity. Build registration uses the existing
Hamiltonian unit test executable and pytest modules; no new end-to-end CTest
integration-test directory was introduced by the committed change set.

The existing progress log reports that the focused ten-case MFR suite passed
468 assertions in each real/full, complex/full, real/mixed, and complex/mixed
configuration, plus additional Hamiltonian/Nexus/qmca tests. This is historical
evidence, not a fresh four-build rerun in this audit. Initial build failures
involving ``MinimalParticlePool.h`` were recorded before the later successful
precision-matrix work; do not interpret that older note as the final status.

Retained smoke runs in ``mfr_test_runs`` include LDA/PZ diamond and PBE helium,
legacy/batched paths, Jastrow/no-Jastrow, standalone/default/comparison, and setup
rejections. The current local analysis helper ``mfr_test_runs/mfr_energies.py``
is restricted to no-Jastrow data; interactive plotting and commented-out saving
reflect later local edits. It is not the authoritative general qmca interface.

Completed hBN VMC comparisons are recorded in the external ``01_vmc_summary.md``:
no-Jastrow standard-minus-MFR offsets approximately 4.95/3.41 mHa for Gamma/
off-Gamma; with Jastrow approximately 8.04/3.71 mHa. These offsets are statistically
resolved. The Jastrow comparison includes possible omitted nonlocal-Jastrow
contributions; it is not solely an interpolation consistency test.

Extension boundaries and recommended follow-on work
---------------------------------------------------

* New DFT codes can target the existing file contract for compatible local
  semilocal fields. Reader validation alone cannot establish the field's physical
  origin or agreement with the actual QMC determinant. Add cross-code gauge,
  occupation, ionic-operator, electron-count, and twist checks.
* The reader accepts commensurate supercell lattices for repeating the field,
  but does **not** scale ``TotalEnergyMF`` by a replication factor. General
  primitive-field-to-supercell use therefore needs additional energy/electron-
  count logic or a correctly matched file. Lattice acceptance alone is not a
  complete supercell implementation.
* The KS metadata are currently provenance only. No automatic validation checks
  that the occupied orbital subset and twist match the field's eigenvalue sum.
  Multiple-k averaged SCF constants are not interchangeable with a single-twist
  determinant constant.
* New functionals should expose operator classes explicitly. Hybrid Fock and
  projector terms require nonlocal action/localization residuals; generalized-KS
  meta-GGA can require differential operators. A scalar ``MFRPotential`` grid
  plus an eigenvalue sum is insufficient in these cases. Version the schema and
  update the estimator identity, mode validation, and tests together.
* A nonlocal-correlation functional may have a local functional derivative;
  assess its actual KS operator rather than treating every nonlocal energy
  functional as a nonlocal one-electron operator.
* Noncollinear/spin-orbit support requires matrix/spinor fields and mapping;
  the current two-group scalar-channel approach is collinear only.
* NLCC/pseudopotential consistency, generalized-overlap USPP/PAW treatments,
  environmental fields, and nonstandard electrostatics need a jointly defined
  producer/consumer contract. The current schema lacks sufficient provenance
  to infer all these choices automatically.
* Add unequal-grid/skew-lattice tests, supercell-constant tests, KS-occupation
  mismatch rejection, broader metadata-negative cases, and deterministic
  VMC/DMC integration checks before expanding the support claim. Assess spline
  and nonlocal-quadrature convergence against the no-Jastrow residual.

For a baseline diff and targeted source search::

   git diff 86e852d4f 72ab2e527 -- src/QMCHamiltonians src/QMCDrivers nexus/nexus
   rg -n 'MFRPotential|KineticJastrowResidual|mfr_qmc|LocalEnergyMFR' src nexus/nexus

This audit changed documentation only. It did not modify operators, rebuild
executables, resubmit jobs, or rerun statistical calculations.
