MFR Implementation Progress
===========================

2026-09-28: Contract and source integration
-------------------------------------------

* Corrected the implementation contract: ``MFRPotential`` is the positive
  :math:`\sum_i V_{ee}^{\mathrm{MF}}(\mathbf r_i,\sigma_i)` in Hartree.
  The negative sign belongs only in the derived ``LocalEnergyMFR`` arithmetic.
* Inspected the existing QE exporter, QMCPACK external-potential and MPC spline
  implementations, Hamiltonian registration, Nexus XML model/generator, and
  ``qmca`` scalar-data analysis.
* Added ``QMCHamiltonians/MFRPotential.{h,cpp}``.  The initial implementation
  validates schema version 1, reads the QE lattice and one/two spin channels,
  converts Rydberg to Hartree, transposes the QE HDF5 grid ordering into
  einspline ordering, constructs periodic 3D cubic B-splines, and evaluates the
  positive per-configuration potential.
* Registered ``type="mfr"`` as an auxiliary QMCPACK operator named
  ``MFRPotential``.  It will be written to scalar output without contributing
  to QMCPACK's physical ``LocalEnergy``.
* Added the Nexus ``extpot``/MFR XML model and the ``mfr_potential`` argument to
  ``generate_qmcpack_input``.  The generator validates/relativizes the path and
  emits a positive-scale MFR element without changing physical interactions or
  correction selection.
* The old ``build_real`` CMake cache referred to a removed CMake installation
  and obsolete options.  Configured a clean isolated ``build_mfr_real`` tree;
  its first compilation is in progress.

Open arithmetic dependency
--------------------------

The agreed MFR expression needs ``TotalEnergyMF`` and
``KineticJastrowResidual`` in addition to ``ElecElec`` and ``MFRPotential``.
The current scalar file contains total ``Kinetic`` but not the determinant-only
kinetic term or its Jastrow residual, and the current QE MFR HDF5 schema does
not contain ``TotalEnergyMF``.  The C++ reader/Nexus work is independent of
this issue.  The ``qmca`` implementation must not silently substitute total
``Kinetic`` for ``KineticJastrowResidual``; the data source for these two
quantities needs to be made explicit before finalizing that calculation.

2026-09-28: QE data production and Nexus validation
---------------------------------------------------

* Rebuilt both QE executables requested for this work: ``pw.x`` and
  ``pw2qmcpack.x``.
* Created ``mfr_test_runs/scf_mfr`` and retained ``scf.in``, ``scf.out``,
  ``p2q.in``, ``p2q.out``, ``C.BFD.upf``, the complete ``pwscf.save`` tree,
  ``pwscf.mfr.h5``, and ``pwscf.pwscf.h5``.
* The LDA SCF completed normally.  The MFR file contains one float64 spin
  channel on a 45 x 45 x 45 periodic grid and the expected version-1 metadata.
* The full pre-existing Nexus ``test_qmcpack_input.py`` suite passed after the
  generator change (15 tests at that point).  Added a focused MFR test covering
  positive scale, path relativization, retained physical potentials, explicit
  MPC exclusion, and missing-file validation; it passes.
* Added ``mfr_test_runs/mfr_vmc.py``.  It is generate-only by default, supports
  environment overrides for executable/resources/run length, and generates
  longer Jastrow and no-Jastrow legacy VMC workflows.  Generated XML was
  checked to reference ``../scf_mfr/pwscf_output`` for both orbital and MFR
  HDF5 files, retain ElecElec/IonIon/PseudoPot, and omit MPC.
* A fresh ``build_mfr_real`` was needed because the old build cache referenced
  a removed CMake installation and obsolete QMCPACK options.  The new build
  reached and compiled ``MFRPotential.cpp`` after one current-API lattice alias
  correction; the overall build is continuing.

2026-09-28: QMCPACK build and deterministic smoke runs
------------------------------------------------------

* Corrected handling of QE's ``grid_dimensions`` metadata: qeh5 stores the
  three integers as an HDF5 array datatype in a scalar attribute, rather than
  as a rank-one attribute dataspace.
* The real-valued ``qmcpack`` executable builds successfully with the new
  operator.
* The three-block Jastrow and no-Jastrow VMC smoke runs both completed normally
  in their dedicated ``mfr_test_runs`` subdirectories.  Each log reports the
  45 x 45 x 45 grid, one spin channel, and the exact 0.5 Ry-to-Ha conversion.
  Each scalar file contains finite ``LocalEnergy`` and ``MFRPotential`` values.
* Ran an additional same-seed three-block Jastrow control without the MFR XML
  element.  Every standard scalar energy column, including ``LocalEnergy``, is
  bit-for-bit identical to the corresponding MFR-observable run; the maximum
  absolute difference is 0.0.  This confirms auxiliary registration does not
  modify the physical Hamiltonian.
* The complete Nexus qmcpack-input suite now passes: 16 tests, including the
  new MFR-specific generator test.
* The isolated complex-valued ``build_mfr_comp`` configuration also builds
  ``qmcpack_complex`` successfully.

2026-09-28: Validation completion and qmca boundary
---------------------------------------------------

* Added validation of the QE grid-order metadata, nonempty functional metadata,
  full periodicity, and integer commensurability between the QMCPACK simulation
  cell and the QE field lattice.  Equivalent integer supercells are accepted;
  unrelated cells are rejected.
* Rebuilt the real-valued executable and repeated the short Jastrow VMC in the
  isolated ``diamond_vmc_jastrow_validation`` directory.  It completed normally,
  logged the PZ functional and 45 x 45 x 45 grid, and wrote finite
  ``MFRPotential`` samples.
* A two-rank MPI Jastrow smoke run completed successfully and wrote finite MFR
  samples, providing a concurrency/clone-path check in addition to the serial
  Jastrow and no-Jastrow runs.
* The rebuilt real-valued core Hamiltonian unit suite passed (one test, zero
  failures), and the final complex-valued executable build completed.
* Added qmca abbreviations/display ordering for ``MFRPotential`` and the future
  ``LocalEnergyMFR`` field.  The full qmca test module passes (14 tests); its
  focused case confirms that ``-q mfr`` reads and statistically processes an
  MFR scalar column.  The derived energy is still gated by the missing inputs
  below.
* The implementation plan was corrected so it no longer describes
  ``LocalEnergyMFR`` as derivable from two existing scalar columns.  The exact
  agreed arithmetic requires ``TotalEnergyMF`` and
  ``KineticJastrowResidual``; neither is presently available through the
  stipulated interfaces.  The derived qmca calculation remains intentionally
  unimplemented pending a decision about those data sources.

2026-09-28: Schema-v2 energy contract and completed MFR arithmetic
-----------------------------------------------------------------

* Extended the QE exporter to schema version 2.  ``total_energy_mf`` is now
  explicitly defined as the occupied KS eigenvalue sum plus ion--ion energy,
  not the DFT functional total.  The latter is retained separately as
  ``dft_total_energy`` provenance.
* Added ``/mfr/ks/eigenvalues``, ``occupations``, ``kpoints``,
  ``kpoint_weights``, and ``spin_index``.  QE writes the full band table and
  occupation weights so occupied orbitals and their k-point/spin mapping are
  recoverable without an ambiguous ragged dataset.
* Moved the QE export to the converged final-energy path and rebuilt both
  ``pw.x`` and ``pw2qmcpack.x``.  The retained final SCF output is
  ``scf_mfr/scf_mfr_final.out``.  Its HDF5 file has
  ``TotalEnergyMF=-19.93700378007214 Ry``,
  ``ks_eigenvalue_sum=5.614330672766295 Ry``, and
  ``ion_ion_energy=-25.551334452838432 Ry``; the identity closes exactly.
* Added QMCPACK ``KineticJastrowResidual``.  It reevaluates fermionic and
  nonfermionic wavefunction derivatives separately and records
  ``T[DJ]/(DJ)-T[D]/D`` including their gradient cross term.  It is an
  auxiliary observable and returns exactly zero for a determinant-only trial
  wavefunction.
* The MFR reader now requires schema 2, validates the exact TotalEnergyMF
  definition, converts it from Ry to Ha, and writes ``MFRPotential`` and
  ``TotalEnergyMF``.  Hamiltonian construction also adds the residual scalar.
* Implemented qmca arithmetic after unit conversion:
  ``LocalEnergyMFR = TotalEnergyMF + KineticJastrowResidual + ElecElec -
  MFRPotential``.  The original ``LocalEnergy`` remains unchanged.
* Rebuilt real and complex QMCPACK successfully.  The deterministic core
  Hamiltonian test passed.  The full qmca module passed 14 tests and the full
  qmcpack-input module passed 16 tests.
* Final retained serial smoke runs are in
  ``diamond_vmc_mfr_energy_final_jastrow`` and
  ``diamond_vmc_mfr_energy_final_no_jastrow``.  Both completed normally.
  qmca reports respectively ``LocalEnergy=-10.44(13)`` versus
  ``LocalEnergyMFR=-10.49(12)`` and ``LocalEnergy=-10.07(31)`` versus
  ``LocalEnergyMFR=-10.04(31)``.  These are smoke statistics only.
  The no-Jastrow residual is exactly zero in every block.
* A two-rank QE SCF/export check also completed normally and is retained as
  ``scf_mfr/scf_mfr_mpi2.out`` with ``pwscf.mfr.mpi2.h5``.  It produced the
  same schema, occupations, and converged field; expected parallel reduction
  differences were about ``5.1e-5 Ry`` in the grid and ``6.7e-6 Ry`` in KS
  eigenvalues.  ``pwscf.mfr.h5`` was restored to the serial-final file used by
  the reported QMC smoke runs and by ``mfr_vmc.py``.

2026-09-28: Nexus driver style cleanup
----------------------------------------

* Restyled ``mfr_test_runs/mfr_vmc.py`` to match the repository Nexus
  examples: aligned keyword arguments, lowercase local settings, direct
  ``generate_qmcpack`` calls, and ``run_project()`` registration behavior.
* Removed environment-variable and ``pathlib`` configuration.  All paths are
  relative strings, and run/resource controls are ordinary local variables.
* Python compilation and a generate-only Nexus execution both completed
  successfully.  The generated inputs retain relative ``../scf_mfr`` orbital
  and MFR HDF5 references, the pseudopotential, VMC-only calculations, and no
  MPC term.

2026-09-28: Standalone MFR energy analysis
-------------------------------------------

* Added ``mfr_test_runs/mfr_energies.py``.  It reads a no-Jastrow
  ``scalar.dat`` file and uses the same ``nexus.numerics.simstats`` routine as
  qmca to report ``LocalEnergy``, ``LocalEnergyMFR``, and the blockwise
  ``TotalEnergyMF_stat = LocalEnergy - ElecElec + MFRPotential``.
* It reports the stored constant ``TotalEnergyMF`` without a statistical
  uncertainty, supports optional initial-block exclusion with ``-e``, checks
  required columns and constancy, and rejects Jastrow data.
* Its standard and MFR means/error bars agree with qmca on the retained
  no-Jastrow smoke output.

2026-09-28: bcc-helium batched MFR test
------------------------------------------

* Added and ran a two-atom conventional bcc He cell with ``a=5 bohr`` under
  ``mfr_test_runs/he_scf_mfr`` using the supplied PBE ccECP pseudopotential,
  a 200 Ry cutoff, and explicit Gamma k-point.
* The first QE gamma-trick conversion was retained as a diagnostic;
  ``pw2qmcpack`` requires an explicit zero k-point.  The explicit-Gamma SCF
  converged in seven iterations, exported a 45 cubed schema-v2 field, and the
  subsequent orbital conversion completed normally.
* The QE file reports ``TotalEnergyMF=-10.029403582206756 Ry`` for the final
  explicit-Gamma calculation and its KS-sum plus ion-ion identity closes
  exactly.
* Added ``mfr_test_runs/he_mfr_vmc.py`` and ran its single Jastrow VMC in
  ``he_mfr_runs/he_vmc_jastrow``.  It uses the batched driver, four walkers,
  double-precision orbital splines, meshfactor 2, and a nonzero RPA-generated
  two-body Jastrow.  MPC is absent.
* The 100-block run completed successfully and directly validates the batched
  MFR operator path.  After ten equilibration blocks qmca reports
  ``LocalEnergy=-5.957(63) Ha`` and ``LocalEnergyMFR=-5.878(13) Ha``.
  ``KineticJastrowResidual=0.131(20) Ha`` is nonzero as intended.  The local
  He XML pseudopotential produces no ``NonLocalECP`` scalar column.

2026-09-28: MFR energy trace plots
-----------------------------------

* Updated ``mfr_test_runs/mfr_energies.py`` to produce two PNG trace plots
  by default.  One overlays ``LocalEnergy`` and ``LocalEnergyMFR``; the other
  overlays ``TotalEnergyMF_stat`` and the noiseless ``TotalEnergyMF``.
* Plot files are written beside the scalar input with
  ``.local_energy_trace.png`` and ``.total_energy_mf_trace.png`` suffixes.
  The existing stdout statistics are unchanged, ``-e`` applies to both
  statistics and traces, and ``--no-plots`` disables image output.

2026-09-28: Standalone MFR-QMC driver mode
------------------------------------------

* Added the tri-state ``mfr_qmc`` request (auto/yes/no) and the resolved
  Hamiltonian enum states standard, standard-with-MFR-observables, and
  standalone MFR-QMC.  An MFR potential with no explicit driver parameter
  selects standalone mode; ``mfr_qmc=no`` preserves comparison mode.
* Standalone setup requires an MFR potential and accepts only ``Kinetic`` and
  the electronic Coulomb ``ElecElec`` as physical Hamiltonian operators.
  Pseudopotential/electron-ion, ion-ion, MPC, and other physical terms are
  rejected before walker cloning.  ``mfr_qmc=yes`` without the MFR potential
  is also rejected.
* The active walker energy in standalone mode is
  ``TotalEnergyMF + KineticJastrowResidual + ElecElec - MFRPotential`` in both
  scalar/legacy and batched/multi-walker evaluation paths.  Hamiltonian
  clones retain the resolved mode.
* Disabled conventional combined-energy traces in standalone mode and reject
  per-particle LocalEnergy listeners because their standard operator
  decomposition cannot represent the MFR constant and residual safely.
* Nexus now accepts ``mfr_qmc`` for VMC/DMC driver elements and as a
  ``generate_qmcpack_input`` argument.  Its default standalone generation
  emits only electronic Coulomb plus the MFR external potential and removes
  MPC; explicit false retains the conventional ionic/pseudopotential
  Hamiltonian for comparison.  ``he_mfr_vmc.py`` explicitly selects
  standalone mode and now generates a Jastrow wavefunction.
* ``build_mfr_real`` completed successfully.  The legacy-driver, batched-driver,
  Nexus qmcpack-input, and qmca test suites all pass.  Nexus regression coverage
  includes standalone default generation, explicit comparison generation, and
  rejection of true without a potential.
* Retained short helium runs under ``mfr_test_runs/he_mfr_runs`` cover
  batched determinant-only, batched Slater--Jastrow, legacy Slater--Jastrow,
  omitted-parameter auto mode, explicit comparison mode, incompatible ionic
  terms, and a missing MFR potential.  Both valid driver paths reproduce the
  component MFR expression within 5e-11 Ha at scalar-file precision; the
  Slater--Jastrow residual is nonzero.

* A direct ``qmca -q 'e emfr'`` check on the standalone Jastrow scalar
  output reports identical ``LocalEnergy`` and ``LocalEnergyMFR`` values,
  ``-5.723 +/- 0.024 Ha`` for this two-block smoke sample.
* The changed Hamiltonian sources also compiled successfully into the complex
  ``qmcham`` library.  The pre-existing full ``build_mfr_comp`` tree could not
  complete because its integration-test target lacks ``MinimalParticlePool.h``;
  this failure is outside the MFR sources and occurred after ``qmcham`` linked.
\n2026-10-01: deterministic MFR coverage and precision matrix\n----------------------------------------------------------\n\n* Changed ``MFRPotential`` interpolation return and ``TotalEnergyMF`` storage to\n  ``FullPrecRealType`` so mixed-precision builds retain the double-precision QE\n  field and total-energy data.\n* Added ``test_MFRPotential.cpp`` to the deterministic Hamiltonian unit target.\n  Its generated 4 x 4 x 4 HDF5 fixtures cover one- and two-spin fields, periodic\n  nonconstant interpolation, cloning, unit conversion, invalid metadata, missing\n  paths, and incompatible spin grouping.\n* Added analytic ``KineticJastrowResidual`` checks for determinant-only,\n  Slater--Jastrow cross-term, and invalid nonfermionic-only wavefunctions.\n* Added mode-resolution and rejection coverage for electron-ion, ion-ion,\n  pseudopotential, MPC, missing Coulomb, and missing residual operators.\n* Added exact standalone arithmetic checks for both scalar and multiwalker\n  evaluation, plus comparison-mode energy invariance and clone-mode retention.\n* Configured fresh real/mixed and complex/mixed build trees and enabled unit\n  tests in the complex/full tree.  The final MFR suite passes in real/full,\n  complex/full, real/mixed, and complex/mixed builds: 468 assertions in 10 test\n  cases in each configuration.  The complete real/full Hamiltonian unit test,\n  focused Nexus MFR generator test, and focused qmca LocalEnergyMFR test pass.\n
2026-11-03 17:40 EST -- batched determinant log-Laplacian precision
=================================================================

A separate definite cancellation defect was reproduced and fixed in
src/QMCWaveFunctions/Fermion/DiracDeterminantBatched.cpp:602--613.
computeGL previously formed lap-dot(grad,grad) in the SPO value precision,
then added it to double particle log derivatives. FullPrecGrad/FullPrecValue
now hold the intermediates before this subtraction, matching the legacy
path. SPO storage and dot accumulation remain unchanged; cached Jastrow
residual evaluation still avoids rebuilding the determinant.

src/QMCWaveFunctions/tests/test_DiracDeterminantBatched.cpp adds the
[gl_precision] test using the existing ConstantSPOSet fixture. It supplies
one-orbital Taylor jets with phi=1, Lap(phi)=1 and large exactly representable
real/complex gradients. The inverse is perfectly conditioned and kinetic
energy is analytically-1/2. Both unchanged full builds pass27 assertions;
old mixed-real/mixed-complex fail6/9 assertions. After the production fix,
all four pass27. Initial test-engine and executable-location harness mistakes
are retained; corrected invocations use the existing host OMPTARGET engine.

All four CPU applications rebuild. Full determinant suites pass532/938
assertions in real/complex builds (both precisions). MFR Hamiltonian suites
pass16321/16226 assertions, and full driver suites pass478/678/472/672 for
real/complex/mixed-real/mixed-complex. These include existing mode, cache,
buffer and restart checks; they are not GPU acceptance.

Matched24-case integration inputs retain their hashes. Full-precision and
standalone MFR paths/energies are preserved at the recorded checks. Those
inputs use legacy determinants even with batched drivers, so they do not
exercise the corrected computeGL path. Six additional application cases
explicitly enable batched determinants and pass arithmetic/MFR checks.
One frozen-before mixed batched comparison has identical VMC coordinates;
the fix changes conventional kinetic energy by up to2.29e-7Ha, with zero
MFR field/residual change. Before binaries and failed analytic tests are
preserved in sol/lifecycle/batched_gl_precision_20261103.

This verified cancellation fix did NOT remove the larger mixed spline/
kinetic-reference discrepancy exposed by the legacy-determinant matrix.
The earlier attribution of that entire discrepancy to this defect was too
broad. That failed check remains open, along with mixed weighted-scalar
reporting and all broader native meta, physical, lifecycle and performance
requirements. No Goal completion or changed physical acceptance criterion.

2026-11-03 17:40 EST -- native complex TPSS handoff and open diagnostics
======================================================================

The current RMG executable reproduces the off-Gamma He TPSS failure with
Pulay mixing0.5. Mixing0.1 stays bounded and reaches native energy stopping;
linear0.1 stays bounded but does not converge in40 steps. Two subsequent
strict-RMS1e-8 runs exhaust their120/200 iteration limits without publishing
MFR output. A multigrid control fails. These failed strict studies remain
failed; charge/tau mixing consistency and native complex convergence remain
open. No SCF/operator regularization or tolerance change was made in RMG.

A separate import fixture uses native energy criterion1e-12, more stringent
than the original1e-9. It completes at9.85e-13 with potential RMS4.00e-6,
so it is explicitly an energy-converged format/import fixture, not a passed
strict-RMS reference. It exports common binary64 real-space u/q fields on
64^3 grids and a companion core file. All4 states pair exactly with ES-HDF;
independent reconstruction verifies the exported bare-core action. The
native source is complex, unpolarized TPSS with one twist and a local XML
He pseudopotential. Native spin meta remains guarded.

The corrected-QMC application matrix contains24 cases: full/mixed complex,
legacy/batched drivers, no-J/J2, comparison and standalone core/meta modes,
with VMC followed by DMC. Six additional J2 cases explicitly set determinant
batch=yes, distinguishing batched determinants from batched drivers. Both
sets pass raw arithmetic and the unchanged independent MFR field/action
trace limits. There are2496+624 observations; maximum assembly error is
1.78e-15Ha. The largest independent MFR trace error is1.74e-6Ha. Full build
trace limits remain1e-8*(1+scale), mixed2e-5*(1+scale). J2 coefficients are
frozen smoke fixtures; no statistical or optimized-wavefunction claim.

The same independent checker also records two FAILED diagnostics. Mixed
conventional determinant kinetic energy differs from a double-arithmetic
cubic reference by up to2.10e-4Ha, exceeding the attempted fixed bound.
Mixed builds force float orbital coefficients despite precision=double;
modeling coefficient quantization alone does not explain this discrepancy.
Float kernel arithmetic/coordinate transforms and derivative accumulation
need further isolation. Full trace comparisons pass. No failed mixed kinetic
check is reclassified as passed, and no numerical bound was loosened.

In mixed batched DMC, the conventional scalar IonIon differs from the stable
raw-walker inferred constant by up to3.26e-7Ha; the original5e-10 scalar/raw
check remains flagged failed. Raw TotalEnergyMF and IonIonMF are constant
and match the HDF metadata/selected occupied sum at the checked bounds.
Weighted scalar reporting remains a separate open diagnostic. The checker
first assumed IonIon was logged per walker; it is actually omitted there
and retained in scalar.dat. The failed harness attempts are preserved.

Physical accuracy remains OPEN. Unshifted standard-minus-core RMS ranges
0.0387--0.1571Ha; meta-minus-core RMS0.0080--0.0123Ha over the comparison
series. No offset fit, clipping, physical acceptance or long-statistics
claim was made. Coarse action/orbital/field resolution, native residuals
and near-node behavior require further convergence tests.

Records: runs/functional_extension/rmg/complex_meta_*_20261103 and QMCPACK
mfr_test_runs/functional_extension/rmg_complex_meta_{pipeline,
gl_precision_fixed,batched_determinants}_20261103. Current numerical reports
are qmc_validation.json, qmc_validation.batched_determinants.json and
summary.json in sol/rmg_meta/complex_native_pipeline_20261103.

2026-11-03 18:16 EST -- scalar weight normalization correction
=============================================================

The mixed batched-DMC scalar constant drift is now explained and corrected.
ScalarEstimatorBase/LocalEnergyEstimator accumulate weighted sums in double,
but EstimatorManagerCrowd held the total weight in QMCTraits::RealType
(float in mixed builds), and EstimatorManagerNew narrowed it again in its
PropertyCache before normalization. The old raw weight sums versus printed
weights independently predict both the constant drift and varying-energy
mean error to output precision; native_before_prediction.json preserves
that calculation and the original source-file hashes.

src/Estimators/EstimatorManagerCrowd.h now stores and returns block_weight_
in FullPrecRealType. src/Estimators/EstimatorManagerNew.h uses
Vector<FullPrecRealType> for PropertyCache, retaining denominator precision
through crowd collection, MPI reduction and division. Estimator sums,
normalization formula, observable names, input syntax and output format are
unchanged. The change applies to every scalar column, including energy
second moments and TotalEnergyMF/IonIonMF; it adds no MFR special case.

The new [scalar_weight_precision] case in tests/test_EstimatorManagerCrowd.cpp
uses actual crowd accumulation, main-estimator collection, normalization
and output. Two blocks, uneven crowds, fractional weights, constant energy
and a varying potential are checked against an independent long-double
reference. A separate exact-weight-argument section isolates the manager
cache. The old code passes all32 assertions in full real/complex and fails20
in each mixed build. The fixed code passes32 in all four. The registered
focused two-rank CTest passes in all four; MPI rank0 has32 assertions and
rank1 has16. A CTest resource lock serializes estimator tests sharing files.

All four applications rebuild. Full estimator suites pass1523 assertions/
107 cases in real and1293/114 in complex, at both precisions. Driver suites
pass478/678/472/672 assertions for real/complex/mixed-real/mixed-complex.
The existing three-rank estimator suite is rebuilding its previously unbuilt
dependencies; its result is pending at this timestamp and is not claimed.

Nine fresh corrected application cases cover real/complex full/mixed VMC
and DMC, conventional comparison and standalone meta mode, plus actual
MPI2/crowds2. All934 raw observations and metadata constants are finite.
Independently weighted walker records reproduce every checked scalar column,
including LocalEnergy_sq, within the printed precision; maximum column error
is4.88e-10. All constant-column errors are <=3.15e-11Ha under the original
5e-10 check. A frozen-before executable/control reproduces constant drift
up to5.69e-7Ha. No shift, fit, or tolerance relaxation was used to accept the
corrected records. Inputs, outputs, dependency identities and binary hashes
are retained in qmcpack_dev_new5/mfr_test_runs/functional_extension/
scalar_weight_precision_20261103 and this checkpoint.

This closes the identified scalar-normalization cause from the17:40 meta
checkpoint for the checked scope. It does not close the separate mixed
spline/kinetic-reference discrepancy, native meta convergence/physical
accuracy, spin-meta implementation, or the full functional/workflow/state/
performance acceptance matrix. Goal and F1--F18 scope remain active.

2026-11-03 18:22 EST -- final scalar normalization review
--------------------------------------------------------

The existing three-rank estimator suites now PASS in all four builds,
superseding the18:16 pending-build statement. All focused unit/MPI checks,
full estimator suites, full driver suites and the nine corrected native
application cases have completed. Source diff whitespace checks pass.

For the completed scope, the mixed weighted-scalar discrepancy is CLOSED:
full-precision sums and normalization now use consistent weights. The
frozen-before control and failed20/32 mixed assertions remain unchanged
on disk. The independent reconstruction covers constant and varying scalar
columns, second moments, actual MPI2/crowds2, and metadata constants. This
is a producer-independent QMCPACK correction; no QE/RMG source or common
export schema changed for this fix.

The separate mixed spline/kinetic-reference diagnostic, native spin-meta
implementation, strict complex meta SCF convergence, physical accuracy and
remaining functional/state/workflow/performance matrix remain OPEN. The
read-only spin-meta inventory and exploratory backend probe provide the
next bounded implementation foundation; backend discrepancies are not
reclassified as accepted. The full Goal stays active and unchanged.


2026-11-03 18:35 EST -- native polarized meta fields (partial G4)
---------------------------------------------------------------------------

Implemented canonical XClib/rank-local spin mapping, correctly sized Cartesian
correlation flux, per-spin kinetic density exchange, native divergence and
GetTe spin reduction. The unpolarized branch is unchanged; frozen complex
restart controls have byte-identical payloads/diagnostics. New independent
manufactured-functional test passes256 checks and rejects three mutations.
All15 focused RMG operator/field/MPI regressions pass.

Cold-start complex TPSS refinement1 FAILS with nonfinite energy/diagonalization.
PBE-seeded balanced-spin TPSS refinement2 reaches the native energy stop
(8.62e-13Ha) but final RMS3.49e-6 FAILS the unchanged1e-8 strict criterion.
These are native smoke results only. Main.cpp spin-meta export guard remains
in place. Spin-resolved exported field/action oracles and QMC hand-off,
unequal/fully polarized channels and strict SCF/physical acceptance remain open.
Earlier backend derivative/limit failures remain open, without weakened bounds.

Detailed source snapshots, diffs, frozen executables, logs and review:
functional_extension_validation/sol/rmg_meta/spin_native_20261103/.
All native inputs/outputs: runs/functional_extension/rmg/
spin_meta_native_probes_20261103/. Goal scope/status unchanged: active.


2026-11-03 18:45 EST -- spin-meta exports and two-spin QMC hand-off (partial G4-G6)
------------------------------------------------------------------------------

Main.cpp now permits implemented collinear two-spin meta export, superseding
the18:35 staging-guard statement; unsupported AFM/scalar/meta-U/meta-hybrid
remain rejected. Balanced2/2 and unequal3/1 complex native TPSS exports
pass common schema/state pairing and all28 independent bare-core action
oracles (max7.31e-14). Both native energy criteria pass but strict RMS1e-8
FAILS (4.28e-6 and4.17e-5), so native physical acceptance remains open.

All40 full/mixed complex noJ/J2 legacy/batched VMC/DMC integration cases
finish, with4160 finite raw observations and correct MF constants/provider
arithmetic. Balanced independent MFR traces pass. Two unequal mixed legacy
VMC differential checks FAIL the unchanged2e-5*(1+scale) bound: maxima
2.81801e-5(noJ)/2.80928e-5Ha(J2). Full complex passes; mixed conventional
kinetic strict checks also remain failed(16 series total). Original failed
validator/log and separate all-series diagnostics are retained; no tolerance
was loosened. Native float spline/phase/derivative evaluation needs isolation.

Actual wrong-spin and wrong-reference QMC inputs reject before scalar
output. Physical pointwise residuals remain open; no shifts/clipping or
statistical acceptance. Full scope unchanged, Goal active. Detailed review:
functional_extension_validation/sol/rmg_meta/spin_export_20261103/review.rst.


2026-11-03 18:58 EST -- mixed spline representation discrepancy explained
---------------------------------------------------------------------------

The two unequal mixed meta differential failures were comparison-representation
differences, not evidence of wrong spin/q contraction. Native phase rotation
precedes float coefficient rounding; the earlier independent oracle rounded
unrotated coefficients and evaluated in double. Actual save_coefs replays
have bit-identical raw trajectories/properties. Independent phase/prefilter
reconstruction matches>99.99995% of checked native float coefficients exactly.

All40 balanced/unequal full/mixed noJ/J2 legacy/batched VMC/DMC cases
(4160 observations) pass matched-representation field/core/kinetic checks at
unchanged1e-8/full and2e-5/mixed bounds. Reconstructed tables plus the native
float kernel and independent matrix contraction give maxQ error3.81e-8Ha
and maxkinetic1.023e-6Ha. Original failed double-reference reports remain
retained and explicitly distinguished; no tolerance was loosened. No
production code changed. Strict SCF, backend derivatives and physical grid
accuracy remain open; full Goal remains active.

Correction: both preceding spin exports contain7 bands/spin (14 actions each,
28 total), not4/16. Actual oracle arrays always covered all stored bands.
Reports and plans corrected, with original prose preserved.

Next cost candidate: accepted determinant G/L reuse for the meta q contraction
using Q_L=-0.5*sum[q*(L_D+G_D.G_D)+grad(q).G_D], with G_D/L_D obtained by
subtracting nonfermion log derivatives from current accepted P.G/P.L. Keep
state/reference checks and prove precision/state/move equivalence and timing
against immutable current baselines before acceptance (existing G7/F16-F18).

Evidence: functional_extension_validation/sol/lifecycle/
native_float_spline_reference_20261103/review.rst.


2026-11-03 19:24 EST -- accepted meta derivative reuse (bounded G7/F16-F18 contribution)
------------------------------------------------------------------------------

MFRMetaFields now contracts q with accepted determinant log derivatives,
obtained from total P.G/P.L minus nonfermion cached GL. It removes repeated
SPO VGL, inverse copying and quadratic action scratch, using per-clone O(N)
scratch and a fine timer. Formula: Q_L=-0.5*sum[q*(L_D+G_D.G_D)+grad(q).G_D].
Reference/state checks and scalar/input formats remain unchanged.

All4 app builds, affected MFR suites(real16833/complex16738 assertions) and
full driver suites PASS. Tight independent cofactor/native-VGL tests cover
J/non-J, complex/skew/boundary, both determinants, first/repeated nonmutation
and existing moves/clones/buffers. Native49 candidate contexts (real/Gamma,
balanced/unequal complex, VMC/DMC, MPI2/crowds2) preserve accepted R/G/L
bit-for-bit across5197 observations; original precision bounds pass.

Three interleaved original-baseline timing pairs/target show coarse VMC
medians1.137x(He complex),1.004x(He mixed),1.011x(C real),1.041x(C complex).
Mixed He/real carbon are near noise. Matched fine-instrumented evaluator
trials demonstrate2.22--2.88x meta-evaluator speedup. Both benchmark forms
preserve final configs and physical scalars. Persistent linear full-precision
scratch can exceed prior payload for tiny mixed cases; no universal RSS
improvement or whole-run speedup is claimed.

Detailed source/code/validation/performance records:
functional_extension_validation/sol/cost/meta_accepted_gl_20261103/review.rst.
The full Goal remains active. Strict SCF, physical convergence, remaining
functional/state/workflow and full cost-matrix requirements remain open.


2026-11-03 19:48 EST -- native96 meta physical-validation failures (G4/F5/F7-F9)
------------------------------------------------------------------------------

Fresh PBE96 two-grid and single-grid seeds pass strict RMS1e-8
(7.80e-9/8.12e-9). All3 TPSS96 FD12 attempts fail dsyevd(exit137)
without MFR output: two-grid mixing0.5/0.1 and single-grid0.1.
Negative fine tau is observed in the two-grid path; single-grid removes
that interpolation issue but still exhibits tauW/tau~60 and divergent q.
No96-grid QMC was launched; the strict-reference harness rejects it.

Independent64-grid FD12 equation control passes occupied RMS4.78e-9.
New analytic uniform-limit probe FAILS: actual XClibs nspin1/2 TPSS
exchange returns0 instead of-.08638235734 at rho=.2. Direct metax energy
matches analytic LDA, but limiting derivatives signal floating exceptions,
so deleting guards is not a fix. Unpolarized C++ additionally discards
small-gradient full fields where the backend returns finite nonzero energy.

Source audit finds charge-only SCF mixing with tau recomputed from current
orbitals, and ordinary non-positive-preserving KED prolongation. Plan
follow-up: analytic finite functional limits; matched rho/tau input/output
state and mixing history; variational fine-grid meta construction/adjoint
transfer; meta-field/operator convergence safeguards; rerun the preserved
controls before physical QMC acceptance. No clipping/tolerance relaxation.
Also audit finer physical bare-core action export independently, starting
from the local-He control and preserving common format/state semantics.

Detailed source anchors, terminal runs, independent probes and follow-up:
functional_extension_validation/sol/rmg_meta/grid96_convergence_20261103/
review.rst. Goal remains active with all F1-F18 requirements unchanged.

2026-11-04 15:59 EST: committed preceding MFR meta/lifecycle checkpoint 44f22a6aa63add9b5538a70bdd051bf148b34225; cache correctness checkpoint follows with four-build and16-context/4104-observation evidence. Controlled performance and physical meta acceptance remain open. Future accepted milestones include local qp_mfr commits.


2026-11-04 bounded action-import milestone: verified startup benefit
--------------------------------------------------------------------

Source/cache correctness is committed in fd61a3077, following the broader
paired-meta/lifecycle checkpoint44f22a6aa. All four CPU variants passed focused
MFR tests. Sixteen real/complex, full/mixed, legacy/batched moving VMC/DMC
before/after contexts matched4104 raw walker observations and every scalar
column except BlockCPU exactly; five actual QE-meta invalid-input controls
rejected the expected ionic/provider combinations.

Three interleaved repeats per immutable binary on one explicitly bound CPU
now verify the large Ti1200 meshfactor2 combined core/meta comparison import.
Median startup falls from111.1139 to92.6679 seconds (1.19905x); paired speedups
are1.17024x,1.32853x,1.18351x. Baseline range108.4433--120.2102s and candidate
range90.4836--93.8849s are retained. Median peak RSS falls by219.0703 MiB,
6.1711%. All1241 half-second rank-affinity samples show CPU8; the monitor uses
CPU19. Inputs are pre-warmed and hash-identical. Each paired raw walker array
and all non-timing scalar columns match exactly. The first launcher-bound
series is excluded after an observed affinity violation and remains archived.

This accepts a bounded warm-cache startup/memory benefit, not a per-sample,
production-efficiency, physical meta-accuracy or complete F18 claim. The host
was not exclusively reserved, and repeat spread is reported. No further
startup timing expansion is planned for this checkpoint. The remaining
physical/functional/state/workflow criteria stay open. Evidence, reproduction
scripts, resource logs and immutable binary/source hashes are under
/home/j1k/projects/mstd/34_mfrqmc/work/05_hubbard_u_impl/functional_extension_validation/
sol/cost/action_import_reuse_20261104/{application_matrix,startup_benchmark_rankpin_v2}/.


2026-11-04 16:30 EST -- fresh native Nexus QE-meta workflow milestone
---------------------------------------------------------------------

A fresh local Nexus graph scheduled high-cutoff two-atom He r2SCAN SCF,
pw2qmcpack and paired pw2mfr_core conversion, followed by batched comparison
VMC with double orbitals and meshfactor2. The600/9600 Ry SCF strictly
converged in77 iterations,8.4e-14 Ry against unchanged1e-12 Ry target.
Explicit Gamma k points avoid the optimized-gamma exporter guard. Prefix/
outdir, native flags, actual result paths and both MFR dependencies bind
the generated QMC input to the fresh converter outputs. All four native
programs completed. Schema/state identity and core/meta coefficients pass;
comparison mode retains the physical local ECP and ion-ion terms.

The actual qmca CLI emits LocalEnergy, LocalEnergyMFR, LocalEnergyMFRMeta
and MetaMinusAction. The DatAnalyzer API's derived arrays match independent
per-block arithmetic exactly, and its mean/variance/error/kappa match direct
simstats calls. This is a two-block software/dependency/statistics regression,
not useful physical statistics or acceptance of the economical meta estimator.
Earlier preview CLI-option failure is preserved; no native job was submitted
by that preview. The corrected twelve-band RMG Ti+U core-only control is
running separately; the inherited invalid scalar-export setup is preserved
as a correctly rejected attempt. No RMG convergence/physical result is inferred.

Reproduction and immutable input/output/source hashes:
/home/j1k/projects/mstd/34_mfrqmc/work/05_hubbard_u_impl/functional_extension_validation/
sol/nexus_native_qe_r2scan_20261104/ (he_r2scan_nexus.py, validation.json,
qmca_stats_validation.json, native/preview logs and final_manifest.json).
This closes the selected fresh QE-meta native scheduling/hand-off subcase,
not all F13 combinations, the physical criterion or the complete Goal.
