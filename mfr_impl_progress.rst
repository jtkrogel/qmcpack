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