MFR Potential Reader and Periodic B-Spline Implementation Plan
================================================================

Objective and scope
-------------------

Implement a QMCPACK one-body operator that reads the converged mean-field
electron--electron potential written by the Quantum ESPRESSO MFR exporter and
constructs periodic three-dimensional cubic B-splines.  QMCPACK will evaluate
the positive mean-field potential at every electron position as an auxiliary observable
named ``MFRPotential``.  It must not add this operator to QMCPACK's physical
Hamiltonian or change the existing ``LocalEnergy`` estimator.  QMCPACK will additionally write ``TotalEnergyMF`` and
``KineticJastrowResidual`` as auxiliary scalar fields.  The ``qmca``
post-processing program will combine these with ``ElecElec`` and
``MFRPotential`` to produce the derived ``LocalEnergyMFR`` quantity, while
preserving the original ``LocalEnergy`` unchanged.

The QE file contains

.. math::

   V_{ee}^{\mathrm{MF}}(\mathbf r,\sigma)
   = v_H(\mathbf r) + v_{xc,\sigma}(\mathbf r)

on QE's dense real-space FFT grid.  It does not contain the ionic potential.
The first target is the one-cell diamond-carbon VMC input derived from
``tests/solids/diamondC_1x1x1_pp/det_qmc_vmc_dmc_tm.in.xml``.  The initial
runs are smoke tests only; longer statistical validation will be performed
separately by the user.

Source assessment
-----------------

``QMCHamiltonians/HarmonicExternalPotential`` demonstrates the required
``OperatorDependsOnlyOnParticleSet`` interface: input parsing, scalar
``evaluate``, trace support, and cloning.  ``HamiltonianFactory.cpp`` creates
external potentials from ``<extpot>`` elements and registers them with the
Hamiltonian.

``QMCHamiltonians/MPC`` supplies the closest periodic spline example.  It
creates a ``UBspline_3d_d`` over the unit cube with ``PERIODIC`` boundary
conditions, maps Cartesian positions to wrapped reduced coordinates with
``CrystalLattice::toUnit_floor``, and evaluates the spline with
``eval_UBspline_3d_d``.

There is also an existing ``GridExternalPotential``.  It is useful as an I/O
and external-operator example but is not directly sufficient for MFR:

* it assumes one common grid size and one common Cartesian interval;
* it does not obtain the grid or lattice from the file;
* it has only one potential channel and therefore cannot represent LSDA/GGA
  spin-up and spin-down potentials;
* it applies the simulation-cell minimum image rather than mapping through
  the lattice belonging to the sampled periodic field; and
* its trace evaluation currently computes ``dot(r,r)`` rather than evaluating
  the loaded spline.

The orbital code in ``QMCWaveFunctions/BsplineFactory`` is substantially more
general than this operator needs.  It establishes the production conventions
for HDF5 reading, periodic unit-cube grids, spline ownership, and lattice
coordinate conversion.  The MFR potential needs one or two scalar
``UBspline_3d_d`` objects rather than an orbital ``MultiBspline`` or an SPOSet.

The einspline tests confirm the required periodic convention: each ``Ugrid``
has ``start=0``, ``end=1``, and ``num=N``; the samples are located at
``i/N`` and the endpoint at one is excluded.

Proposed operator and input
---------------------------

Add a dedicated ``MFRPotential`` class in
``src/QMCHamiltonians/MFRPotential.{h,cpp}``, derived from
``OperatorDependsOnlyOnParticleSet``.  A separate class keeps the QE schema,
unit conversion, spin semantics, and lattice validation out of the legacy
general-purpose grid operator.

Register a new external-potential type in ``HamiltonianFactory.cpp`` and add
the source to ``src/QMCHamiltonians/CMakeLists.txt``.  Unlike the existing
harmonic and grid external potentials, register this operator with
``QMCHamiltonian::addOperator(..., "MFRPotential", false)``.  The ``false``
places it in ``auxH``: it is evaluated and written as an observable, but is not
included in the physical Hamiltonian sum that produces ``LocalEnergy``.

The XML contract is:

.. code-block:: xml

   <extpot type="mfr" name="MFRPotential"
           href="diamond.mfr.h5" scale="1.0"/>

``href`` is required.  The observable name is fixed to, or defaults to and is
validated as, ``MFRPotential`` so that the scalar-file/post-processing contract
is unambiguous.  ``scale`` defaults to ``1.0`` and the MFR input uses ``scale="1.0"``.  The
``MFRPotential`` column is therefore the positive, Ry-to-Hartree converted
quantity :math:`\sum_i V_{ee}^{\mathrm{MF}}(\mathbf r_i,\sigma_i)`.  The
minus sign is applied only by ``qmca`` when it forms the MFR local energy; it
is not embedded in either the HDF5 data or the QMCPACK observable.
Schema version 2 also carries :math:`E_{\mathrm{tot}}^{\mathrm{MF}}` as
the occupied KS eigenvalue sum plus the ion--ion energy.  It is exposed as the
constant ``TotalEnergyMF`` scalar field.  The DFT functional total is retained
separately as provenance and must not be substituted for this mean-field
Hamiltonian eigenvalue.

HDF5 schema handling
--------------------

Read and validate schema version 2:

* the potential and grid attributes listed above;
* ``total_energy_mf``, ``total_energy_mf_units``, and the exact definition
  ``sum occupied KS eigenvalues + ion-ion energy``;
* provenance attributes ``ks_eigenvalue_sum``, ``ion_ion_energy``, and
  ``dft_total_energy`` in Rydberg;
* dataset ``/mfr/lattice_vectors`` and one or two ``/mfr/v_hxc/spin_N``
  datasets; and
* ``/mfr/ks/eigenvalues``, ``occupations``, ``kpoints``,
  ``kpoint_weights``, and ``spin_index``.  The full eigenvalue table plus
  occupations preserves the occupied-orbital selection and k-point mapping.

Use ``hdf_archive`` for the file and datasets.  QMCPACK has no evident
high-level attribute reader, so add a small local RAII/helper layer around the
HDF5 C attribute calls, or factor such a helper into the HDF support if it is
generally useful.  Attribute strings written by QE are fixed-length and
space-padded and must be trimmed.

Fail with a clear uniform error for a missing file or dataset, an unsupported
schema, non-double/rank-3 potential data, dimensions smaller than required for
cubic interpolation, unknown units, an unexpected quantity, inclusion of
ions, or a spin-channel mismatch.  Log the file, logical dimensions,
functional, channel count, scale, and conversion applied.

QE stores the potential in Rydbergs while QMCPACK energies are Hartree.
Multiply every sample by exactly ``0.5 * scale`` before spline construction.

Grid and memory-order conversion
--------------------------------

The QE exporter records logical dimensions ``(nr1,nr2,nr3)`` and stores the
Fortran grid with ``i`` fastest.  As seen through the HDF5 C API, the rank-3
dataset shape is ``(nr3,nr2,nr1)`` and its last index is the fast ``i`` index.
QMCPACK ``Array`` and einspline input use the last C++ array index as the
fastest index, while spline coordinate zero corresponds to the first array
index.  The input must therefore be explicitly reordered rather than passed
directly to ``create_UBspline_3d_d``:

.. code-block:: c++

   spline_values(i,j,k) = qe_dataset(k,j,i);

for ``0 <= i < nr1``, ``0 <= j < nr2``, and ``0 <= k < nr3``.  A deliberately
non-cubic fixture and an index-coded grid should test this conversion so an
axis permutation cannot pass accidentally.

Create a separate spline for each stored spin channel using three independent
unit grids and ``PERIODIC`` boundary conditions.  No duplicate endpoint is
added.

Lattice and coordinate mapping
------------------------------

Read the QE lattice and construct a ``CrystalLattice`` owned by the operator.
For each electron, map its Cartesian position through that lattice with
``toUnit_floor`` and evaluate the spline on ``[0,1)^3``.  This is preferable
to using the QMCPACK simulation cell directly: it remains correct if the two
inputs use equivalent primitive bases or if a primitive potential is tiled
through a commensurate QMCPACK supercell.

The current QE schema attribute says that lattice vectors are columns of the
logical Fortran matrix.  With the present QE/qeh5 write path, a C HDF5 reader
observes those vectors as dataset rows, which is the convention expected by
``CrystalLattice::set`` and by ESHDF ``primitive_vectors``.  Treat this as an
explicit schema-version-2 conversion and test it against the diamond input.
The QE metadata wording should be corrected or clarified separately so later
schema versions do not remain ambiguous.

Validate commensurability between the field lattice and the QMCPACK periodic
simulation cell.  For the identity-cell target this should reduce to equality
within a tight tolerance.  More generally, transform each QMCPACK lattice
vector into QE reduced coordinates and require values close to integers; this
allows an integer supercell of the QE field but rejects unrelated cells.

Spin behavior
-------------

For one channel, use ``spin_1`` for every electron.  For two channels, require
two electron particle groups and map QMCPACK group 0 (normally ``u``) to
``spin_1`` and group 1 (normally ``d``) to ``spin_2``.  Evaluate the spline
selected by ``P.getGroupID(i)``.  Reject extra groups or inconsistent channel
counts rather than silently choosing a potential.

Evaluation, ownership, and traces
---------------------------------

``evaluate`` sums the selected, converted positive spline value over all
electrons and stores the result in ``value_``.  Because the operator is
auxiliary, QMCPACK records that value in the ``MFRPotential`` scalar column but
does not include it in ``LocalEnergy`` or ``LocalPotential``.  The trace path
must perform the identical interpolation per electron and record those
individual values; it must not copy the current ``GridExternalPotential``
trace implementation.

Own splines with ``std::shared_ptr<UBspline_3d_d>`` and ``destroy_Bspline``.
Clones on a rank may share immutable spline coefficients, but
``makeClone(ParticleSet& P)`` must construct a clone rebound to ``P`` rather
than copying a reference to the original particle set.  Keep all spline data
immutable after construction.

The initial implementation reads and validates the small field file once per
MPI rank, then shares immutable splines among walker clones on that rank.  A
two-rank smoke run exercises this path.  A future scalability optimization may
centralize the read and broadcast contiguous grid buffers without changing the
file or XML contract.

Scalar-output and ``qmca`` processing
-------------------------------------

QMCPACK writes three auxiliary scalar fields without changing its physical
Hamiltonian or original ``LocalEnergy``:

* ``MFRPotential`` is the raw, positive-signed (not necessarily numerically
  positive) :math:`\sum_i V_{ee}^{\mathrm{MF}}(\mathbf r_i,\sigma_i)`;
* ``TotalEnergyMF`` is the constant schema-v2 value converted from Rydberg to
  Hartree; and
* ``KineticJastrowResidual`` is :math:`T[DJ]/(DJ)-T[D]/D`, including the
  determinant--Jastrow gradient cross term, and is exactly zero when no
  nonfermionic wavefunction factor is present.

The required block-by-block qmca arithmetic is::

   LocalEnergyMFR = TotalEnergyMF + KineticJastrowResidual
                    + ElecElec - MFRPotential

The explicit minus sign is applied only in qmca.  Derivation occurs after qmca
unit conversion so all terms have common units.  If any required field is
absent, qmca leaves its established behavior unchanged and does not invent a
partial MFR energy.  ``LocalEnergyMFR`` participates in normal quantity
selection and statistical analysis while ``LocalEnergy`` remains untouched.
Focused tests cover the arithmetic and sign.

Nexus input generation
----------------------

Extend ``nexus/nexus/qmcpack_input.py`` so the XML object model can represent
the new element.  Add an ``extpot`` element/factory and an MFR-specific
``QIxml`` class with the supported ``type``, ``name``, ``href``, and ``scale``
attributes; add ``extpot`` to ``hamiltonian.elements`` and to the appropriate
class, type, and plural registries.  Reading and writing XML should preserve
the element shown above.

Add ``mfr_potential=None`` to ``gen_basic_input_defaults``.  When a filepath is
provided to ``generate_qmcpack_input``/``generate_basic_input``:

* validate its existence when ``check_paths=True`` and issue a focused error
  naming ``mfr_potential`` if it is missing;
* make the path relative to ``run_path`` when requested, consistently with the
  existing orbital-file handling;
* pass it into an added ``mfr_potential`` argument of
  ``generate_hamiltonian``; and
* append an ``<extpot type="mfr" name="MFRPotential" ... scale="1.0"/>``
  element without removing or changing any physical pair potentials.

The new argument controls only MFR XML generation.  It must not silently
change ``interactions``, estimators, or correction selection.  In particular,
Nexus currently chooses MPC by default for periodic basic inputs; callers that
want the MFR smoke-test Hamiltonian must explicitly request no MPC (for
example, with ``corrections=[]``) while retaining the ordinary ElecElec,
IonIon, and electron--ion/pseudopotential terms.

Add focused tests in ``nexus/nexus/tests/test_qmcpack_input.py`` for direct XML
parse/write round trips, exact XML generated through ``mfr_potential``, path
validation and ``run_path`` relativization, coexistence with all ordinary
physical potentials, and no-MPC generation when the caller explicitly
disables that correction.

Implementation sequence
-----------------------

1. Add the ``MFRPotential`` interface, schema constants, HDF5 loader,
   dimension/lattice validation, unit conversion, axis reorder, and periodic
   spline construction.
2. Implement scalar evaluation, spin selection, per-particle traces, and
   clone rebinding.
3. Register ``type="mfr"`` in ``HamiltonianFactory`` as the auxiliary
   ``MFRPotential`` observable and add the new source to CMake.  Verify that
   its scalar value is present but the physical ``LocalEnergy`` is unchanged.
4. Add focused QMCPACK unit coverage for periodic interpolation at grid points
   and off-grid points, boundary wrapping, non-cubic axis order, Ry-to-Hartree
   conversion, one- and two-channel selection, cloning, lattice mismatch, and
   malformed schema handling.  Use a tiny deterministic HDF5 fixture; these
   are correctness checks, not statistical QMC tests.
5. Extend the Nexus XML model and ``generate_qmcpack_input`` path with the
   ``mfr_potential`` argument, then run the focused Nexus input tests.
6. Export ``TotalEnergyMF`` and KS metadata in QE schema v2, add the
   ``KineticJastrowResidual`` auxiliary operator, and extend qmca to derive and
   report ``LocalEnergyMFR`` with focused CLI/data-analysis tests.
7. Build at least the real-valued configuration used by the diamond example;
   also compile the complex configuration because the Hamiltonian operator is
   independent of wavefunction value type.
8. Prepare two short diamond VMC inputs derived from
   ``det_qmc_vmc_dmc_tm.in.xml``.  Remove all DMC sections and the MPC pair
   potential only; retain all other existing physical potential terms.  Retain
   one short VMC section and add the MFR external potential.  One input retains
   J1/J2 and the other removes both Jastrows.
9. Run only smoke tests sufficient to establish clean startup, HDF5 loading,
   spline creation, finite ``LocalEnergy``/``MFRPotential`` values, and normal
   completion.  Validate ``qmca`` production of ``LocalEnergyMFR`` from all four required
   scalar inputs.  Do not
   interpret their short-run energy statistics.

Run-directory policy
--------------------

All implementation smoke runs will be isolated below
``/home/j1k/repositories/qmcpack_dev_new5/mfr_test_runs`` so they cannot
collide with the user's runs.  Use dedicated subdirectories such as:

::

   mfr_test_runs/diamond_vmc_jastrow/
   mfr_test_runs/diamond_vmc_no_jastrow/

Each directory should contain or link only the required XML, orbital HDF5,
pseudopotential, and MFR HDF5 inputs.  Generated scalar, log, configuration,
and checkpoint files remain inside that directory.  No runs should be made
directly in the source test directory or at the top of ``mfr_test_runs``.

Smoke-run acceptance criteria
-----------------------------

* QMCPACK reports the MFR schema, dimensions, lattice, channel count, and
  Ry-to-Hartree conversion without warnings or HDF5 errors.
* ``scalar.dat`` gains finite ``MFRPotential``, ``TotalEnergyMF``, and
  ``KineticJastrowResidual`` columns but no ``LocalEnergyMFR`` column; QMCPACK's original ``LocalEnergy`` remains the
  standard fully interacting energy.
* ``qmca`` reports both ``LocalEnergy``
  and ``LocalEnergyMFR``; the latter uses the positive ``MFRPotential`` with
  a minus sign and follows the stated MFR arithmetic.
* Both the Jastrow and no-Jastrow VMC examples complete normally without MPC
  or any DMC section, while retaining every other original potential.
* At least one run exercises the batched driver if that is the intended
  production path; legacy execution may be used as a secondary smoke check.
* A serial run and a small MPI run load the same schema and complete, with no
  per-rank file race or clone/reference failure.

Deferred validation
-------------------

Longer statistical comparisons, MFR energy identities, variance changes,
time-step studies, and production scaling are intentionally deferred to the
user.  Support for noncollinear potentials, more than two channels,
non-periodic grids, lattice interpolation between unrelated cells, and GPU
offload of this small one-body spline are also outside the initial scope.
