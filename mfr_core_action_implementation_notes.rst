QMCPACK MFR implementation inventory
====================================

Status and source identity
--------------------------

This final implementation inventory is an interface and evidence record for
the QMCPACK consumer as frozen on 2026-11-02. Source anchors refer to the
immutable project snapshot at
``parallel_validation/final_sources/qmcpack/``, not to a mutable checkout.
The combined QE/QMCPACK/Nexus snapshot manifest is
``parallel_validation/final_sources/manifest.json``, SHA256
``dde148594c1148c3181f01cda8a1ae6ce356760308ab2cc8d3cd4e06a2542303``.
The immutable copy used by the final Luna endpoint evidence is
``parallel_validation/luna_restart/frozen_endpoint/source_manifest.json``.
The binary manifest in that directory has SHA256
``7f9dfa4c0fd0eb9be06adb39f770fca54a452d6a750fc1ac3579e10a8d754395``.

Key source hashes in the final snapshot:

* ``src/QMCHamiltonians/MFRCoreAction.cpp``:
  ``47ee66b839f94690cce8dc5bc07ca7a613ddc9846228ce56d5c531cc0bd4d6d6``
* ``src/QMCHamiltonians/MFRCoreAction.h``:
  ``a6ef8e609d5aaebfdde3ac7617ec5128a89aee5a6199a94306a4f18518a718b6``
* ``src/QMCHamiltonians/MFRPotential.cpp``:
  ``d19aa458335a0215bb80a0e10d25fcecfa5fdfbfa29d0053479dd11190c58511``
* ``src/QMCHamiltonians/QMCHamiltonian.cpp``:
  ``0f60b9f453bab09978c75f8844a890702f6e52513758bdb6d1ae0c1fdc57a789``
* ``src/QMCHamiltonians/tests/test_MFRCoreAction.cpp``:
  ``326460e3c42bfad95623095b4aec3fb3c29bb1ae38675224c5c6f45ab45a6ea6``

The older repository-local scalar-route notes remain untouched. This project
record covers both that preserved scalar API and the current action API.

Route overview and API flow
---------------------------

The action path connects the DFT producer, orbital builder, Hamiltonian and
sample analysis through explicit files and source-state metadata:

#. QE ``pw2mfr_core.x`` reads the converged plane-wave save used for the
   ES-HDF orbitals, assembles the bare core operator and writes
   ``<prefix>.mfr_core.h5``. Its data are described in the paired QE
   inventory. The writer sets ``complete=1`` only after writing the full
   table to a temporary HDF5 file, then renames it into place. Its completion
   message and ``JOB DONE.`` are both required before Nexus marks the new
   dependency result successful.
#. Nexus ``Pw2mfrCore`` exposes that file as the
   ``mfr_core_action`` result. It requires the same prefix and outdir as the
   dependent pwscf job and rejects a stale file without the current-run
   completion message (``nexus/nexus/qmcpack_converters.py:577-710``).
#. ``generate_qmcpack(mfr_core_action=...)`` emits an
   ``<extpot type="mfr_core_action" name="MFRCoreAction" href="..."/>``.
   Nexus substitutes the dependency result path and rejects simultaneous
   scalar and action payloads (``nexus/nexus/qmcpack.py:871-880`,
   ``nexus/nexus/qmcpack_input.py:7152-7169, 9849-9895``).
#. ``SplineSetReader`` builds a source map from the exact B-spline SPO
   ordering: QE spin, twist, band, the ES-HDF source path, orbital gauge and
   phase metadata, and the ion ParticleSet
   (``src/QMCWaveFunctions/BsplineFactory/SplineSetReader.cpp:101-126``).
   ``MFRCoreAction::put`` loads the table and pairs each occupied source
   orbital to both its ES-HDF coefficients and the QMC ionic source.
   ``HamiltonianFactory`` adds the action and
   ``KineticJastrowResidual`` as auxiliary operators
   (``src/QMCHamiltonians/HamiltonianFactory.cpp:169-176``).
#. At an energy boundary, the action contracts each determinant's inverse
   transpose with the corresponding bare action values. QMCPACK writes the
   action observables; Nexus ``qmca`` reconstructs the MFR energy sample by
   sample so block statistics retain covariance.

For a new DFT producer to interoperate, it must export the same selected
Bloch-state coefficients and operator action with the conventions below.
A filename or band index by itself is not a sufficient pairing key. In
particular, the current reader is tied to the QE schema and B-spline source
map; this document does not define a producer-neutral interchange standard.

Action HDF5 contract
--------------------

The action route represents
``T + bare ionic local + bare norm-conserving nonlocal`` applied to each
selected QE orbital. QE assembles the nonlocal part with bare ``dvan``, not
effective ``deeq``. The operator excludes Hartree, XC, Hubbard U and DFT
double-counting terms. The current producer validation and Fourier oracle
results are in the QE inventory and
``parallel_validation/sol/qe_full_operator/``; the producer test is
separate from this consumer record.

The schema is a periodic Bloch-action grid, with Hartree energy, bohr lengths,
and float64 values. A retained polarized carbon payload has root attributes
including:

* ``schema_family=mfr_core_action``, ``schema_version=1``,
  ``complete=1``, and
  ``representation=periodic_bloch_core_action_grid``;
* ``energy_units=hartree``, ``length_units=bohr``,
  ``orbital_normalization=sum_G c_G exp(+i G.r); no volume factor``, and
  ``bloch_phase=exp(+2 pi i reduced_k.reduced_r)``;
* ``grid_order=HDF5: z,y,x; x fastest``,
  ``periodic_grid_endpoint=excluded``, and integer
  ``grid_dimensions=[nx,ny,nz]``;
* ``spin_channels``, ``number_of_kpoints``,
  ``number_of_states``, ``ion_ion_energy``, and ``ion_species``.

The file layout is:

* ``/lattice_vectors``: float64 ``(3,3)``, lattice vectors in bohr.
* ``/ion_positions``: float64 ``(nat,3)``, positions in bohr;
  ``/species_N`` groups carry fixed-string ``element`` and numeric
  ``valence_charge`` attributes.
* ``/kpoint_K``: reduced k vector attribute. Each
  ``/kpoint_K/spin_S`` group contains integer ``gvectors`` of shape
  ``(ng,3)``.
* ``ion_species`` values are one-based indices into zero-based ``/species_N``
  groups; k-point, spin and state group suffixes are zero-based.
* Each ``/kpoint_K/spin_S/state_B`` group contains an eigenvalue attribute,
  G-space orbital identity arrays ``orbital_g_real`` and
  ``orbital_g_imag``, and separate float64 grids ``action_real`` and
  ``action_imag`` of shape ``(nz,ny,nx)``. Complex values are stored as
  two real datasets, not a compound-complex dataset.

The action arrays are periodic parts; the reader interpolates them with
periodic double-precision cubic splines, then restores the Bloch/gauge phase.
The periodic upper endpoint is omitted. Each QMC and future-producer
implementation must use the same axis order, normalization, phase convention,
state indexing and ion constant meaning. The consumer verifies schema,
representation, units, normalization, phase, grid order, completion,
dimensions, finite values and lattice geometry. It does not authenticate the
snapshot MD5, validate a pseudopotential checksum, or interpret every
provenance attribute such as ``functional`` and ``hubbard_enabled``.

An early synthetic fixture omitted the trailing NUL used in QE fixed-string
ES-HDF names. The first physical smoke then exposed a real reader issue: the
generic archive string read retained QE's NUL and rejected a valid species
label. The final code uses a bounded fixed-string dataset reader, trims NUL
and padding, and tests QE-style padded strings plus invalid shapes
(``MFRCoreAction.cpp:44-75, 593-598``;
``src/QMCHamiltonians/tests/test_MFRCoreAction.cpp``). Those early fixture/reader failures are
corrected history, not current gaps.

Orbital, ion and cell pairing
-----------------------------

``MFRCoreAction::put`` reads the href and starts a fresh state snapshot,
clearing row caches and observables before loading it
(``MFRCoreAction.cpp:418-447``). The header reader validates fixed-length
scalar string attributes, typed numeric attributes, and
expected dataset ranks and shapes. The ES-HDF species-name dataset reader
additionally requires one fixed string no longer than 256 bytes
(``MFRCoreAction.cpp:35-368, 449-513``). It checks the primitive cell as a
fully periodic exact cell, not a supercell or a tiled cell, with a
``1e-8`` reduced-vector tolerance
(``MFRCoreAction.cpp:515-530``). A two-spin file needs two electron groups.

The B-spline source map preserves the orbital band ordering. In real builds,
a complex orbital can map to a real/imaginary time-reversal pair; complex
builds retain a complex source. Rotation phase, duplex use, mesh, half-G and
the source ParticleSet are stored with that map
(``src/QMCWaveFunctions/BsplineFactory/BsplineSet.h:44-60, 94-159``).
For each occupied state, the consumer performs these checks:

* selected spin, twist and band are within the exported range;
* action and ES-HDF reduced k vectors agree within ``1e-10``;
* integer G vectors are unique and map one-to-one, and selected orbital
  coefficients agree within ``1e-12``;
* the same action ion geometry agrees with ES-HDF geometry modulo periodic
  images to ``1e-8 bohr``;
* species names and valence charges agree between the action and ES-HDF
  metadata, then the action geometry is matched to the actual QMC source ion
  ParticleSet. The QMC species need a ``charge`` and either an
  ``atomicnumber`` or the matching element label. Charge tolerance is
  ``1e-8``; coordinate tolerance is
  ``max(1e-8, 4*epsilon(RealType)*max(1,max_cell_vector_length))``; ``RealType``
  is double in full-precision builds and float in mixed-precision builds.

The ion source comes from the B-spline builder's ``SourcePtcl``, so these
checks do not rely only on the action file's atom ordering or the XML href.
The QMC XML must describe the same physical primitive cell, ions, species and
occupied spin orbitals as the exported action. An explicit app-level paired
geometry failure and a 5-up/3-down control are retained in the L3 package.

``readState`` compares the action coefficients with
``/electrons/kpoint_K/spin_S/state_B/psi_g`` from the paired ES-HDF file.
The consumer applies the full coordinate before wrapping the periodic spline
table and evaluates the Bloch phase from the unwrapped position. The real
duplex path reapplies the QMC twist and saved rotation phase. The endpoint
regression for points just below the periodic upper face exposed that float
narrowing could round a wrapped coordinate to 1; promoting before wrapping
and retaining double reduced coordinates fixes the scalar and action routes.
The odd-grid tests at ``+/-2e-9 bohr`` pass in full and mixed precision.
The earlier mixed-precision loss was about ``3e-9/6e-9 Ha``; it is not a
current failure.

Supported consumer scope
------------------------

The current action API requires one Slater determinant product with a fixed
B-spline SPO set and a complete source map. It rejects multideterminants and
requires exactly one SlaterDet wavefunction component
(``MFRCoreAction.cpp:761-795``). It supports one or two collinear spin
channels; a two-channel action requires two QMC electron groups. The QMC cell
must be fully periodic and equal the exported primitive cell. Anisotropic
nonperiodic, tiled/supercell, spinor, arbitrary independent band rotation or
generalized state reordering paths are not established as supported.

The QE producer currently supports converged LDA/PBE SCF saves, explicit k
points, collinear norm-conserving pseudopotentials, one image/pool/band group,
and the documented on-site Dudarev U settings with supported projectors. It
rejects additional fields and non-3D periodic operators, unsupported J,
background or constraint terms, PAW and other unsupported reference modes.
These limits originate in the producer and must be read with its own inventory.
The consumer validates the arrays it uses; it is not a substitute for the
producer's reference-input validation.

The ``TotalEnergyMF`` observable on the action route is determinant-specific:
it starts with the exported ion-ion constant and adds the selected occupied
KS eigenvalues. It is not an SCF occupation-weighted sum and is not a DFT
total energy. The action file has no separate DFT total-energy field. The
same-named metadata in the preserved scalar schema is a separate scalar-route
quantity.

Trace, observables and row-cache behavior
-----------------------------------------

For each determinant, the consumer evaluates

.. math::

   C = \sum_\sigma \operatorname{Tr}(M_\sigma^{-1}G_\sigma),

where ``M_ij=phi_j(r_i)`` and ``G_ij=(h_core phi_j)(r_i)``. The
complex product is algebraic and uses no Hermitian conjugation
(``src/QMCWaveFunctions/Fermion/OneBodyOperatorTrace.hpp:16-35``). Before
contraction, ``evaluateTrace`` completes pending wavefunction updates and
copies the determinant's inverse transpose. The action exports
``MFRCoreAction``, ``MFRCoreActionImag``, ``TotalEnergyMF`` and
``IonIonMF`` (``MFRCoreAction.cpp:848-912, 922-958``).

``MFRCoreAction`` keeps immutable state tables in shared data and an
instance-local row cache for each determinant
(``MFRCoreAction.h:73-97``). At an energy boundary it compares accepted
electron coordinates with the cached row coordinates and reevaluates changed
rows only. This also refreshes rows after configuration restoration or walker
reassignment because the accepted coordinates no longer match. Clones share
immutable action splines but start with their own empty row cache
(``MFRCoreAction.cpp:960-975``). A repeated ``put()`` creates a new
shared table snapshot, resets rows/counts/values, and leaves existing clones
on their old immutable tables (``MFRCoreAction.cpp:418-437``).

Caching avoids recomputing unchanged action rows; the inverse transpose is
still copied and the trace contracted at each energy boundary, both
``O(N^2)``. The code adds no second inverse and no proposal-time flush.
The separate ``KineticJastrowResidual`` still evaluates fermionic and
nonfermionic wavefunction component gradients/laplacians
(``src/QMCHamiltonians/KineticJastrowResidual.cpp:34-80``).

Hamiltonian arithmetic and mode behavior
----------------------------------------

The action-route estimator is

.. math::

   E_L^{\rm MFR} =
   \operatorname{Re} C + \Delta T_J + V_{ee} + E_{II}^{\rm MF}.

The imaginary trace is a diagnostic and does not enter the real local-energy
sum. In standalone mode, QMCPACK replaces ``LocalEnergy`` with this action
expression. In comparison mode, the usual Hamiltonian ``LocalEnergy`` is
retained and MFR quantities remain auxiliary observables
(``QMCHamiltonian.cpp:177-271``). The scalar route has its distinct
``TotalEnergyMF + KineticJastrowResidual + ElecElec - MFRPotential``
arithmetic (``QMCHamiltonian.cpp:274-300``).

The driver accepts ``mfr_qmc=auto|yes|no``
(``src/QMCDrivers/QMCDriverInput.cpp:39-75, 133-156``):

* ``yes`` without either MFR payload is an error.
* A payload with ``auto`` or ``yes`` selects standalone MFR mode.
* A payload with ``no`` retains the conventional Hamiltonian and records
  MFR observables.
* No payload with ``auto`` or ``no`` selects the standard mode.

The regular Hamiltonian factory adds ``BareKineticEnergy`` as ``Kinetic``
(``HamiltonianFactory.cpp:100-106``), and adds
``KineticJastrowResidual`` when it creates an MFR payload. The mode resolver
explicitly requires ``KineticJastrowResidual`` and an electron-electron Coulomb
operator named ``ElecElec``. It permits ``Kinetic`` and ``ElecElec`` and rejects
other physical Hamiltonian terms. The resolver does not separately reject a
missing ``Kinetic`` for a Hamiltonian assembled through the direct API; normal
factory-generated inputs include it. Explicit pseudopotential, electron-ion,
ion-ion, MPC or other physical Hamiltonian terms are rejected. Nexus
suppresses ionic corrections and MPC estimators for standalone input and
defaults to standalone when a
payload is supplied; set ``mfr_qmc=False`` to request comparison mode
(``QMCHamiltonian.cpp:201-230``,
``nexus/nexus/qmcpack_input.py:9849-9862``). The action contributes
``IonIonMF``; no explicit ``IonIon`` pairpot is permitted in that mode.

Nexus ``qmca`` derives ``LocalEnergyMFR`` on each saved sample before
block statistics. It uses

``MFRCoreAction + KineticJastrowResidual + ElecElec + IonIonMF``

when action fields exist, otherwise it uses the scalar formula with a minus
sign before ``MFRPotential``
(``nexus/nexus/bin/qmca:387-399``). The real action enters the sum;
``MFRCoreActionImag`` remains a diagnostic.

Preserved scalar route
----------------------

The scalar route remains the separate, economical
``src/QMCHamiltonians/MFRPotential.{h,cpp}`` operator. Its payload is
opt-in through QE ``write_mfr_potential`` and is disabled for Hubbard-U
runs. It samples a periodic electron field and does not load per-orbital
action grids or evaluate a determinant trace.

The scalar schema v2 has ``quantity=Vee_MF = v_H + v_xc``, Rydberg units,
Fortran ordering (``i`` fastest), an excluded periodic endpoint, a
``(3,3)`` lattice and datasets
``/mfr/v_hxc/spin_1`` and, when polarized,
``/mfr/v_hxc/spin_2`` with shape ``(nz,ny,nx)``. It also records
``total_energy_mf`` in Rydberg as occupied KS eigenvalues plus ion-ion
energy. QMCPACK converts the potential samples by ``0.5*scale`` to Hartree,
and converts ``total_energy_mf`` by ``0.5`` regardless of the potential scale.
It interpolates the field at electrons; ``V_H+V_xc`` is stored with its plus
sign, and the scalar MFR energy subtracts the sampled field
(``MFRPotential.cpp:152-279, 317-349``). Scalar-only XML does not load
action state tables or calculate the action trace.

Nexus distinguishes these APIs: ``mfr_potential=...`` emits
``<extpot type="mfr" name="MFRPotential" href="..." scale="1.0"/>``,
whereas ``mfr_core_action=...`` emits the separate action element without
a scalar scale parameter. Do not conflate their HDF5 schemas, stored
``TotalEnergyMF`` provenance or energy signs.

Verification and retained evidence
----------------------------------

The final four-build MFR suite used the source snapshots and build matrix
recorded in
``parallel_validation/sol/build_matrix_endpoint/manifest.json``
(SHA256 ``d0159fe59ed038d480a99367b0385a2cc36ffc0850f3b28674770a3df3d8ff69``)
and ``numerical_summary.json`` (SHA256
``2762a62b3686012bda96bded19a6f2eca92b4ec0c64376e0bf3da0d29c1cd69a``).
All four build and test commands returned zero:

.. code-block:: sh

   cmake --build /home/j1k/repositories/qmcpack_dev_new5/build_mfr_real -j4 --target test_hamiltonian_ham qmcpack
   /home/j1k/repositories/qmcpack_dev_new5/build_mfr_real/src/QMCHamiltonians/tests/test_hamiltonian_ham '[mfr]' --rng-seed 17 --turn-on-printout

Repeat with ``build_mfr_comp``, ``build_mfr_mixed_real`` and
``build_mfr_mixed_comp`` for the complex/full and both mixed-precision
configurations. The test executable must run from its matching
``src/QMCHamiltonians/tests`` working directory so fixtures resolve.
Build/test logs are in ``parallel_validation/sol/build_matrix_endpoint/``.

The real/full and real/mixed suites each report 7286 assertions in 19 cases;
the complex/full and complex/mixed suites each report 7191 assertions in 18
cases. These totals include HDF5 fixture setup and are not independent
physics-check counts. Across the matrix, matched inverse/action contraction
errors are below ``5e-16`` in full precision and below ``2e-7`` in mixed
precision. The summary keeps matched algebra, residual, continuum-action and
finite-spline errors separate; per-mesh maximum continuum-core errors range from
``6.38e-6`` to ``2.83e-4 Ha``, and per-mesh maximum DLA spline errors from
``9.08e-4`` to ``4.97e-3 Ha`` for the recorded 16/32 point examples.
Cases cover real/complex, full/mixed, skew cells, Gamma/off-Gamma and real
antiperiodic twists, unequal spin populations, fixed Jastrow/no-Jastrow,
occupied-space rotations, clones, multiwalker state, reload and cache refresh.

Six selected Nexus/qmca MFR tests passed, with 28 others deselected; the log
is ``parallel_validation/sol/mfr_nexus_selected.log``. Ten full-real
physical smoke cases and the same ten mixed-real cases passed. They cover
LDA/PBE scalar and action paths, U=0 and finite-U LDA/PBE, the spin-polarized
5-up/3-down carbon case, and no-Jastrow/fixed-J2 examples. Each short smoke
uses one rank/thread, at most eight walkers and three 3-step blocks. They are
application smoke checks, not statistically converged results.

The separate-process restart results are retained in
``parallel_validation/sol/restart_endpoint/``; the primary independent
raw HDF/XML/hash review is
``parallel_validation/sol/mfr_restart_endpoint_review.py``. Real/legacy
standalone continuation restored eight walkers and RNG, matched entry
coordinates within ``2.423e-12 bohr`` and per-walker energy within
``6.513e-12 Ha``, then ran short branching DMC. Complex/batched comparison
restored eight walkers and RNG, matched entry coordinates within
``3.830e-12 bohr`` and its derived MFR energy at output precision. The
complex standard nonlocal ``LocalEnergy`` changed by
``1.45143e-4 Ha`` and is not a deterministic restart invariant. No
bitwise continuous-versus-split trajectory identity or MPI repartition
claim is made; the complex walker log does not contain per-walker auxiliary
MFR fields.

Luna's final copied-payload application package is
``parallel_validation/luna_restart/negative_validation/``. Against the
endpoint-safe real executable, the matched 5-up/3-down control passed and
eight negative inputs exited before sampling at the intended guard:
action/ES-HDF and QMC ion geometry, selected orbital coefficients, band
range, primitive-cell/tile, representation, determinant spin mapping,
missing payload in standalone mode, and explicit ionic/pseudopotential terms
in standalone mode. Original action, orbital and pseudopotential SHA256s
remain unchanged. The package pins the executable/source manifests, commands,
logs, failed attempts, private mutated HDF5 files and their hashes.
L2/L3 here are short smoke and guard tests, not long-run statistics.

Historical finite-grid and pseudopotential limits
-------------------------------------------------

Short historical comparisons remain useful with their sampling and
representation limits attached. For He LDA at
``ecutwfc/ecutrho=80/2560 Ry`` and orbital ``meshfactor=3``, the
30-block action-standard differences are ``-0.000315 +/- 0.000283 Ha``
without a Jastrow and ``+0.000070 +/- 0.000256 Ha`` with fixed Jastrow.
At ``ecutrho=320 Ry`` the residual is about ``-0.022 Ha`` and is not
removed by refining only the QMC orbital mesh. The action grid follows the QE
SCF dense grid; it has no independent consumer resampling control. A compact
PBE-He comparison is ``-0.0000518 +/- 0.000728 Ha``; its scalar-route
mean ``+0.01373 +/- 0.06935 Ha`` is statistically unresolved.

The finite-U carbon action-standard differences are 20-block estimates, not
the deterministic DLA oracle values:

* PBE U=0, determinant-only: ``-0.005055 +/- 0.002821 Ha``.
* PBE U=3 eV, determinant-only: ``-0.003643 +/- 0.003074 Ha``.
* PZ/LDA U=3 eV, determinant-only: ``-0.002824 +/- 0.002575 Ha``.
* PBE U=3 eV, fixed J2: ``-0.002262 +/- 0.005816 Ha``.
* PZ/LDA U=3 eV, fixed J2: ``-0.007319 +/- 0.003123 Ha``.

The last value is ``2.34 SEM`` from zero and is not universally within
one or two standard errors. For PBE, increasing the orbital mesh factor from
3 to 4 decreases RMS local-energy residual from ``0.01218`` to
``0.01142 Ha`` at U=0 and ``0.01130`` to ``0.00929 Ha`` at U=3 eV.
These short statistics do not establish a pointwise identity for a finite
grid.

QE's BFD UPF local channel and QMCPACK XML local channel differ by about
``6.7e-5 Ha-bohr``. The older PZ-VBC UPF has a malformed header and radial
table count; the retained XML was generated from a temporary copy with the
count fixed, while QE reads the original UPF. The local channels differ by
about ``1.33e-3 Ha-bohr`` below 2 bohr, and ``ppconvert`` labels its XML
header PBE. This mismatch limits interpretation of the PZ/LDA+U comparison;
it is not proof of pseudopotential equivalence.

Remaining limits and producer-port questions
--------------------------------------------

The final deterministic action/DLA identity tests pass in the declared
real/complex and full/mixed matrix, with interpolation errors reported
separately. They do not imply that every finite-grid application sample equals
the conventional pseudopotential local energy. QE's independent Fourier test
validates kinetic, local, bare nonlocal and complete action against its
analytic construction, but does not independently construct QE Hubbard
projectors. Keep that producer boundary with the QE inventory.

For another DFT producer, define a versioned schema profile for the exact
core operator, reference restrictions, source-state identity, ion metadata,
grid discretization and phase conventions. The current consumer does not
verify snapshot XML MD5 against source input, an ES-HDF content hash, or
pseudopotential identity; it pairs the actual numerical orbital data instead.
The action grid and orbital mesh are independent numerical resolutions but
the consumer currently exposes no action-grid resampling API. Generalized
spinors, independent band rotations, symmetrically expanded cells and tiled
supercells need explicit source maps and tests before they can be treated as
supported.

Reproduction and source files
-----------------------------

The matrix, endpoint smoke, restart and L3 logs plus manifests listed above
are the exact reproduction records. The QMC MFR unit command may be run from
the corresponding test working directory after building the matching target;
this Luna documentation task did not build binaries, run QE/SCF, or run
additional scientific tests. The four QMCPACK builds, MFR suites, Nexus/qmca
tests and physical/restart invocations were performed by Sol and are cited
as retained evidence.

The previous baseline QMC inventory was reviewed and accepted as L1; this
file replaces only its project-local final source/evidence refresh. The older
repository-local scalar note
``/home/j1k/repositories/qmcpack_dev_new5/mfr_qmcpack_implementation_notes.rst``
was not edited.

Meta-GGA two-field alternative and hybrid extension (2026-11-03)
----------------------------------------------------------------

The economical generalized-KS meta-GGA alternative stores two real scalar
fields per distinct spin: ``u = v_H + v_xc^mult`` and
``q = partial f_xc / partial tau`` for ``tau = sum |grad phi|^2/2``.  In the
QE convention, ``u`` is the converged ``v%of_r`` converted from Ry to Ha;
``q`` corresponds to ``v%kin_r`` under the convention used by
``h_psi_meta.f90`` and is dimensionless, not an energy field.  This mapping
must be checked against an explicit QE differential-action oracle, including
the interpolation from the dense grid to ``dffts``.  The QMC spline of ``q``
provides both ``q`` and its Cartesian gradient.

For a determinant ``D``, define

.. math::

   Q_L[D] = -\frac12\sum_i\left[q_{\sigma_i}(r_i)
   \frac{\nabla_i^2 D}{D} + \nabla q_{\sigma_i}(r_i)\cdot
   \frac{\nabla_i D}{D}\right].

The economical estimate is ``TotalEnergyMF + KineticJastrowResidual +
ElecElec - sum_i u_sigma(r_i) - Q_L[D]``.  The finite-basis residual
``Delta_MF = sum_sigma Tr(M_sigma^-1 R_sigma)`` must be added for an exact
continuum identity, where ``R_j = (h_ref - epsilon_j) phi_j``.  The two
scalar fields alone cannot reconstruct ``Delta_MF`` pointwise.  Thus this
route is explicitly approximate and its paired error must be measured against
the existing bare-core-action route on identical configurations, with and
without Jastrow.  No empirical constant should be used to conceal its
fluctuations.  It needs a distinct schema and observable name; the current
``MFRPotential`` is the LDA/GGA Hartree-plus-XC scalar field and must retain
that meaning.

The planned QE meta-GGA producer must require a fully converged, supported
collinear norm-conserving reference; export the two fields on the declared
dense periodic grid with spin, units, tau convention, selected-orbital and
ionic metadata.  It must keep ``write_mfr_potential`` unchanged for LDA/GGA.
QMCPACK must evaluate the meta-GGA differential contraction using its
Slater inverse and orbital gradients/Laplacians.  The coefficient acts on the
determinant only; the physical Jastrow kinetic residual and DMC diffusion
retain the ordinary constant-mass form.  Validate the q-gradient spline and
spin mapping in real/complex full/mixed builds; compare to the exact action
route on a common set of walkers and under single-particle updates.

Hybrids use the existing bare-core-action representation and QMCPACK reader:
no Fock-action spline or exchange application is needed in QMCPACK.  QE must
accept a converged hybrid save in ``pw2mfr_core.x``, retain the same paired
orbital coefficients, and compute only kinetic, ionic local and bare ionic
nonlocal actions.  A hybrid reference needs an additional outer-exchange-loop
completion check: QE's general SCF convergence flag may be set when the EXX
iteration limit is reached solely to write restart data.  Successful and
iteration-limited PBE0/HSE saves must both be tested.  Exchange fraction,
screening, divergence and grid settings should be recorded as provenance.
The existing unsupported spinor, PAW/USPP, background and nonlocal-correlation
operators remain subject to their explicit guards.


2026-11-03 meta-field consumer development checkpoint
------------------------------------------------------

New src/QMCHamiltonians/MFRMetaFields.h/.cpp reads common meta v1 u/q
fields on independently specified periodic binary64 grids. It evaluates
analytic q gradients with einspline's new scalar value/gradient routine,
transforms fractional derivatives to Cartesian coordinates and contracts
Q phi with the completed Slater inverse using determinant-only SPO VGL.
Native SPO output precision is promoted for contraction. No ECP is evaluated
by this operator. MFRCoreAction's metadata-only payload mode reuses geometry,
selected spin/band/twist and ES-HDF coefficient pairing without action splines.

HamiltonianFactory registers mfr_meta_fields/MFRMetaFields. QMCHamiltonian
selects a scalar/core/meta energy provider, preserves conventional comparison
LocalEnergy, rejects ambiguous standalone core+meta combinations and applies
TotalEnergyMF + KineticJastrowResidual + ElecElec - MFRMetaU -
MFRMetaDifferential in standalone meta mode. Imaginary differential values
are separately reported. Explicit property indices support core/meta
shared constants in either XML order; cloned Hamiltonians retain the provider.

Focused unit tests pass real/complex full/mixed, covering skew-cell field
values/gradients, corrupted inputs, analytic contractions, Jastrow, twists,
legacy/batched determinants, clones, energy signs and ambiguity guards.
Real/full QE TPSS import/VMC/DMC smoke tests execute, but physical two-field
low-error validation FAILS the current carbon pilots; native projected
SCF convergence does not establish pointwise replacement. See project
functional_extension_validation/sol/qe_meta/meta_pointwise_pilot.rst.
RMG meta export, full lifecycle/driver/logging coverage, qmca meta output,
workflow result dependencies and required cost work remain open. Earlier
immutable final inventories refer to their older source snapshot only.

2026-11-03 13:36 EST -- Import phase and residual cost updates
-------------------------------------------------------------

MFRCoreAction.cpp:731-773 skips unused phase FFTs/temporary vectors for
META_FIELDS, while retaining state/orbital/ion identity checks. Core-action
phase handling is unchanged. test_MFRCoreAction.cpp:877 verifies metadata
import leaves existing SPO phase state intact. All4 focused builds pass.

src/QMCHamiltonians/KineticJastrowResidual.cpp:34-82 now uses current total
log gradients P.G and nonfermionic evaluateGL(...,false), instead of
calling fermionic/nonfermionic evaluateLog. P.G must be current at the
accepted energy boundary, the same precondition as BareKineticEnergy.
For J=sum nonfermionic logs and G_D=P.G-G_J, it computes
-1/(2m) Re[L_J+G_J.G_J+2G_D.G_J], with algebraic complex dots.
Fermionic Laplacians and reconstruction/inversion are unnecessary.
Member scratch gradients/Laplacians in the header are resized/reused;
no-J still returns0 before their allocation. Scratch is O(N), owned per
operator/walker, not a new grid per clone. The timer is thread-local to
avoid shared flat-timer state across concurrently evaluated walkers.

src/QMCHamiltonians/tests/test_MFRPotential.cpp:419 adds independent
complex/unequal-mass arithmetic, multiple factors, repeated evaluation,
unchanged P.G/P.L and component call counts. Its fake component now obeys
the additive derivative API. Existing Slater/Jastrow analytic values first
initialize total wavefunction derivatives. No production tolerance changed.

Validation
----------

Final focused MFR unit matrix passes: real/full and real/mixed22cases,
16033assertions; complex/full and complex/mixed21cases,15938assertions.
The first complex/mixed build rejected a double scalar literal in the
independent oracle, corrected to Value(2); failure log retained.

Scalar/core/meta before/after fixed72 (scalar random-initialized but fixed
with the same seed) runs have identical inputs. Across28800 walker
measurements per repetition, final energy max differences0/8.89e-16/1.78e-15
Ha; field differences and repeat logs retained. Actual moving legacy and
batched VMC/DMC pairs (12processes) agree in energy<=1.78e-15Ha and positions
<=4.45e-16bohr. Full field errors retained; scalar batched unwrapped phase
can differ2pi, with exp(i delta_phase) error<1e-12. This does not establish
all restart/reset/buffer/precision/near-node lifecycles. Two-thread legacy
scalar and two-crowd meta VMC/DMC final-binary smoke tests succeed; raw
MFR arithmetic agrees<=1.78e-15Ha.

Performance and reproducibility
------------------------------

Immutable instrumented pre-optimization and final executable/source copies
are in before/ and after/, with SHA256 manifests. The intermediate static-
timer candidate was superseded by a thread-local timer; its timed binary
was not retained before relink, so use the final set for reproducible
acceptance. Initial noisy and intermediate controlled observations remain
separately labeled. Final interleaved3repetitions use72walkers,20blocks,
20steps, timestep1e-100, one pinned CPU/thread and fine timers. These are
fixed-position cost measurements, not statistical estimates or realistic
trajectory-efficiency measurements. No concurrent parent builds in final
measurements; shared-machine noise remains visible in the repeat spread.

Final median residual-inclusive speedups: scalar47.84x, core43.58x,
meta42.32x. Final median VMC execution-time speedups: scalar1.127x,
core1.049x, meta1.056x. See mfr_residual_bench_final_comparison.json for
all times/calls/error fields. Baseline profiles show determinant inverse
and SPO VGL work nested under residual at both Hamiltonian and auxiliary
boundaries; those inverse/recompute child calls disappear in the candidate.
The duplicate residual boundary itself remains, now inexpensive, and needs
a supported lifecycle-aware scheduling design before further elimination.

The targeted bottleneck benefit and small-He sweep benefit are demonstrated.
Do not generalize to larger systems, complex/mixed production, MPI scaling,
DMC time-to-error or the required broader batch/layout/memory acceptance.
An exact rearrangement does not itself reduce physical estimator variance.
Physical meta approximation gaps and native functional support are independent
requirements and remain open.

Project evidence: functional_extension_validation/sol/cost/kinetic_residual_20261103.

2026-11-03 larger cost and long-range regression checkpoint
----------------------------------------------------------

Derivative reuse has immutable moving Ti12e core and hBN16e scalar
real/complex VMC timings: median speedups1.095x/1.052x/1.129x. Walker paths
and saved RNG agree exactly, and per-walker fields pass fixed/moving
full-precision comparison. Detailed hashes, schedules, spread, memory,
parent validators and corrected complex-layout parser records are under
functional_extension_validation/sol/cost/larger_derivative_reuse_20261103
in the Hubbard implementation project. These do not establish physical
meta accuracy or broad mixed/driver/lifecycle acceptance.

A CPU-only StructFact host-dispatch experiment was evaluated and rejected:
Ti/hBN medians0.9907x/0.9492x fail the frozen performance gate. Production
src/Particle/LongRange/StructFact.cpp is restored exactly; it contains no
retained optimization from this experiment. The independent Fourier oracle
in src/Particle/LongRange/tests/test_StructFact.cpp, test case "StructFact
batched phases and per-particle storage", is retained. It covers cubic/skew
cells, unequal species, both coordinate kinds, changed walker positions
and per-particle storage. All4 real/complex full/mixed long-range suites
pass16,183 assertions in8cases. An actual rebuilt real batched standalone
meta VMC/DMC restoration comparison matches every raw field exactly.
All candidate/baseline sources, executables, run leaves and failures are
retained under sol/cost/structure_factor_host_dispatch_20261103 and
mfr_test_runs/functional_extension/structure_factor_dispatch_benchmark.

2026-11-03 15:32 EST -- Actual walker-buffer and precision lifecycle checkpoint
-----------------------------------------------------------------------------

Extended the analytic periodic determinant/DLA test in
src/QMCHamiltonians/tests/test_MFRCoreAction.cpp, starting near lines754
and1142. Added production Walker registration/allocation, accepted delayed
moves, wavefunction-buffer update, deep walker copy, further accepted moves,
loadWalker/copyFromBuffer and cached-GL reconstruction. Buffers are declared
before the wavefunction because legacy determinant matrices reference them.
The no-J test varies both determinant implementations, cubic/skew cells and
periodic/Bloch boundary cases at mesh16, in all four precision/complex builds.

Restoration asserts exact positions, gradients/Laplacians, log magnitude,
unit phase, core/meta traces and standalone Hamiltonian energies, no-J kinetic
residual and precise changed-row cache counts. An independent2x2 inverse/
actual-SPO action contraction checks the restored inverse before evaluateLog
can hide a failed buffer reload. Fresh determinant recomputation then agrees.
This tests buffer restoration through both determinant implementations, not
the batched QMC driver's distinct population/recompute protocol. Production
Jastrow buffer internals are not claimed covered by this no-J unit fixture.

A deliberate test-only omission of checkpoint copyFromBuffer fails100
assertions on real/full, including stale derivatives and core/meta energies.
Positive source is restored exactly. Four isolated positive focused suites
pass: real/full and real/mixed22cases/16,321 assertions; complex/full and
complex/mixed21cases/16,226 assertions. Original bounds1e-10 full/2e-5 mixed
are retained. An initial parallel invocation mistakenly shared the source
working directory, causing HDF fixture filename collisions in two variants.
All four first-attempt logs are retained; the final separate working-directory
runs pass. No production code was changed for the buffer test.

Ten new short production-Jastrow standalone application cases complete:
mixed-real scalar/core/meta, each legacy/batched; full and mixed complex
actual off-Gamma meta, each legacy/batched. Each has72 VMC observations;
DMC has32 except scalar mixed-real legacy34, so replication is exercised.
All1,042 logged observations satisfy independent signed MFR arithmetic
(max3.56e-15Ha), finite numeric fields and actual per-walker movement.
TotalEnergyMF agrees exactly with independently summed selected export
KS eigenvalues plus ion-ion constant (scalar Ry metadata converted to Ha).
These prove scoped production integration, not physical two-field accuracy,
continuous-vs-split trajectory equivalence or deterministic branch selection.

Five separate-process checkpoint pairs add missing combinations: mixed-real
scalar batched/core legacy/meta legacy, mixed-complex off-Gamma meta batched,
and full-complex off-Gamma meta legacy. For each, a fixed-position VMC loader
probe recovers the original DMC checkpoint coordinates bit-for-bit as a
multiset; another independent process continues DMC directly. All10 processes
pass raw MFR arithmetic/finiteness. Logs confirm RNG stream reload; source
walker/RNG checkpoint hashes remain unchanged. RNG stream correctness beyond
the observed production reload is not inferred from the message alone.

New unresolved legacy driver reset issue
----------------------------------------

An initial combined tiny-timestep loader-VMC then normal-DMC restart succeeds
for scalar batched but aborts for legacy core in
SimpleFixedNodeBranch::resetRun: cannot initialize WalkerController. All
failed outputs are retained. Direct DMC continuation passes. Separate normal-
timestep.02->.02 restart VMC/DMC reproductions abort in both mfr_qmc=yes and
comparison mfr_qmc=no. Comparison uses standard T+EE with the same auxiliary
core provider, not the original physical ionic Hamiltonian; it suffices to
show the reset failure is not exclusive to standalone energy replacement.
SimpleFixedNodeBranch.cpp has no worktree diff. Its restored DMC-mode flag
and null-controller check around lines549-594 are the next source path to
investigate; the exact fix and broader reset acceptance remain open.
Separating probes is not claimed to fix or accept this failed transition.

The common schema, native functional/operator, physical meta convergence,
production Jastrow-buffer, deterministic DMC branch-selection and remaining
restart/precision matrix requirements stay open; the Goal remains active.

Evidence: sol/lifecycle/{walker_buffer,precision_driver}_20261103.
New run trees: functional_extension/precision_driver_lifecycle_20261103,
precision_restart_lifecycle_20261103 (initial failed transition retained),
precision_restart_lifecycle_20261103_split_probes, and
legacy_restart_transition_diagnostic_20261103.

2026-11-03 15:44 EST -- Legacy restart VMC->DMC controller reset fixed
--------------------------------------------------------------------

Source investigation confirms a legitimate restored state: DMC flags and
statistics survive restart, but WalkerController is process-local and absent.
QMCMain.cpp calls the reused engine's resetRun before the next DMC driver
runs initWalkerController. The old resetRun abort therefore incorrectly
assumed that a historical DMC flag guaranteed a local controller.

Changed src/QMCDrivers/SimpleFixedNodeBranch.cpp near line592: after reading
new section parameters, return/defer controller-specific work if the local
controller is absent. Keep restart flags, energy/variance history, reference/
trial energy and parsed parameters. The existing population-aware
initWalkerController creates it later. An already-created controller still
uses the original reset path. No DMC energy formula, population policy,
pseudopotential treatment or producer format changes in this fix.

Added deterministic regression in tests/test_fixed_node_branch.cpp near
line57. It reproduces restored DMC/no-controller state, preserves histories
and parameters across deferred reset, initializes a real controller using
actual walker offsets, then changes timestep/warmup and verifies the same
controller is reused/reset. The initial real/full regression aborts as
expected on the old production code; that log is retained.

Final complete driver suites pass in all4 CPU variants: full real478
assertions/35cases, full complex678/37, mixed real472/34, mixed complex672/36.
The focused regression passes28 assertions per rank on an actual2-rank MPI
population. Earlier21-assertion positive logs predate the added active-
controller continuation check and are superseded by final suites/MPI logs.

Twelve actual checkpoint->VMC->DMC processes complete: all4 variants times
standalone normal.02, comparison normal.02 and standalone tiny1e-100 loader
probe followed by DMC.02. All produce finite raw walker logs and satisfy
independent standard T+EE or MFR core+J-residual+EE+ion constant arithmetic
(max error1.78e-15Ha). The four tiny probes reproduce saved checkpoint
coordinates bit-for-bit. Source walker/RNG checkpoint hashes remain unchanged;
logs show RNG reload and deferred controller initialization. Original failed
normal/tiny/comparison attempts remain untouched. Comparison uses auxiliary
MFR plus standard T+EE, so it diagnoses generic driver reset behavior, not
an original ionic-Hamiltonian physical energy comparison.

This resolves the specific legacy reset failure recorded at15:32. Broader
state/precision/restart invariants, production Jastrow-buffer oracles,
deterministic DMC branch-selection checks and native functional/physical-
accuracy requirements remain open. No continuous/split stochastic trajectory
identity is inferred from these short transition tests. Goal stays active.

Evidence: sol/lifecycle/legacy_branch_restart_20261103. Application outputs
are at qmcpack_dev_new5/mfr_test_runs/functional_extension/
legacy_restart_transition_fixed_20261103; older failures remain in the
precision_restart_lifecycle_20261103 and legacy_restart_transition_diagnostic_20261103 trees.

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


2026-11-04 accepted cache correctness checkpoint and branch policy
------------------------------------------------------------------

Immutable action grids are reused across determinants sharing source states.
Reference binding is tracked separately from partial grid preload, preserving
all per-SPO geometry/coefficient/phase checks and clone/reconfiguration
semantics. The targeted suites pass in all four real/complex full/mixed CPU
builds (22 real cases/16890 assertions;21 complex cases/16795 assertions).
Sixteen moving VMC/DMC before/after contexts compare4104 walker observations
and all non-timing scalar columns exactly; five actual QE-meta ionic-term/
provider rejection cases pass. Controlled repeated startup/memory benchmarking
is still running and is not accepted at this checkpoint. The two-field Ti
meta route remains physically unaccepted (1200 Ry mesh2 RMS88.37 mHa).

Evidence and immutable binaries:
functional_extension_validation/sol/cost/action_import_reuse_20261104/
in /home/j1k/projects/mstd/34_mfrqmc/work/05_hubbard_u_impl/.
At each subsequent bounded acceptance milestone, commit the validated source,
tests and associated status records to qp_mfr rather than accumulating a
multi-day implementation diff. Keep physical/performance acceptance explicit.


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

2026-11-04 17:16 EST -- bind native RMG completion markers to result paths
-----------------------------------------------------------------------

Nexus rmg.py previously bound only the historical meta writer marker to its
expected output path. Scalar/core completion accepted a marker naming any
file, provided a nonempty expected file already existed. A completed retry
could therefore release a stale scalar/core result after writing another
prefix. check_sim_status now normalizes and compares each enabled writer's
reported path with its get_result path. Plain and native progress-prefixed
messages remain accepted, including absolute and normalized relative paths.
Unrelated text containing the marker cannot complete a result. The latest
log, native convergence, nonempty file and job-finished guards remain active.
Existing meta behavior is retained; RMG meta development remains excluded.

Five new regressions fail by assertion against the snapshotted old method
in a process-local baseline injection. The corrected implementation passes
all20 selected scalar/core/dependency/meta-preservation/simulation tests.
Actual native input/log rechecks accept the converged Ti Bessel20/Bessel40
runs and reject the retained failed U-occupation-guard run. Native data are
read only. This proves a bounded hand-off safety correction, not fresh
Nexus scheduling or physical core-standard acceptance.

Evidence and source snapshots are under functional_extension_validation/
sol/nexus_rmg_marker_binding_20261104/. Initial harness failures are retained:
the parameterized form was incompatible with Nexus's single-fixture
isolation decorator, and the first native-log harness reused an identifier.
The final baseline and corrected suites use valid isolated cases and unique
identifiers. No C++ executable changes or rebuilds are needed for this Python
completion-guard correction. The separate64^3 Ti SCF remains pending.

2026-11-04 17:19 EST -- native retry numbering companion guard
-------------------------------------------------------------

RMG InitIo.cpp:456 uses %02d log numbering, which expands beyond two digits.
Nexus's former two-digit-only glob ignored a failed .100.log and could
release a stale successful .99.log. The companion regression reproduces
that assertion failure. rmg.py now accepts only numeric native log suffixes
of at least two digits and selects the maximum numerically; similarly named
unrelated files are excluded. All21 affected completion/dependency tests
pass, and the same three real native success/failure cases retain their
expected status. Evidence and before/after source are in
sol/nexus_rmg_marker_binding_20261104/numeric_retry/. This is completion-guard
coverage, not new physical acceptance or RMG meta development.
