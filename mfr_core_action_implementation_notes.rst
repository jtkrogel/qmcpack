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
