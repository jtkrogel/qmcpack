Independent-twist launches and explicit CPU family restart
========================================================

``python3 utils/psiformer_twist_workflow.py MANIFEST --binary QMCPACK --output DIRECTORY``
executes one ordinary XML per fixed twist, serially, in a fresh ``attempt_*``
directory. See ``tests/molecules/LiH_ae_psiformer_complex_cpu/independent_twists.json``.
The binary may be a complex build for all entries; optional real Gamma retains
its real initialization profile. Neither Gamma nor its weight forces a paired
complex head. Prepared XML/logs/raw scalar blocks/checkpoint-artifact hashes,
input and executable hashes, exact fractional representatives, actual skew-cell
Cartesian kappa and normalized weights are retained in ``result.json``.

Input v1 requires self-contained bohr-unit periodic AE sole-PsiFormer XML,
identical physical N/spin/cell/ordered ions/masses/Hamiltonian across entries,
unique safe IDs and distinct positive RNG seeds. Each entry explicitly supplies
the final VMC measurement series and discarded block count. Inputs contain only
``vmc_batch`` or CPU ``wf_train`` followed by ``vmc_batch``. CPU paired training
continues to require both positive finite ``max_clone_numeric_bytes`` and
``maximum_retained_bytes``; ordinary production guards remain authoritative.
GPU entries are initialized-only inference and retain their existing two caps.

Fractional twists are supercell reciprocal components: for the XML lattice ROW
matrix A, ``A*kappa=2*pi*t``. The single-state outer phase is applied by QMCPACK,
not this runner. Representatives are NEVER silently wrapped modulo reciprocal
vectors. Independently trained plus/minus states are not assumed identical or
conjugate; no time-reversal/space-group reduction is performed.

Only fixed-parameter final VMC ``LocalEnergy`` blocks enter the estimator, never
optimization histories or amplitudes. Nonnegative weights are explicitly
normalized by their positive sum (zero-weight entries are still reported).
Existing QMCPACK ``check_scalars.simstats`` supplies autocorrelation-aware MC
errors. For explicitly independent streams the total MC error is the quadrature
of normalized weighted per-twist errors. Shared/correlated sampling is not
supported. This is RAW finite-supercell Ewald energy, not a finite-size correction
or twist-grid integration error. Two-block checks demonstrate execution, NOT
converged means or reliable uncertainty estimates. Existing real scalar output
does not expose imaginary local-energy statistics; none are fabricated here.

The reported ``block_energy_variance`` is the variance of VMC BLOCK-energy
observations, not the per-walker local-energy variance. Console output streams
directly to disk, and file hashes are computed in bounded chunks. Input XML is
parsed and identified from the same bytes; imported head paths and external-file
attributes are rejected even for optional real Gamma.

Every launch still uses a fresh attempt; failed/partial attempts cannot become an
aggregate. Explicit ``--restart-result PRIOR/result.json --iterations N`` is
available ONLY when every prior entry is paired CPU training followed by VMC.
The prior result must be complete and match the exact manifest/grid, binary,
template hashes, physical metadata, profile, initialization/RNG seeds and selected
root stage identity. No caller-supplied checkpoint filename, real-Gamma/GPU or
initialized-only restart, imported parameter XML or automatic attempt reuse.

The existing lazy ``h5py`` training dependency reads only small selected manifest/
progress metadata. One named manifest selects its root envelope and ALL rank
shards, never the latest filename glob. All hashes must match prior provenance.
The selected family is COPIED, not hardlinked, into each fresh private
``restart_family`` directory and verified BEFORE ANY subprocess starts. This
uses explicit additional DISK space for the complete model/sampler payloads;
hashing/copying is bounded-memory, not process-RSS admission. Source families are
untouched. Production C++ remains authoritative for complete schema/physical
stage/optimizer/sampler topology validation before parameter publication.

Only the target iteration limit may increase; other training/sampler controls
remain unchanged. Equal completed target restores with zero optimizer work then
measures through following VMC; no new checkpoint is expected, so the private
input snapshot is recorded as its effective reusable family. Increased target
records the newly published selected family. Root/stage progress counts must
agree, and requested N cannot be below completed count. Result records parent
result hash and per-entry stage/family identities. GPU checkpoint import/training
remains unavailable. Stage7 convergence/symmetry/scientific qualification is not
implied by tiny execution and restart gates.

Examples use existing tiny LiH inputs, no DFT/ECP or large-system assets. Numeric
caps are per ordinary job and are not process-RSS/adaptive100--300-electron or
aggregate concurrent-job memory qualifications.
