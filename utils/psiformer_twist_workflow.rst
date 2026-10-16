Independent-twist prepare/launch/aggregate (first cut)
====================================================

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

This slice DOES NOT support restart, imported walkers/parameters, checkpoint
family consumption or automatic completed-attempt reuse. Every launch is fresh;
failed/partial attempts stay marked incomplete and cannot become an aggregate.
Recorded checkpoint filenames/hashes are provenance ONLY, not validated restart
admission. CPU trained-state restore followed by VMC has a separate tiny actual
handoff qualification; safe per-family workflow restart is a later bounded cut.
GPU checkpoint import/training is unavailable. Stage7 as a whole is not complete.

Examples use existing tiny LiH inputs, no DFT/ECP or large-system assets. Numeric
caps are per ordinary job and are not process-RSS/adaptive100--300-electron or
aggregate concurrent-job memory qualifications.
