Optional WFTrain work counters
==============================

Set ``PSIFORMER_WORK_COUNTER_TRACE`` to a fresh path prefix to write optional
per-rank JSONL rows named ``PREFIX.rankRANK.pidPID.jsonl``. Remove the setting
to keep the counter writer inactive. The sidecar does not modify
``training_info.csv`` or require MPI reductions. Rank totals may be summed for
work, but this file contains no elapsed time.

Each ``work_counters`` row records a phase/iteration, process-local monotonic
timestamp, a driver/callsite ``source`` label, and rank-local batch counts. The
writer defaults to ``WFTrainBatched.cpp``; callers for other VMC flows can pass
their source label on each row. ``qmc_section`` carries the caller-supplied
root/section name and defaults to ``unknown``; reducers keep sections distinct
even when they share a process, phase and iteration. Older rows without this
field are read as ``unknown``. Proposal rounds are sampler substeps per crowd.
Ratio and drift-gradient fields count API batches, not walker-electron
operations. Accepted/rejected fields count move decisions. Hamiltonian fields
count walker Hamiltonian evaluations; the ECP-bearing field counts such walker
evaluations routed through a configured physical ECP Hamiltonian, not
quadrature points or ECP kernel launches. Retained configurations count
stored walker samples in WFTrain snapshots or VMC sample collection. KFAC
factor/inverse fields count successfully published candidate factor generations
and newly built inverse generations; proposal
rejection does not erase the recorded work, and inverse-period reuse does not
count as a refresh.

Requested and computed VALUE/gradient/Laplacian product batch counts are
separate and may overlap: one full evaluation contributes to every product it
computes. If a compatibility path cannot identify what it computed,
``computed_product_status_unknown_batches`` records that call rather than
claiming it was VALUE-only. Counts are API-level work labels; they are not
configuration counts or exclusive timer durations.

``pre_move_full_spatial_handoff_batches`` counts the specific accepted-state
FULL VGL replay before production movement when the sampler finds gradients or
Laplacians invalid. It is distinct from proposal VALUE requests. A VALUE_ONLY
request records returned value work and marks any extra compatibility products
unknown when the API does not report whether it fell back to FULL.

The fine ``spatial_evaluation_api`` and ``hamiltonian_evaluation_api`` timers
measure the enclosing native host calls. They may include a wait performed by
that API, but do not certify asynchronous device completion; native timer scopes
record the controlling/master thread only. ``initial_evaluation`` and
``completed_iteration_wall`` are outer host wall scopes around synchronous
driver calls. The latter continues through iteration inference/checkpoint work.
The existing ``training_info.csv`` ``total_seconds`` field is still sampled
before optional post-iteration inference/checkpoint and keeps its old meaning.

Schema version 1 records use ``kind=work_counters`` and require all fields
defined by ``utils/psiformer_work_report.py``. The reducer validates rows and
sums only counts across the supplied rank files by phase and iteration. It
preserves individual records and reports missing, duplicate or malformed data.
This sidecar is opt-in telemetry. The sampled atomic updates and per-row file
flush should be measured separately from ordinary runs when enabled.
