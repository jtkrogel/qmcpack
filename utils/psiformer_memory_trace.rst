PsiFormer memory attribution
===========================

Enable the optional trace before process startup::

  mkdir -p /path/to/run/memory
  export PSIFORMER_MEMORY_TRACE=/path/to/run/memory/trace
  # Launch the usual application command and input.
  python3 utils/psiformer_memory_report.py /path/to/run/memory/trace.*.jsonl > report.json

Use a fresh prefix for each run. Files are named
``PREFIX.rankRANK.pidPID.jsonl``. Rank comes from OMPI_COMM_WORLD_RANK,
PMI_RANK, or SLURM_PROCID, in that order. Missing or malformed rank is explicitly
``unknown``; it is not guessed to be zero or a subcommunicator rank. When rank
is unavailable on multiple hosts, use a separate prefix directory per host to
avoid PID collisions. The existing ``PSIFORMER_GPU_MEMORY_SUMMARY=1`` control
is independent and remains available.

Tracing changes no admission limits, allocation policy, model arithmetic,
optimizer recurrence, timer level, or synchronization policy. The disabled path
does not construct the sink, open files, query devices, synchronize streams,
allocate a metadata registry, or acquire a trace lock. Preparation-site device
labels are literals. Enabled recording is serialized and flushed per event;
there is no per-electron, per-parameter, or per-sample trace. History is streamed
instead of retained in the application.

Owned capacity and coverage
---------------------------

Device physical allocation events wrap the existing DeviceBuffer ownership
transitions. Each actual allocation has a fresh process-local ID. A move retains
the ID; shared model images share it; views acquire no capacity. Growth records
the replacement while the predecessor is still live. Admission still occurs
before malloc, failed malloc still unwinds the existing reservation, and a failed
free leaves its original allocation ID charged even when a replacement was
published. The rejected event names the exact source member and request before
malloc. Its parent identifies any predecessor allocation. Direct use of the
metrics helper outside DeviceBuffer is not independently traced.

Named device coverage includes real model metadata and parameter images,
positions/features/attention, retained layer states, orbital jets, determinant
matrices/LU/inverses/solve, determinant products and spatial traces, outputs and
status, score reverse/JVP/contractions, target kinetic forward/reverse owners,
complex value/spatial/score/kinetic/products, and device Adam. Mixed FP32 affine
weights/source/target are physical shadows; their events report ``fp32`` while
FP64 islands report ``fp64``. Struct/status buffers use ``opaque_pod``. Other
DeviceBuffer users still emit physical events with an explicit unlabelled source;
this includes some standalone validation, coordinate/sampler, nonlocal, and
SR/PCG helpers. Vendor-private workspaces are outside this ownership scope.

Packed spatial-jet manifests describe value, gradient, and Laplacian plane
families using existing SpatialJetLayout extents. ``parent`` is the physical
allocation and alias group; ``offset_bytes`` and ``bytes`` bound the view.
``shape`` records configuration, plane, and element dimensions plus element
strides. A strided view's byte extent includes gaps, which are not additional
owned bytes. A view lives until the parent release; graph reuse can change its
logical contents. Model context events link successful inference buffers to
their shared parameter allocation, resource address, and generation. Generation
zero without a context means unspecified; a known initial model can also have
generation zero. Resource addresses are process-local provenance, not stable
identifiers across runs. Layer tape allocations have separate physical IDs;
the current manifest does not enumerate each retained layer's logical views.

HOST capacity observations use existing vector-capacity APIs at preparation or
diagnostic boundaries. DirectBatchWorkspace reports dense/sparse logical data,
outputs, sparse tile positions, value/full/active workspaces and its shared
spatial arena separately. Workspace moves preserve observation identity; a
staged replacement records actual capacity before retiring the previous owner.
Retained crowd snapshots report actual R/G/L and scalar vector capacities with
crowd identity. Real derivative operators report per-operator copied positions,
gradients/masses and sample-product banks; their shared host score/kinetic lane
and parameter channels are charged once on the shared scratch owner. Its device
component is excluded using the same session's current device capacity, never a
peak or forecast. Complex CPU derivative lanes, copied sample/product banks,
and value/full/active clone workspaces use existing exact diagnostics. Adam
reports its existing host moment mirrors. KFAC reports accepted/pending factors,
exported inverse banks, prepared preconditioner storage, and solve vectors.

Host observations are boundary snapshots, not interceptions of std::vector
allocations. They can miss temporary coexistence within a growth operation,
factor update, checkpoint restore, or exception before the next observation.
Their reconstructed peaks are partial boundary-observed capacity estimates.
Concurrent or delayed observations/releases can also change reconstructed
coexistence, so these are not unconditional mathematical lower bounds.
Immutable host model/parameter images, ParticleSet internals other than copied
R/G/L, teacher/cache objects, host ECP replay banks, generic optimizer workspaces,
allocator bookkeeping, runtime heaps, and library storage remain untracked.
Do not interpret tracked totals as process fit or subtract them from RSS.

Phases and timestamps
---------------------

Phase scopes cover sampling, retained configuration/product preparation,
objective/update work, affine observations, factor update/inversion, KFAC solve,
candidate/publication/refresh and checkpointing. They supplement the existing
phase reports and timers; they do not add MPI collectives. A host scope marks
entry/exit and submission/lifetime boundaries, not device completion. Existing
stream synchronization remains authoritative.

The report reconstructs independent entry live bytes, phase-local peaks and exit
live bytes for each host scope, keyed by scope ID and thread and separated by
memory category/backend/device. Pre-existing retained owners are included.
These are process-owned totals observed during each thread's scope, including
concurrent activity from other threads. Overlapping scopes and their peaks are
not additive; they never reset a shared historical counter. Saved owner IDs
describe the live set reconstructed in trace order.

Concurrent threads update native ownership before acquiring the trace sink lock.
Their records can therefore be reordered relative to physical allocation/free
transitions. Reconstructed ``peak_bytes`` and ``peak_owner_ids_in_trace_order``
are explicitly trace-order observations, not guaranteed concurrent physical
peaks/live sets. Device records also sample the existing native live/interval-peak
counters without GPU calls. The report presents ``native_counter_peak_bytes``
separately as the maximum observed native interval high water and
``native_counter_last_live_bytes`` as the last sampled native live value. Atomic
counter fields are not a coherent transaction during concurrent activity.
Reconcile live totals at a quiescent boundary; no extra allocator lock or stream
synchronization was introduced for attribution.

Each file begins with schema version 1, PID, rank, wall-clock nanoseconds since
the Unix epoch, and monotonic nanoseconds. Event monotonic timestamps are taken
under the sink lock. Align another source using::

  event_wall_ns = begin.wall_ns + event.mono_ns - begin.mono_ns

The clock pair has ordinary sampling uncertainty, and cross-host clock skew
still matters. Join native counters, RSS samples and vendor samples at matching
boundaries/timestamps, preserving each measurement's scope. A native owned peak,
a sampled whole-device peak, and process RSS are different quantities; subtracting
their unrelated maxima does not measure library overhead. Boundary RSS events
are observations only and never participate in owned-capacity sums.

Schema, failures and reader bounds
---------------------------------

All records are JSON objects, one per line. Normal events contain ``schema``,
``kind``, ``mono_ns``, ``thread``, ``id``, ``owner``, ``category``, ``backend``,
``device``, ``bytes``, ``logical_bytes``, ``source``, ``line``, ``parent``,
``offset_bytes``, ``dtype``, ``shape``, ``generation``, ``resource`` and ``crowd``.
Device ownership records additionally carry ``native_snapshot``,
``native_live_bytes`` and ``native_peak_bytes``; other records set
``native_snapshot`` false. RSS boundary generation denotes the iteration.
Unknown device/crowd is -1, unknown resource/parent is 0, and unavailable textual
metadata is ``unknown``. Source on host records is the observation site.
For device allocation requests logical bytes are the requested prepared extent;
they do not claim a smaller currently active evaluation extent. Optional extra
fields may be added within schema 1; readers reject unsupported schema versions.

Kinds are request/admitted/rejected/allocated/allocation_failed/release/free_failed,
capacity, ownership, context, view/invalid_view, phase_begin/phase_end, boundary,
snapshot_failed/manifest_failed, plus begin/end lifecycle records. Allocated
adds physical bytes; capacity replaces an owner's previous host observation;
release removes capacity only after successful device free. All other kinds
are descriptive. Admission failures and failed frees are expected evidence,
not schema errors. The reader preserves their owner/source and live-set context.

Startup path errors warn once and disable file recording while scientific
execution continues. Runtime I/O/metadata errors also warn once, suppress further
recording, and never replace original allocation exceptions or destructor
behavior. Dropped event counts are included in an exit marker if it can still be
written. Per-event flushing reduces buffered loss but cannot guarantee data after
SIGKILL, host failure, disk-full, or failed writes. The sink intentionally has
process lifetime so late static destructors cannot access a destroyed logger.
An atexit marker can precede such destructors; events after the final marker make
the reader explicitly report an incomplete trace. Open host scopes also indicate
incompleteness. A killed run can still contain useful earlier live sets/rejections.

The reader streams input and defaults to at most 100000 allocation/phase metadata
owner entries, 100000 simultaneous active phase entries, and 1000 detailed
phases/views/rejections/failures per output collection.
Use ``--max-metadata`` and ``--max-details`` to adjust these bounds. Reaching the
metadata bound stops reconstruction explicitly; reaching a detail bound truncates
those details while retaining byte totals and emits a warning. Nonzero exit code
2 means incomplete or truncated reporting. Each process is reported separately;
the tool does not combine unrelated process/device peaks.

Focused validation
------------------

``test_psiformer_memory_trace`` is a CPU-only deterministic runtime adapter that
exercises the production DeviceBuffer code. ``test_psiformer_memory_report`` uses
tiny generated traces and that executable for disabled/enabled behavior, growth
overlap, rejection, reservation rollback, malloc/free failures, moves/sharing,
packed aliases, overflow, nested/concurrent phases, recording concurrency,
capacity replacement, schema errors, output bounds, path errors, /dev/full
runtime errors, escaping and late static destruction. Run::

  cmake --build BUILD --target test_psiformer_memory_trace
  ctest --test-dir BUILD -R 'deterministic-unit_test_psiformer_memory_(trace|report)$' --output-on-failure

Hardware validation and repeatable application overhead measurements should use
the same frozen inputs/model versions with tracing off/on. Enabled overhead
includes serialization and file flushes; filesystem, MPI startup, cache state
and small-run noise can dominate. A single small test is not evidence of exactly
zero disabled overhead or of whole-device accounting completeness.
