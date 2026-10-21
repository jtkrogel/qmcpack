#!/usr/bin/env python3
"""Stream a schema-1 PsiFormer memory trace into a scope-aware JSON report."""
import argparse
import collections
import json
import sys


KINDS = {
    'begin', 'end', 'request', 'admitted', 'rejected', 'allocation_failed',
    'allocated', 'release', 'free_failed', 'capacity', 'view', 'invalid_view',
    'phase_begin', 'phase_end', 'boundary', 'snapshot_failed', 'manifest_failed',
    'context', 'ownership',
}


def summarize(path, max_metadata=100000, max_details=1000):
    """Do not merge processes or subtract unrelated whole-device/RSS maxima."""
    owners, live, peaks, live_sets = {}, collections.Counter(), {}, {}
    native_peaks, native_last_live = {}, {}
    active, phases, rejected, views, failures = {}, [], [], [], []
    warnings = []
    header = None
    last_kind, last_time, events, ended_count = None, 0, 0, None
    truncated = False

    def detail(target, item):
        nonlocal truncated
        if len(target) < max_details:
            target.append(item)
        else:
            truncated = True

    def domain(e):
        return f"{e['category']}:{e['backend']}:{e['device']}"

    def update(e, capacity):
        ident = e['id']
        previous = owners.get(ident)
        if previous and (previous['domain'] != domain(e)):
            raise ValueError('allocation identity changed memory domain')
        if not previous:
            if len(owners) >= max_metadata:
                raise ValueError('metadata bound reached; report is incomplete')
            previous = owners[ident] = dict(owner=e['owner'], domain=domain(e), bytes=0,
                high_water_bytes=0, source=e['source'], line=e['line'], dtype=e['dtype'],
                resource=e.get('resource', 0), crowd=e.get('crowd', -1), generation=e['generation'])
        key = previous['domain']
        live[key] += capacity - previous['bytes']
        previous['bytes'] = capacity
        previous['high_water_bytes'] = max(previous['high_water_bytes'], capacity)
        if live[key] > peaks.get(key, 0):
            peaks[key] = live[key]
            live_sets[key] = [i for i, o in owners.items() if o['domain'] == key and o['bytes']]
        # Independent peaks for each overlapping thread's host scope, per domain.
        # These are process-owned totals observed during that scope, never sums of peaks.
        for phase in active.values():
            phase['peak_bytes'][key] = max(phase['peak_bytes'].get(key, 0), live[key])

    try:
        with open(path, encoding='utf-8') as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    e = json.loads(line)
                    if type(e.get('schema')) is not int or e.get('schema') != 1 or e.get('kind') not in KINDS:
                        raise ValueError('unsupported schema or event kind')
                    kind = e['kind']
                    last_kind = kind
                    if kind == 'begin':
                        if header is not None or events:
                            raise ValueError('multiple or misplaced begin events')
                        for field in ('wall_ns', 'mono_ns', 'pid', 'rank'):
                            if field not in e:
                                raise ValueError(f'missing header field {field}')
                        header = e
                        last_time = e['mono_ns']
                        continue
                    if header is None:
                        raise ValueError('missing begin event')
                    if kind == 'end':
                        ended_count = e.get('events')
                        if e.get('dropped_events', 0):
                            warnings.append(f"trace writer dropped {e['dropped_events']} events")
                        if ended_count != events:
                            raise ValueError('end event count does not match stream')
                        continue
                    events += 1
                    for field in ('mono_ns', 'thread', 'id', 'owner', 'category', 'backend',
                                  'device', 'bytes', 'logical_bytes', 'source', 'line', 'parent',
                                  'offset_bytes', 'dtype', 'shape', 'generation'):
                        if field not in e:
                            raise ValueError(f'missing event field {field}')
                    for field in ('bytes', 'logical_bytes', 'offset_bytes', 'id', 'parent'):
                        if type(e[field]) is not int or e[field] < 0:
                            raise ValueError(f'invalid nonnegative integer {field}')
                    if e['mono_ns'] < last_time:
                        raise ValueError('nonmonotonic event timestamps')
                    last_time = e['mono_ns']
                    if e.get('native_snapshot'):
                        key = domain(e)
                        native_peaks[key] = max(native_peaks.get(key, 0), e['native_peak_bytes'])
                        native_last_live[key] = e['native_live_bytes']
                    if kind in ('allocated', 'capacity'):
                        if kind == 'allocated' and e['id'] in owners:
                            raise ValueError('duplicate physical allocation identity')
                        update(e, e['bytes'])
                    elif kind == 'release':
                        if e['id'] not in owners:
                            raise ValueError('release has no recorded capacity owner')
                        # Physical identity retains its allocation-time provenance.
                        previous = owners[e['id']]
                        category, backend, device = previous['domain'].split(':')
                        release = dict(e, category=category, backend=backend, device=int(device))
                        update(release, 0)
                    elif kind == 'rejected':
                        detail(rejected, dict(e, owned_live_bytes=dict(live)))
                    elif kind in ('free_failed', 'allocation_failed', 'snapshot_failed', 'manifest_failed', 'invalid_view'):
                        detail(failures, e)
                        if kind in ('snapshot_failed', 'manifest_failed', 'invalid_view'):
                            detail(warnings, f'line {line_number}: {kind}; attribution is incomplete')
                    elif kind in ('context', 'ownership') and e['id'] in owners:
                        owner = owners[e['id']]
                        owner['resource'] = e.get('resource', 0)
                        if kind == 'context':
                            owner['generation'] = e['generation']
                            owner['model_allocation_id'] = e['parent']
                    elif kind == 'view':
                        if e['parent'] not in owners:
                            raise ValueError('view has no physical owner')
                        owner = owners[e['parent']]
                        if e['offset_bytes'] + e['bytes'] > owner['bytes']:
                            raise ValueError('packed view exceeds its owner capacity')
                        detail(views, e)
                    elif kind == 'phase_begin':
                        if e['id'] in active or len(active) >= max_metadata:
                            raise ValueError('duplicate phase or active phase bound reached')
                        active[e['id']] = dict(id=e['id'], owner=e['owner'], thread=e['thread'],
                            begin_ns=e['mono_ns'], entry_live_bytes=dict(live), peak_bytes=dict(live))
                    elif kind == 'phase_end':
                        if e['id'] not in active:
                            raise ValueError('phase end has no begin')
                        phase = active.pop(e['id'])
                        if phase['thread'] != e['thread']:
                            raise ValueError('phase ended on a different thread')
                        phase.update(end_ns=e['mono_ns'], exit_live_bytes=dict(live))
                        detail(phases, phase)
                except (ValueError, TypeError, KeyError) as error:
                    warnings.append(f'line {line_number}: {error}')
                    break
    except OSError as error:
        warnings.append(str(error))
    if last_kind != 'end' or ended_count != events:
        warnings.append('missing final end marker (crash, SIGKILL, I/O loss, or late static destruction)')
    if active:
        warnings.append(f'{len(active)} unfinished host phases')
    if truncated:
        warnings.append('detail output truncated at configured bound; byte totals remain valid')
    return dict(schema=1, path=str(path), process=header, complete=not warnings,
        warnings=warnings, detail_truncated=truncated, events=events,
        scopes='instrumented physical capacity only; host snapshots are boundary observations',
        phase_scope='process-owned totals per memory domain during each thread host scope; overlaps are not additive; no device completion implied',
        peak_semantics='trace-order reconstruction; concurrent event reordering can differ from physical native high water; host growth is boundary-sampled',
        peak_bytes=peaks, final_live_bytes=dict(live), peak_owner_ids_in_trace_order=live_sets,
        native_counter_peak_bytes=native_peaks, native_counter_last_live_bytes=native_last_live,
        owners=owners, phases=phases, rejected_requests=rejected, packed_views=views, failures=failures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', nargs='+')
    parser.add_argument('--max-metadata', type=int, default=100000)
    parser.add_argument('--max-details', type=int, default=1000)
    args = parser.parse_args()
    if args.max_metadata < 1 or args.max_details < 1:
        parser.error('bounds must be positive')
    reports = [summarize(path, args.max_metadata, args.max_details) for path in args.trace]
    json.dump(reports[0] if len(reports) == 1 else reports, sys.stdout, indent=2)
    print()
    return 0 if all(report['complete'] for report in reports) else 2


if __name__ == '__main__':
    sys.exit(main())
