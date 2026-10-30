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
    owners, owner_groups, live, peaks, live_sets = {}, {}, collections.Counter(), {}, {}
    native_peaks, native_last_live = {}, {}
    active, phases, rejected, views, failures = {}, [], [], [], []
    requests, completed_requests = {}, []
    event_counts = collections.Counter()
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

    def group_key(e):
        # Resource addresses and model generations stay on allocation records;
        # grouping by the stable source/role exposes repeated object lifetimes.
        return json.dumps([e['owner'], domain(e), e['source'], e['line'], e['dtype'], e['shape'],
                           e.get('crowd', -1)], separators=(',', ':'))

    def ensure_group(e):
        key = group_key(e)
        if key not in owner_groups:
            if len(owner_groups) >= max_metadata:
                raise ValueError('owner-group bound reached; lifecycle report is incomplete')
            owner_groups[key] = dict(owner=e['owner'], domain=domain(e), source=e['source'], line=e['line'],
                dtype=e['dtype'], shape=e['shape'], crowd=e.get('crowd', -1), allocation_count=0,
                free_count=0, allocated_bytes=0, freed_bytes=0, equal_size_reallocation_count=0,
                open_allocations=0, capacity_owner_count=0, capacity_owner_release_count=0,
                capacity_change_count=0,
                capacity_increase_bytes=0, capacity_decrease_bytes=0, request_count=0,
                allocation_failure_count=0, rejected_request_count=0, unrequested_allocation_count=0,
                size_lifecycles={})
        return owner_groups[key]

    def update(e, capacity, lifecycle_group_key=None):
        ident = e['id']
        previous = owners.get(ident)
        if previous and (previous['domain'] != domain(e)):
            raise ValueError('allocation identity changed memory domain')
        if not previous:
            if len(owners) >= max_metadata:
                raise ValueError('metadata bound reached; report is incomplete')
            owner_group_key = lifecycle_group_key or group_key(e)
            previous = owners[ident] = dict(owner=e['owner'], domain=domain(e), bytes=0,
                high_water_bytes=0, source=e['source'], line=e['line'], dtype=e['dtype'], shape=e['shape'],
                resource=e.get('resource', 0), crowd=e.get('crowd', -1), generation=e['generation'],
                first_seen_ns=e['mono_ns'], last_seen_ns=e['mono_ns'], allocation_ns=None,
                release_ns=None, allocation_lifetime_ns=None, allocation_count=0, free_count=0,
                allocated_bytes=0, freed_bytes=0, free_failed_count=0, capacity_change_count=0,
                capacity_increase_bytes=0, capacity_decrease_bytes=0,
                capacity_live_since_ns=None, observed_capacity_lifetime_ns=0,
                capacity_release_ns=None, group_key=owner_group_key, allocation_mode=None)
            group = owner_groups.get(owner_group_key)
            if group is None:
                group = ensure_group(e)
                previous['group_key'] = group_key(e)
            if e['kind'] == 'capacity':
                group['capacity_owner_count'] += 1
        key = previous['domain']
        old_capacity = previous['bytes']
        live[key] += capacity - previous['bytes']
        previous['bytes'] = capacity
        previous['high_water_bytes'] = max(previous['high_water_bytes'], capacity)
        previous['last_seen_ns'] = e['mono_ns']
        if live[key] > peaks.get(key, 0):
            peaks[key] = live[key]
            live_sets[key] = [i for i, o in owners.items() if o['domain'] == key and o['bytes']]
        # Independent peaks for each overlapping thread's host scope, per domain.
        # These are process-owned totals observed during that scope, never sums of peaks.
        for phase in active.values():
            phase['peak_bytes'][key] = max(phase['peak_bytes'].get(key, 0), live[key])
        return previous, old_capacity

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
                    event_counts[kind] += 1
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
                    if kind == 'request':
                        if e['id'] in requests or e['id'] in owners:
                            raise ValueError('duplicate allocation request identity')
                        group = ensure_group(e)
                        group['request_count'] += 1
                        requests[e['id']] = dict(owner=e['owner'], domain=domain(e), source=e['source'],
                            category=e['category'], backend=e['backend'], device=e['device'],
                            line=e['line'], dtype=e['dtype'], shape=e['shape'], resource=e.get('resource', 0),
                            crowd=e.get('crowd', -1), generation=e['generation'], requested_bytes=e['bytes'],
                            request_ns=e['mono_ns'], outcome='pending', _group_key=group_key(e))
                    elif kind == 'allocated':
                        if kind == 'allocated' and e['id'] in owners:
                            raise ValueError('duplicate physical allocation identity')
                        request = requests.pop(e['id'], None)
                        if request and request['domain'] != domain(e):
                            raise ValueError('allocation outcome changed request memory domain')
                        owner, _ = update(e, e['bytes'], request['_group_key'] if request else None)
                        if request:
                            request.update(outcome='allocated', resolved_ns=e['mono_ns'], allocated_bytes=e['bytes'])
                            owner['request_ns'] = request['request_ns']
                            owner['requested_bytes'] = request['requested_bytes']
                            request.pop('_group_key', None)
                            detail(completed_requests, request)
                        else:
                            owner['request_ns'] = None
                            owner['requested_bytes'] = None
                            owner_groups[owner['group_key']]['unrequested_allocation_count'] += 1
                        owner['allocation_ns'] = e['mono_ns']
                        owner['allocation_count'] += 1
                        owner['allocated_bytes'] += e['bytes']
                        owner['allocation_mode'] = ('device_owned_capacity' if e['category'] == 'device_capacity'
                                                    else 'tracked_host_container_payload')
                        group = owner_groups[owner['group_key']]
                        group['allocation_count'] += 1
                        group['allocated_bytes'] += e['bytes']
                        size_key = str(e['bytes'])
                        size = group['size_lifecycles'].setdefault(size_key,
                            dict(bytes=e['bytes'], allocation_count=0, free_count=0,
                                 active_allocations=0, sequential_equal_size_reallocations=0))
                        if size['allocation_count'] and size['active_allocations'] == 0:
                            size['sequential_equal_size_reallocations'] += 1
                            group['equal_size_reallocation_count'] += 1
                        size['allocation_count'] += 1
                        size['active_allocations'] += 1
                    elif kind == 'capacity':
                        owner, old_capacity = update(e, e['bytes'])
                        owner['capacity_change_count'] += 1
                        group = owner_groups[owner['group_key']]
                        group['capacity_change_count'] += 1
                        delta = e['bytes'] - old_capacity
                        if delta > 0:
                            owner['capacity_increase_bytes'] += delta
                            group['capacity_increase_bytes'] += delta
                            if old_capacity == 0:
                                owner['capacity_live_since_ns'] = e['mono_ns']
                        elif delta < 0:
                            owner['capacity_decrease_bytes'] += -delta
                            group['capacity_decrease_bytes'] += -delta
                            if e['bytes'] == 0 and owner['capacity_live_since_ns'] is not None:
                                owner['observed_capacity_lifetime_ns'] += e['mono_ns'] - owner['capacity_live_since_ns']
                                owner['capacity_live_since_ns'] = None
                    elif kind == 'release':
                        if e['id'] not in owners:
                            raise ValueError('release has no recorded capacity owner')
                        # Physical identity retains its allocation-time provenance.
                        previous = owners[e['id']]
                        if previous['release_ns'] is not None or previous['capacity_release_ns'] is not None:
                            raise ValueError('allocation identity released more than once')
                        category, backend, device = previous['domain'].split(':')
                        release = dict(e, category=category, backend=backend, device=int(device))
                        owner, old_capacity = update(release, 0)
                        if owner['allocation_ns'] is not None:
                            owner['release_ns'] = e['mono_ns']
                            owner['allocation_lifetime_ns'] = e['mono_ns'] - owner['allocation_ns']
                            owner['free_count'] += 1
                            owner['freed_bytes'] += old_capacity
                            group = owner_groups[owner['group_key']]
                            group['free_count'] += 1
                            group['freed_bytes'] += old_capacity
                            size = group['size_lifecycles'][str(old_capacity)]
                            if size['active_allocations'] <= 0:
                                raise ValueError('allocation lifecycle free has no active allocation')
                            size['active_allocations'] -= 1
                            size['free_count'] += 1
                        else:
                            if owner['capacity_live_since_ns'] is not None:
                                owner['observed_capacity_lifetime_ns'] += e['mono_ns'] - owner['capacity_live_since_ns']
                                owner['capacity_live_since_ns'] = None
                            owner['capacity_release_ns'] = e['mono_ns']
                            owner_groups[owner['group_key']]['capacity_owner_release_count'] += 1
                    elif kind == 'rejected':
                        request = requests.pop(e['id'], None)
                        if request:
                            if request['domain'] != domain(e):
                                raise ValueError('rejection outcome changed request memory domain')
                            request.update(outcome='rejected', resolved_ns=e['mono_ns'], rejected_bytes=e['bytes'])
                            owner_groups[request['_group_key']]['rejected_request_count'] += 1
                            request.pop('_group_key', None)
                            detail(completed_requests, request)
                        detail(rejected, dict(e, owned_live_bytes=dict(live)))
                    elif kind in ('free_failed', 'allocation_failed', 'snapshot_failed', 'manifest_failed', 'invalid_view'):
                        detail(failures, e)
                        if kind == 'free_failed' and e['id'] in owners:
                            owners[e['id']]['free_failed_count'] += 1
                        if kind == 'allocation_failed':
                            request = requests.pop(e['id'], None)
                            if request and request['domain'] != domain(e):
                                raise ValueError('allocation failure changed request memory domain')
                            group = owner_groups[request['_group_key']] if request else ensure_group(e)
                            group['allocation_failure_count'] += 1
                            if request:
                                request.update(outcome='allocation_failed', resolved_ns=e['mono_ns'])
                                request.pop('_group_key', None)
                                detail(completed_requests, request)
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
    if requests:
        warnings.append(f'{len(requests)} allocation requests have no recorded outcome')
    if truncated:
        warnings.append('detail output truncated at configured bound; byte totals remain valid')
    open_allocations, open_capacity_owners = [], []
    for ident, owner in owners.items():
        group = owner_groups[owner['group_key']]
        if owner['allocation_ns'] is not None and owner['release_ns'] is None:
            group['open_allocations'] += 1
            open_allocations.append(dict(id=ident, owner=owner['owner'], domain=owner['domain'],
                bytes=owner['bytes'], source=owner['source'], line=owner['line'],
                free_failed_count=owner['free_failed_count']))
        if owner['capacity_live_since_ns'] is not None:
            owner['observed_capacity_lifetime_ns'] += max(0, last_time - owner['capacity_live_since_ns'])
        if owner['allocation_ns'] is None and owner['bytes'] > 0:
            open_capacity_owners.append(dict(id=ident, owner=owner['owner'], domain=owner['domain'],
                bytes=owner['bytes'], source=owner['source'], line=owner['line']))
    return dict(schema=1, path=str(path), process=header, complete=not warnings,
        warnings=warnings, detail_truncated=truncated, events=events,
        scopes='instrumented physical capacity only; host snapshots are boundary observations',
        lifecycle_scope='allocated/release pairs cover selected owned payloads; host_capacity changes are observations, not allocator interception',
        phase_scope='process-owned totals per memory domain during each thread host scope; overlaps are not additive; no device completion implied',
        peak_semantics='trace-order reconstruction; concurrent event reordering can differ from physical native high water; host growth is boundary-sampled',
        peak_bytes=peaks, final_live_bytes=dict(live), peak_owner_ids_in_trace_order=live_sets,
        native_counter_peak_bytes=native_peaks, native_counter_last_live_bytes=native_last_live,
        event_counts=dict(event_counts), owners=owners, owner_groups=owner_groups,
        open_allocations=open_allocations, open_capacity_owners=open_capacity_owners,
        pending_allocation_requests=[{key: value for key, value in request.items() if key != '_group_key'}
                                     for request in requests.values()],
        allocation_requests=completed_requests,
        phases=phases, rejected_requests=rejected, packed_views=views, failures=failures)


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
