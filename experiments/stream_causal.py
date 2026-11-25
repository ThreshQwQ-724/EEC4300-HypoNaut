"""Stream all causal records, count GTs, check samples, retain complete small instances.

Does NOT claim a full soundness/completeness check of millions of causal graphs.
Memory use is bounded by <=32 retained graphs per current observation plus small exports.
"""
import hashlib
import json
import time
from collections import Counter
from common import DATA, RESULTS, COMMIT, save
import ijson
import networkx as nx


class HashReader:
    def __init__(self, stream):
        self.stream = stream
        self.hasher = hashlib.sha256()
        self.bytes_read = 0

    def read(self, n=-1):
        data = self.stream.read(n)
        self.hasher.update(data)
        self.bytes_read += len(data)
        return data


def check_graph(graph, observations, metadata):
    nodes = metadata['nodes']
    edges = [tuple(e) for e in graph['edges']]
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    return (set(graph['nodes']) == set(nodes) and set(g.nodes) == set(nodes)
            and len(edges) == len(set(edges))
            and len(edges) <= metadata['max_edges']
            and nx.is_directed_acyclic_graph(g)
            and all({n: int(n in nx.descendants(g, obs['perturbed_node'])) for n in nodes}
                    == obs['effects'] for obs in observations))


def main():
    source = DATA / 'causal/datasets/node06/n6_all_observations.json'
    summaries, small, errors = [], [], []
    checked = 0
    start = time.monotonic()
    with source.open('rb') as stream:
        reader = HashReader(stream)
        events = iter(ijson.parse(reader))
        metadata = None
        for prefix, event, value in events:
            if prefix == 'metadata' and event == 'start_map':
                builder = ijson.common.ObjectBuilder()
                builder.event(event, value)
                depth = 1
                while depth:
                    p, e, v = next(events)
                    builder.event(e, v)
                    depth += (e in ('start_map', 'start_array')) - (e in ('end_map', 'end_array'))
                metadata = builder.value
                continue
            if prefix.startswith('datasets_by_n_observations.') and prefix.count('.') == 2 and event == 'start_map':
                row_prefix = prefix
                row_builder = ijson.common.ObjectBuilder()
                row_builder.event(event, value)
                retained, count = [], 0
                while True:
                    p, e, v = next(events)
                    if p == row_prefix + '.ground_truth_graphs' and e == 'start_array':
                        # Keep an empty array in the metadata builder, stream graphs separately.
                        row_builder.event('start_array', None)
                        while True:
                            gp, ge, gv = next(events)
                            if gp == p and ge == 'end_array':
                                break
                            if gp == p + '.item' and ge == 'start_map':
                                count += 1
                                collect = count <= 32
                                gb = ijson.common.ObjectBuilder() if collect else None
                                if collect:
                                    gb.event(ge, gv)
                                depth = 1
                                while depth:
                                    _, ee, vv = next(events)
                                    if collect:
                                        gb.event(ee, vv)
                                    depth += (ee in ('start_map', 'start_array')) - (ee in ('end_map', 'end_array'))
                                if collect:
                                    retained.append(gb.value)
                        row_builder.event('end_array', None)
                        continue
                    row_builder.event(e, v)
                    if p == row_prefix and e == 'end_map':
                        break
                row = row_builder.value
                row_id = row['observation_set_id']
                if row['n_compatible_graphs'] != count:
                    errors.append({'id': row_id, 'error': 'declared_count_mismatch'})
                to_check = retained if count <= 32 else retained[:2]
                for graph in to_check:
                    checked += 1
                    if not check_graph(graph, row['observations'], metadata):
                        errors.append({'id': row_id, 'error': 'invalid_sampled_graph', 'graph': graph})
                summaries.append({'id': row_id, 'n_observations': row['n_observations'],
                                  'n_ground_truths': count, 'n_graphs_validated': len(to_check)})
                if 2 <= count <= 32 and len(small) < 20:
                    row['ground_truth_graphs'] = retained
                    small.append(row)
                if len(summaries) % 25 == 0:
                    print(f'{len(summaries)} observations; {reader.bytes_read/1e9:.2f} GB read; '
                          f'{time.monotonic()-start:.0f}s', flush=True)
        fingerprint = reader.hasher.hexdigest()
    groups = {}
    for n in sorted({r['n_observations'] for r in summaries}):
        counts = [r['n_ground_truths'] for r in summaries if r['n_observations'] == n]
        groups[str(n)] = {'instances': len(counts), 'gt_min': min(counts),
                          'gt_max': max(counts), 'gt_mean': sum(counts)/len(counts),
                          'total_gt_entries': sum(counts)}
    if len(summaries) != metadata['total_observation_sets']:
        errors.append({'error': 'metadata_observation_count_mismatch'})
    report = {'source': str(source), 'source_sha256': fingerprint,
              'size_bytes': source.stat().st_size, 'upstream_commit': COMMIT,
              'metadata': metadata, 'observation_sets': len(summaries), 'groups': groups,
              'graph_entries_counted': sum(r['n_ground_truths'] for r in summaries),
              'graph_entries_validated': checked, 'errors': errors,
              'scope': 'Full JSON scan and per-record counts; sampled validity only; completeness NOT verified.',
              'seconds': round(time.monotonic()-start, 2), 'instances': summaries}
    save('causal_audit.json', report)
    subset = {'metadata': {**metadata, 'total_observation_sets': len(small),
                          'subset_note': 'First up to 20 file-order instances with 2..32 GTs; smoke test only.'},
              'datasets_by_n_observations': {}}
    for row in small:
        subset['datasets_by_n_observations'].setdefault(str(row['n_observations']), []).append(row)
    save('causal_small_subset.json', subset)
    print(json.dumps({k:v for k,v in report.items() if k not in ('instances','metadata')}, indent=2))
    if errors:
        raise SystemExit('Causal audit found errors; see results/causal_audit.json')


if __name__ == '__main__':
    main()
