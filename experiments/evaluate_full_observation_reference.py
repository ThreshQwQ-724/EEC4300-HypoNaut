"""Exact reference for complete singleton perturbations, using observations only."""
import argparse
from itertools import combinations
import json
from pathlib import Path
import time

import networkx as nx

from run_feedback_study import graph_line
from run_local_causal import write_json


def compatible_edge_sets(nodes, max_edges, observations):
    """Enumerate all DAGs with the measured reachability and edge bound."""
    effects = {row['perturbed_node']: row['effects'] for row in observations}
    if len(observations) != len(nodes) or set(effects) != set(nodes):
        raise ValueError('Reference requires one observation per node')
    if any(set(row) != set(nodes) or any(v not in (0, 1) for v in row.values()) for row in effects.values()):
        raise ValueError('Observations must provide one binary outcome per node')
    reachable = {(a, b) for a in nodes for b in nodes if effects[a][b] == 1}
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(reachable)
    if not nx.is_directed_acyclic_graph(graph) or set(nx.transitive_closure_dag(graph).edges) != reachable:
        raise ValueError('Observations do not define a DAG reachability relation')
    required = set(nx.transitive_reduction(graph).edges)
    optional = sorted(reachable - required)
    for n in range(min(len(optional), max_edges - len(required)) + 1):
        for extra in combinations(optional, n):
            yield tuple(sorted(required.union(extra)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    results = []
    for size in ('n4', 'n6'):
        dataset = json.loads((args.run_dir/f'{size}_test.json').read_text(encoding='utf-8'))
        nodes = dataset['metadata']['nodes']
        max_edges = dataset['metadata']['max_edges']
        for item in dataset['datasets']:
            started = time.perf_counter()
            computed = list(compatible_edge_sets(nodes, max_edges, item['observations']))
            candidates = computed[:8]
            proposals = [graph_line(candidates[i % len(candidates)]) for i in range(8)]
            elapsed = time.perf_counter() - started
            truth = {tuple(sorted(map(tuple, row['edges']))) for row in item['ground_truth_graphs']}
            assert set(computed) == truth and len(computed) == len(truth)
            results.append({'size': size, 'item': item['observation_set_id'], 'n_truth': len(truth),
                            'valid_rate': 1.0, 'novelty_rate': len(candidates)/8,
                            'recovery_rate': len(candidates)/len(truth), 'proposals': proposals,
                            'exact_match_with_reference_set': True, 'seconds': elapsed})
    write_json(args.run_dir/'full_observation_reference.json', results)
    print(f'Exact observation-only reference matches the full answer sets for {len(results)} tasks.')


if __name__ == '__main__':
    main()
