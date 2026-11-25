"""Replay a bounded slot-five repair, then explore from the repaired graph."""
import argparse
from itertools import permutations
from pathlib import Path
from statistics import mean
import time

import networkx as nx

from common import load_upstream
from run_feedback_study import graph_line, observations_check
from run_local_causal import write_json
from run_verified_graph_search import branch, read, replay_score

REPAIR_SLOT = 5


def repair_one_edge(benchmark, observations, prefix):
    nodes = benchmark.nodes
    edges = tuple(permutations(sorted(nodes), 2))
    parents = set()
    for row in prefix:
        parsed = benchmark.parse_llm_response(row['answer'])
        if parsed is not None:
            parents.add(tuple(sorted(parsed.edges)))
    ordered = sorted(parents, key=lambda candidate: (
        len(observations_check(nodes, observations, candidate)), candidate))
    checked = set(parents)
    checks = 0
    for parent in ordered:
        for edge in edges:
            candidate = tuple(sorted(set(parent) ^ {edge}))
            if candidate in checked or len(candidate) > benchmark.max_edges:
                continue
            checked.add(candidate)
            checks += 1
            graph = nx.DiGraph()
            graph.add_nodes_from(nodes)
            graph.add_edges_from(candidate)
            if nx.is_directed_acyclic_graph(graph) and not observations_check(nodes, observations, candidate):
                return candidate, {'parent': parent, 'edge': edge,
                                   'action': 'remove' if edge in parent else 'add'}, checks
    return None, None, checks


def recorded_call(row):
    return {'response': {'message': {'content': row['answer']}, 'done_reason': row['done_reason'],
                         'prompt_eval_count': row['prompt_tokens'], 'eval_count': row['output_tokens']},
            'seconds': row['seconds']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    path = args.run_dir
    config = read(path/'config.json')
    source = Path(config['source'])
    module = load_upstream('causal', 'run_causal_benchmark.py')
    data = {'n4': source/'n4_test.json', 'n6': source/'n6_test.json',
            'partial_n4': path/'partial_test.json'}
    benchmarks = {key: module.CausalBenchmarkEnhanced(str(file)) for key, file in data.items() if file.exists()}
    instances = {key: {r['observation_set_id']: r for r in benchmark.all_observation_sets}
                 for key, benchmark in benchmarks.items()}
    trials = []
    for previous in read(path/'results.json'):
        group, item = previous['group'], previous['item']
        benchmark = benchmarks[group]
        instance = instances[group][item]
        observations = instance['observations']
        base_rows = previous['baseline_rows']
        prefix = base_rows[:REPAIR_SLOT-1]
        attempted = not any(row['valid'] for row in prefix)
        calls = [recorded_call(row) for row in base_rows]
        repair, edit, checks = None, None, 0
        started = time.perf_counter()
        if attempted:
            repair, edit, checks = repair_one_edge(benchmark, observations, prefix)
        elapsed = time.perf_counter()-started
        if repair is not None:
            calls[REPAIR_SLOT-1] = {'response': {'message': {'content': graph_line(repair)},
                                               'done_reason': 'one_edge_repair'}, 'seconds': elapsed}
        modified, trace = branch(benchmark, observations, calls)
        if repair is not None:
            trace[REPAIR_SLOT-1] = {'source': 'one_edge_repair', 'edit': edit, 'checks': checks}
        official, rows = replay_score(benchmark, instance, modified)
        if repair is not None:
            assert rows[REPAIR_SLOT-1]['new_valid']
        for row, step in zip(rows, trace):
            if step['source'] == 'new_local_graph':
                assert row['valid'] and row['new_valid']
        trials.append({'group': group, 'item': item, 'gt': previous['gt'], 'attempted': attempted,
                       'repair_succeeded': repair is not None, 'repair_checks': checks,
                       'repair_seconds': elapsed, 'baseline': previous['baseline'], 'search': previous['search'],
                       'early_repair_search': official, 'rows': rows, 'trace': trace})
    summaries = {}
    subsets = {'complete': [r for r in trials if r['group'] in ('n4', 'n6')]}
    subsets.update({key: [r for r in trials if r['group'] == key] for key in benchmarks})
    for group, runs in subsets.items():
        if not runs:
            continue
        summaries[group] = {'tasks': len(runs), 'attempts': sum(r['attempted'] for r in runs),
                            'successes': sum(r['repair_succeeded'] for r in runs),
                            'repair_checks': sum(r['repair_checks'] for r in runs)}
        for method in ('baseline', 'search', 'early_repair_search'):
            summaries[group][method] = {'rr': mean(r[method]['recovery_rate'] for r in runs),
                                        'vr': mean(r[method]['valid_rate'] for r in runs),
                                        'distinct': sum(r[method]['n_recovered_gts'] for r in runs)}
        summaries[group]['paired_rr'] = [
            sum((r['early_repair_search']['recovery_rate']-r['baseline']['recovery_rate'])*sign > 0
                for r in runs) for sign in (1, -1)]
    write_json(path/'early_repair_results.json', {'summary': summaries, 'trials': trials})
    for key, summary in summaries.items():
        print(key, summary)


if __name__ == '__main__':
    main()
