"""Evaluate single-edge exploration after the first validated model proposal."""
import argparse
from collections import deque
import csv
from datetime import datetime
from itertools import combinations, permutations
import json
from pathlib import Path
import random
from statistics import mean
import time

import networkx as nx
import requests

from common import HERE, digest, load_upstream
from run_constraint_feedback import OPTIONS, extra_feedback, extract_edge_graph
from run_feedback_study import graph_line, make_feedback, observations_check
from run_four_groups import BASE_URL, MODEL, SEEDS, score_calls
from run_local_causal import write_json
from run_thinking_trigger_study import EXCLUDED, THINKING_OPTIONS

SLOTS = 8
PARTIAL_SEED = 991126


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


class VerifiedFrontier:
    """Breadth-first exploration of the valid component containing the anchor."""

    def __init__(self, nodes, max_edges, observations, anchor):
        self.nodes = nodes
        self.max_edges = max_edges
        self.observations = observations
        self.edges = tuple(permutations(sorted(nodes), 2))
        self.pending = deque([(anchor, iter(self.edges))])
        self.checked = {anchor}
        self.anchor = anchor
        self.checks = 0

    def next_graph(self):
        while self.pending:
            parent, edits = self.pending[0]
            for edge in edits:
                added = edge not in parent
                candidate = tuple(sorted(set(parent) ^ {edge}))
                if candidate in self.checked or len(candidate) > self.max_edges:
                    continue
                self.checked.add(candidate)
                self.checks += 1
                graph = nx.DiGraph()
                graph.add_nodes_from(self.nodes)
                graph.add_edges_from(candidate)
                if not nx.is_directed_acyclic_graph(graph):
                    continue
                if observations_check(self.nodes, self.observations, candidate):
                    continue
                self.pending.append((candidate, iter(self.edges)))
                return candidate, {'parent': parent, 'edge': edge, 'action': 'add' if added else 'remove'}
            self.pending.popleft()
        return self.anchor, None


def select_partial_cases(source):
    original = read(HERE/'data/causal_n4_full_observations.json')
    trigger = read(HERE/'runs/qwen3_thinking_trigger_20261112_132335/config.json')
    excluded = EXCLUDED | {key for key, _ in trigger['cases']}
    excluded.update(key for size, key, *_ in read(source/'config.json')['cases'] if size == 'n4')
    nodes = original['metadata']['nodes']
    all_dags = []
    edges = tuple(permutations(nodes, 2))
    for count in range(7):
        for candidate in combinations(edges, count):
            graph = nx.DiGraph()
            graph.add_nodes_from(nodes)
            graph.add_edges_from(candidate)
            if nx.is_directed_acyclic_graph(graph):
                outcome = {a: tuple(int(b in nx.descendants(graph, a)) for b in nodes) for a in nodes}
                all_dags.append((candidate, outcome))
    assert len(all_dags) == 543
    rng = random.Random(PARTIAL_SEED)
    available = sorted((r for r in original['datasets'] if r['observation_set_id'] not in excluded),
                       key=lambda r: r['observation_set_id'])
    picked = []
    used_parents, used_keys = set(), set()
    for n_obs in (2, 3):
        pool = [(parent, obs) for parent in available for obs in combinations(parent['observations'], n_obs)]
        rng.shuffle(pool)
        for parent, obs in pool:
            parent_id = parent['observation_set_id']
            key = tuple(sorted((o['perturbed_node'], tuple(o['effects'][n] for n in nodes)) for o in obs))
            if parent_id in used_parents or key in used_keys:
                continue
            compatible = [candidate for candidate, outcome in all_dags
                          if all(outcome[a] == values for a, values in key)]
            if len(compatible) < 4:
                continue
            used_parents.add(parent_id)
            used_keys.add(key)
            picked.append({'observation_set_id': f'{parent_id}_obs{n_obs}', 'parent_id': parent_id,
                           'n_observations': n_obs, 'observations': list(obs),
                           'n_compatible_graphs': len(compatible),
                           'ground_truth_graphs': [{'nodes': nodes, 'edges': [list(e) for e in candidate]}
                                                   for candidate in compatible]})
            if sum(row['n_observations'] == n_obs for row in picked) == 3:
                break
    assert len(picked) == 6
    return {'metadata': {**original['metadata'], 'max_edges': 6, 'total_observation_sets': len(picked)},
            'datasets': picked}


def replay_score(benchmark, instance, calls):
    responses = iter(calls)

    class Replay:
        def query_with_usage(self, prompt):
            response = next(responses)['response']
            return {'response': response['message']['content'], 'cost': 0,
                    'usage': {'prompt_tokens': response.get('prompt_eval_count', 0),
                              'completion_tokens': response.get('eval_count', 0)}}

    official = benchmark.evaluate_single_observation_set(Replay(), instance, SLOTS, False, max_retries=1)
    rows, counts = score_calls(benchmark, instance, calls)
    assert not official['errors'] and all(official[k] == v for k, v in counts.items())
    return official, rows


def branch(benchmark, observations, baseline):
    """Consume the recorded prefix until the first valid graph, then search."""
    calls, trace = [], []
    frontier = None
    for original in baseline:
        if frontier is None:
            calls.append(original)
            trace.append({'source': 'model_prefix'})
            parsed = benchmark.parse_llm_response(original['response']['message']['content'])
            if parsed is not None and not observations_check(benchmark.nodes, observations, parsed.edges):
                frontier = VerifiedFrontier(benchmark.nodes, benchmark.max_edges, observations,
                                            tuple(sorted(parsed.edges)))
            continue
        started = time.perf_counter()
        previous_checks = frontier.checks
        candidate, edit = frontier.next_graph()
        calls.append({'response': {'message': {'content': graph_line(candidate)}, 'done_reason': 'local_search'},
                      'seconds': time.perf_counter()-started})
        trace.append({'source': 'new_local_graph' if edit else 'frontier_exhausted',
                      'edit': edit, 'checks': frontier.checks-previous_checks})
    return calls, trace


def generate_partial_baseline(benchmark, instance, seed, folder, session):
    calls, history, parsed_history = [], [], []
    last_graph = None
    for slot in range(1, SLOTS+1):
        observations = instance['observations']
        prompt = benchmark.create_prompt(observations, parsed_history)
        prompt += make_feedback('combined', benchmark.nodes, observations, history)
        if history:
            feedback, _ = extra_feedback('diagnostics', benchmark.nodes, observations, history, last_graph, None)
            prompt += feedback
        think = slot == SLOTS and not any(
            candidate is not None and not observations_check(benchmark.nodes, observations, candidate)
            for candidate in history)
        options = {**OPTIONS, 'seed': seed, **(THINKING_OPTIONS if think else {})}
        payload = {'model': MODEL, 'think': think, 'stream': False, 'keep_alive': '10m',
                   'options': options, 'messages': [{'role': 'user', 'content': prompt}]}
        path = folder/f'call_{slot:02d}.json'
        if path.exists():
            call = read(path)
            assert call['request'] == payload
        else:
            started = time.perf_counter()
            response = session.post(BASE_URL+'/api/chat', json=payload, timeout=(10, 300))
            response.raise_for_status()
            call = {'request': payload, 'response': response.json(), 'seconds': time.perf_counter()-started}
            write_json(path, call)
        calls.append(call)
        answer = call['response']['message']['content']
        parsed = benchmark.parse_llm_response(answer)
        if parsed is not None:
            parsed_history.append(parsed)
        history.append(list(parsed.edges) if parsed is not None else None)
        last_graph = extract_edge_graph(answer, benchmark.nodes)
    return calls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=HERE/'runs/qwen3_project_evaluation_20261112_145049')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    output = args.output or HERE/'runs'/('verified_graph_search_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True, exist_ok=True)
    module = load_upstream('causal', 'run_causal_benchmark.py')
    old_config = read(args.source/'config.json')
    partial = select_partial_cases(args.source) if args.partial else None
    session = requests.Session()
    session.trust_env = False
    if partial:
        model_response = session.get(BASE_URL+'/api/tags', timeout=10)
        model_response.raise_for_status()
        model = next(m for m in model_response.json()['models'] if m['name'] == MODEL)
        assert model['digest'] == old_config['model']['digest']
        write_json(output/'partial_test.json', partial)
    config = {'source': str(args.source.resolve()), 'source_summary_sha': digest(args.source/'summary.json'),
              'partial_seed': PARTIAL_SEED if partial else None, 'runner_sha': digest(__file__),
              'evaluator_sha': digest(module.__file__), 'slots': SLOTS,
              'model': old_config['model'], 'options': OPTIONS, 'thinking_overrides': THINKING_OPTIONS}
    path = output/'config.json'
    if path.exists():
        assert read(path) == config
    else:
        write_json(path, config)
    print('OUTPUT_DIR=' + str(output.resolve()), flush=True)
    old = {(r['size'], r['item']): r for r in read(args.source/'summary.json')['results'] if r['method'] == 'slot_8'}
    results = []
    datasets = [(size, args.source/f'{size}_test.json') for size in ('n4', 'n6')]
    if partial:
        datasets.append(('partial_n4', output/'partial_test.json'))
    for group, dataset in datasets:
        benchmark = module.CausalBenchmarkEnhanced(str(dataset))
        for i, instance in enumerate(benchmark.all_observation_sets):
            item = instance['observation_set_id']
            if group == 'partial_n4':
                seed = SEEDS[i % len(SEEDS)]
                folder = output/group/item
                folder.mkdir(parents=True, exist_ok=True)
                baseline = generate_partial_baseline(benchmark, instance, seed, folder, session)
            else:
                reference = old[group, item]
                seed = reference['seed']
                baseline = []
                for row in reference['rows']:
                    folder = 'diagnostics' if row['shared'] else 'slot_8'
                    baseline.append(read(args.source/group/item/folder/f"call_{row['query']:02d}.json"))
            before, before_rows = replay_score(benchmark, instance, baseline)
            if group != 'partial_n4':
                assert before['recovery_rate'] == old[group, item]['official']['recovery_rate']
            proposed, trace = branch(benchmark, instance['observations'], baseline)
            after, after_rows = replay_score(benchmark, instance, proposed)
            for row, step in zip(after_rows, trace):
                if step['source'] == 'new_local_graph':
                    assert row['valid'] and row['new_valid']
            results.append({'group': group, 'item': item, 'seed': seed,
                            'gt': len(instance['ground_truth_graphs']),
                            'baseline': before, 'search': after, 'baseline_rows': before_rows,
                            'search_rows': after_rows, 'trace': trace})
            write_json(output/'results.json', results)
            print(f"{group}/{item} {before['n_recovered_gts']} -> {after['n_recovered_gts']} "
                  f"of {len(instance['ground_truth_graphs'])}", flush=True)
    summarize(output, results)


def summarize(output, results):
    summary = {}
    subsets = {'complete': [r for r in results if r['group'] in ('n4', 'n6')]}
    for group in sorted({r['group'] for r in results}):
        subsets[group] = [r for r in results if r['group'] == group]
    for group, selected in subsets.items():
        stats = {}
        for method in ('baseline', 'search'):
            rows = [row for r in selected for row in r[f'{method}_rows']]
            stats[method] = {k: mean(r[method][k] for r in selected)
                             for k in ('recovery_rate', 'valid_rate', 'novelty_rate')}
            stats[method].update({'unique_valid': sum(r[method]['n_recovered_gts'] for r in selected),
                                  'model_calls': len(rows) if method == 'baseline' else sum(
                                      step['source'] == 'model_prefix' for r in selected for step in r['trace']),
                                  'seconds': sum(row['seconds'] for row in rows),
                                  'output_tokens': sum(row['output_tokens'] for row in rows)})
        deltas = [r['search']['recovery_rate']-r['baseline']['recovery_rate'] for r in selected]
        summary[group] = {'tasks': len(selected), **stats, 'wins': sum(d > 0 for d in deltas),
                          'ties': sum(d == 0 for d in deltas), 'losses': sum(d < 0 for d in deltas)}
    write_json(output/'summary.json', summary)
    fields = ['group', 'item', 'seed', 'gt', 'query', 'answer', 'source', 'valid', 'new_valid',
              'cumulative_recovered', 'edit', 'checks']
    with (output/'search_calls.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for result in results:
            for row, trace in zip(result['search_rows'], result['trace']):
                writer.writerow({**{k: result[k] for k in fields[:4]}, **row, **trace})
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
