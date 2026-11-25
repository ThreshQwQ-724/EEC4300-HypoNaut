"""Evaluate reference-graph constraints and concrete graph diagnostics."""
import argparse
from collections import defaultdict
import csv
from datetime import datetime
import json
from pathlib import Path
import random
import re
from statistics import mean
import time

import networkx as nx
import requests

from common import COMMIT, HERE, load_upstream, digest
from run_feedback_study import graph_line, make_feedback, observations_check, verify_checker
from run_four_groups import BASE_URL, MODEL, SEEDS, SHARED, score_calls
from run_local_causal import write_json

GROUPS = {
    'control': (False, False),
    'retention': (True, False),
    'diagnostics': (False, True),
    'both': (True, True),
}
DEVELOPMENT = ('n4_219', 'n4_085', 'n4_182', 'n4_116')
OPTIONS = {**SHARED, 'temperature': .6, 'top_p': .95}


def extract_edge_graph(answer, nodes):
    """Read a graph line for diagnostics even when it contains a cycle."""
    match = re.search(r'^\s*Graph\s*:\s*([^\r\n]+)', answer, re.I | re.M)
    if not match:
        return None
    line = match[1].strip().rstrip('.;').replace('→', '->')
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    if line.lower() == 'no edges':
        return graph
    edges = []
    for part in line.split(','):
        edge = re.fullmatch(r'\s*(\w+)\s*->\s*(\w+)\s*', part)
        if not edge or any(node not in nodes for node in edge.groups()):
            return None
        edges.append(edge.groups())
    graph.add_edges_from(sorted(edges))
    return graph


def cycle_edges(graph):
    if graph is not None and not nx.is_directed_acyclic_graph(graph):
        return nx.find_cycle(graph)
    return []


def extra_feedback(group, nodes, observations, history, graph, reference):
    """Return prompt additions and the updated reference without accessing labels."""
    retain, diagnose = GROUPS[group]
    lines = []
    if retain and history[-1] is not None:
        candidate = history[-1]
        mismatches = observations_check(nodes, observations, candidate)
        missed = {(m['source'], m['target']) for m in mismatches}
        if reference is None or missed < reference[1]:
            reference = (candidate, missed)
    if retain and reference is not None:
        edges, missed = reference
        matched = defaultdict(list)
        for obs in observations:
            source = obs['perturbed_node']
            for target, expected in obs['effects'].items():
                if target != source and (source, target) not in missed:
                    matched[(source, expected)].append(target)
        lines.append('Reference graph from an earlier proposal: ' + graph_line(edges))
        lines.append('Preserve these verified effects when changing the edge set:')
        for (source, expected), targets in matched.items():
            relation = 'must reach' if expected else 'must not reach'
            lines.append(f'- {source} {relation} each of {", ".join(targets)}.')
        lines.append('Also satisfy all remaining observations. You may change reference edges.')
    if diagnose and graph is not None:
        cycle = cycle_edges(graph)
        if cycle:
            chain = ' -> '.join([cycle[0][0]] + [target for _, target in cycle])
            lines.append(f'The last graph contains this directed cycle: {chain}. '
                         'Change at least one edge in that cycle and check all observations.')
        elif history[-1] is not None:
            graph = nx.DiGraph()
            graph.add_nodes_from(nodes)
            graph.add_edges_from(sorted(history[-1]))
            for m in observations_check(nodes, observations, history[-1]):
                source, target = m['source'], m['target']
                if m['predicted']:
                    path = ' -> '.join(nx.shortest_path(graph, source, target))
                    lines.append(f'Forbidden path in the last graph: {path}. Block every path from {source} to {target}.')
                else:
                    reachable = ', '.join(sorted(nx.descendants(graph, source))) or 'no nodes'
                    lines.append(f'Missing path from {source} to {target}. In the last graph, {source} reaches {reachable}.')
    if not lines:
        return '', reference
    return '\n\nSpecific checks:\n' + '\n'.join(lines) + '\nReturn only the next Graph: line.', reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    mod = load_upstream('causal', 'run_causal_benchmark.py')
    data = HERE / 'data/causal_n4_full_observations.json'
    benchmark = mod.CausalBenchmarkEnhanced(str(data))
    index = {r['observation_set_id']: r for r in benchmark.all_observation_sets}
    rng = random.Random(731126)
    fresh = [rng.choice(sorted(k for k, r in index.items()
                               if k not in DEVELOPMENT and len(r['ground_truth_graphs']) == count))
             for count in (2, 4, 8)]
    instances = [index[k] for k in (*DEVELOPMENT, *fresh)]
    verification = verify_checker(benchmark, instances)
    nodes = benchmark.nodes
    if args.verify_only:
        obs = index['n4_219']['observations']
        cycle = extract_edge_graph('Graph: A->B, B->A', nodes)
        feedback, ref = extra_feedback('both', nodes, obs, [None], cycle, None)
        assert 'A -> B -> A' in feedback and ref is None
        bad = extract_edge_graph('Graph: B->A', nodes)
        feedback, ref = extra_feedback('both', nodes, obs, [list(bad.edges)], bad, None)
        assert 'Forbidden path in the last graph: B -> A' in feedback
        assert 'Missing path from A to B' in feedback
        good = extract_edge_graph('Graph: A->B, B->C, C->D', nodes)
        _, ref = extra_feedback('both', nodes, obs, [list(good.edges)], good, ref)
        assert ref[1] == set()
        _, unchanged = extra_feedback('both', nodes, obs, [list(bad.edges)], bad, ref)
        assert unchanged == ref
        print(json.dumps(verification))
        return
    output = args.output or HERE / 'runs' / ('qwen3_constraints_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.trust_env = False
    def api(route, payload=None):
        response = (session.get(BASE_URL + route, timeout=10) if payload is None
                    else session.post(BASE_URL + route, json=payload, timeout=(10, 300)))
        response.raise_for_status()
        return response.json()

    config = dict(model=next(m for m in api('/api/tags')['models'] if m['name'] == MODEL),
                  version=api('/api/version'), options=OPTIONS, think=False, seeds=SEEDS,
                  groups={k: list(v) for k, v in GROUPS.items()}, queries=8, new_selection_seed=731126,
                  development=list(DEVELOPMENT), fresh=fresh, dataset_sha=digest(data), upstream=COMMIT,
                  evaluator_sha=digest(mod.__file__), runner_sha=digest(__file__),
                  feedback_helper_sha=digest(HERE / 'run_feedback_study.py'),
                  scoring_helper_sha=digest(HERE / 'run_four_groups.py'))
    if (output / 'config.json').exists():
        assert json.loads((output / 'config.json').read_text(encoding='utf-8')) == config
    else:
        write_json(output / 'config.json', config)
    write_json(output / 'checker_verification.json', verification)
    print('OUTPUT_DIR=' + str(output.resolve()), flush=True)
    results = []
    first_answers = defaultdict(set)
    for instance in instances:
        item = instance['observation_set_id']
        obs = instance['observations']
        for seed_index, seed in enumerate(SEEDS):
            groups = list(GROUPS)
            offset = (seed_index + len(results) // 12) % len(groups)
            for group in groups[offset:] + groups[:offset]:
                folder = output / 'calls' / f'{item}_{seed}_{group}'
                folder.mkdir(parents=True, exist_ok=True)
                calls, history, parsed_history, events = [], [], [], []
                reference, last_graph = None, None
                rejected = set()
                for query in range(1, 9):
                    base = benchmark.create_prompt(obs, parsed_history)
                    extra = ''
                    if history:
                        extra, reference = extra_feedback(group, nodes, obs, history, last_graph, reference)
                    prompt = base + make_feedback('combined', nodes, obs, history) + extra
                    payload = {'model': MODEL, 'think': False, 'stream': False, 'keep_alive': '10m',
                               'options': {**OPTIONS, 'seed': seed}, 'messages': [{'role': 'user', 'content': prompt}]}
                    path = folder / f'call_{query:02}.json'
                    if path.exists():
                        call = json.loads(path.read_text(encoding='utf-8'))
                        assert call['request'] == payload
                    else:
                        start = time.perf_counter()
                        response = api('/api/chat', payload)
                        call = {'request': payload, 'response': response, 'seconds': time.perf_counter() - start}
                        write_json(path, call)
                    calls.append(call)
                    answer = call['response']['message'].get('content', '')
                    if query == 1:
                        first_answers[item, seed].add(answer)
                    parsed = benchmark.parse_llm_response(answer)
                    history.append(list(parsed.edges) if parsed is not None else None)
                    if parsed is not None:
                        parsed_history.append(parsed)
                    last_graph = extract_edge_graph(answer, nodes)
                    cycle = cycle_edges(last_graph)
                    repeated_cycle = False
                    if cycle:
                        key = tuple(sorted(last_graph.edges))
                        repeated_cycle = key in rejected
                        rejected.add(key)
                    events.append({'cycle': cycle, 'repeated_cyclic_graph': repeated_cycle,
                                   'reference_mismatches': len(reference[1]) if reference is not None else None})

                class Replay:
                    index = 0
                    def query_with_usage(self, prompt):
                        call = calls[self.index]
                        assert call['request']['messages'][0]['content'].startswith(prompt)
                        self.index += 1
                        return {'response': call['response']['message'].get('content', '')}

                official = benchmark.evaluate_single_observation_set(Replay(), instance, 8, False, max_retries=1)
                rows, counts = score_calls(benchmark, instance, calls)
                assert not official['errors'] and all(official[k] == v for k, v in counts.items())
                for row, event in zip(rows, events):
                    row.update(event)
                result = dict(item=item, group=group, seed=seed, official=official, rows=rows)
                write_json(folder / 'result.json', result)
                results.append(result)
                print(f'{len(results)}/84 {item} {seed} {group} VR={official["valid_rate"]:.3f} '
                      f'RR={official["recovery_rate"]:.3f} cycles={sum(bool(r["cycle"]) for r in rows)}', flush=True)
    summarize(output, results, config, first_answers)


def summarize(output, results, config, first_answers):
    subsets = {'development': config['development'], 'fresh': config['fresh']}
    subsets.update({item: [item] for item in config['development'] + config['fresh']})
    summary = {}
    for subset, items in subsets.items():
        summary[subset] = {}
        for group in GROUPS:
            selected = [r for r in results if r['item'] in items and r['group'] == group]
            rows = [d for r in selected for d in r['rows']]
            summary[subset][group] = {
                **{k: mean(r['official'][k] for r in selected)
                   for k in ('valid_rate', 'novelty_rate', 'recovery_rate')},
                'calls': len(rows), 'cycles': sum(bool(d['cycle']) for d in rows),
                'repeated_cycles': sum(d['repeated_cyclic_graph'] for d in rows),
                'parsed_duplicates': sum(d['duplicate'] for d in rows),
                'parse_failures': sum(not d['parsed'] for d in rows),
                'truncations': sum(d['done_reason'] == 'length' for d in rows),
                **{k: sum(d[k] for d in rows) for k in ('prompt_tokens', 'output_tokens', 'seconds')},
            }
    write_json(output / 'summary.json', {'metrics': summary, 'paired_first_answers_matching':
                                        sum(len(v) == 1 for v in first_answers.values()),
                                        'paired_first_answer_sets': len(first_answers)})
    fields = ['item', 'group', 'seed', 'query', 'answer', 'parsed', 'valid', 'duplicate', 'new_valid',
              'cumulative_recovered', 'cycle', 'repeated_cyclic_graph', 'reference_mismatches',
              'done_reason', 'prompt_tokens', 'output_tokens', 'seconds']
    with (output / 'all_calls.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for result in results:
            writer.writerows({**{k: result[k] for k in ('item', 'group', 'seed')}, **r}
                             for r in result['rows'])


if __name__ == '__main__':
    main()
