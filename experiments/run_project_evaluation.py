"""Frozen held-out causal evaluation of diagnostic feedback and slot-eight thinking."""
import argparse
import csv
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import random
import re
from statistics import mean
import time

import requests

from common import COMMIT, DATA, HERE, RESULTS, digest, load_upstream
import ijson
from run_constraint_feedback import OPTIONS, extra_feedback, extract_edge_graph
from run_feedback_study import make_feedback, verify_checker
from run_four_groups import BASE_URL, MODEL, SEEDS, score_calls
from run_local_causal import write_json
from run_thinking_trigger_study import EXCLUDED, THINKING_OPTIONS
from stream_causal import check_graph

METHODS = ('b_baseline', 'diagnostics', 'slot_8')
TEST_SEED = 981126


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def select_cases():
    n4 = read(HERE/'data/causal_n4_full_observations.json')
    trigger_study = read(HERE/'runs/qwen3_thinking_trigger_20261112_132335/config.json')
    excluded = EXCLUDED | {item for item, _ in trigger_study['cases']}
    rng = random.Random(TEST_SEED)
    chosen4 = []
    for count in (2, 4, 8):
        pool = [r for r in n4['datasets'] if r['observation_set_id'] not in excluded
                and len(r['ground_truth_graphs']) == count]
        chosen4.extend(rng.sample(sorted(pool, key=lambda r: r['observation_set_id']), 4))
    source = DATA/'causal/datasets/node06/n6_all_observations.json'
    audit = read(RESULTS/'causal_audit.json')
    assert source.stat().st_size == audit['size_bytes']
    with source.open('rb') as stream:
        metadata = next(ijson.items(stream, 'metadata'))
        stream.seek(max(0, source.stat().st_size - 16 * 1024**2))
        tail = stream.read()
    marker, = re.finditer(rb'^    "6": (\[)', tail, re.M)
    segment = tail[marker.start(1):].decode('utf-8')
    n6, end = json.JSONDecoder().raw_decode(segment)
    assert re.fullmatch(r'\s*}\s*}\s*', segment[end:])
    declared = {r['id']: r['n_ground_truths'] for r in audit['instances'] if r['n_observations'] == 6}
    assert len(n6) == len(declared) == 100
    assert all(r['n_observations'] == 6 and len(r['ground_truth_graphs']) == declared[r['observation_set_id']]
               for r in n6)
    chosen6 = []
    for low, high in ((1, 1), (2, 8), (9, 219)):
        pool = [r for r in n6 if low <= len(r['ground_truth_graphs']) <= high]
        chosen6.extend(rng.sample(sorted(pool, key=lambda r: r['observation_set_id']), 2))
    for item in chosen6:
        graphs = item['ground_truth_graphs']
        assert all(check_graph(g, item['observations'], metadata) for g in graphs)
        assert len({tuple(sorted(map(tuple, g['edges']))) for g in graphs}) == len(graphs)
    subsets = {'n4': {'metadata': {**n4['metadata'], 'max_edges': 6, 'total_observation_sets': len(chosen4)},
                     'datasets': chosen4},
               'n6': {'metadata': {**metadata, 'total_observation_sets': len(chosen6)}, 'datasets': chosen6}}
    provenance = {'n4_excluded': sorted(excluded), 'n6_previous_audit_sha': audit['source_sha256'],
                  'n6_file_size': source.stat().st_size, 'n6_selection': 'full six-observation block',
                  'n6_exact_block_sha': sha256(segment[:end].encode('utf-8')).hexdigest()}
    return subsets, provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    subsets, provenance = select_cases()
    cases = [(size, r['observation_set_id'], SEEDS[i % len(SEEDS)], len(r['ground_truth_graphs']))
             for size, subset in subsets.items() for i, r in enumerate(subset['datasets'])]
    if args.verify_only:
        print(json.dumps(cases))
        return
    output = args.output or HERE/'runs'/('qwen3_project_evaluation_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True, exist_ok=True)
    module = load_upstream('causal', 'run_causal_benchmark.py')
    benchmarks, lookup = {}, {}
    for size, dataset in subsets.items():
        path = output/f'{size}_test.json'
        if path.exists():
            assert read(path) == dataset
        else:
            write_json(path, dataset)
        benchmark = module.CausalBenchmarkEnhanced(str(path))
        benchmarks[size] = benchmark
        lookup[size] = {r['observation_set_id']: r for r in benchmark.all_observation_sets}
    checks = verify_checker(benchmarks['n4'], subsets['n4']['datasets'])
    write_json(output/'checker_verification.json', checks)
    session = requests.Session()
    session.trust_env = False

    def api(route, payload=None):
        response = (session.get(BASE_URL+route, timeout=10) if payload is None
                    else session.post(BASE_URL+route, json=payload, timeout=(10, 300)))
        response.raise_for_status()
        return response.json()

    config = {'cases': [list(c) for c in cases], 'methods': list(METHODS), 'selection_seed': TEST_SEED,
              'ordinary_options': OPTIONS, 'thinking_overrides': THINKING_OPTIONS, 'queries': 8,
              'model': next(m for m in api('/api/tags')['models'] if m['name'] == MODEL),
              'ollama': api('/api/version'), 'upstream': COMMIT, 'evaluator_sha': digest(module.__file__),
              'n4_source_sha': digest(HERE/'data/causal_n4_full_observations.json'),
              'test_shas': {size: digest(output/f'{size}_test.json') for size in subsets},
              'source_hashes': {name: digest(HERE/name) for name in ('run_project_evaluation.py',
                  'run_constraint_feedback.py', 'run_feedback_study.py', 'run_four_groups.py',
                  'run_thinking_trigger_study.py', 'stream_causal.py')}, **provenance}
    path = output/'config.json'
    if path.exists():
        assert read(path) == config
    else:
        write_json(path, config)
    print('OUTPUT_DIR=' + str(output.resolve()), flush=True)
    results = []

    def rollout(size, item, seed, gt, method, control=None):
        benchmark = benchmarks[size]
        instance = lookup[size][item]
        observations = instance['observations']
        folder = output/size/item/method
        folder.mkdir(parents=True, exist_ok=True)
        trigger = control is not None and not any(r['valid'] for r in control[1][:7])
        calls, history, shares = [], [], []
        last_graph = None

        class Adapter:
            def query_with_usage(self, base_prompt):
                nonlocal last_graph
                query = len(calls) + 1
                prompt = base_prompt
                if method != 'b_baseline':
                    prompt += make_feedback('combined', benchmark.nodes, observations, history)
                    if history:
                        feedback, _ = extra_feedback('diagnostics', benchmark.nodes, observations, history, last_graph, None)
                        prompt += feedback
                think = trigger and query == 8
                options = {**OPTIONS, 'seed': seed}
                if think:
                    options.update(THINKING_OPTIONS)
                payload = {'model': MODEL, 'think': think, 'stream': False, 'keep_alive': '10m',
                           'options': options, 'messages': [{'role': 'user', 'content': prompt}]}
                path = folder/f'call_{query:02d}.json'
                shared = control is not None and not think
                if shared:
                    call = control[0][query-1]
                    assert call['request'] == payload
                elif path.exists():
                    call = read(path)
                    assert call['request'] == payload
                else:
                    start = time.perf_counter()
                    response = api('/api/chat', payload)
                    call = {'request': payload, 'response': response, 'seconds': time.perf_counter()-start}
                    write_json(path, call)
                calls.append(call)
                shares.append(shared)
                r = call['response']
                answer = r['message'].get('content', '')
                parsed = benchmark.parse_llm_response(answer)
                history.append(list(parsed.edges) if parsed is not None else None)
                last_graph = extract_edge_graph(answer, benchmark.nodes)
                pin, pout = r.get('prompt_eval_count', 0), r.get('eval_count', 0)
                return {'response': answer, 'usage': {'prompt_tokens': pin, 'completion_tokens': pout,
                                                     'total_tokens': pin+pout}, 'cost': 0}

        official = benchmark.evaluate_single_observation_set(Adapter(), instance, 8, False, max_retries=1)
        assert len(calls) == 8 and not official['errors'], official['errors']
        rows, counts = score_calls(benchmark, instance, calls)
        assert all(official[k] == v for k, v in counts.items())
        for row, call, shared in zip(rows, calls, shares):
            row['think'] = call['request']['think']
            row['shared'] = shared
        result = dict(size=size, item=item, seed=seed, gt=gt, method=method, triggered=trigger,
                      official=official, rows=rows)
        write_json(folder/'result.json', result)
        results.append(result)
        print(f'{len(results)}/54 {size}/{item} {method} think={trigger} '
              f'VR={official["valid_rate"]:.3f} RR={official["recovery_rate"]:.3f} '
              f'last_tokens={rows[-1]["output_tokens"]} last_stop={rows[-1]["done_reason"]} '
              f'last_seconds={rows[-1]["seconds"]:.1f}', flush=True)
        return calls, rows

    controls = {}
    for case in cases:
        rollout(*case, 'b_baseline')
        controls[case[:2]] = rollout(*case, 'diagnostics')
    for case in cases:
        rollout(*case, 'slot_8', controls[case[:2]])

    metrics = {}
    for size in ('n4', 'n6', 'all'):
        metrics[size] = {}
        for method in METHODS:
            runs = [r for r in results if r['method'] == method and (size == 'all' or r['size'] == size)]
            rows = [row for r in runs for row in r['rows']]
            thinking = [r for r in rows if r['think']]
            metrics[size][method] = {
                **{k: mean(r['official'][k] for r in runs) for k in ('valid_rate','novelty_rate','recovery_rate')},
                'tasks': len(runs), 'solved': sum(r['official']['n_recovered_gts'] > 0 for r in runs),
                'unique_valid_sum': sum(r['official']['n_recovered_gts'] for r in runs),
                'triggered': len(thinking), 'thinking_valid': sum(r['valid'] for r in thinking),
                'thinking_truncated': sum(r['done_reason'] == 'length' for r in thinking),
                **{k: sum(r[k] for r in rows) for k in ('seconds','prompt_tokens','output_tokens')}}
    actual = len(list(output.glob('*/*/*/call_*.json')))
    assert actual == sum(not d['shared'] for r in results for d in r['rows'])
    write_json(output/'summary.json', {'metrics': metrics, 'results': results, 'actual_requests': actual})
    fields = ['size','item','seed','gt','method','query','think','shared','answer','parsed','valid','duplicate',
              'new_valid','cumulative_recovered','seconds','prompt_tokens','output_tokens','done_reason']
    with (output/'all_calls.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for result in results:
            writer.writerows({**{k: result[k] for k in fields[:5]}, **row} for row in result['rows'])
    print(json.dumps({'metrics': metrics, 'actual_requests': actual}), flush=True)


if __name__ == '__main__':
    main()
