"""Temperature tuning with a preselected validation split and fixed query budget."""
import argparse
from collections import Counter
import csv
from datetime import datetime
import json
from pathlib import Path
import random
from statistics import mean
import time

import requests

from common import COMMIT, HERE, digest, load_upstream
from run_constraint_feedback import OPTIONS, extra_feedback, extract_edge_graph, cycle_edges
from run_feedback_study import make_feedback, observations_check, verify_checker
from run_four_groups import BASE_URL, MODEL, SEEDS, score_calls
from run_local_causal import write_json

POLICIES = ('baseline', 'low', 'high', 'adaptive')
DEV = ('n4_219', 'n4_182', 'n4_134', 'n4_199')
PREVIOUS = set(DEV) | {'n4_085', 'n4_116', 'n4_152'}
NAMES = dict(zip(POLICIES, ('基线 0.60', '低温 0.35', '高温 0.90', '按上一答在 0.60 / 0.90 间切换')))


def temperature(policy, last_problem):
    if policy == 'adaptive':
        return .90 if last_problem else .60
    return {'baseline': .60, 'low': .35, 'high': .90}[policy]


def metrics(results):
    rows = [row for result in results for row in result['rows']]
    return {
        **{metric: mean(r['official'][metric] for r in results)
           for metric in ('valid_rate', 'novelty_rate', 'recovery_rate')},
        'calls': len(rows), 'cycles': sum(row['cycle'] for row in rows),
        'duplicates': sum(row['duplicate'] for row in rows),
        'truncations': sum(row['done_reason'] == 'length' for row in rows),
        **{key: sum(row[key] for row in rows) for key in ('prompt_tokens', 'output_tokens', 'seconds')},
        'temperatures': dict(Counter(str(row['temperature']) for row in rows)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    module = load_upstream('causal', 'run_causal_benchmark.py')
    data = HERE / 'data/causal_n4_full_observations.json'
    benchmark = module.CausalBenchmarkEnhanced(str(data))
    index = {row['observation_set_id']: row for row in benchmark.all_observation_sets}
    rng = random.Random(891126)
    validation = [rng.choice(sorted(key for key, row in index.items()
                                    if key not in PREVIOUS and len(row['ground_truth_graphs']) == n))
                  for n in (2, 4, 8)]
    checker = verify_checker(benchmark, [index[key] for key in (*DEV, *validation)])
    if args.verify_only:
        assert [temperature('adaptive', x) for x in (False, True, False)] == [.6, .9, .6]
        print(json.dumps(checker))
        return
    output = args.output or HERE / 'runs' / ('qwen3_temperature_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.trust_env = False

    def api(route, payload=None):
        response = (session.get(BASE_URL + route, timeout=10) if payload is None
                    else session.post(BASE_URL + route, json=payload, timeout=(10, 300)))
        response.raise_for_status()
        return response.json()

    config = dict(model=next(m for m in api('/api/tags')['models'] if m['name'] == MODEL),
                  ollama=api('/api/version'), options=OPTIONS, think=False, seeds=SEEDS, calls_per_run=8,
                  development=list(DEV), validation=validation, validation_selection_seed=891126,
                  policies={key: [temperature(key, False), temperature(key, True)] for key in POLICIES},
                  selection='Best nonbaseline on development by RR, then VR, then policy order.',
                  dataset_sha=digest(data), upstream_commit=COMMIT, evaluator_sha=digest(module.__file__),
                  source_hashes={name: digest(HERE/name) for name in (
                      'run_temperature_study.py', 'run_constraint_feedback.py',
                      'run_feedback_study.py', 'run_four_groups.py')})
    config_path = output/'config.json'
    if config_path.exists():
        assert json.loads(config_path.read_text(encoding='utf-8')) == config
    else:
        write_json(config_path, config)
    write_json(output/'checker_verification.json', checker)
    print('OUTPUT_DIR=' + str(output.resolve()), flush=True)
    results = []

    def evaluate(split, items, policies):
        for item_index, item in enumerate(items):
            instance = index[item]
            observations = instance['observations']
            for seed_index, seed in enumerate(SEEDS):
                shift = (item_index + seed_index) % len(policies)
                for policy in policies[shift:] + policies[:shift]:
                    folder = output / split / f'{item}_{seed}_{policy}'
                    folder.mkdir(parents=True, exist_ok=True)
                    history, calls, settings = [], [], []
                    seen = set()
                    last_graph = None
                    last_problem = False

                    class Adapter:
                        def query_with_usage(self, base_prompt):
                            nonlocal last_graph, last_problem
                            query = len(calls) + 1
                            extra = ''
                            if history:
                                extra, _ = extra_feedback('diagnostics', benchmark.nodes, observations,
                                                          history, last_graph, None)
                            prompt = base_prompt + make_feedback('combined', benchmark.nodes, observations, history) + extra
                            temp = temperature(policy, last_problem)
                            payload = {'model': MODEL, 'think': False, 'stream': False, 'keep_alive': '10m',
                                       'options': {**OPTIONS, 'temperature': temp, 'seed': seed},
                                       'messages': [{'role': 'user', 'content': prompt}]}
                            path = folder/f'call_{query:02d}.json'
                            if path.exists():
                                call = json.loads(path.read_text(encoding='utf-8'))
                                assert call['request'] == payload
                            else:
                                start = time.perf_counter()
                                response = api('/api/chat', payload)
                                call = {'request': payload, 'response': response, 'seconds': time.perf_counter()-start}
                                write_json(path, call)
                            calls.append(call)
                            settings.append({'temperature': temp, 'previous_problem': last_problem})
                            answer = call['response']['message'].get('content', '')
                            hypothesis = benchmark.parse_llm_response(answer)
                            last_graph = extract_edge_graph(answer, benchmark.nodes)
                            if hypothesis is None:
                                last_problem = True
                                history.append(None)
                            else:
                                key = tuple(sorted(hypothesis.edges))
                                bad = observations_check(benchmark.nodes, observations, key)
                                last_problem = bool(bad) or key in seen
                                seen.add(key)
                                history.append(list(key))
                            response = call['response']
                            pin, pout = response.get('prompt_eval_count', 0), response.get('eval_count', 0)
                            return {'response': answer, 'usage': {'prompt_tokens': pin, 'completion_tokens': pout,
                                                                 'total_tokens': pin+pout}, 'cost': 0}

                    official = benchmark.evaluate_single_observation_set(Adapter(), instance, 8, False, max_retries=1)
                    assert len(calls) == 8 and not official['errors'], official['errors']
                    rows, counts = score_calls(benchmark, instance, calls)
                    assert all(official[key] == value for key, value in counts.items())
                    for row, setting in zip(rows, settings):
                        row.update(setting)
                        row['cycle'] = bool(cycle_edges(extract_edge_graph(row['answer'], benchmark.nodes)))
                    result = dict(split=split, item=item, seed=seed, policy=policy, official=official, rows=rows)
                    write_json(folder/'result.json', result)
                    results.append(result)
                    print(f'{len(results)}/66 {split} {item} {seed} {policy} '
                          f'VR={official["valid_rate"]:.3f} RR={official["recovery_rate"]:.3f}', flush=True)

    evaluate('development', DEV, POLICIES)
    dev = {policy: metrics([r for r in results if r['policy'] == policy]) for policy in POLICIES}
    chosen = max(POLICIES[1:], key=lambda key: (dev[key]['recovery_rate'], dev[key]['valid_rate']))
    selection = {'challenger': chosen, 'development': dev}
    write_json(output/'selection.json', selection)
    print('SELECTED=' + chosen + ' ' + json.dumps(dev), flush=True)
    evaluate('validation', validation, ('baseline', chosen))
    val = {policy: metrics([r for r in results if r['split'] == 'validation' and r['policy'] == policy])
           for policy in ('baseline', chosen)}
    per_item = {item: {policy: metrics([r for r in results if r['item'] == item and r['policy'] == policy])
                      for policy in (POLICIES if item in DEV else ('baseline', chosen))}
                for item in (*DEV, *validation)}
    write_json(output/'summary.json', {**selection, 'validation': val, 'per_item': per_item})
    fields = ['split', 'item', 'seed', 'policy', 'query', 'temperature', 'previous_problem',
              'answer', 'parsed', 'valid', 'duplicate', 'cycle', 'new_valid', 'cumulative_recovered',
              'prompt_tokens', 'output_tokens', 'seconds', 'done_reason']
    with (output/'all_calls.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for result in results:
            writer.writerows({**{key: result[key] for key in fields[:4]}, **row} for row in result['rows'])
    print('VALIDATION=' + json.dumps(val), flush=True)


if __name__ == '__main__':
    main()
