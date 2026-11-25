"""Compare fixed and per-query seeds with the same diagnostic prompt."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import random
import time

import requests

from common import COMMIT, HERE, digest, load_upstream
from run_constraint_feedback import OPTIONS, extra_feedback, extract_edge_graph, cycle_edges
from run_feedback_study import make_feedback, verify_checker
from run_four_groups import BASE_URL, MODEL, SEEDS, score_calls
from run_local_causal import write_json
from run_temperature_study import DEV, PREVIOUS, metrics

GROUPS = ('fixed_seed', 'vary_seed')
DIFFICULT = ('n4_182', 'n4_184', 'n4_098')
STRIDE = 1009


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    module = load_upstream('causal', 'run_causal_benchmark.py')
    data = HERE/'data/causal_n4_full_observations.json'
    benchmark = module.CausalBenchmarkEnhanced(str(data))
    index = {r['observation_set_id']: r for r in benchmark.all_observation_sets}
    excluded = PREVIOUS | set(DEV) | {'n4_184', 'n4_204', 'n4_098'}
    rng = random.Random(921126)
    fresh = [rng.choice(sorted(key for key, r in index.items()
                              if key not in excluded and len(r['ground_truth_graphs']) == count))
             for count in (2, 4, 8)]
    verification = verify_checker(benchmark, [index[item] for item in (*DIFFICULT, *fresh)])
    output = args.output or HERE/'runs'/('qwen3_seed_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.trust_env = False

    def api(route, payload=None):
        response = (session.get(BASE_URL + route, timeout=10) if payload is None
                    else session.post(BASE_URL + route, json=payload, timeout=(10, 300)))
        response.raise_for_status()
        return response.json()

    config = dict(model=next(m for m in api('/api/tags')['models'] if m['name'] == MODEL),
                  ollama=api('/api/version'), options=OPTIONS, think=False, seeds=SEEDS,
                  calls_per_run=8, seed_stride=STRIDE, difficult=list(DIFFICULT), fresh=fresh,
                  fresh_selection_seed=921126, excluded=sorted(excluded), dataset_sha=digest(data),
                  upstream_commit=COMMIT, evaluator_sha=digest(module.__file__),
                  source_hashes={name: digest(HERE/name) for name in ('run_seed_study.py',
                      'run_temperature_study.py', 'run_constraint_feedback.py',
                      'run_feedback_study.py', 'run_four_groups.py')})
    path = output/'config.json'
    if path.exists():
        assert json.loads(path.read_text(encoding='utf-8')) == config
    else:
        write_json(path, config)
    write_json(output/'checker_verification.json', verification)
    print('OUTPUT_DIR='+str(output.resolve()), flush=True)
    results = []
    for split, items in [('difficult', DIFFICULT), ('fresh', fresh)]:
        for item_index, item in enumerate(items):
            instance = index[item]
            obs = instance['observations']
            for seed_index, seed in enumerate(SEEDS):
                order = GROUPS if (seed_index + item_index) % 2 == 0 else GROUPS[::-1]
                for group in order:
                    folder = output/split/f'{item}_{seed}_{group}'
                    folder.mkdir(parents=True, exist_ok=True)
                    history, graphs, calls = [], [], []

                    class Adapter:
                        def query_with_usage(self, base_prompt):
                            offset = len(calls)
                            extra = ''
                            if history:
                                extra, _ = extra_feedback('diagnostics', benchmark.nodes, obs, history, graphs[-1], None)
                            prompt = base_prompt + make_feedback('combined', benchmark.nodes, obs, history) + extra
                            sample_seed = seed + (STRIDE*offset if group == 'vary_seed' else 0)
                            payload = {'model': MODEL, 'think': False, 'stream': False, 'keep_alive': '10m',
                                       'options': {**OPTIONS, 'seed': sample_seed},
                                       'messages': [{'role': 'user', 'content': prompt}]}
                            path = folder/f'call_{offset+1:02d}.json'
                            if path.exists():
                                call = json.loads(path.read_text(encoding='utf-8'))
                                assert call['request'] == payload
                            else:
                                start = time.perf_counter()
                                response = api('/api/chat', payload)
                                call = {'request': payload, 'response': response, 'seconds': time.perf_counter()-start}
                                write_json(path, call)
                            calls.append(call)
                            response = call['response']
                            answer = response['message'].get('content', '')
                            hypothesis = benchmark.parse_llm_response(answer)
                            history.append(list(hypothesis.edges) if hypothesis is not None else None)
                            graphs.append(extract_edge_graph(answer, benchmark.nodes))
                            pin, pout = response.get('prompt_eval_count', 0), response.get('eval_count', 0)
                            return {'response': answer, 'usage': {'prompt_tokens': pin, 'completion_tokens': pout,
                                                                 'total_tokens': pin+pout}, 'cost': 0}

                    official = benchmark.evaluate_single_observation_set(Adapter(), instance, 8, False, max_retries=1)
                    assert len(calls) == 8 and not official['errors'], official['errors']
                    rows, counts = score_calls(benchmark, instance, calls)
                    assert all(official[key] == value for key, value in counts.items())
                    for row, graph, call in zip(rows, graphs, calls):
                        row['cycle'] = bool(cycle_edges(graph))
                        row['temperature'] = OPTIONS['temperature']
                        row['sample_seed'] = call['request']['options']['seed']
                    result = dict(split=split, item=item, seed=seed, group=group, official=official, rows=rows)
                    write_json(folder/'result.json', result)
                    results.append(result)
                    print(f'{len(results)}/36 {split} {item} {seed} {group} '
                          f'VR={official["valid_rate"]:.3f} RR={official["recovery_rate"]:.3f}', flush=True)
    subsets = {'difficult': DIFFICULT, 'fresh': fresh, **{item: [item] for item in (*DIFFICULT, *fresh)}}
    summary = {label: {group: metrics([r for r in results if r['group'] == group and r['item'] in items])
                       for group in GROUPS} for label, items in subsets.items()}
    write_json(output/'summary.json', summary)
    fields = ['split', 'item', 'seed', 'group', 'query', 'sample_seed', 'answer', 'parsed', 'valid',
              'duplicate', 'cycle', 'new_valid', 'cumulative_recovered', 'prompt_tokens', 'output_tokens',
              'seconds', 'done_reason']
    with (output/'all_calls.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for result in results:
            writer.writerows({**{key: result[key] for key in fields[:4]}, **row} for row in result['rows'])
    print(json.dumps({key: summary[key] for key in ('difficult', 'fresh')}), flush=True)


if __name__ == '__main__':
    main()
