"""Replay seven fixed-seed proposals and replace the eighth with thinking."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import requests

from common import HERE, digest, load_upstream
from run_four_groups import BASE_URL, MODEL, score_calls
from run_local_causal import write_json

CASES = (('difficult', 'n4_182', 33550337), ('difficult', 'n4_184', 33550336),
         ('fresh', 'n4_174', 33550338))


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('control', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or HERE/'runs'/('qwen3_thinking_fallback_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.trust_env = False
    saved = read(args.control/'config.json')
    response = session.get(BASE_URL+'/api/tags', timeout=10)
    response.raise_for_status()
    model = next(m for m in response.json()['models'] if m['name'] == MODEL)
    response = session.get(BASE_URL+'/api/version', timeout=10)
    response.raise_for_status()
    assert model['digest'] == saved['model']['digest'] and response.json() == saved['ollama']
    module = load_upstream('causal', 'run_causal_benchmark.py')
    data = HERE/'data/causal_n4_full_observations.json'
    assert digest(data) == saved['dataset_sha'] and digest(module.__file__) == saved['evaluator_sha']
    benchmark = module.CausalBenchmarkEnhanced(str(data))
    index = {r['observation_set_id']: r for r in benchmark.all_observation_sets}
    config = {'control': str(args.control.resolve()), 'cases': [list(case) for case in CASES],
              'change': {'think': True, 'num_ctx': 16384, 'num_predict': 8192},
              'runner_sha': digest(__file__), 'control_config_sha': digest(args.control/'config.json')}
    path = output/'config.json'
    if path.exists():
        assert read(path) == config
    else:
        write_json(path, config)
    print('OUTPUT_DIR='+str(output.resolve()), flush=True)
    results = []
    for split, item, seed in CASES:
        source = args.control/split/f'{item}_{seed}_fixed_seed'
        before = [read(source/f'call_{i:02d}.json') for i in range(1, 9)]
        original_rows, counts = score_calls(benchmark, index[item], before)
        assert counts['n_valid'] == 0
        old = before[-1]['request']
        payload = {**old, 'think': True, 'options': {**old['options'], 'num_ctx': 16384, 'num_predict': 8192}}
        path = output/f'{item}_{seed}_call_08.json'
        if path.exists():
            replacement = read(path)
            assert replacement['request'] == payload
        else:
            start = time.perf_counter()
            response = session.post(BASE_URL+'/api/chat', json=payload, timeout=(10, 300))
            response.raise_for_status()
            replacement = {'request': payload, 'response': response.json(), 'seconds': time.perf_counter()-start}
            write_json(path, replacement)
        rows, counts = score_calls(benchmark, index[item], before[:7] + [replacement])
        result = {'item': item, 'seed': seed, 'control_last': original_rows[-1], 'thinking_last': rows[-1],
                  'valid_rate': counts['n_valid']/8,
                  'recovery_rate': counts['n_recovered_gts']/len(index[item]['ground_truth_graphs'])}
        results.append(result)
        write_json(output/'summary.json', results)
        print(f"{item} valid={rows[-1]['valid']} RR={result['recovery_rate']:.3f} "
              f"stop={rows[-1]['done_reason']} tokens={rows[-1]['output_tokens']} "
              f"seconds={rows[-1]['seconds']:.2f} answer={rows[-1]['answer']!r}", flush=True)


if __name__ == '__main__':
    main()
