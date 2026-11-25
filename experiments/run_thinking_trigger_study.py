"""Compare thinking at slot 5 or 8 after a shared non-thinking prefix."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import random
from statistics import mean
import time

import requests

from common import COMMIT, HERE, digest, load_upstream
from run_constraint_feedback import OPTIONS, extra_feedback, extract_edge_graph
from run_feedback_study import make_feedback, verify_checker
from run_four_groups import BASE_URL, MODEL, SEEDS, score_calls
from run_local_causal import write_json
from run_seed_study import PREVIOUS

POLICIES = {'ordinary': None, 'slot_5': 5, 'slot_8': 8}
THINKING_OPTIONS = {'num_ctx': 16384, 'num_predict': 8192}
EXCLUDED = PREVIOUS | {'n4_184', 'n4_204', 'n4_098', 'n4_033', 'n4_078', 'n4_174'}
SELECTION_SEED = 971126


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    module = load_upstream('causal', 'run_causal_benchmark.py')
    data = HERE/'data/causal_n4_full_observations.json'
    benchmark = module.CausalBenchmarkEnhanced(str(data))
    index = {r['observation_set_id']: r for r in benchmark.all_observation_sets}
    rng = random.Random(SELECTION_SEED)
    items = []
    for n in (2, 4, 8):
        pool = sorted(key for key, r in index.items() if key not in EXCLUDED and len(r['ground_truth_graphs']) == n)
        items.extend(rng.sample(pool, 2))
    cases = [(item, SEEDS[i % len(SEEDS)]) for i, item in enumerate(items)]
    verification = verify_checker(benchmark, [index[key] for key in items])
    if args.verify_only:
        print(json.dumps({'cases': cases, 'checker': verification}))
        return
    output = args.output or HERE/'runs'/('qwen3_thinking_trigger_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.trust_env = False

    def api(route, payload=None):
        response = (session.get(BASE_URL+route, timeout=10) if payload is None
                    else session.post(BASE_URL+route, json=payload, timeout=(10, 300)))
        response.raise_for_status()
        return response.json()

    config = {
        'cases': [list(case) for case in cases], 'excluded': sorted(EXCLUDED), 'selection_seed': SELECTION_SEED,
        'model': next(m for m in api('/api/tags')['models'] if m['name'] == MODEL),
        'ollama': api('/api/version'), 'ordinary_options': OPTIONS, 'thinking_overrides': THINKING_OPTIONS,
        'policies': POLICIES, 'calls_per_policy_per_item': 8, 'upstream_commit': COMMIT,
        'dataset_sha': digest(data), 'evaluator_sha': digest(module.__file__),
        'source_hashes': {name: digest(HERE/name) for name in ('run_thinking_trigger_study.py',
            'run_constraint_feedback.py', 'run_feedback_study.py', 'run_four_groups.py',
            'run_seed_study.py', 'run_temperature_study.py')},
    }
    path = output/'config.json'
    if path.exists():
        assert read(path) == config
    else:
        write_json(path, config)
    write_json(output/'checker_verification.json', verification)
    print('OUTPUT_DIR=' + str(output.resolve()), flush=True)
    results = []

    def rollout(item, seed, policy, baseline_calls=None, baseline_rows=None):
        instance = index[item]
        observations = instance['observations']
        slot = POLICIES[policy]
        trigger = slot if slot and not any(row['valid'] for row in baseline_rows[:slot-1]) else None
        folder = output / item / policy
        folder.mkdir(parents=True, exist_ok=True)
        calls, history, reused = [], [], []
        last_graph = None

        class Adapter:
            def query_with_usage(self, base_prompt):
                nonlocal last_graph
                query = len(calls) + 1
                feedback = ''
                if history:
                    feedback, _ = extra_feedback('diagnostics', benchmark.nodes, observations, history, last_graph, None)
                prompt = base_prompt + make_feedback('combined', benchmark.nodes, observations, history) + feedback
                think = query == trigger
                options = {**OPTIONS, 'seed': seed}
                if think:
                    options.update(THINKING_OPTIONS)
                payload = {'model': MODEL, 'think': think, 'stream': False, 'keep_alive': '10m',
                           'options': options, 'messages': [{'role': 'user', 'content': prompt}]}
                path = folder/f'call_{query:02d}.json'
                shared = baseline_calls is not None and (trigger is None or query < trigger)
                if shared:
                    call = baseline_calls[query-1]
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
                reused.append(shared)
                response = call['response']
                answer = response['message'].get('content', '')
                parsed = benchmark.parse_llm_response(answer)
                history.append(list(parsed.edges) if parsed is not None else None)
                last_graph = extract_edge_graph(answer, benchmark.nodes)
                pin, pout = response.get('prompt_eval_count', 0), response.get('eval_count', 0)
                if think:
                    print(f'THINK {item} {policy} slot={query} stop={response.get("done_reason")} '
                          f'tokens={pout} time={call["seconds"]:.1f}s', flush=True)
                return {'response': answer, 'usage': {'prompt_tokens': pin, 'completion_tokens': pout,
                                                     'total_tokens': pin+pout}, 'cost': 0}

        official = benchmark.evaluate_single_observation_set(Adapter(), instance, 8, False, max_retries=1)
        assert len(calls) == 8 and not official['errors'], official['errors']
        rows, counts = score_calls(benchmark, instance, calls)
        assert all(official[k] == v for k, v in counts.items())
        for row, shared, call in zip(rows, reused, calls):
            row['shared_from_ordinary'] = shared
            row['think'] = call['request']['think']
        assert sum(row['think'] for row in rows) == int(trigger is not None)
        result = {'item': item, 'seed': seed, 'policy': policy, 'trigger_slot': trigger, 'official': official, 'rows': rows}
        write_json(folder/'result.json', result)
        results.append(result)
        write_json(output/'progress.json', {'done': len(results), 'planned': 18, 'results': results})
        print(f'{len(results)}/18 {item} {policy} trigger={trigger} VR={official["valid_rate"]:.3f} '
              f'RR={official["recovery_rate"]:.3f}', flush=True)
        return calls, rows

    baselines = {}
    for item, seed in cases:
        baselines[item] = rollout(item, seed, 'ordinary')
    for i, (item, seed) in enumerate(cases):
        order = ('slot_5', 'slot_8') if i % 2 == 0 else ('slot_8', 'slot_5')
        for policy in order:
            rollout(item, seed, policy, *baselines[item])

    aggregates = {}
    for policy in POLICIES:
        chosen = [r for r in results if r['policy'] == policy]
        rows = [row for r in chosen for row in r['rows']]
        thinking = [row for row in rows if row['think']]
        aggregates[policy] = {
            **{key: mean(r['official'][key] for r in chosen)
               for key in ('valid_rate', 'novelty_rate', 'recovery_rate')},
            'triggered': len(thinking), 'thinking_valid': sum(row['valid'] for row in thinking),
            'thinking_truncated': sum(row['done_reason'] == 'length' for row in thinking),
            'items_with_valid': sum(r['official']['n_valid'] > 0 for r in chosen),
            'after_trigger_new_valid': sum(row['new_valid'] for r in chosen if r['trigger_slot']
                                          for row in r['rows'][r['trigger_slot']:]),
            **{key: sum(row[key] for row in rows) for key in ('prompt_tokens', 'output_tokens', 'seconds')},
        }
    actual = len(list(output.glob('*/*/call_*.json')))
    assert actual == sum(not row['shared_from_ordinary'] for r in results for row in r['rows'])
    write_json(output/'summary.json', {'groups': aggregates, 'results': results, 'actual_model_requests': actual})
    print(json.dumps({'groups': aggregates, 'actual_model_requests': actual}), flush=True)


if __name__ == '__main__':
    main()
