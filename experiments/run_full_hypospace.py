"""Run all supplied HypoSpace cases with domain-specific Qwen methods and controls."""
import argparse
from itertools import product
from pathlib import Path
from statistics import mean

import requests

from common import DATA, HERE, Replay, digest, load_upstream
from offline_3d import enumerate_physics, from_heights, to_response
from run_boolean_feedback_pilot import SOURCE as BOOLEAN, inspect_answer
from run_constraint_feedback import OPTIONS
from run_final_qwen_path_repair import run as run_causal
from run_formal_agent_debate import REFERENCE, request as raw_request
from run_formal_causal_comparison import DATA as CAUSAL, LIMIT, NODES, program_anchor
from run_four_groups import BASE_URL, MODEL, SEEDS
from run_local_causal import write_json
from run_qwen_guided_edits import decode, request, schema
from run_qwen_rr_budget import read
from validated_evaluators import corrected_3d_class, corrected_boolean_class


SLOTS = 8
VOXEL = DATA/'3d/datasets/3d_grid3_h3_tp3.json'
METRICS = {'rr': 'recovery_rate', 'vr': 'valid_rate', 'nr': 'novelty_rate'}
COUNTS = {'boolean': 80, '3d': 84, 'causal': 510}


def cost(calls):
    return {'calls': len(calls), 'tokens': sum(c['response'].get('eval_count', 0) for c in calls),
            'input_tokens': sum(c['response'].get('prompt_eval_count', 0) for c in calls),
            'request_seconds': sum(c['seconds'] for c in calls)}


def score(bench, task, answers):
    replay = Replay(answers)
    row = bench.evaluate_single_observation_set(replay, task, SLOTS, False, max_retries=1)
    assert replay.calls == SLOTS
    return {key: row[field] for key, field in METRICS.items()} | {'found': row['n_recovered_gts']}


def boolean_trial(session, folder, seed, task, module, corrected, original, formulas, feedback):
    obs = [module.BooleanObservation(row['inputs'], row['output']) for row in task['observations']]
    used, seen, calls, answers = set(), {}, [], []
    catalog = '\n'.join(f'{i}. {formula}' for i, formula in enumerate(formulas, 1))
    last = ''
    for slot in range(1, SLOTS+1):
        remaining = [i for i in range(1, len(formulas)+1) if i not in used]
        prompt = ('Choose a new Boolean expression matching every observation. Return a JSON choice ID.\n'
                  'Observations:\n' + '\n'.join(row.to_string() for row in obs)
                  + '\nPreviously selected IDs: ' + str(sorted(used)) + '\nCatalog:\n' + catalog)
        if feedback and last:
            prompt += '\nVerifier on the previous choice: ' + last
        label = 'first' if slot == 1 else 'feedback' if feedback else 'grammar'
        call = request(session, folder/f'{label}_{slot:02}.json', prompt, seed+slot,
                       schema({'choice': {'type': 'integer', 'enum': remaining}}))
        choice = decode(call).get('choice')
        formula = None
        if type(choice) is int and choice in remaining:
            used.add(choice)
            formula = formulas[choice-1]
        last, key = inspect_answer(corrected, module, formula, obs, seen)
        if key is not None:
            seen[key] = formula
        calls.append(call)
        answers.append('Expression: '+formula if formula is not None else '')
    return score(corrected, task, answers) | cost(calls) | {
        'answers': answers, 'upstream': score(original, task, answers)}


def voxel_trial(session, folder, seed, task, bench, structured):
    obs = task['observation']
    size, height = (bench.metadata[key] for key in ('grid_size', 'max_height'))
    heights = [''.join(map(str, combo)) for combo in product(range(1, height+1), repeat=obs.count('1'))]
    answers, calls, history = [], [], []
    used = set()
    for slot in range(1, SLOTS+1):
        if structured:
            prompt = (f'Top projection is {obs}, row-major on a {size}x{size} grid. Each 1 is a column '
                      f'with height 1 to {height}, in row-major order. Choose one new height string '
                      'as JSON choice. Previously submitted: '+str(sorted(used)))
            remaining = [h for h in heights if h not in used]
            call = request(session, folder/f'heights_{slot:02}.json', prompt, seed+slot,
                           schema({'choice': {'type': 'string', 'enum': remaining}}))
            choice = decode(call).get('choice')
            answer = ''
            if isinstance(choice, str) and choice in remaining:
                used.add(choice)
                answer = to_response(from_heights(obs, tuple(map(int, choice))), size)
        else:
            prompt = bench.create_prompt(obs, history)
            call = raw_request(session, folder/f'direct_{slot:02}.json',
                               [{'role': 'user', 'content': prompt}], OPTIONS, seed+slot, 256)
            answer = call['response']['message'].get('content', '')
            parsed = bench.parse_llm_response(answer)
            if parsed is not None:
                history.append(parsed)
        calls.append(call)
        answers.append(answer)
    return score(bench, task, answers) | cost(calls) | {'answers': answers}


def tasks_and_trial(domain):
    if domain == 'boolean':
        module = load_upstream('boolean', 'boolean_benchmark.py')
        from boolean_dataset import BooleanDiscoveryGame
        original = module.BooleanBenchmarkRefined(str(BOOLEAN))
        corrected = corrected_boolean_class(module)(str(BOOLEAN))
        grammar = BooleanDiscoveryGame.generate_all_expressions(
            corrected.variables, corrected.operators, corrected.max_depth, corrected.mechanistic_opts)
        formulas = sorted(expr.formula for expr in grammar)
        assert len(formulas) == len(set(formulas)) == 65

        def trial(session, folder, seed, task):
            return {method: boolean_trial(session, folder, seed, task, module, corrected, original, formulas, feedback)
                    for method, feedback in (('grammar', False), ('feedback', True))}

        return original.all_observation_sets, 'observation_set_id', trial, digest(BOOLEAN), digest(module.__file__)
    if domain == '3d':
        module = load_upstream('3d', 'run_3d_benchmark.py')
        bench = corrected_3d_class(module)(str(VOXEL))

        def trial(session, folder, seed, task):
            runs = {method: voxel_trial(session, folder, seed, task, bench, structured)
                    for method, structured in (('direct', False), ('heights', True))}
            size, height = (bench.metadata[k] for k in ('grid_size', 'max_height'))
            candidates = enumerate_physics(task['observation'], height)
            runs['program'] = score(bench, task, [to_response(g, size) for g in candidates[:SLOTS]]) | cost([])
            return runs

        return bench.all_observation_sets, 'observation_id', trial, digest(VOXEL), digest(module.__file__)
    module = load_upstream('causal', 'run_causal_benchmark.py')
    bench = module.CausalBenchmarkEnhanced(str(REFERENCE/'observations.json'))

    def trial(session, folder, seed, task):
        obs = task['observations']
        anchor = program_anchor(NODES, LIMIT, obs)
        assert anchor is not None
        # RR here is provisional until the original answer file membership audit.
        truth = task['n_ground_truths']
        runs = {}
        for method in ('reselect', 'repair'):
            r = run_causal(obs, anchor, SLOTS, method, session, folder, seed, bench)
            r.update(rr=r['found']/truth, vr=r['found']/SLOTS, nr=r['found']/SLOTS)
            runs[method] = r
        return runs

    return read(CAUSAL), 'id', trial, digest(CAUSAL), digest(module.__file__)


def summarize(rows):
    return {method: {**{k: mean(r['runs'][method][k] for r in rows) for k in METRICS},
                     **{k: sum(r['runs'][method].get(k, 0) for r in rows)
                        for k in ('found', 'calls', 'tokens', 'request_seconds')}}
            for method in rows[0]['runs']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain', choices=COUNTS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    domain, output = args.domain, args.output/args.domain
    output.mkdir(parents=True, exist_ok=True)
    tasks, id_key, trial, input_sha, scorer_sha = tasks_and_trial(domain)
    assert len(tasks) == len({t[id_key] for t in tasks}) == COUNTS[domain]
    model_digest = read(REFERENCE/'config.json')['model']
    config = {'domain': domain, 'cases': [t[id_key] for t in tasks], 'slots': SLOTS,
              'model': MODEL, 'model_digest': model_digest, 'input_sha': input_sha,
              'scorer_sha': scorer_sha, 'seeds': SEEDS, 'options': OPTIONS, 'think': False,
              'code_sha': {name: digest(HERE/name) for name in (Path(__file__).name,
                  'run_final_qwen_path_repair.py', 'run_qwen_path_repair.py', 'run_qwen_guided_edits.py',
                  'validated_evaluators.py', 'offline_3d.py', 'run_formal_agent_debate.py')}}
    path = output/'config.json'
    if path.exists() and read(path) != config:
        raise ValueError('Output already contains a different configuration')
    write_json(path, config)
    rows = []
    with requests.Session() as session:
        session.trust_env = False
        response = session.get(BASE_URL+'/api/tags', timeout=10)
        response.raise_for_status()
        assert next(m for m in response.json()['models'] if m['name'] == MODEL)['digest'] == model_digest
        for i, task in enumerate(tasks):
            case = task[id_key]
            folder = output/str(case)
            folder.mkdir(exist_ok=True)
            result_path = folder/'result.json'
            if result_path.exists():
                row = read(result_path)
            else:
                seed = SEEDS[i%len(SEEDS)]
                runs = trial(session, folder, seed, task)
                row = {'id': case, 'seed': seed, 'observations': task.get('n_observations'), 'runs': runs}
                tmp = result_path.with_suffix('.tmp')
                write_json(tmp, row)
                tmp.replace(result_path)
            rows.append(row)
            if (i+1)%10 == 0 or i+1 == len(tasks):
                print(f'{domain}: {i+1}/{len(tasks)}', flush=True)
    write_json(output/'summary.json', {'methods': summarize(rows), 'cases': len(rows)})


if __name__ == '__main__':
    main()
