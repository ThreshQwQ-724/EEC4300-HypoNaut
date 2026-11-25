"""Separate prefix notation from verifier feedback on the same Boolean tasks."""
import argparse
from datetime import datetime
from pathlib import Path
import re
from statistics import mean
from time import perf_counter

import requests

from common import HERE, digest, load_upstream
from run_boolean_feedback_pilot import (
    METRICS, SLOTS, SOURCE, feedback_prompt, inspect_answer, replay,
)
from run_constraint_feedback import OPTIONS
from run_four_groups import BASE_URL, MODEL
from run_local_causal import write_json
from run_qwen_rr_budget import read
from validated_evaluators import corrected_boolean_class

PREVIOUS = HERE/'runs/boolean_feedback_pilot_20261113_164512'
MODES = ('prefix_only', 'prefix_feedback')
RULES = (
    '\n\nUse only prefix notation for this answer. A variable is x or y. '
    'Write (NOT a), (AND a b), (OR a b), or (NOR a b), where a and b are expressions. '
    'Depth counts each operator once, including NOR. Maximum depth is two. '
    'NOR is true only when both arguments are false. '
    'For example, (NOR (AND x y) (NOT x)) has depth two. '
    'Return one line beginning with Expression: followed by the prefix expression. '
    'Use no commas or infix operators. Any earlier expressions may appear in translated form.'
)


def translate(answer):
    """Parse an exact prefix tree and serialize to the benchmark's infix notation."""
    line = answer.strip()
    if line.lower().startswith('expression:'):
        line = line[len('expression:'):].strip()
    if not line or len(line) > 240 or not re.fullmatch(r'[a-zA-Z()\s]+', line):
        return ''
    tokens = re.findall(r'[A-Za-z]+|[()]', line)
    stream = iter(tokens)

    def parse(depth):
        token = next(stream)
        if token in ('x', 'y'):
            return token
        if token != '(' or depth >= 2:
            raise ValueError('invalid prefix or depth')
        op = next(stream)
        if op not in ('NOT', 'AND', 'OR', 'NOR'):
            raise ValueError('unknown operator')
        left = parse(depth+1)
        if op == 'NOT':
            result = f'NOT ({left})'
        else:
            right = parse(depth+1)
            infix = 'OR' if op == 'NOR' else op
            result = f'(({left}) {infix} ({right}))'
            if op == 'NOR':
                result = 'NOT '+result
        if next(stream) != ')':
            raise ValueError('missing close parenthesis')
        return result

    try:
        expression = parse(0)
        return 'Expression: '+expression if next(stream, None) is None else ''
    except (ValueError, StopIteration):
        return ''


def trial(module, benchmark, task, mode, folder, seed, api):
    observations = [module.BooleanObservation(o['inputs'], o['output']) for o in task['observations']]
    history, answers, calls = [], [], []
    valid = {}
    previous = 'No earlier answer.'
    for slot in range(1, SLOTS+1):
        prompt = benchmark.create_prompt(observations, history)
        if mode == 'prefix_feedback' and slot > 1:
            prompt += feedback_prompt(benchmark, valid, previous).replace(
                'Use NOT (x OR y) style for NOR and count NOR as one grammar operation. ', '')
        prompt += RULES
        payload = {'model': MODEL, 'think': False, 'stream': False, 'keep_alive': '10m',
                   'options': {**OPTIONS, 'seed': seed, 'num_predict': 128},
                   'messages': [{'role': 'user', 'content': prompt}]}
        path = folder/f'{mode}_{slot:02d}.json'
        if path.exists():
            call = read(path)
            assert call['request'] == payload
        else:
            started = perf_counter()
            response = api('/api/chat', payload)
            call = {'request': payload, 'response': response, 'seconds': perf_counter()-started}
            write_json(path, call)
        calls.append(call)
        translated = translate(call['response']['message'].get('content', ''))
        answers.append(translated)
        expr = benchmark.parse_llm_response(translated)
        if expr:
            history.append(expr)
        previous, key = inspect_answer(benchmark, module, expr, observations, valid)
        if key is not None:
            valid.setdefault(key, expr)
    return answers, calls, valid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    previous = read(PREVIOUS/'config.json')
    assert previous['source_sha'] == digest(SOURCE)
    module = load_upstream('boolean', 'boolean_benchmark.py')
    original = module.BooleanBenchmarkRefined(str(SOURCE))
    corrected = corrected_boolean_class(module)(str(SOURCE))
    tasks = {row['observation_set_id']: row for row in original.all_observation_sets}
    output = args.output or HERE/'runs'/('boolean_prefix_pilot_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.trust_env = False

    def api(route, payload=None):
        response = (session.get(BASE_URL+route, timeout=10) if payload is None
                    else session.post(BASE_URL+route, json=payload, timeout=(10, 180)))
        response.raise_for_status()
        return response.json()

    model = next(m for m in api('/api/tags')['models'] if m['name'] == MODEL)
    assert model['digest'] == previous['model']
    config = dict(model=model['digest'], version=api('/api/version'), slots=SLOTS,
                  cases=previous['cases'], previous_config_sha=digest(PREVIOUS/'config.json'),
                  code_sha=digest(__file__), helper_sha=digest(HERE/'run_boolean_feedback_pilot.py'),
                  evaluator_sha=digest(module.__file__), corrected_sha=digest(HERE/'validated_evaluators.py'))
    path = output/'config.json'
    if path.exists():
        assert read(path) == config
    else:
        write_json(path, config)
    print('OUTPUT_DIR='+str(output.resolve()), flush=True)
    records = []
    for index, item in enumerate(previous['cases']):
        before = read(PREVIOUS/item/'result.json')
        seed = before['seed']
        folder = output/item
        folder.mkdir(exist_ok=True)
        scores = before['scores']
        costs = {}
        for mode in (MODES if index % 2 == 0 else MODES[::-1]):
            answers, calls, valid = trial(module, corrected, tasks[item], mode, folder, seed, api)
            for name, benchmark in [('original', original), ('corrected', corrected)]:
                scores[name][mode] = replay(benchmark, tasks[item], answers)
            assert scores['corrected'][mode]['n_recovered_gts'] == len(valid)
            costs[mode] = {'parsed_prefix': sum(bool(answer) for answer in answers), 'answers': answers,
                           'seconds': sum(c['seconds'] for c in calls),
                           'output_tokens': sum(c['response'].get('eval_count', 0) for c in calls)}
        record = dict(item=item, seed=seed, scores=scores, costs=costs, gt=before['gt'])
        write_json(folder/'result.json', record)
        records.append(record)
        print(f"{index+1}/12 {item}: {[scores['corrected'][m]['n_recovered_gts'] for m in MODES]}", flush=True)
    assert len(list(output.glob('*/prefix_*.json'))) == 192
    conditions = ('official_prompt', 'mechanism_feedback', *MODES, 'grammar_reference')
    summary = {version: {mode: {metric: mean(r['scores'][version][mode][metric] for r in records)
                               for metric in (*METRICS, 'n_recovered_gts')}
                         for mode in conditions} for version in ('original', 'corrected')}
    summary['prefix_accepted'] = {mode: sum(r['costs'][mode]['parsed_prefix'] for r in records) for mode in MODES}
    for version in ('original', 'corrected'):
        differences = [r['scores'][version]['prefix_feedback']['recovery_rate'] -
                       r['scores'][version]['prefix_only']['recovery_rate'] for r in records]
        summary[version]['feedback_pair'] = [sum(d > 0 for d in differences), sum(d == 0 for d in differences),
                                             sum(d < 0 for d in differences)]
    write_json(output/'summary.json', summary)
    print(summary, flush=True)


if __name__ == '__main__':
    main()
