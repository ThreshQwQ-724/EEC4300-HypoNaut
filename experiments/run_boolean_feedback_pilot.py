"""Paired Qwen pilot for mechanism-aware feedback on Boolean HypoSpace."""
import argparse
import ast
from collections import Counter
from datetime import datetime
from pathlib import Path
import random
import re
from statistics import mean
from time import perf_counter

import requests

from common import DATA, HERE, Replay, digest, load_upstream
from run_constraint_feedback import OPTIONS
from run_four_groups import BASE_URL, MODEL, SEEDS
from run_local_causal import write_json
from run_qwen_rr_budget import MODEL_REFERENCE, read
from validated_evaluators import corrected_boolean_class

SELECTION_SEED = 1016126
SLOTS = 8
MODES = ('official_prompt', 'mechanism_feedback')
METRICS = ('valid_rate', 'novelty_rate', 'recovery_rate')
SOURCE = DATA/'boolean/datasets/boolean_full_d2.json'
EXPRESSION = re.compile(r'(?:(?:AND|OR|NOT|NOR|x|y)\b|[(),\s])+')


def safe_answer(benchmark, answer):
    parsed = benchmark.parse_llm_response(answer)
    if parsed and len(parsed) <= 240 and EXPRESSION.fullmatch(parsed):
        return 'Expression: '+parsed
    return ''


def inspect_answer(benchmark, module, expr, observations, seen):
    if not expr or not benchmark._in_space(expr):
        return 'The previous output did not parse as an allowed expression within depth two.', None
    key = benchmark.get_expression_mechanistic_key(expr)
    expression = module.BooleanExpression(expr, benchmark.variables, benchmark.operators)
    for obs in observations:
        actual = expression.evaluate(obs.inputs)
        if actual != obs.output:
            return (f'At {obs.inputs}, the previous expression returned {actual}; expected {obs.output}. '
                    'Check this input and every other observed input before submitting.'), None
    if key in seen:
        return 'The previous expression repeated an already accepted mechanism.', key
    return 'The previous expression matched all observations and was structurally new.', key


def feedback_prompt(benchmark, valid, previous):
    roots = Counter(key[0] for key in valid)
    covered = ', '.join(f'{op}: {count}' for op, count in sorted(roots.items())) or 'none'
    examples = '\n'.join('Expression: '+expr for expr in valid.values()) or 'none'
    return ('\n\nVerifier feedback: '+previous+
            '\nVerified mechanisms already submitted:\n'+examples+
            '\nVerified root operator counts: '+covered+
            '\nExplore a different operator or nested structure, especially a root with few or zero '
            'verified mechanisms. Keep all observed outputs and the allowed depth. Reordering AND/OR '
            'inputs or repeating an identical operand does not create a new mechanism. Structurally '
            'different formulas may agree on all observed inputs. Use NOT (x OR y) style for NOR '
            'and count NOR as one grammar operation. Return one Expression: line.')


def trial(module, benchmark, task, mode, folder, seed, api):
    observations = [module.BooleanObservation(o['inputs'], o['output']) for o in task['observations']]
    history, answers, calls = [], [], []
    valid = {}
    previous = 'No earlier answer.'
    for slot in range(1, SLOTS+1):
        prompt = benchmark.create_prompt(observations, history)
        if mode == 'mechanism_feedback' and slot > 1:
            prompt += feedback_prompt(benchmark, valid, previous)
        payload = {'model': MODEL, 'think': False, 'stream': False, 'keep_alive': '10m',
                   'options': {**OPTIONS, 'seed': seed, 'num_predict': 128},
                   'messages': [{'role': 'user', 'content': prompt}]}
        path = folder/f'{mode}_{slot:02d}.json'
        if path.exists():
            call = read(path)
            assert call['request'] == payload
        else:
            start = perf_counter()
            response = api('/api/chat', payload)
            call = {'request': payload, 'response': response, 'seconds': perf_counter()-start}
            write_json(path, call)
        calls.append(call)
        answer = safe_answer(benchmark, call['response']['message'].get('content', ''))
        answers.append(answer)
        expr = benchmark.parse_llm_response(answer)
        if expr:
            history.append(expr)
        previous, key = inspect_answer(benchmark, module, expr, observations, valid)
        if key is not None:
            valid.setdefault(key, expr)
    return answers, calls, valid


def replay(benchmark, task, answers):
    result = benchmark.evaluate_single_observation_set(Replay(answers), task, SLOTS, False, max_retries=1)
    return {k: result[k] for k in (*METRICS, 'n_recovered_gts', 'n_valid', 'n_unique_all')}


def grammar_reference(benchmark, module, task, grammar):
    observations = [module.BooleanObservation(o['inputs'], o['output']) for o in task['observations']]
    compatible = [expr for _, expr in grammar if all(expr.evaluate(obs.inputs) == obs.output for obs in observations)]
    assert len(compatible) == len(task['ground_truth_expressions'])
    assert compatible
    # Leave unused slots empty when the space has fewer than eight mechanisms.
    answers = ['Expression: '+expr.formula for expr in compatible[:SLOTS]]
    return replay(benchmark, task, answers+['']*(SLOTS-len(answers)))


def report(output, records):
    names = ('original', 'corrected')
    summary = {version: {mode: {metric: mean(r['scores'][version][mode][metric] for r in records)
                               for metric in (*METRICS, 'n_recovered_gts')}
                         for mode in (*MODES, 'grammar_reference')} for version in names}
    summary['paired'] = {}
    for version in names:
        delta = [r['scores'][version]['mechanism_feedback']['recovery_rate'] -
                 r['scores'][version]['official_prompt']['recovery_rate'] for r in records]
        summary['paired'][version] = {'delta_rr': mean(delta), 'win_tie_loss': [
            sum(d > 0 for d in delta), sum(d == 0 for d in delta), sum(d < 0 for d in delta)]}
    write_json(output/'summary.json', summary)
    lines = ['# Qwen 布尔机制反馈开发试验', '',
             '从 HypoSpace 布尔数据按观测数 1–4 各抽三题。12 题均用两个提示条件各生成 8 次，共 192 次 Qwen3 8B 调用。每题配对使用同一个采样种子。两组均采用安全词法检查，保留未通过检查的槽位，原始响应完整保存。', '',
             '增强提示在第二槽开始反馈上一个表达式的观测反例、文法失败或重复，并提示探索此前较少出现的根算子。反馈只依赖题面和已经提交的表达式。原版和修正版评分器分别重放相同回答。程序参考枚举公开文法，再按观测过滤，采用同一提交预算。', '',
             '| 评分器 | 生成方法 | VR | NR | RR | 题均找回机制数 |',
             '|---|---|---:|---:|---:|---:|']
    labels = {'official_prompt': 'Qwen 官方提示', 'mechanism_feedback': 'Qwen 机制反馈',
              'grammar_reference': '程序文法枚举'}
    for version in names:
        for mode, d in summary[version].items():
            lines.append(f"| {version} | {labels[mode]} | {d['valid_rate']:.2%} | "
                         f"{d['novelty_rate']:.2%} | {d['recovery_rate']:.2%} | {d['n_recovered_gts']:.2f} |")
    lines += ['', '修正版保持数据生成器的 NOR 深度和公开文法。评分器修正不计为模型提升。12 题用于方向筛选，每题一个种子，后续需要独立验证。']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(summary, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    module = load_upstream('boolean', 'boolean_benchmark.py')
    from boolean_dataset import BooleanDiscoveryGame
    original = module.BooleanBenchmarkRefined(str(SOURCE))
    corrected = corrected_boolean_class(module)(str(SOURCE))
    rng = random.Random(SELECTION_SEED)
    tasks = [r for count in range(1, 5) for r in rng.sample(sorted(
        (t for t in original.all_observation_sets if t['n_observations'] == count),
        key=lambda t: t['observation_set_id']), 3)]
    grammar = sorted(((repr(expr.mechanistic_key(**corrected.mechanistic_opts)), expr)
                     for expr in BooleanDiscoveryGame.generate_all_expressions(
                         corrected.variables, corrected.operators, corrected.max_depth,
                         corrected.mechanistic_opts)), key=lambda pair: pair[0])
    for task in tasks:
        expected = {ast.literal_eval(g['mechanistic_key']) for g in task['ground_truth_expressions']}
        compatible = {ast.literal_eval(key) for key, expr in grammar if all(
            expr.evaluate(obs['inputs']) == obs['output'] for obs in task['observations'])}
        assert expected == compatible
    if args.verify_only:
        print([t['observation_set_id'] for t in tasks], len(grammar))
        assert all(grammar_reference(corrected, module, t, grammar)['n_recovered_gts']
                   == min(SLOTS, len(t['ground_truth_expressions'])) for t in tasks)
        return
    output = args.output or HERE/'runs'/('boolean_feedback_pilot_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.trust_env = False

    def api(route, payload=None):
        response = (session.get(BASE_URL+route, timeout=10) if payload is None
                    else session.post(BASE_URL+route, json=payload, timeout=(10, 180)))
        response.raise_for_status()
        return response.json()

    model = next(m for m in api('/api/tags')['models'] if m['name'] == MODEL)
    assert model['digest'] == read(MODEL_REFERENCE)['model']['digest']
    config = dict(model=model['digest'], version=api('/api/version'), slots=SLOTS, seeds=SEEDS,
                  selection_seed=SELECTION_SEED, cases=[r['observation_set_id'] for r in tasks],
                  source_sha=digest(SOURCE), code_sha=digest(__file__), evaluator_sha=digest(module.__file__),
                  corrected_sha=digest(HERE/'validated_evaluators.py'), options={**OPTIONS, 'num_predict': 128})
    path = output/'config.json'
    if path.exists():
        assert read(path) == config
    else:
        write_json(path, config)
    print('OUTPUT_DIR='+str(output.resolve()), flush=True)
    records = []
    for i, task in enumerate(tasks):
        item = task['observation_set_id']
        folder = output/item
        folder.mkdir(exist_ok=True)
        seed = SEEDS[i % len(SEEDS)]
        scores = {'original': {}, 'corrected': {}}
        costs = {}
        for mode in (MODES if i % 2 == 0 else MODES[::-1]):
            answers, calls, valid = trial(module, corrected, task, mode, folder, seed, api)
            scores['original'][mode] = replay(original, task, answers)
            scores['corrected'][mode] = replay(corrected, task, answers)
            assert scores['corrected'][mode]['n_recovered_gts'] == len(valid)
            costs[mode] = {'seconds': sum(c['seconds'] for c in calls),
                           'tokens': sum(c['response'].get('eval_count', 0) for c in calls)}
        for name, benchmark in [('original', original), ('corrected', corrected)]:
            scores[name]['grammar_reference'] = grammar_reference(benchmark, module, task, grammar)
        record = dict(item=item, seed=seed, gt=len(task['ground_truth_expressions']), scores=scores, costs=costs)
        write_json(folder/'result.json', record)
        records.append(record)
        print(f"{i+1}/12 {item} corrected "
              f"{[scores['corrected'][m]['n_recovered_gts'] for m in MODES]} of {record['gt']}", flush=True)
    assert len(list(output.glob('*/*_prompt_*.json'))) + len(list(output.glob('*/*_feedback_*.json'))) == 192
    report(output, records)


if __name__ == '__main__':
    main()
