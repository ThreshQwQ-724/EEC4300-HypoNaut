"""Compare Boolean grammar-constrained Qwen selection with and without feedback."""
import argparse
from pathlib import Path
import random
from statistics import mean

import requests

from common import HERE, digest, load_upstream
from run_boolean_feedback_pilot import METRICS, SLOTS, SOURCE, inspect_answer, replay
from run_four_groups import BASE_URL, MODEL, SEEDS
from run_local_causal import write_json
from run_qwen_guided_edits import decode, request, schema
from run_qwen_rr_budget import read
from validated_evaluators import corrected_boolean_class

PRIOR = HERE/'runs/boolean_feedback_pilot_20261113_164512'
PREFIX = HERE/'runs/boolean_prefix_pilot_20261113_164814'
MODES = ('grammar_only', 'grammar_feedback')


def trial(benchmark, module, item, formulas, mode, seed, folder, session):
    observations = [module.BooleanObservation(row['inputs'], row['output']) for row in item['observations']]
    menu = '\n'.join(f'{i}. {formula}' for i, formula in enumerate(formulas, 1))
    answers, calls, choices, used, accepted = [], [], [], set(), {}
    feedback = 'No earlier answer.'
    for slot in range(1, SLOTS+1):
        remaining = [i for i in range(1, len(formulas)+1) if i not in used]
        prompt = (
            'Choose one Boolean expression that matches every observed input-output pair. '
            'The numbered catalog includes incompatible expressions. NOT negates its operand. '
            'Select one ID and return it as JSON choice.\nObservations:\n'
            + '\n'.join(obs.to_string() for obs in observations)
            + '\nAlready selected IDs: ' + (', '.join(map(str, sorted(used))) or 'none')
            + '\nCatalog:\n' + menu)
        if mode == 'grammar_feedback' and slot > 1:
            prompt += '\nVerifier on the previous selection: ' + feedback
        call = request(session, folder/f'{mode}_{slot:02}.json', prompt, seed, schema({
            'choice': {'type': 'integer', 'enum': remaining}}))
        choice = decode(call).get('choice')
        formula = formulas[choice-1] if type(choice) is int and choice in remaining else None
        if formula is not None:
            used.add(choice)
        feedback, key = inspect_answer(benchmark, module, formula, observations, accepted)
        if key is not None:
            accepted[key] = formula
        calls.append(call)
        answers.append('Expression: '+formula if formula is not None else '')
        choices.append(choice if formula is not None else None)
    return answers, choices, calls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE/'runs/boolean_constrained_pilot_20261113')
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    module = load_upstream('boolean', 'boolean_benchmark.py')
    from boolean_dataset import BooleanDiscoveryGame
    original = module.BooleanBenchmarkRefined(str(SOURCE))
    corrected = corrected_boolean_class(module)(str(SOURCE))
    previous = read(PRIOR/'config.json')
    assert previous['source_sha'] == digest(SOURCE)
    all_tasks = {r['observation_set_id']: r for r in original.all_observation_sets}
    # Freeze a fresh stratum-balanced follow-up before invoking the model.
    rng = random.Random(1017126)
    cases = [item for n in range(1, 5) for item in rng.sample(sorted(
        i for i, r in all_tasks.items() if r['n_observations'] == n and i not in previous['cases']), 3)]
    grammar = BooleanDiscoveryGame.generate_all_expressions(
        corrected.variables, corrected.operators, corrected.max_depth, corrected.mechanistic_opts)
    formulas = sorted(expr.formula for expr in grammar)
    assert len(formulas) == len(set(formulas)) == 65
    assert all(corrected._in_space(formula) for formula in formulas)
    session = requests.Session()
    session.trust_env = False
    response = session.get(BASE_URL+'/api/tags', timeout=10)
    response.raise_for_status()
    model = next(m for m in response.json()['models'] if m['name'] == MODEL)
    assert model['digest'] == previous['model']
    config = {'model': model['digest'], 'cases': cases, 'catalog': formulas, 'slots': SLOTS,
              'selection_seed': 1017126, 'source_sha': digest(SOURCE), 'runner_sha': digest(__file__),
              'scorer_sha': digest(HERE/'validated_evaluators.py'),
              'upstream_sha': digest(module.__file__), 'request_helper_sha': digest(HERE/'run_qwen_guided_edits.py')}
    path = output/'config.json'
    if path.exists():
        assert read(path) == config
    else:
        write_json(path, config)
    print('OUTPUT_DIR='+str(output.resolve()), flush=True)
    records = []
    for i, item in enumerate(cases):
        task = all_tasks[item]
        folder = output/item
        folder.mkdir(exist_ok=True)
        seed = SEEDS[i % len(SEEDS)]
        scores = {'original': {}, 'corrected': {}}
        costs = {}
        for mode in MODES if i % 2 == 0 else MODES[::-1]:
            answers, choices, calls = trial(corrected, module, task, formulas, mode, seed, folder, session)
            for version, benchmark in [('original', original), ('corrected', corrected)]:
                scores[version][mode] = replay(benchmark, task, answers)
            costs[mode] = {'choices': choices, 'accepted_choices': sum(choice is not None for choice in choices),
                           'tokens': sum(c['response'].get('eval_count', 0) for c in calls),
                           'seconds': sum(c['seconds'] for c in calls)}
        gt = len(task['ground_truth_expressions'])
        ceiling = min(SLOTS, gt)/gt
        record = dict(item=item, seed=seed, gt=gt, ceiling=ceiling, scores=scores, costs=costs)
        write_json(folder/'result.json', record)
        records.append(record)
        print(f"{i+1}/12 {item} {[scores['corrected'][m]['n_recovered_gts'] for m in MODES]} of {gt}", flush=True)
    assert len(list(output.glob('*/grammar_*.json'))) == 192
    summary = {v: {m: {k: mean(r['scores'][v][m][k] for r in records)
                       for k in (*METRICS, 'n_recovered_gts')} for m in MODES}
               for v in ('original', 'corrected')}
    summary['ceiling'] = mean(r['ceiling'] for r in records)
    summary['choices'] = {m: sum(r['costs'][m]['accepted_choices'] for r in records) for m in MODES}
    diffs = [r['scores']['corrected']['grammar_feedback']['recovery_rate'] -
             r['scores']['corrected']['grammar_only']['recovery_rate'] for r in records]
    summary['paired'] = [sum(d>0 for d in diffs), sum(d==0 for d in diffs), sum(d<0 for d in diffs)]
    write_json(output/'summary.json', summary)
    lines = ['# 布尔公开文法约束解码试验', '',
             '从上一轮未用的题目中按 1–4 条观测各取三题，每题八槽，每个条件 96 次调用。两组共享同一份 65 项公开文法目录。目录不按题面过滤。每次使用 JSON 编号约束，排除已经选过的编号。反馈组另提供上次选择的观测反例或成功信息。原版与修正版重放完全相同的公式。', '',
             '| 评分器 | 策略 | VR | NR | RR | 找回机制数 |',
             '|---|---|---:|---:|---:|---:|']
    for v in ('original', 'corrected'):
        for m, row in summary[v].items():
            lines.append(f"| {v} | {m} | {row['valid_rate']:.2%} | {row['novelty_rate']:.2%} | "
                         f"{row['recovery_rate']:.2%} | {round(12*row['n_recovered_gts'])} |")
    lines += ['', f"修正版八槽 RR 上限 {summary['ceiling']:.2%}。反馈组相对无反馈组逐题胜平负为 "
              + '/'.join(map(str, summary['paired'])) + '。',
              '', '前两次开发试验与本试验使用不同题目。公开文法的完整枚举为可直接求解的参考，约束解码结果只衡量 Qwen 从未过滤目录选取合法表达式的能力。']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(summary, flush=True)


if __name__ == '__main__':
    main()
