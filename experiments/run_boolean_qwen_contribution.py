"""Measure Qwen's selection contribution with the same Boolean pruning rules."""
import argparse
from collections import Counter
from math import comb
from pathlib import Path
from statistics import mean

import requests

from run_boolean_counterexample_ablation import (
    BASE_URL, HERE, MODEL, OUTPUT as PILOT, ROOT, SLOTS, SOURCE, atomic,
    corrected_boolean_class, digest, load_upstream, read, score, trial,
)


OUTPUT = HERE/'runs/boolean_qwen_contribution_20261114'
METHODS = {'all': 'Qwen 选编号', 'program': '程序选最小编号'}
SCORERS = ('corrected', 'upstream')


def inspect_menus(task, catalog, trace):
    # Counterfactual scoring only. These labels are never passed into the selector.
    observations = task['observations']
    compatible = {i for i, expr in catalog.items()
                  if all(expr.evaluate(o['inputs']) == o['output'] for o in observations)}
    assert len(compatible) == len(task['ground_truth_expressions'])
    return [{'slot': step['slot'], 'menu_size': len(step['options']),
             'compatible_count': len(compatible.intersection(step['options'])),
             'qwen_valid': step['choice'] in compatible,
             'program_valid': min(step['options']) in compatible,
             'different_choice': step['choice'] != min(step['options'])}
            for step in trace]


def summarize(rows):
    result = {}
    for name in METHODS:
        runs = [r['runs'][name] for r in rows]
        result[name] = {'scores': {}, 'cost': {key: sum(r['cost'].get(key, 0) for r in runs)
            for key in ('calls', 'seconds', 'input_tokens', 'output_tokens', 'evaluations', 'lookups')}}
        for version in SCORERS:
            scores = [r['scores'][version] for r in runs]
            result[name]['scores'][version] = {
                **{key: mean(s[key] for s in scores) for key in ('recovery_rate', 'valid_rate', 'novelty_rate')},
                'hits': sum(s['n_recovered_gts'] for s in scores)}
    paired = {}
    for version in SCORERS:
        differences = Counter((a > b)-(a < b) for row in rows
            for a, b in [(row['runs']['all']['scores'][version]['n_recovered_gts'],
                          row['runs']['program']['scores'][version]['n_recovered_gts'])])
        wins, ties, losses = (differences[i] for i in (1, 0, -1))
        n = wins+losses
        paired[version] = {'wtl': [wins, ties, losses],
                           'sign_test_p': min(1., 2*sum(comb(n,k) for k in range(min(wins,losses)+1))/2**n)}
    menus = [s for row in rows for s in row['same_state']]
    return {'methods': result, 'paired': paired, 'cases': len(rows),
            'ceiling': mean(min(SLOTS,r['truth'])/r['truth'] for r in rows),
            'same_state': {'menus': len(menus), 'menus_with_legal': sum(s['compatible_count'] > 0 for s in menus),
                'qwen_hits': sum(s['qwen_valid'] for s in menus),
                'program_hits': sum(s['program_valid'] for s in menus),
                'random_expected_hits': sum(s['compatible_count']/s['menu_size'] for s in menus),
                'qwen_only': sum(s['qwen_valid'] and not s['program_valid'] for s in menus),
                'program_only': sum(s['program_valid'] and not s['qwen_valid'] for s in menus)}}


def report(output, rows):
    cohorts = {'全量': rows, '扩展 56 题': [r for r in rows if not r['pilot']],
               '原消融 24 题': [r for r in rows if r['pilot']]}
    cohorts.update({f'{n} 条观测': [r for r in rows if r['observations'] == n] for n in range(1, 5)})
    summary = {label: summarize(items) for label,items in cohorts.items()}
    lines = ['# Boolean 全量 Qwen 选编号贡献验证', '',
             '原始 80 题，每题 8 槽。复用上一轮 24 题原始结果，并对其余 56 题跑相同策略。'
             '56 题中含此前其他实验使用过的题。两组每轮都验证被选公式、移除已选编号，并使用全部反例剪枝。'
             '唯一决策差异是 Qwen 选编号或程序取最小剩余编号。评分使用原版与 NOR 修正版。', '',
             '| 样本 | 题数 | 评分器 | 选择策略 | RR | VR | NR | 命中 | 模型调用 |',
             '|---|---:|---|---|---:|---:|---:|---:|---:|']
    for label, block in summary.items():
        for version in SCORERS:
            for method, name in METHODS.items():
                run = block['methods'][method]
                d = run['scores'][version]
                lines.append(f"| {label} | {block['cases']} | {version} | {name} | {d['recovery_rate']:.4%} | "
                             f"{d['valid_rate']:.2%} | {d['novelty_rate']:.2%} | {d['hits']} | {run['cost']['calls']} |")
    lines += ['', '## 同状态复核', '',
              '在 Qwen 实际遇到的每个菜单上，事后比较 Qwen、程序首选和均匀随机的期望命中。'
              '这里只比较当前一步，标签不参与实际选择，不模拟替代策略后续会遇到的菜单。', '',
              '| 样本 | 菜单 | 含合法选项 | Qwen 命中 | 程序命中 | 随机期望命中 | 仅 Qwen 对 | 仅程序对 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for label, block in summary.items():
        d = block['same_state']
        lines.append(f"| {label} | {d['menus']} | {d['menus_with_legal']} | {d['qwen_hits']} | "
                     f"{d['program_hits']} | {d['random_expected_hits']:.2f} | {d['qwen_only']} | {d['program_only']} |")
    lines += ['', '## 逐题配对与成本', '']
    for label in ('全量', '扩展 56 题', '原消融 24 题'):
        block = summary[label]
        for version in SCORERS:
            p = block['paired'][version]
            lines.append(f"- {label}，{version} 合法图数胜／平／负 {'／'.join(map(str,p['wtl']))}，"
                         f"双侧符号检验 p={p['sign_test_p']:.4f}。")
    overall = summary['全量']
    for method, name in METHODS.items():
        cost = overall['methods'][method]['cost']
        lines.append(f"- {name}共 {cost['calls']} 次请求（含原 24 题），输入 {cost['input_tokens']:,} token，"
                     f"输出 {cost['output_tokens']:,} token，请求耗时 {cost['seconds']:.2f} 秒，"
                     f"计算 {cost['evaluations']:,} 个公式–观测单元。")
    lines += ['', f"修正版程序枚举的 8 槽 RR 上限为 {overall['ceiling']:.4%}。"
              '最小编号属于一个具体程序策略。原版和修正版对同一批提交评分。', '']
    atomic(output/'summary.json', summary)
    (output/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    module = load_upstream('boolean', 'boolean_benchmark.py')
    from boolean_dataset import BooleanDiscoveryGame
    original = module.BooleanBenchmarkRefined(str(SOURCE))
    corrected = corrected_boolean_class(module)(str(SOURCE))
    grammar = sorted(BooleanDiscoveryGame.generate_all_expressions(
        corrected.variables, corrected.operators, corrected.max_depth, corrected.mechanistic_opts),
        key=lambda expr: expr.formula)
    catalog = dict(enumerate(grammar, 1))
    tasks = {t['observation_set_id']: t for t in original.all_observation_sets}
    seeds = {r['id']:r['seed'] for r in read(ROOT/'boolean'/'results.json')}
    pilot_config = read(PILOT/'config.json')
    pilot = {r['id']:r for r in read(PILOT/'results.json')}
    assert set(pilot) == set(pilot_config['cases']) and set(pilot) <= set(tasks) == set(seeds)
    assert len(tasks) == 80 and len(pilot) == 24 and len(catalog) == 65
    assert digest(SOURCE) == pilot_config['data_sha'] and digest(module.__file__) == pilot_config['evaluator_sha']
    assert all(digest(HERE/name) == sha for name,sha in pilot_config['code_sha'].items())
    assert all(seeds[c] == pilot_config['seeds'][c] for c in pilot)
    cfg = {'cases': list(tasks), 'seeds': seeds, 'slots': SLOTS, 'data_sha': digest(SOURCE), 'model': MODEL,
           'model_digest': pilot_config['model_digest'], 'pilot_sha': digest(PILOT/'results.json'),
           'pilot_config_sha': digest(PILOT/'config.json'), 'runner_sha': digest(__file__)}
    config_path = output/'config.json'
    if config_path.exists() and read(config_path) != cfg:
        raise ValueError('Output directory contains a different experiment configuration')
    atomic(config_path, cfg)
    rows = []
    with requests.Session() as session:
        session.trust_env = False
        tags = session.get(BASE_URL+'/api/tags', timeout=10)
        tags.raise_for_status()
        assert next(m for m in tags.json()['models'] if m['name'] == MODEL)['digest'] == cfg['model_digest']
        for index, (case, task) in enumerate(tasks.items(), 1):
            folder = output/case
            folder.mkdir(exist_ok=True)
            path = folder/'result.json'
            if path.exists():
                row = read(path)
            else:
                runs = {}
                for method in METHODS:
                    if case in pilot:
                        runs[method] = pilot[case]['runs'][method]
                        continue
                    answers, cost, trace = trial(task,catalog,folder,seeds[case],session,method)
                    runs[method] = {'answers': answers, 'cost': cost, 'trace': trace,
                                    'scores': {k:score(b,task,answers) for k,b in
                                               (('corrected',corrected),('upstream',original))}}
                row = {'id': case, 'observations': task['n_observations'], 'pilot': case in pilot,
                       'truth': len(task['ground_truth_expressions']), 'runs': runs,
                       'same_state': inspect_menus(task, catalog, runs['all']['trace'])}
                atomic(path, row)
            assert row['id'] == case
            rows.append(row)
            if index % 8 == 0:
                print(f'Boolean {index}/{len(tasks)}', flush=True)
    atomic(output/'results.json', rows)
    report(output, rows)


if __name__ == '__main__':
    main()
