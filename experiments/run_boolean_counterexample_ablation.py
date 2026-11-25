"""Compare first and all counterexamples with matched selection and a program control."""
import argparse
from collections import Counter
from pathlib import Path
import random
from statistics import mean

import requests

from common import HERE, digest, load_upstream
from run_all_domain_evaluation import ROOT, SLOTS, atomic, pad, score, usage
from run_boolean_feedback_pilot import SOURCE
from run_domain_improvement_pilot import pick
from run_four_groups import BASE_URL, MODEL
from run_qwen_guided_edits import read
from validated_evaluators import corrected_boolean_class


OUTPUT = HERE/'runs/boolean_counterexample_ablation_20261114'
SEED = 14112027
METHODS = {'first': 'Qwen＋首条反例', 'all': 'Qwen＋全部反例', 'program': '程序选编号＋全部反例'}


def trial(task, catalog, folder, seed, session, method):
    observations = task['observations']
    eligible = set(catalog)
    answers, calls, trace = [], [], []
    feedback = 'No previous selection.'
    checks = Counter()
    # Cache identical truth-table cells. Read only input/output pairs in the task.
    cells = {}

    def evaluate(candidate, index):
        key = candidate, index
        if key not in cells:
            cells[key] = catalog[candidate].evaluate(observations[index]['inputs'])
            checks['evaluations'] += 1
        checks['lookups'] += 1
        return cells[key]

    for slot in range(1, SLOTS+1):
        if not eligible:
            break
        options = sorted(eligible)
        if method == 'program':
            choice = options[0]
        else:
            prompt = ('Select one formula that matches all observed inputs and outputs. Return its JSON ID. '
                      'The menu may contain incompatible formulas.\nObservations:\n'
                      + '\n'.join(f'{o["inputs"]} -> {o["output"]}' for o in observations)
                      + '\nVerifier feedback: '+feedback+'\nRemaining formulas:\n'
                      + '\n'.join(f'{i}. {catalog[i].formula}' for i in options))
            path = folder/('shared_01.json' if slot == 1 else f'{method}_{slot:02}.json')
            choice, call = pick(session, path, seed, prompt, options)
            calls.append(call)
        answers.append('Expression: '+catalog[choice].formula if choice is not None else '')
        counterexamples, removed = [], []
        if choice is not None:
            eligible.remove(choice)
            for index, obs in enumerate(observations):
                actual = evaluate(choice, index)
                if actual != obs['output']:
                    counterexamples.append({'index': index, 'actual': int(actual)})
                    if method == 'first':
                        break
            removed = [i for i in sorted(eligible) if any(
                evaluate(i, c['index']) != observations[c['index']]['output'] for c in counterexamples)]
            eligible.difference_update(removed)
            feedback = ('\n'.join(f'At {observations[c["index"]]["inputs"]}, the last formula returned '
                                  f'{c["actual"]}; expected {observations[c["index"]]["output"]}.'
                                  for c in counterexamples)
                        or 'The last formula matched all observations.')
        trace.append({'slot': slot, 'options': options, 'choice': choice, 'removed': removed,
                      'counterexamples': counterexamples})
    return pad(answers), usage(calls) | dict(checks), trace


def summarize(output, rows):
    stats = {}
    lines = ['# Boolean 首条与全部反例消融', '',
             '从原始 80 题的 2、3、4 条观测层各随机抽 8 题，排除前两轮小样本中的题号。'
             '这些题有既有全量基线，未用于前两轮反例剪枝调试。每题 8 个提交槽，三组共用公开文法、'
             '公式编号和观测。Qwen 两组使用相同提示模板、模型参数及种子，第一槽共享同一请求。', '',
             '首条组在发现第一个错误时停止诊断。全部组检查所选公式的所有观测并反馈所有错误。'
             '只剪除违反本轮反例的未选公式。程序组使用相同全部反例规则，每槽取最小剩余编号。'
             '程序组的挑选顺序是确定性参照。槽位耗尽或菜单为空即停止。', '',
             '| 评分器 | 方法 | RR | VR | NR | 合法命中 | 调用 | 输入 token | 请求秒数 |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for method, label in METHODS.items():
        runs = [r['runs'][method] for r in rows]
        d = {key: sum(r['cost'].get(key, 0) for r in runs)
             for key in ('calls', 'seconds', 'input_tokens', 'output_tokens', 'evaluations', 'lookups')}
        d['removed'] = sum(len(s['removed']) for r in runs for s in r['trace'])
        d['counterexamples'] = sum(len(s['counterexamples']) for r in runs for s in r['trace'])
        d['scores'] = {}
        for version in ('corrected','upstream'):
            scores = [r['scores'][version] for r in runs]
            result = {k: mean(s[k] for s in scores) for k in ('recovery_rate','valid_rate','novelty_rate')}
            result['hits'] = sum(s['n_recovered_gts'] for s in scores)
            d['scores'][version] = result
            lines.append(f"| {version} | {label} | {result['recovery_rate']:.4%} | {result['valid_rate']:.2%} | "
                         f"{result['novelty_rate']:.2%} | {result['hits']} | {d['calls']} | "
                         f"{d['input_tokens']:,} | {d['seconds']:.2f} |")
        stats[method] = d
    paired = {}
    lines += ['', '## 逐题配对与程序工作量', '']
    for left, right in (('all','first'), ('all','program')):
        for version in ('corrected','upstream'):
            diff = [r['runs'][left]['scores'][version]['n_recovered_gts'] -
                    r['runs'][right]['scores'][version]['n_recovered_gts'] for r in rows]
            wtl = [sum(d>0 for d in diff), sum(d==0 for d in diff), sum(d<0 for d in diff)]
            paired[f'{left}_vs_{right}_{version}'] = wtl
            lines.append(f"- {version}，{METHODS[left]}对{METHODS[right]}的合法命中数胜／平／负为 "
                         f"{'／'.join(map(str,wtl))}。")
    for method, d in stats.items():
        lines.append(f"- {METHODS[method]}提取 {d['counterexamples']} 条反例，排除 {d['removed']} 个候选，"
                     f"计算 {d['evaluations']} 个公式–观测单元，含缓存共访问 {d['lookups']} 次。")
    strata = {n: {method: mean(r['runs'][method]['scores']['corrected']['recovery_rate']
                                 for r in rows if r['observations'] == n) for method in METHODS} for n in (2,3,4)}
    lines += ['', 'RR 按题宏平均。请求成本按各方法独立运行统计。修正版修正 NOR 的文法深度判定；'
              '两种评分器接收相同输出。程序组用于衡量共享剪枝规则下 Qwen 的选图贡献。', '',
              '此前的剪枝实现将循环退出放在无剪枝分支中。开启剪枝时会继续检查后续观测，'
              '所以此前结果用于说明开启剪枝的整体收益。本轮使用显式首条退出和全部收集。', '']
    atomic(output/'summary.json', {'methods': stats, 'paired': paired, 'strata': strata})
    (output/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(stats, paired, flush=True)


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
        key=lambda e: e.formula)
    catalog = dict(enumerate(grammar, 1))
    assert len(catalog) == 65 and all(corrected._in_space(e.formula) for e in grammar)
    previous = {c for d in ('domain_followup_20261114','domain_improvement_pilot_20261114')
                for c in read(HERE/'runs'/d/'boolean'/'config.json')['cases']}
    tasks = {t['observation_set_id']:t for t in original.all_observation_sets}
    rng = random.Random(SEED)
    cases = [c for n in (2,3,4) for c in rng.sample(sorted(c for c,t in tasks.items()
             if c not in previous and t['n_observations'] == n), 8)]
    source_config = read(ROOT/'boolean'/'config.json')
    seeds = {r['id']: r['seed'] for r in read(ROOT/'boolean'/'results.json')}
    assert digest(SOURCE) == source_config['source_sha']
    config = {'cases': cases, 'selection_seed': SEED, 'slots': SLOTS, 'data_sha': digest(SOURCE),
              'model': MODEL, 'model_digest': source_config['digest'], 'evaluator_sha': digest(module.__file__),
              'code_sha': {name: digest(HERE/name) for name in (Path(__file__).name,'validated_evaluators.py',
                          'run_domain_improvement_pilot.py','run_qwen_guided_edits.py')},
              'seeds': {c: seeds[c] for c in cases}}
    path = output/'config.json'
    if path.exists() and read(path) != config:
        raise ValueError('Output directory has a different experiment configuration')
    atomic(path, config)
    rows = []
    with requests.Session() as session:
        session.trust_env = False
        tags = session.get(BASE_URL+'/api/tags', timeout=10)
        tags.raise_for_status()
        assert next(m for m in tags.json()['models'] if m['name'] == MODEL)['digest'] == config['model_digest']
        for i, case in enumerate(cases,1):
            task = tasks[case]
            folder = output/case
            folder.mkdir(exist_ok=True)
            runs = {}
            for method in (('first','all','program') if i % 2 else ('all','first','program')):
                answers,cost,trace = trial(task,catalog,folder,seeds[case],session,method)
                runs[method] = {'answers': answers, 'trace': trace, 'cost': cost,
                                'scores': {key:score(b,task,answers) for key,b in
                                           (('corrected',corrected),('upstream',original))}}
            assert runs['first']['trace'][0]['choice'] == runs['all']['trace'][0]['choice']
            row = {'id': case, 'observations': task['n_observations'],
                   'truth': len(task['ground_truth_expressions']), 'runs': runs}
            atomic(folder/'result.json',row)
            rows.append(row)
            print(f'{i}/{len(cases)} {case} '
                  + str([runs[m]['scores']['corrected']['n_recovered_gts'] for m in METHODS]),flush=True)
    atomic(output/'results.json',rows)
    summarize(output,rows)


if __name__ == '__main__':
    main()
