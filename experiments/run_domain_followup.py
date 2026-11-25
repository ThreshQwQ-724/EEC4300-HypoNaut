"""Follow-up experiments on disjoint HypoSpace samples."""
import argparse
from itertools import combinations, product
from pathlib import Path
import random
from statistics import mean

import requests

from common import HERE, digest
from run_all_domain_evaluation import ROOT, SLOTS, atomic, score, usage
from run_domain_improvement_pilot import OUTPUT as PILOT, boolean, causal, spatial, score_causal
from run_four_groups import BASE_URL, MODEL
from run_qwen_guided_edits import decode, read, request, schema


OUTPUT = HERE/'runs/domain_followup_20261114'
DOMAINS = ('boolean', '3d', 'causal')
SEED = 14112026
LABELS = {'feedback_only': '反例反馈', 'pruned': '反例反馈＋剪枝',
          'balanced': '无放回＋总高度分层', 'maximin': '分层＋柱高距离',
          'batch': '一次生成 8 个不同编号', 'base': '原路径反馈重选', 'fill': '原 Qwen＋空槽补位'}


def spatial_batch(task, session, folder, seed, benchmark):
    from offline_3d import from_heights, to_response

    obs = task['observation']
    size = benchmark.metadata['grid_size']
    height = benchmark.metadata['max_height']
    cells = [tuple(v+1 for v in divmod(i, size)) for i, x in enumerate(obs) if x == '1']
    catalog = dict(enumerate(product(range(1, height+1), repeat=len(cells)), 1))
    prompt = (f'The occupied columns are {cells} in row-column order. Below are height tuples. '
              'Choose exactly eight DIFFERENT IDs. Aim to cover as many total heights as possible '
              'and to spread the selected heights across all columns. Return a JSON object with the '
              'array choices containing eight distinct integer IDs.\nCandidates:\n'
              + '\n'.join(f'{i}. {values}, total {sum(values)}' for i, values in catalog.items()))
    call = request(session, folder/'batch.json', prompt, seed, schema({
        'choices': {'type': 'array', 'minItems': SLOTS, 'maxItems': SLOTS, 'uniqueItems': True,
                    'items': {'type': 'integer', 'enum': list(catalog)}}}))
    choices = decode(call).get('choices')
    submitted, used = [], set()
    for i in choices[:SLOTS] if isinstance(choices, list) else ():
        values = catalog[i] if type(i) is int and i in catalog and i not in used else None
        submitted.append(values)
        if values is not None:
            used.add(i)
    submitted.extend([None]*(SLOTS-len(submitted)))
    answers = [to_response(from_heights(obs, values), size) if values else '' for values in submitted]
    return answers, usage([call]), [{'heights': values} for values in submitted]


def fill_causal(task, prior, benchmark):
    from run_feedback_study import graph_line
    from solve_partial_graph import solve

    original = prior['answers']
    answers = list(original)
    remaining = [i for i, value in enumerate(answers) if not value]
    if not remaining:
        return answers, prior['cost'] | {'added': 0, 'extra_checks': 0}, []
    obs = task['observations']
    previous = {tuple(sorted(benchmark.parse_llm_response(answer).edges)) for answer in answers if answer}
    pool = solve(benchmark.nodes, benchmark.max_edges, obs, slots=SLOTS, constructive=True)
    candidates = [p['edges'] for p in pool['proposals'] if p['edges'] not in previous]
    filled = []
    for i, graph in zip(remaining, candidates):
        answers[i] = graph_line(graph)
        filled.append(i+1)
    assert all(not old or old == new for old, new in zip(original, answers, strict=True))
    return answers, prior['cost'] | {'added': len(filled), 'extra_checks': pool['checks']}, filled


def execute(domain, output):
    baseline_path = ROOT/domain/'results.json'
    baseline = {r['id']: r for r in read(baseline_path)}
    earlier = read(PILOT/domain/'config.json')
    old_config = read(ROOT/domain/'config.json')
    output.mkdir(parents=True, exist_ok=True)
    if domain == 'boolean':
        tasks, prune, scorers, strata, _, source = boolean(ablation=True)
        _, control, _, _, _, _ = boolean(prune=False, ablation=True)
        per_stratum = 4
    elif domain == '3d':
        tasks, experiment, scorers, strata, _, source = spatial(methods=('balanced', 'maximin'))
        per_stratum = 4
    else:
        tasks, _, scorers, strata, _, source = causal()
        per_stratum = 3
    assert old_config['source_sha'] == digest(source)
    eligible = set(tasks) - set(earlier['cases'])
    rng = random.Random(SEED)
    selected = [(case, 'stratified') for level in sorted({strata(tasks[c]) for c in eligible})
                for case in rng.sample(sorted(c for c in eligible if strata(tasks[c]) == level), per_stratum)]
    if domain == 'causal':
        picked = {c for c, _ in selected}
        gaps = sorted(c for c in eligible-picked
                      if baseline[c]['runs']['enhanced']['scores']['upstream']['n_recovered_gts']
                      < min(SLOTS, baseline[c]['truth']))
        selected.extend((c, 'gap') for c in rng.sample(gaps, 8))
    config = {'cases': [c for c, _ in selected], 'groups': dict(selected), 'seed': SEED, 'slots': SLOTS,
              'baseline_sha': digest(baseline_path), 'previous_sample_sha': digest(PILOT/domain/'config.json'),
              'runner_sha': digest(__file__), 'pilot_code_sha': digest(HERE/'run_domain_improvement_pilot.py'),
              'model_digest': old_config['digest']}
    path = output/'config.json'
    if path.exists() and read(path) != config:
        raise ValueError('Output directory has different settings')
    atomic(path, config)
    rows = []
    with requests.Session() as session:
        session.trust_env = False
        if domain != 'causal':
            response = session.get(BASE_URL+'/api/tags', timeout=10)
            response.raise_for_status()
            assert next(m for m in response.json()['models'] if m['name'] == MODEL)['digest'] == config['model_digest']
        for i, (case, group) in enumerate(selected, 1):
            task, previous = tasks[case], baseline[case]
            seed = previous['seed']
            folder = output/case
            folder.mkdir(exist_ok=True)
            result = folder/'result.json'
            if result.exists():
                row = read(result)
            else:
                methods = {}
                if domain == 'boolean':
                    for name, runner in (('feedback_only', control), ('pruned', prune)):
                        target = folder/name
                        target.mkdir(exist_ok=True)
                        methods[name] = runner(task, session, target, seed)
                elif domain == '3d':
                    methods = experiment(task, session, folder, seed)
                    methods['batch'] = spatial_batch(task, session, folder, seed, scorers['corrected'])
                else:
                    methods['fill'] = fill_causal(task, previous['runs']['enhanced'], scorers['upstream'])
                runs = {'base': previous['runs']['enhanced']}
                for name, (answers, cost, trace) in methods.items():
                    scores = {key: score_causal(b, task, answers) if domain == 'causal' else score(b, task, answers)
                              for key, b in scorers.items()}
                    runs[name] = {'answers': answers, 'cost': cost, 'scores': scores, 'trace': trace}
                row = {'id': case, 'group': group, 'stratum': strata(task), 'seed': seed,
                       'truth': previous['truth'], 'runs': runs}
                atomic(result, row)
            rows.append(row)
            print(f'{domain}: {i}/{len(selected)} {case}', flush=True)
    atomic(output/'results.json', rows)


def spatial_metrics(trace):
    heights = sorted({tuple(step['heights']) for step in trace if step['heights'] is not None})
    pairs = [sum(abs(a-b) for a,b in zip(x,y)) for x,y in combinations(heights, 2)]
    return {'height_totals': len({sum(x) for x in heights}), 'pairwise': mean(pairs) if pairs else 0,
            'minimum': min(pairs, default=0)}


def report(root):
    lines = ['# HypoSpace 三题型小样本复测', '',
             '每题固定 8 槽，沿用同一题的 Qwen3:8B 种子，三类样本均排除上一轮小样本。'
             'Boolean 按观测数各抽 4 题，3D 按相邻占据格对数各抽 4 题，因果图按干预数各抽 3 题，'
             '再另抽 8 道已知存在可补空槽的题做压力测试。', '',
             'Boolean 两组使用同一提示模板和反例反馈，开关仅控制是否移除违反反例的编号。'
             '3D 比较总高度分层、分层内最大化与已选组合的最小柱高距离、一次输出八个不同编号。'
             '因果图重放已有路径反馈答案，仅在推理全部结束后从约束构造池补空槽，已有答案和位置原样保留。', '',
             '| 题型 | 样本 | 方法 | RR | VR | NR | 合法图数 | 模型调用 |',
             '|---|---:|---|---:|---:|---:|---:|---:|']
    summary = {}
    for domain in DOMAINS:
        rows = read(root/domain/'results.json')
        primary = 'upstream' if domain == 'causal' else 'corrected'
        methods = LABELS | {'base': '上次全量 Qwen' if domain != 'causal' else LABELS['base']}
        blocks = {'分层': [row for row in rows if row['group'] == 'stratified']}
        if domain == 'causal':
            blocks['失分压力'] = [row for row in rows if row['group'] == 'gap']
        summary[domain] = {}
        for group, cases in blocks.items():
            anchor = {'boolean': 'feedback_only', '3d': 'balanced', 'causal': 'base'}[domain]
            block = {}
            for name in cases[0]['runs']:
                runs = [r['runs'][name] for r in cases]
                scores = {scorer: {k: mean(r['scores'][scorer][k] for r in runs)
                                   for k in ('recovery_rate','valid_rate','novelty_rate')} for scorer in runs[0]['scores']}
                metrics = scores[primary]
                diffs = [r['runs'][name]['scores'][primary]['n_recovered_gts'] -
                         r['runs'][anchor]['scores'][primary]['n_recovered_gts'] for r in cases]
                entry = {'scores': scores, 'hits': sum(r['scores'][primary]['n_recovered_gts'] for r in runs),
                         'wtl': [sum(d>0 for d in diffs), sum(d==0 for d in diffs), sum(d<0 for d in diffs)],
                         **{key: sum(r['cost'].get(key, 0) for r in runs)
                            for key in ('calls','seconds','output_tokens','extra_checks','added')}}
                if domain == '3d' and name != 'base':
                    profiles = [spatial_metrics(r['trace']) for r in runs]
                    entry['diversity'] = {key: mean(p[key] for p in profiles) for key in profiles[0]}
                block[name] = entry
                lines.append(f"| {domain} {group} | {len(cases)} | {methods[name]} | "
                             f"{metrics['recovery_rate']:.4%} | {metrics['valid_rate']:.2%} | "
                             f"{metrics['novelty_rate']:.2%} | {entry['hits']} | {entry['calls']} |")
            summary[domain][group] = block
    lines += ['', '表中 Boolean 和 3D 采用修正版评分器。原版得分、请求耗时和逐题胜平负保存在 summary.json。'
              '因果调用数为原 Qwen 流程的历史调用，空槽补位阶段新增模型调用为零。', '']
    atomic(root/'summary.json', summary)
    (root/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('domain', choices=(*DOMAINS, 'summary'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.domain == 'summary':
        report(args.output)
    else:
        execute(args.domain, args.output/args.domain)


if __name__ == '__main__':
    main()
