"""Compare batched spatial diversity and preserve-and-fill causal policies."""
import argparse
from collections import Counter
from itertools import combinations, product
from pathlib import Path
import random
from statistics import mean
from time import perf_counter

import requests

from common import HERE, digest
from run_all_domain_evaluation import ROOT, SLOTS, atomic, score, usage
from run_domain_followup import OUTPUT as FOLLOWUP, fill_causal, spatial_batch, spatial_metrics
from run_domain_improvement_pilot import OUTPUT as PILOT, causal, pick, score_causal, spatial
from run_four_groups import BASE_URL, MODEL
from run_qwen_guided_edits import read, signature


OUTPUT = HERE/'runs/spatial_causal_followup_20261114'
SEED = 15112026
LABELS = {'maximin': '逐槽分层＋柱高距离', 'batch': '一次批量选编号',
          'combined': '分层距离方案＋一次选方案',
          'base': '原 Qwen 路径反馈', 'fill': '顺序补位',
          'diverse': '同池预测多样性补位', 'wide': '扩池预测多样性补位'}


def portfolios(catalog):
    """Shortlist high-distance sets that represent every attainable total height."""
    by_total = {}
    for values in catalog:
        by_total.setdefault(sum(values), []).append(values)
    groups = list(by_total.values())
    assert len(groups) == SLOTS-1
    distances = {(a,b): sum(abs(x-y) for x,y in zip(a,b,strict=True)) for a,b in product(catalog, repeat=2)}
    ranked = []
    for group in groups:
        for selection in product(*(list(combinations(g, 2)) if g is group else [(v,) for v in g]
                                   for g in groups)):
            values = tuple(v for subset in selection for v in subset)
            pairs = [distances[a,b] for a,b in combinations(values, 2)]
            ranked.append((-sum(pairs), -min(pairs), values))
    ranked.sort()
    return [values for _,_,values in ranked[:8]]


def combined_spatial(task, session, folder, seed, benchmark, shortlist):
    from offline_3d import from_heights, to_response

    obs = task['observation']
    size = benchmark.metadata['grid_size']
    cells = [tuple(v+1 for v in divmod(i, size)) for i,x in enumerate(obs) if x == '1']
    prompt = (f'Three occupied columns, in row-column order: {cells}. Each plan contains eight '
              'distinct height tuples, from ground level up, and covers all total heights. '
              'Choose one plan to produce diverse shapes. Prefer a balanced spread of heights across '
              'the three column positions. Return the plan ID only as JSON choice.\nPlans:\n'
              + '\n'.join(f'{i}. {values}' for i,values in enumerate(shortlist,1)))
    choice, call = pick(session, folder/'combined.json', seed, prompt, list(range(1,len(shortlist)+1)))
    selected = shortlist[choice-1] if choice is not None else [None]*SLOTS
    answers = [to_response(from_heights(obs, values), size) if values is not None else '' for values in selected]
    return answers, usage([call]), [{'heights': values, 'plan': choice} for values in selected]


def causal_pattern_fill(task, prior, benchmark, pool_size):
    from run_feedback_study import graph_line
    from solve_partial_graph import solve

    answers = prior['answers'].copy()
    empty = [i for i, answer in enumerate(answers) if not answer]
    if not empty:
        return answers, prior['cost'] | {'added': 0, 'extra_checks': 0}, []
    nodes = benchmark.nodes
    obs = task['observations']
    unknown = sorted(set(nodes)-{o['perturbed_node'] for o in obs})
    # An empty prediction vector makes all choices equivalent on this metric.
    if not unknown:
        return fill_causal(task, prior, benchmark)
    chosen = {tuple(sorted(benchmark.parse_llm_response(answer).edges)) for answer in answers if answer}
    pool = solve(nodes, benchmark.max_edges, obs, slots=pool_size, constructive=True)
    remaining = [p['edges'] for p in pool['proposals'] if p['edges'] not in chosen]
    predictions = {g: tuple(map(frozenset, signature(g, nodes, unknown))) for g in chosen.union(remaining)}
    known = {predictions[g] for g in chosen}
    filled = []
    for i in empty:
        if not remaining:
            break

        def rank(g):
            pattern = predictions[g]
            nearest = min((sum(len(a ^ b) for a,b in zip(pattern,p,strict=True)) for p in known), default=0)
            return pattern in known, -nearest

        graph = min(remaining, key=rank)
        remaining.remove(graph)
        known.add(predictions[graph])
        answers[i] = graph_line(graph)
        filled.append(i+1)
    return answers, prior['cost'] | {'added': len(filled), 'extra_checks': pool['checks']}, filled


def prediction_metrics(answers, benchmark, unknown):
    patterns = [tuple(map(frozenset, signature(parsed.edges, benchmark.nodes, unknown))) for text in answers
                if (parsed := benchmark.parse_llm_response(text)) is not None]
    distances = [sum(len(a ^ b) for a,b in zip(x,y,strict=True)) for x,y in combinations(patterns, 2)]
    return {'patterns': len(set(patterns)), 'pairwise': mean(distances) if distances else 0}


def cases_for(domain, tasks, stratum, baseline):
    excluded = set(read(PILOT/domain/'config.json')['cases']) | set(read(FOLLOWUP/domain/'config.json')['cases'])
    eligible = set(tasks)-excluded
    rng = random.Random(SEED)
    groups = {level: sorted(c for c in eligible if stratum(tasks[c]) == level)
              for level in sorted({stratum(tasks[c]) for c in eligible})}
    cases = [(case, '分层') for group in groups.values() for case in rng.sample(group, 4 if domain == '3d' else 3)]
    if domain == 'causal':
        picked = {c for c,_ in cases}
        gaps = sorted(c for c in eligible-picked if tasks[c]['n_observations'] < 6
                      and baseline[c]['runs']['enhanced']['scores']['upstream']['n_recovered_gts']
                      < min(SLOTS, baseline[c]['truth']))
        cases.extend((c, '部分观测失分') for c in rng.sample(gaps,12))
    return cases


def execute(domain, output):
    output.mkdir(parents=True, exist_ok=True)
    baseline_path = ROOT/domain/'results.json'
    baseline = {r['id']: r for r in read(baseline_path)}
    prior_config = read(ROOT/domain/'config.json')
    if domain == '3d':
        tasks, experiment, scorers, stratum, _, source = spatial(methods=('maximin',))
        height = scorers['corrected'].metadata['max_height']
        shortlist = portfolios(list(product(range(1,height+1), repeat=3)))
    else:
        tasks, _, scorers, stratum, _, source = causal()
        shortlist = []
    assert prior_config['source_sha'] == digest(source)
    cases = cases_for(domain, tasks, stratum, baseline)
    config = {'cases': dict(cases), 'slots': SLOTS, 'seed': SEED, 'model_digest': prior_config['digest'],
              'baseline_sha': digest(baseline_path), 'source_sha': digest(source), 'runner_sha': digest(__file__),
              'dependencies': {name: digest(HERE/name) for name in ('run_domain_followup.py',
                  'run_domain_improvement_pilot.py', 'solve_partial_graph.py', 'run_qwen_guided_edits.py')},
              'spatial_shortlist': shortlist, 'causal_pool_sizes': [8,16]}
    # Round-trip tuples into JSON arrays before comparing a resumed config.
    path = output/'config.json'
    if path.exists():
        config['spatial_shortlist'] = [[list(v) for v in plan] for plan in shortlist]
        if read(path) != config:
            raise ValueError('Output directory has different settings')
    atomic(path, config)
    rows = []
    with requests.Session() as session:
        session.trust_env = False
        if domain == '3d':
            tags = session.get(BASE_URL+'/api/tags', timeout=10)
            tags.raise_for_status()
            assert next(m for m in tags.json()['models'] if m['name'] == MODEL)['digest'] == config['model_digest']
        for index,(case,group) in enumerate(cases,1):
            task = tasks[case]
            old = baseline[case]
            seed = old['seed']
            prior = old['runs']['enhanced']
            folder = output/case
            folder.mkdir(exist_ok=True)
            path = folder/'result.json'
            if path.exists():
                row = read(path)
            else:
                runs = {}
                unknown = []
                if domain == '3d':
                    trials = experiment(task, session, folder, seed)
                    trials['batch'] = spatial_batch(task, session, folder, seed, scorers['corrected'])
                    trials['combined'] = combined_spatial(task, session, folder, seed, scorers['corrected'], shortlist)
                else:
                    benchmark = scorers['upstream']
                    unknown = sorted(set(benchmark.nodes)-{o['perturbed_node'] for o in task['observations']})
                    trials = {}
                    for name, fill in (('fill', fill_causal),
                                       ('diverse', lambda t,p,b: causal_pattern_fill(t,p,b,8)),
                                       ('wide', lambda t,p,b: causal_pattern_fill(t,p,b,16))):
                        start = perf_counter()
                        answers,cost,trace = fill(task, prior, benchmark)
                        trials[name] = (answers,cost | {'fill_seconds': perf_counter()-start},trace)
                    runs['base'] = prior | {'diversity': prediction_metrics(prior['answers'],benchmark,unknown)}
                for name,(answers,cost,trace) in trials.items():
                    scores = {key: score(b,task,answers) if domain == '3d' else score_causal(b,task,answers)
                              for key,b in scorers.items()}
                    diversity = (spatial_metrics(trace) if domain == '3d' else
                                 prediction_metrics(answers,scorers['upstream'],unknown))
                    if domain == 'causal':
                        assert all(not a or a == b for a,b in zip(prior['answers'],answers,strict=True))
                        assert cost['calls'] == prior['cost']['calls']
                    runs[name] = {'answers': answers, 'cost': cost, 'scores': scores, 'trace': trace,
                                  'diversity': diversity}
                row = {'id': case, 'group': group, 'seed': seed, 'truth': old['truth'], 'unknown': unknown, 'runs': runs}
                atomic(path,row)
            rows.append(row)
            print(f'{domain} {index}/{len(cases)} {case}',flush=True)
    atomic(output/'results.json',rows)


def summarize(root):
    lines = ['# HypoSpace 3D 与因果图小样本复测', '',
             '各题 8 槽，两组样本均排除前两轮小样本。3D 按相邻占据格对数每层取 4 题。'
             '因果图按干预数每层取 3 题，并另抽 12 道有空槽且仍可增加合法图的部分观测题做压力对照。', '',
             '3D 组合版用公开柱高空间构造 8 套备选，每套含 8 张不同结构，覆盖全部 7 种总高度。'
             '程序按两两柱高距离筛方案，Qwen 一次选择方案编号。'
             '程序筛选计算一次并跨题复用，调用耗时只统计 Ollama 请求。', '',
             '| 3D 方法 | RR | VR / NR | 合法图 | 调用 | 请求秒数 | 平均总高度种类 | 平均两两距离 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    summary = {}
    for domain in ('3d','causal'):
        rows = read(root/domain/'results.json')
        assert [r['id'] for r in rows] == list(read(root/domain/'config.json')['cases'])
        by_group = {}
        if domain == 'causal':
            lines += ['', '因果图重放原 Qwen 输出，只填空槽。同池版与顺序版均用原来的 8 图构造池，'
                      '扩池版最多建 16 图。多样性排序优先选择新的未观测干预预测模式，再比较可达性距离。'
                      '完全观测题沿用原补位。三种补位均无需额外模型调用。', '',
                      '| 因果组 | 方法 | RR | VR / NR | 合法图 | 新增检查 | 部分观测题平均模式数 | 平均两两距离 |',
                      '|---|---|---:|---:|---:|---:|---:|---:|']
        for group in dict.fromkeys(r['group'] for r in rows):
            cases = [r for r in rows if r['group'] == group]
            by_group[group] = {}
            for method in cases[0]['runs']:
                runs = [r['runs'][method] for r in cases]
                scorer = 'corrected' if domain == '3d' else 'upstream'
                scores = [r['scores'][scorer] for r in runs]
                profiles = [r['runs'][method]['diversity'] for r in cases if domain == '3d' or r['unknown']]
                d = {key: mean(s[key] for s in scores) for key in ('recovery_rate','valid_rate','novelty_rate')}
                d.update({key: sum(r['cost'].get(key,0) for r in runs)
                          for key in ('calls','seconds','extra_checks','added','output_tokens','fill_seconds')})
                d['hits'] = sum(s['n_recovered_gts'] for s in scores)
                d['diversity'] = {key: mean(p[key] for p in profiles) for key in profiles[0]}
                by_group[group][method] = d
                prefix = f'| {LABELS[method]}' if domain == '3d' else f'| {group}（{len(cases)}题） | {LABELS[method]}'
                rates = f" | {d['recovery_rate']:.4%} | {d['valid_rate']:.2%} / {d['novelty_rate']:.2%} | {d['hits']} | "
                cost = f"{d['calls']} | {d['seconds']:.2f}" if domain == '3d' else str(d['extra_checks'])
                spread = d['diversity']['height_totals' if domain == '3d' else 'patterns']
                lines.append(prefix + rates + cost + f" | {spread:.3f} | {d['diversity']['pairwise']:.3f} |")
        summary[domain] = by_group
    lines += ['', '因果图模式指所有未观测干预节点的联合可达性，模式统计只包含部分观测题。'
              '失分样本按已有空槽抽取，结果用于诊断。不同方案在同一题上保留原 Qwen 提交及其位置。', '']
    atomic(root/'summary.json',summary)
    (root/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('domain', choices=('3d','causal','summary'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.domain == 'summary':
        summarize(args.output)
    else:
        execute(args.domain,args.output/args.domain)


if __name__ == '__main__':
    main()
