"""Evaluate Boolean probe ranking, 3D shape-aware selection, and batched causal ranking."""
import argparse
from collections import Counter, deque
from itertools import combinations, permutations, product
from pathlib import Path
import random
from statistics import mean

import requests

from common import HERE, digest, load_upstream
from graph_completion import complete_observation_count
from run_all_domain_evaluation import ROOT, SLOTS, atomic, pad, score, usage
from run_boolean_counterexample_ablation import SOURCE, OUTPUT as BOOLEAN_PILOT
from run_boolean_qwen_contribution import OUTPUT as BOOLEAN_BASE
from run_domain_improvement_pilot import causal, pick, score_causal, spatial
from run_exploratory_uot import make_edits, ranked_groups
from run_feedback_study import graph_line
from run_four_groups import BASE_URL, MODEL
from run_independent_rewire import verified
from run_numbered_agent_search import build_menu
from run_qwen_guided_edits import decode, read, request, schema
from run_spatial_causal_followup import (
    OUTPUT as SPATIAL_PILOT, causal_pattern_fill, combined_spatial, portfolios, prediction_metrics,
)
from validated_evaluators import corrected_boolean_class


OUT = HERE/'runs/three_domain_optimization_20261114'
SEED = 14112028


def boolean_trial(task, catalog, table, folder, seed, session, use_model):
    obs = task['observations']
    expected = tuple(int(o['output']) for o in obs)
    remaining = set(catalog)
    answers, calls, trace = [], [], []
    for slot in range(SLOTS):
        if not remaining:
            break
        counts = Counter(table[i] for i in remaining)
        n = len(remaining)

        def priority(i):
            # A common output pattern favors immediate validity. Balanced splits can
            # reveal more about the remaining catalog when the formula fails.
            votes = [sum(count for pattern, count in counts.items() if pattern[j] == value)
                     for j, value in enumerate(table[i])]
            return -sum(votes), -sum(v*(n-v) for v in votes), i

        menu = sorted(remaining, key=priority)[:8]
        choice = menu[0]
        if use_model:
            prompt = ('Choose one formula matching all observations. These are candidates, and their '
                      'validity has not been checked. Return only its JSON choice ID.\nObservations:\n'
                      + '\n'.join(f'{o["inputs"]} -> {o["output"]}' for o in obs)
                      + '\nCandidates:\n' + '\n'.join(f'{i}. {catalog[i].formula}' for i in sorted(menu)))
            choice, call = pick(session, folder/f'{slot+1:02}.json', seed, prompt, sorted(menu))
            calls.append(call)
        answers.append('Expression: '+catalog[choice].formula if choice is not None else '')
        failures = []
        if choice is not None:
            failures = [j for j, value in enumerate(table[choice]) if value != expected[j]]
            remaining.remove(choice)
            remaining = {i for i in remaining if all(table[i][j] == expected[j] for j in failures)}
        trace.append({'menu': menu, 'choice': choice, 'counterexamples': failures})
    return pad(answers), usage(calls), trace


def boolean_cases(root, session, *, full=False):
    module = load_upstream('boolean', 'boolean_benchmark.py')
    from boolean_dataset import BooleanDiscoveryGame
    original = module.BooleanBenchmarkRefined(str(SOURCE))
    corrected = corrected_boolean_class(module)(str(SOURCE))
    grammar = sorted(BooleanDiscoveryGame.generate_all_expressions(
        corrected.variables, corrected.operators, corrected.max_depth, corrected.mechanistic_opts),
        key=lambda e: e.formula)
    catalog = dict(enumerate(grammar, 1))
    assert len(catalog) == 65
    tasks = {t['observation_set_id']: t for t in original.all_observation_sets}
    baseline = {r['id']: r for r in read(BOOLEAN_BASE/'results.json')}
    seeds = read(BOOLEAN_BASE/'config.json')['seeds']
    excluded = set(read(BOOLEAN_PILOT/'config.json')['cases'])
    rng = random.Random(SEED)
    ids = list(tasks) if full else [c for n in range(1,5) for c in rng.sample(sorted(
        c for c in tasks if c not in excluded and tasks[c]['n_observations'] == n), 8)]
    for index, case in enumerate(ids, 1):
        task = tasks[case]
        folder = root/case
        folder.mkdir(parents=True, exist_ok=True)
        table = {i: tuple(int(e.evaluate(o['inputs'])) for o in task['observations']) for i,e in catalog.items()}
        runs = {'current': baseline[case]['runs']['all'], 'min_id': baseline[case]['runs']['program']}
        for name in ('ranked_program', 'ranked_qwen'):
            answers,cost,trace = boolean_trial(task,catalog,table,folder,seeds[case],session,name=='ranked_qwen')
            runs[name] = {'scores': {k: score(b,task,answers) for k,b in
                                    (('corrected',corrected),('upstream',original))},
                          'answers': answers, 'cost': cost, 'trace': trace}
        print(f'boolean {index}/{len(ids)}', flush=True)
        yield {'id': case, 'stratum': task['n_observations'], 'runs': runs}


def height_patterns(plan, edges):
    return len({tuple(h[a]-h[b] for a,b in edges) for h in plan})


def spatial_cases(root, session, *, full=False):
    from offline_3d import from_heights, to_response
    tasks, _, scorers, stratum, _, _ = spatial()
    benchmark = scorers['corrected']
    prior = {r['id']: r for r in read(ROOT/'3d'/'results.json')}
    excluded = set(read(SPATIAL_PILOT/'3d'/'config.json')['cases'])
    plans = portfolios(list(product(range(1,4),repeat=3)))
    rng = random.Random(SEED)
    ids = list(tasks) if full else [case for n in (0,1,2) for case in rng.sample(sorted(
        c for c,t in tasks.items() if c not in excluded and stratum(t)==n),8)]
    for index,case in enumerate(ids,1):
        task = tasks[case]
        folder = root/case
        folder.mkdir(parents=True, exist_ok=True)
        obs = task['observation']
        seed = prior[case]['seed']
        cells = [divmod(i,3) for i,value in enumerate(obs) if value=='1']
        edges = [(a,b) for a,b in combinations(range(3),2)
                 if sum(abs(x-y) for x,y in zip(cells[a],cells[b]))==1]
        counts = [height_patterns(plan,edges) for plan in plans]
        old, cost, trace = combined_spatial(task,session,folder,seed,benchmark,plans)
        choices = {'current': (trace[0]['plan'], old, cost)}
        choice, call = pick(session,folder/'shape.json',seed,
                            'Choose the plan with the most distinct signed height-difference patterns '
                            'across horizontally adjacent columns. Each plan already covers eight different '
                            'structures and all total heights. Use the supplied pattern counts. Return the ID.\n'
                            + '\n'.join(f'{i}. {plan} distinct patterns: {count}'
                                       for i,(plan,count) in enumerate(zip(plans,counts,strict=True),1)),
                            list(range(1,len(plans)+1)))
        for name, value, cost in (('shape_qwen',choice,usage([call])),
                                  ('shape_program',counts.index(max(counts))+1,usage([]))):
            answers = [to_response(from_heights(obs,h),3) for h in plans[value-1]] if value else ['']*SLOTS
            choices[name] = value, answers, cost
        runs = {name: {'scores': {key: score(b,task,answers) for key,b in scorers.items()},
                       'patterns': counts[value-1] if value else 0, 'plan': value,
                       'answers': answers, 'cost': cost} for name,(value,answers,cost) in choices.items()}
        print(f'3d {index}/{len(ids)}',flush=True)
        yield {'id': case, 'adjacent_pairs': len(edges), 'informative': len(set(counts))>1, 'runs': runs}


def rank_batch(session, folder, slot, seed, menu, observations):
    options = sorted(menu)
    ids = list(range(1,len(options)+1))
    width = min(4,len(options))
    prompt = ('Rank the best candidate DAGs by likelihood of matching all observations. A node '
              'reaches exactly the nodes marked 1 along directed paths. Avoid cycles. Return '
              f'{width} distinct IDs in priority order.\nObservations:\n'
              + '\n'.join(o['string'] for o in observations) + '\nCandidates:\n'
              + '\n'.join(f'{i}. {graph_line(g)}' for i,g in enumerate(options,1)))
    call = request(session,folder/f'batch_{slot:02}.json',prompt,seed,schema({
        'choices': {'type':'array','minItems':width,'maxItems':width,'uniqueItems':True,
                    'items':{'type':'integer','enum':ids}}}))
    chosen = decode(call).get('choices')
    valid = (isinstance(chosen,list) and len(chosen)==width
             and all(type(i) is int and i in ids for i in chosen) and len(set(chosen))==width)
    return [options[i-1] for i in chosen] if valid else menu[:width], call, valid


def causal_batch(task, benchmark, session, folder, seed, model):
    from run_formal_causal_comparison import LIMIT, NODES, forbidden_edges, program_anchor
    obs = task['observations']
    allowed = set(permutations(NODES,2))-forbidden_edges(obs)
    anchor = program_anchor(NODES,LIMIT,obs)
    assert anchor is not None and verified(anchor,benchmark,obs)
    exact = complete_observation_count(NODES,LIMIT,obs)
    emitted, checked = [anchor], {anchor}
    queue, frontier, calls, trace = deque(), {}, [], []
    answers = [graph_line(anchor)]
    for slot in range(2,SLOTS+1):
        if len(emitted)==exact:
            break
        latest = emitted[-1]
        if latest not in frontier:
            frontier[latest] = make_edits(latest,allowed,False)
        new_menu = []
        parsed = None
        if not queue:
            _, entries = ranked_groups(emitted,frontier,checked,slot,False)
            if not entries:
                break
            a,b,_ = build_menu(entries,emitted)
            new_menu = a+b
            if model:
                ranked,call,parsed = rank_batch(session,folder,slot,seed+slot,new_menu,obs)
                calls.append(call)
            else:
                ranked = new_menu[:4]
            queue.extend(ranked)
        probes = []
        answer = ''
        for _ in range(min(2,len(queue))):
            graph = queue.popleft()
            assert graph not in checked
            checked.add(graph)
            ok = verified(graph,benchmark,obs)
            probes.append({'graph':graph,'accepted':ok})
            if ok:
                emitted.append(graph)
                answer = graph_line(graph)
                break
        answers.append(answer)
        trace.append({'slot':slot,'menu':new_menu,'parsed':parsed,'probes':probes})
    cost = usage(calls) | {'checks':len(checked)}
    return {'answers':pad(answers),'cost':cost,'trace':trace}


def causal_cases(root, session, *, full=False):
    from run_final_qwen_path_repair import run as reselect
    from run_formal_causal_comparison import LIMIT, NODES, program_anchor

    tasks, _, scorers, _, _, _ = causal()
    benchmark = scorers['upstream']
    baseline = {r['id']:r for r in read(ROOT/'causal'/'results.json')}
    excluded = set(read(SPATIAL_PILOT/'causal'/'config.json')['cases'])
    rng = random.Random(SEED)
    ids = list(tasks) if full else [case for n in range(1,7) for case in rng.sample(sorted(
        c for c,t in tasks.items() if c not in excluded and t['n_observations']==n),5)]
    for index,case in enumerate(ids,1):
        task = tasks[case]
        old = baseline[case]
        folder = root/case
        folder.mkdir(parents=True, exist_ok=True)
        runs = {'current': old['runs']['enhanced']}
        if full:
            obs = task['observations']
            anchor = program_anchor(NODES, LIMIT, obs)
            assert anchor is not None and verified(anchor, benchmark, obs)
            policy = reselect(obs, anchor, SLOTS, 'reselect', session, folder, old['seed'], benchmark)
            answers = [graph_line(anchor)] + [
                graph_line(step['submitted']) if step['submitted'] is not None else '' for step in policy['trace']]
            answers = pad(answers)
            runs['current'] = {'answers': answers, 'trace': policy['trace'],
                               'scores': {'upstream': score_causal(benchmark, task, answers)},
                               'cost': {'calls': policy['calls'], 'checks': policy['checks'],
                                        'seconds': policy['request_seconds'], 'output_tokens': policy['tokens']}}
        for name in ('batch_program','batch_qwen'):
            run = causal_batch(task,benchmark,session,folder,old['seed'],name=='batch_qwen')
            run['scores'] = {'upstream':score_causal(benchmark,task,run['answers'])}
            runs[name] = run
        unknown = sorted(set(benchmark.nodes)-{o['perturbed_node'] for o in task['observations']})
        for run in runs.values():
            answers,cost,_ = causal_pattern_fill(task,run,benchmark,8)
            assert all(not a or a==b for a,b in zip(run['answers'],answers,strict=True))
            run['filled'] = {'answers':answers,'scores':score_causal(benchmark,task,answers),
                             'cost':cost,'diversity':prediction_metrics(answers,benchmark,unknown)}
        print(f'causal {index}/{len(ids)}',flush=True)
        yield {'id':case,'stratum':task['n_observations'],'truth':old['truth'],'runs':runs}


def summarize(root, *, full=False):
    names = {'current':'当前 Qwen', 'min_id':'程序最小编号', 'ranked_program':'一致性投票程序',
             'ranked_qwen':'一致性排序短菜单＋Qwen', 'shape_qwen':'相邻高度差＋Qwen',
             'shape_program':'相邻高度差程序最优', 'batch_program':'程序批量排序', 'batch_qwen':'Qwen 批量排序'}
    scope = ('覆盖本地全部 674 道题。Boolean 的原策略沿用冻结的 80 题结果，'
             '短菜单重新推理。3D 两种 Qwen 策略和因果图两种 Qwen 策略重新推理。'
             if full else '三组均是原始题库中的探索性样本。题号与已有全量结果有交集。')
    lines = ['# HypoSpace 三题型全量复测' if full else '# HypoSpace 三题型后续消融', '',
             '使用同题同种子、8 个提交槽，沿用 Qwen3:8B 非思考模式。'+scope, '',
             'Boolean 在当前剩余菜单上按候选输出的多数一致性排序，并用分割均衡度打破平局。'
             '程序取首项，Qwen 在同一规则的前八项中选编号。随后沿用全部反例剪枝。'
             '排序读取公式在观测输入上的输出，不使用观测标签，'
             '每题预先计算 65×观测数 个公式输出，剪枝与验收再读取标签。', '',
             '3D 的方法共享八套方案。以正交相邻柱的有向高度差向量种类作为额外指标。'
             '新提示直接提供每套方案的模式数。程序对照取最大值。', '',
             '因果组从同一合法首图出发，排序一次最多给出四个不同候选。'
             '在后续槽依次消耗排序，每槽最多验证两个候选；本槽队列用完时下一槽再补充。'
             '程序按同一菜单的既有顺序取前四项。最终沿用同池预测多样性补位。', '',
             '| 题型 | 方法 | RR | VR / NR | 补位前合法命中 | 调用 | 请求秒数 |',
             '|---|---|---:|---:|---:|---:|---:|']
    summary = {}
    for domain in ('boolean','3d','causal'):
        rows = read(root/domain/'results.json')
        if full:
            expected = {'boolean': 80, '3d': 84, 'causal': 510}[domain]
            assert len(rows) == len({row['id'] for row in rows}) == expected
            assert read(root/domain/'config.json')['full']
        block = {}
        primary = 'upstream' if domain=='causal' else 'corrected'
        for method in rows[0]['runs']:
            runs = [row['runs'][method] for row in rows]
            scores = {}
            for scorer in runs[0]['scores']:
                scores[scorer] = {k:mean(run['scores'][scorer][k] for run in runs)
                                  for k in ('recovery_rate','valid_rate','novelty_rate')}
                scores[scorer]['hits'] = sum(run['scores'][scorer]['n_recovered_gts'] for run in runs)
            diffs = [row['runs'][method]['scores'][primary]['n_recovered_gts'] -
                     row['runs']['current']['scores'][primary]['n_recovered_gts'] for row in rows]
            cost = {key:sum(run['cost'].get(key,0) for run in runs)
                    for key in ('calls','seconds','checks','input_tokens','output_tokens')}
            block[method] = d = {'scores':scores,'cost':cost,
                                 'wtl':[sum(x>0 for x in diffs),sum(x==0 for x in diffs),sum(x<0 for x in diffs)]}
            if domain=='causal':
                d['filled'] = {'rr':mean(run['filled']['scores']['recovery_rate'] for run in runs),
                               'hits':sum(run['filled']['scores']['n_recovered_gts'] for run in runs),
                               'vr':mean(run['filled']['scores']['valid_rate'] for run in runs),
                               'nr':mean(run['filled']['scores']['novelty_rate'] for run in runs),
                               'at_cap':sum(run['filled']['scores']['n_recovered_gts'] == min(SLOTS,row['truth'])
                                            for run, row in zip(runs, rows, strict=True)),
                               'checks':sum(run['filled']['cost']['extra_checks'] for run in runs),
                               'patterns':mean(row['runs'][method]['filled']['diversity']['patterns']
                                               for row in rows if row['stratum']<6)}
            elif domain=='3d':
                d['patterns'] = mean(run['patterns'] for run in runs)
                d['informative_patterns'] = mean(row['runs'][method]['patterns'] for row in rows if row['informative'])
            s = scores[primary]
            lines.append(f"| {domain} ({len(rows)}) | {names[method]} | {s['recovery_rate']:.4%} | "
                         f"{s['valid_rate']:.2%} / {s['novelty_rate']:.2%} | {s['hits']} | "
                         f"{cost['calls']} | {cost['seconds']:.2f} |")
        summary[domain] = block
    lines += ['', '## 额外指标', '']
    for method,d in summary['3d'].items():
        lines.append(f"- 3D {names[method]}平均覆盖 {d['patterns']:.2f} 种相邻高度差模式。"
                     f"仅看方案之间确有差异的题，平均 {d['informative_patterns']:.2f} 种。")
    for method,d in summary['causal'].items():
        f = d['filled']
        lines.append(f"- 因果 {names[method]}完整检查 {d['cost']['checks']} 次，补位再检查 {f['checks']} 次。"
                     f"最终 {f['hits']} 图，RR {f['rr']:.4%}，部分观测平均预测模式 {f['patterns']:.2f}。")
    if full:
        lines += ['', '## 因果图补位后的成绩', '',
                  '| 方法 | RR | VR / NR | 合法图 | 达到上限题数 | 额外检查 |',
                  '|---|---:|---:|---:|---:|---:|']
        for method,d in summary['causal'].items():
            f = d['filled']
            lines.append(f"| {names[method]} | {f['rr']:.4%} | {f['vr']:.2%} / {f['nr']:.2%} | "
                         f"{f['hits']} | {f['at_cap']}/510 | {f['checks']} |")
    lines += ['', '主表使用 Boolean 与 3D 修正版、因果图原版。原版重评分、胜平负和请求成本见 summary.json。', '']
    atomic(root/'summary.json',summary)
    (root/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('domain',choices=('boolean','3d','causal','summary'))
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--full', action='store_true', help='Evaluate all local cases')
    args = parser.parse_args()
    if args.domain=='summary':
        summarize(args.output, full=args.full)
        return
    root = args.output/args.domain
    root.mkdir(parents=True,exist_ok=True)
    source = ROOT/args.domain/'config.json'
    manifest = {'model':MODEL,'digest':read(source)['digest'],'runner_sha':digest(__file__),
                'baseline_sha':digest(source),'seed':SEED,'slots':SLOTS}
    if args.full:
        manifest.update(full=True, reference_results_sha=digest(ROOT/args.domain/'results.json'),
                        boolean_baseline_sha=digest(BOOLEAN_BASE/'results.json'),
                        dependencies={name: digest(HERE/name) for name in (
                            'run_qwen_guided_edits.py', 'run_final_qwen_path_repair.py',
                            'run_spatial_causal_followup.py', 'validated_evaluators.py')})
    path = root/'config.json'
    if path.exists() and read(path)!=manifest:
        raise ValueError('Output directory has a different configuration')
    atomic(path,manifest)
    with requests.Session() as session:
        session.trust_env = False
        tags = session.get(BASE_URL+'/api/tags',timeout=10)
        tags.raise_for_status()
        assert next(m['digest'] for m in tags.json()['models'] if m['name']==MODEL)==manifest['digest']
        rows = []
        for row in {'boolean':boolean_cases,'3d':spatial_cases,'causal':causal_cases}[args.domain](
                root,session,full=args.full):
            atomic(root/row['id']/'result.json',row)
            rows.append(row)
    expected = {'boolean':80,'3d':84,'causal':510} if args.full else {'boolean':32,'3d':24,'causal':30}
    assert len(rows)==len({r['id'] for r in rows})==expected[args.domain]
    atomic(root/'results.json',rows)


if __name__=='__main__':
    main()
