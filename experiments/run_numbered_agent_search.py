"""Compare program ranking and Qwen selection over unverified single-edge edits."""
import argparse
from itertools import permutations
from pathlib import Path
import random
import re
from statistics import mean
from time import perf_counter

import requests

from common import HERE, digest
from graph_completion import complete_observation_count
from run_exploratory_uot import make_edits, ranked_groups
from run_formal_agent_debate import REFERENCE, SECOND_AGENT_SEED_OFFSET
from run_formal_causal_comparison import DATA, LIMIT, MENU, NODES, SLOTS, deletion_safe, forbidden_edges, program_anchor, read, risk, valid
from run_four_groups import BASE_URL, MODEL
from run_local_causal import write_json
from run_qwen_tight_budget import qwen_choice


METHODS = {'program': '程序排序', 'single': '单 Qwen 选编号', 'dual': '双 Qwen 分工选编号'}
FREE_FORM = HERE/'runs/formal_parallel_agent_search_20261113'
SELECTION_SEED = 11132026


def build_menu(ordered, emitted):
    """Prefer additions in A and deletions from other parents in B, without overlap."""
    latest = emitted[-1]
    adds, removes = [], []
    for _, graph, edit in ordered:
        (adds if edit['action'] == 'add' else removes).append(graph)
    edits = {graph: edit for _, graph, edit in ordered}
    a_order = sorted(adds, key=lambda g: edits[g]['parent'] != latest)
    b_order = sorted(removes, key=lambda g: edits[g]['parent'] == latest)
    first = ordered[0][1]
    a = list(dict.fromkeys([first] + a_order + b_order))[:MENU//2]
    b = [g for g in b_order + a_order if g not in a][:MENU//2]
    return a, b, edits


def run(task, anchor, budget, method, session, folder, seed, *, rescue=None):
    start = perf_counter()
    observations = task['observations']
    allowed = set(permutations(NODES, 2)) - forbidden_edges(observations)
    exact = complete_observation_count(NODES, LIMIT, observations)
    emitted, checked = [anchor], {anchor}
    frontier, closures, trace = {}, {}, []
    for slot in range(2, SLOTS+1):
        if len(emitted) == exact:
            break
        latest = emitted[-1]
        if latest not in frontier:
            frontier[latest] = make_edits(latest, allowed, False)
        _, entries = ranked_groups(emitted, frontier, checked, slot, False)
        if not entries:
            break
        order = list(dict.fromkeys(graph for _, graph, _ in entries))
        a, b, edits = build_menu(entries, emitted)
        model_calls, steps = [], []
        queue = [('program' if method == 'program' else 'fallback', graph) for graph in order]

        def choose(role, menu, offset=0):
            if len(menu) < 2:
                return None
            selected, call, parsed = qwen_choice(session, folder/f'{method}_{slot:02}_{role}.json',
                                                  seed+offset, observations, emitted, menu, edits)
            model_calls.append({'role': role, 'choice': selected, 'parsed': parsed,
                                'seconds': call['seconds'], 'tokens': call['response'].get('eval_count', 0),
                                'input_tokens': call['response'].get('prompt_eval_count', 0)})
            return selected if parsed else None

        def probe(graph, origin):
            if graph in checked:
                return False
            checked.add(graph)
            edit = edits[graph]
            traversals = 0
            if edit['action'] == 'add':
                rejection = risk(edit, observations, closures)
            else:
                safe, traversals = deletion_safe(edit, graph, observations)
                rejection = None if safe else 'missing_required_path'
            safe = rejection is None
            accepted = safe and valid(graph, observations)
            assert accepted == safe
            steps.append({'graph': graph, 'origin': origin, 'accepted': accepted, 'full_check': safe,
                          'deletion_traversals': traversals, 'rejection': rejection})
            if accepted:
                emitted.append(graph)
            return accepted

        accepted = False
        if method != 'program':
            menu = a+b if method == 'single' else a
            chosen = choose('single' if method == 'single' else 'A', menu)
            if chosen is not None:
                accepted = probe(chosen, 'model')
        if rescue is not None and steps and not accepted and len(steps) < budget:
            chosen, call = rescue(slot, order, edits, checked, emitted, steps[-1])
            if call is not None:
                model_calls.append(call)
            if chosen is not None:
                accepted = probe(chosen, 'model' if call is not None and call['parsed'] else 'fallback')
        elif method == 'dual' and not accepted and len(steps) < budget and budget > 1:
            remaining = [graph for graph in b if graph not in checked]
            chosen = choose('B', remaining, SECOND_AGENT_SEED_OFFSET)
            if chosen is not None:
                accepted = probe(chosen, 'model')
        if not accepted:
            for origin, graph in queue:
                if len(steps) >= budget or probe(graph, origin):
                    break
        trace.append({'slot': slot, 'menu_a': a, 'menu_b': b, 'program_first': order[0],
                      'calls': model_calls, 'steps': steps})
    elapsed = perf_counter()-start
    steps = [step for row in trace for step in row['steps']]
    calls = [call for row in trace for call in row['calls']]
    assert len(steps) == len(checked)-1
    assert all(len(row['steps']) <= budget for row in trace)
    assert len(set(emitted)) == len(emitted) <= min(SLOTS, task['n_ground_truths'])
    assert all(valid(graph, observations) for graph in emitted)
    counts = {origin: sum(step['accepted'] and step['origin'] == origin for step in steps)
              for origin in ('program', 'model', 'fallback')}
    return {'found': len(emitted), 'rr': len(emitted)/task['n_ground_truths'],
            'vr': len(emitted)/SLOTS, 'nr': len(emitted)/SLOTS, 'graphs': emitted, 'accepted': counts,
            'probes': len(steps), 'full_checks': sum(s['full_check'] for s in steps), 'calls': len(calls),
            'b_calls': sum(c['role'] == 'B' for c in calls), 'valid_choices': sum(c['parsed'] for c in calls),
            'model_probes': sum(s['origin'] == 'model' for s in steps),
            'tokens': sum(c['tokens'] for c in calls), 'input_tokens': sum(c['input_tokens'] for c in calls),
            'request_seconds': sum(c['seconds'] for c in calls), 'seconds': elapsed, 'trace': trace}


def select_cases(tasks, source, cohort):
    if cohort == 'formal':
        return source['cases']
    seen = set()
    for path in (HERE/'runs').glob('*/config.json'):
        if not path.parent.name.startswith('numbered_agent_search_') and read(path).get('all_tasks') is not True:
            seen.update(re.findall(r'\bn[1-6]_\d{3}\b', path.read_text(encoding='utf-8')))
    rng = random.Random(SELECTION_SEED)
    return [task['id'] for n in (2, 3, 4, 5) for task in rng.sample(sorted(
        (t for t in tasks.values() if t['n_observations'] == n and t['id'] not in seen
         and t['n_ground_truths'] >= SLOTS), key=lambda t: t['id']), 4)]


def report(output, records, cohort, budget):
    summary = {}
    lines = ['# HypoSpace 候选编号对照', '',
             f'{len(records)} 道六节点题，每题 8 槽，每个后续槽最多探查 {budget} 个未验证候选。'
             '三组共享程序首图、剪枝、单边邻域与验收规则。一次探查包含风险判断，通过后做完整验证。'
             '编号菜单仅用静态观测剪枝，未预先逐图验证。候选菜单会随各组历史变化。', '',
             '单 Qwen 从两张不重叠的菜单并集选编号，双 Qwen 的 A 优先看增边，B 优先看删边和其他父图。'
             'A 的候选未通过且预算允许时才调用 B，剩余预算由程序排序补位。单探查时双 Qwen 只调用 A。'
             '无效编号交给程序处理。各组找到新合法图后立即提交，每槽最多提交一图。', '',
             '| 方法 | RR | VR / NR | 不同合法图 | 探查 | 完整验证 | 模型验收 / 模型探查 | 程序补位验收 | Qwen 调用（B） | 输出 token | 运行秒数 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for method, label in METHODS.items():
        rows = [record['runs'][method] for record in records]
        d = {key: mean(row[key] for row in rows) for key in ('rr','vr','nr')}
        d.update({key: sum(row[key] for row in rows) for key in ('found','probes','full_checks', 'calls',
                  'b_calls','valid_choices','model_probes','tokens','input_tokens','request_seconds','seconds')})
        d['accepted'] = {origin: sum(row['accepted'][origin] for row in rows)
                         for origin in ('program','model','fallback')}
        summary[method] = d
        lines.append(f"| {label} | {d['rr']:.4%} | {d['vr']:.2%} | {d['found']} | {d['probes']} | "
                     f"{d['full_checks']} | {d['accepted']['model']}/{d['model_probes']} | {d['accepted']['fallback']} | "
                     f"{d['calls']} ({d['b_calls']}) | {d['tokens']} | {d['seconds']:.2f} |")
    comparisons = {}
    for lhs,rhs in (('dual','single'),('dual','program'),('single','program')):
        deltas = [row['runs'][lhs]['found']-row['runs'][rhs]['found'] for row in records]
        wtl = [sum(d>0 for d in deltas),sum(d==0 for d in deltas),sum(d<0 for d in deltas)]
        comparisons[f'{lhs}_vs_{rhs}'] = wtl
        lines += ['', f"{METHODS[lhs]}对比{METHODS[rhs]}，逐题合法图数胜／平／负为 {' / '.join(map(str,wtl))}。"]
    lines += ['', '| 题目 | 合法解数 | 程序 | 单 Qwen | 双 Qwen |', '|---|---:|---:|---:|---:|']
    for row in records:
        values = ' | '.join(str(row['runs'][method]['found']) for method in METHODS)
        lines.append(f"| {row['id']} | {row['truth']:,} | {values} |")
    if cohort == 'formal':
        previous = read(FREE_FORM/'summary.json')['methods']['dual']
        lines += ['', f"同批题历史自由写图双 Agent 找到 {previous['found']} 张合法图，RR 为 {previous['rr']:.4%}，"
                      f"VR 与 NR 均为 {previous['vr']:.2%}，共 {previous['calls']} 次调用。"
                      '改动同时涉及编号表示、候选排序与程序补位。']
    else:
        lines += ['', '紧预算样本从此前模型运行配置未出现的 2–5 次干预题分层抽取，每层 4 题。'
                      '程序全量诊断此前覆盖过这批题。']
    lines += ['', 'RR 按题宏平均。只提交不同合法图，所以 VR 与 NR 相同。耗时为本轮执行时间，缓存重跑时可参考 JSON 的请求秒数。', '']
    write_json(output/'summary.json', {'methods': summary, 'paired': comparisons})
    (output/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cohort', choices=('formal','tight'), required=True)
    parser.add_argument('--budget', type=int, choices=(1, 2, 3))
    args = parser.parse_args()
    budget = args.budget or (3 if args.cohort == 'formal' else 1)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    source = read(REFERENCE/'config.json')
    assert source['source_sha'] == digest(DATA)
    tasks = {task['id']: task for task in read(DATA)}
    path = output/'config.json'
    cases = read(path)['cases'] if path.exists() else select_cases(tasks,source,args.cohort)
    with requests.Session() as session:
        session.trust_env = False
        response = session.get(BASE_URL+'/api/tags', timeout=10)
        response.raise_for_status()
        assert next(row for row in response.json()['models'] if row['name'] == MODEL)['digest'] == source['model']
        config = {'cohort': args.cohort, 'cases': cases, 'selection_seed': SELECTION_SEED, 'budget': budget,
                  'model': MODEL, 'digest': source['model'], 'slots': SLOTS, 'menu_size': MENU,
                  'source_sha': digest(REFERENCE/'config.json'),
                  'code_sha': {name: digest(HERE/name) for name in (Path(__file__).name, 'run_exploratory_uot.py',
                      'run_qwen_tight_budget.py', 'run_qwen_guided_edits.py', 'run_formal_causal_comparison.py')}}
        if path.exists() and read(path) != config:
            raise ValueError('Output directory contains a different experiment configuration')
        write_json(path, config)
        records = []
        for index, case in enumerate(cases):
            task = tasks[case]
            anchor = program_anchor(NODES, LIMIT, task['observations'])
            assert anchor is not None and valid(anchor, task['observations'])
            seed = source['seeds'][index % len(source['seeds'])]
            folder = output/case
            folder.mkdir(exist_ok=True)
            runs = {}
            methods = tuple(METHODS) if index%2 == 0 else tuple(reversed(METHODS))
            for method in methods:
                result_path = folder/f'{method}_result.json'
                if result_path.exists():
                    runs[method] = read(result_path)
                else:
                    runs[method] = run(task, anchor, budget, method, session, folder, seed)
                    tmp = result_path.with_suffix('.tmp')
                    write_json(tmp, runs[method])
                    tmp.replace(result_path)
            record = {'id':case,'truth':task['n_ground_truths'],'seed':seed,'anchor':anchor,'runs':runs}
            write_json(folder/'result.json', record)
            records.append(record)
            print(f'{index+1}/{len(cases)} {case} ' + str({m:runs[m]['found'] for m in METHODS}),flush=True)
        report(output, records, args.cohort, budget)


if __name__ == '__main__':
    main()
