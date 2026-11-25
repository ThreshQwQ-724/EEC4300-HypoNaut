"""Held-out factorial check of addition-first ordering and Qwen."""
import argparse
from itertools import permutations
from pathlib import Path
import random
import re
from statistics import mean
from time import perf_counter

import networkx as nx
import requests

from common import HERE, digest
from run_feedback_study import observations_check
from run_four_groups import BASE_URL, MODEL, SEEDS
from run_local_causal import write_json
from run_partial_graph_pipeline import forbidden_edges, program_anchor
from run_qwen_rr_budget import MODEL_REFERENCE, read
from run_qwen_tight_budget import BUDGETS, LIMIT, MENU, NODES, SLOTS, neighbors, priority, qwen_choice
from stream_causal import check_graph
from validate_graph_rr_budget import DATA

SELECTION_SEED = 1015126
METHODS = ('program', 'program_add', 'qwen', 'qwen_add')
LABELS = dict(zip(METHODS, ('程序原排序', '程序增边优先', 'Qwen 原菜单', 'Qwen 增边优先')))


def select():
    used = set()
    for path in (HERE/'runs').glob('*/config.json'):
        if not path.parent.name.startswith('qwen_addition_validation'):
            used.update(re.findall(r'\bn[1-6]_\d{3}\b', path.read_text(encoding='utf-8')))
    rng = random.Random(SELECTION_SEED)
    tasks = read(DATA)
    return [r for n in range(1, 7) for r in rng.sample(sorted(
        (task for task in tasks if task['n_observations'] == n and task['id'] not in used
         and task['n_ground_truths'] >= SLOTS), key=lambda task: task['id']), 2)]


def run(task, anchor, budget, method, session, folder, seed):
    obs = task['observations']
    blocked = forbidden_edges(obs)
    allowed = tuple(edge for edge in permutations(NODES, 2) if edge not in blocked)
    emitted, checked, pool = [anchor], {anchor}, set()
    trace = []
    started = perf_counter()
    for slot in range(2, SLOTS+1):
        edits = neighbors(emitted, allowed, checked)
        ordered = sorted(edits, key=lambda g: (
            method.endswith('_add') and edits[g]['action'] != 'add', priority(g, emitted)))
        menu = ordered[:MENU]
        if method == 'qwen_add' and any(edits[g]['action'] == 'add' for g in menu):
            menu = [g for g in menu if edits[g]['action'] == 'add']
        chosen = ordered[:budget]
        call, valid_choice = None, None
        if method.startswith('qwen') and len(menu) > 1:
            first, call, valid_choice = qwen_choice(
                session, folder/f'{method}_{budget}_{slot:02}.json', seed, obs, emitted, menu, edits)
            chosen = ([first] + [g for g in ordered if g != first])[:budget]
        checks = []
        for candidate in chosen:
            graph = nx.DiGraph(candidate)
            valid = nx.is_directed_acyclic_graph(graph) and not observations_check(NODES, obs, candidate)
            checked.add(candidate)
            if valid:
                pool.add(candidate)
            checks.append({'graph': candidate, 'valid': valid, **edits[candidate]})
        submitted = min(pool, key=lambda g: priority(g, emitted)) if pool else None
        if submitted is not None:
            pool.remove(submitted)
            emitted.append(submitted)
        trace.append({'slot': slot, 'checked': checks, 'submitted': submitted, 'model_call': call is not None,
                      'valid_choice': valid_choice, 'tokens': call['response'].get('eval_count', 0) if call else 0})
    assert len(emitted) == len(set(emitted)) <= min(SLOTS, task['n_ground_truths'])
    assert all(check_graph({'nodes': NODES, 'edges': g}, obs, {'nodes': NODES, 'max_edges': LIMIT})
               for g in emitted)
    checks = [c for t in trace for c in t['checked']]
    assert len(checks) == (SLOTS-1)*budget
    return {'found': len(emitted), 'rr': len(emitted)/task['n_ground_truths'], 'vr': len(emitted)/SLOTS,
            'checks': len(checks), 'valid_checks': sum(c['valid'] for c in checks),
            'add_checks': sum(c['action'] == 'add' for c in checks),
            'model_calls': sum(t['model_call'] for t in trace),
            'valid_choices': sum(t['valid_choice'] is True for t in trace),
            'tokens': sum(t['tokens'] for t in trace), 'seconds': perf_counter()-started, 'trace': trace}


def report(output, records):
    summary = {}
    lines = ['# 增边优先对 Qwen 的独立复核', '',
             '固定抽样种子 1015126。按干预次数 1–6 各抽两道六节点题，排除已用于推理实验的题目，要求每题至少有 8 个合法图。每题配一个采样种子，四组共享合法首图、观测剪枝、全轮去重和提交排序。首图占一个槽，其余七槽分别最多验证 1 或 2 个候选。', '',
             '程序增边组先排增边候选，再按原多样性排序。Qwen 增边组用同一顺序构建菜单，有增边候选时只允许 Qwen 从增边中选首查项。第二项按对应程序排序补足。Qwen 原菜单组从原多样性排序菜单选首查项。无效编号用该菜单的程序首选。所有组独立验证输出。', '',
             '| 每槽检查 | 方法 | 平均 RR | 合法图总数 | 满 8 图 | 合法检查 / 检查 | 增边检查 | 模型调用 | 输出 token | 秒数 |',
             '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for budget in BUDGETS:
        block = {}
        for method in METHODS:
            rows = [r['runs'][str(budget)][method] for r in records]
            d = {k: sum(row[k] for row in rows) for k in (
                 'found','checks','valid_checks','add_checks','model_calls','valid_choices','tokens','seconds')}
            d.update(rr=mean(row['rr'] for row in rows), vr=mean(row['vr'] for row in rows),
                     full=sum(row['found'] == SLOTS for row in rows))
            block[method] = d
            lines.append(f"| {budget} | {LABELS[method]} | {d['rr']:.4%} | {d['found']} | {d['full']}/12 | "
                         f"{d['valid_checks']}/{d['checks']} | {d['add_checks']} | {d['model_calls']} | "
                         f"{d['tokens']} | {d['seconds']:.2f} |")
        for left, right in (('qwen_add','qwen'), ('program_add','program'), ('qwen_add','program_add')):
            deltas = [r['runs'][str(budget)][left]['found']-r['runs'][str(budget)][right]['found'] for r in records]
            block[f'{left}_vs_{right}'] = {'win_tie_loss': [sum(d>0 for d in deltas),
                                                           sum(d==0 for d in deltas), sum(d<0 for d in deltas)],
                                           'rr_delta': block[left]['rr']-block[right]['rr']}
        summary[str(budget)] = block
    lines += ['', '| 题目 | 合法图总数 | 上限 1（程序 / 程序增边 / Qwen / Qwen 增边） | 上限 2（同顺序） |',
              '|---|---:|---|---|']
    for r in records:
        cells = [' / '.join(str(r['runs'][str(b)][m]['found']) for m in METHODS) for b in BUDGETS]
        lines.append(f"| {r['id']} | {r['truth']} | {cells[0]} | {cells[1]} |")
    lines += ['', '结果来自 12 题、每题一个种子的配对实验。检查次数只计首图之后的图验证。RR 为逐题宏平均，合法图数用于事后评分。逐槽轨迹、模型原始请求、实际参数和配对差值已存档。']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    write_json(output/'summary.json', summary)
    print(summary, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE/'runs/qwen_addition_validation_20261113')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    tasks = select()
    assert len(tasks) == len({r['id'] for r in tasks}) == 12
    if args.verify_only:
        print([(r['id'], r['n_ground_truths']) for r in tasks])
        for r in tasks:
            anchor = program_anchor(NODES, LIMIT, r['observations'])
            assert anchor is not None
            for budget in BUDGETS:
                for method in METHODS[:2]:
                    run(r, anchor, budget, method, None, None, 0)
        return
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.trust_env = False
    response = session.get(BASE_URL+'/api/tags', timeout=10)
    response.raise_for_status()
    model = next(row for row in response.json()['models'] if row['name'] == MODEL)
    assert model['digest'] == read(MODEL_REFERENCE)['model']['digest']
    config = {'model': model['digest'], 'cases': [r['id'] for r in tasks], 'seeds': SEEDS,
              'budgets': list(BUDGETS), 'slots': SLOTS, 'menu': MENU, 'selection_seed': SELECTION_SEED,
              'data_sha': digest(DATA), 'source_hashes': {name: digest(HERE/name) for name in (
                  'validate_qwen_addition_first.py','run_qwen_tight_budget.py',
                  'run_qwen_guided_edits.py','run_partial_graph_pipeline.py')}}
    path = output/'config.json'
    if path.exists():
        assert read(path) == config
    else:
        write_json(path, config)
    print('OUTPUT_DIR='+str(output.resolve()), flush=True)
    records = []
    for i, task in enumerate(tasks):
        folder = output/task['id']
        folder.mkdir(exist_ok=True)
        seed = SEEDS[i % len(SEEDS)]
        anchor = program_anchor(NODES, LIMIT, task['observations'])
        assert anchor is not None
        runs = {str(b): {m: run(task, anchor, b, m, session, folder, seed) for m in METHODS} for b in BUDGETS}
        r = {'id': task['id'], 'truth': task['n_ground_truths'], 'seed': seed, 'anchor': anchor, 'runs': runs}
        write_json(folder/'result.json', r)
        records.append(r)
        print(f"{i+1}/12 {task['id']} "+str([[runs[str(b)][m]['found'] for m in METHODS] for b in BUDGETS]), flush=True)
    report(output, records)


if __name__ == '__main__':
    main()
