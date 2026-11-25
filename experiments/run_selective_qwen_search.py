"""Compare always-on and failure-triggered Qwen under one check per slot."""
import argparse
from itertools import permutations
from pathlib import Path
import random
import re
from statistics import mean
from time import perf_counter

import requests

from common import HERE, digest
from run_four_groups import BASE_URL, MODEL, SEEDS
from run_local_causal import write_json
from run_partial_graph_pipeline import forbidden_edges, program_anchor
from run_qwen_rr_budget import MODEL_REFERENCE, read
from run_qwen_tight_budget import LIMIT, MENU, NODES, SLOTS, neighbors, priority, qwen_choice
from stream_causal import check_graph
from validate_graph_rr_budget import DATA

SELECTION_SEED = 1018126
METHODS = ('program_add', 'qwen_add', 'selective_qwen')
LABELS = dict(zip(METHODS, ('程序增边优先', '每槽 Qwen', '失败后调用 Qwen')))


def select():
    used = set()
    for path in (HERE/'runs').glob('*/config.json'):
        if not path.parent.name.startswith('selective_qwen_search'):
            used.update(re.findall(r'\bn[1-6]_\d{3}\b', path.read_text(encoding='utf-8')))
    rng = random.Random(SELECTION_SEED)
    tasks = read(DATA)
    return [r for n in range(1, 7) for r in rng.sample(sorted(
        (task for task in tasks if task['n_observations'] == n and task['id'] not in used
         and task['n_ground_truths'] >= SLOTS), key=lambda task: task['id']), 4)]


def run(task, anchor, method, session, folder, seed):
    obs = task['observations']
    blocked = forbidden_edges(obs)
    allowed = tuple(edge for edge in permutations(NODES, 2) if edge not in blocked)
    emitted, checked, trace = [anchor], {anchor}, []
    previous_failed = False
    started = perf_counter()
    for slot in range(2, SLOTS+1):
        edits = neighbors(emitted, allowed, checked)
        ordered = sorted(edits, key=lambda graph: (edits[graph]['action'] != 'add', priority(graph, emitted)))
        menu = ordered[:MENU]
        if menu and edits[menu[0]]['action'] == 'add':
            menu = [graph for graph in menu if edits[graph]['action'] == 'add']
        if not menu:
            trace.append({'slot': slot, 'checked': None, 'valid': False, 'model_call': False, 'tokens': 0})
            previous_failed = True
            continue
        ask = len(menu) > 1 and (method == 'qwen_add' or (method == 'selective_qwen' and previous_failed))
        candidate, call, valid_choice = menu[0], None, None
        if ask:
            candidate, call, valid_choice = qwen_choice(
                session, folder/f'{method}_{slot:02}.json', seed, obs, emitted, menu, edits)
        checked.add(candidate)
        valid = check_graph({'nodes': NODES, 'edges': candidate}, obs, {'nodes': NODES, 'max_edges': LIMIT})
        if valid:
            emitted.append(candidate)
        trace.append({'slot': slot, 'checked': candidate, 'valid': valid, 'edit': edits[candidate],
                      'model_call': ask, 'valid_choice': valid_choice, 'previous_failed': previous_failed,
                      'tokens': call['response'].get('eval_count', 0) if call else 0})
        previous_failed = not valid
    assert len(emitted) == len(set(emitted)) <= min(SLOTS, task['n_ground_truths'])
    assert len(emitted) == 1+sum(row['valid'] for row in trace)
    return {'found': len(emitted), 'rr': len(emitted)/task['n_ground_truths'],
            'checks': sum(row['checked'] is not None for row in trace),
            'model_calls': sum(row['model_call'] for row in trace),
            'tokens': sum(row['tokens'] for row in trace), 'seconds': perf_counter()-started, 'trace': trace}


def report(output, rows):
    summary = {}
    lines = ['# 一槽一次检查下的选择性 Qwen 对照', '',
             '固定抽样种子 1018126。按 1–6 次干预各抽四道至少有八个合法图的六节点题，排除已有模型运行配置记录的题目。每题一个采样种子，共 24 题。三组共享已验证的程序首图、增边优先排序、观测剪枝和去重。首图占一槽，之后七槽每槽最多检查一个候选，失败占槽。', '',
             '每槽组在菜单至少有两个候选时调用 Qwen。选择性组仅在上一槽未通过验证且当前菜单至少有两个候选时调用 Qwen，成功后恢复程序排序。该触发规则只使用已经发生的验证结果。各组因历史不同而可能看到不同候选。', '',
             '| 方法 | RR | 不同合法图 | 满 8 图 | 图检查 | Qwen 调用 | 输出 token | 累计秒数 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for method in METHODS:
        runs = [r['runs'][method] for r in rows]
        d = {k: sum(run[k] for run in runs) for k in ('found','checks','model_calls','tokens','seconds')}
        d.update(rr=mean(run['rr'] for run in runs), full=sum(run['found'] == SLOTS for run in runs))
        summary[method] = d
        lines.append(f"| {LABELS[method]} | {d['rr']:.4%} | {d['found']} | {d['full']}/24 | "
                     f"{d['checks']} | {d['model_calls']} | {d['tokens']} | {d['seconds']:.2f} |")
    for left, right in [('qwen_add','program_add'), ('selective_qwen','program_add'), ('selective_qwen','qwen_add')]:
        deltas = [r['runs'][left]['found']-r['runs'][right]['found'] for r in rows]
        summary[f'{left}_vs_{right}'] = {'win_tie_loss': [sum(d > 0 for d in deltas),
            sum(d == 0 for d in deltas), sum(d < 0 for d in deltas)],
            'rr_delta': summary[left]['rr']-summary[right]['rr']}
    lines += ['', '| 题目 | 合法图总数 | 程序 | 每槽 Qwen | 选择性 Qwen |',
              '|---|---:|---:|---:|---:|']
    for row in rows:
        counts = ' | '.join(str(row['runs'][m]['found']) for m in METHODS)
        lines.append(f"| {row['item']} | {row['truth']} | {counts} |")
    lines += ['', 'RR 先逐题计算再平均。图检查不含共同首图的一次校验。所有提交均通过无环、边数、观测与去重检查。结果限于该数据集、一槽一次检查和每题一个种子。']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    write_json(output/'summary.json', summary)
    print(summary, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE/'runs/selective_qwen_search_20261113')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    tasks = select()
    assert len(tasks) == len({r['id'] for r in tasks}) == 24
    if args.verify_only:
        for task in tasks:
            anchor = program_anchor(NODES, LIMIT, task['observations'])
            assert anchor is not None
            result = run(task, anchor, 'program_add', None, None, 0)
            print(task['id'], result['found'], task['n_ground_truths'])
        return
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.trust_env = False
    response = session.get(BASE_URL+'/api/tags', timeout=10)
    response.raise_for_status()
    model = next(row for row in response.json()['models'] if row['name'] == MODEL)
    assert model['digest'] == read(MODEL_REFERENCE)['model']['digest']
    config = {'model': model['digest'], 'cases': [r['id'] for r in tasks], 'seeds': SEEDS, 'slots': SLOTS,
              'checks_per_slot': 1, 'menu': MENU, 'selection_seed': SELECTION_SEED,
              'data_sha': digest(DATA), 'code_hashes': {name: digest(HERE/name) for name in (
                  'run_selective_qwen_search.py','run_qwen_tight_budget.py',
                  'run_qwen_guided_edits.py','run_partial_graph_pipeline.py')}}
    path = output/'config.json'
    if path.exists():
        assert read(path) == config
    else:
        write_json(path, config)
    print('OUTPUT_DIR='+str(output.resolve()), flush=True)
    rows = []
    for index, task in enumerate(tasks):
        folder = output/task['id']
        folder.mkdir(exist_ok=True)
        anchor = program_anchor(NODES, LIMIT, task['observations'])
        assert anchor is not None
        seed = SEEDS[index % len(SEEDS)]
        runs = {m: run(task, anchor, m, session, folder, seed) for m in METHODS}
        record = {'item': task['id'], 'truth': task['n_ground_truths'], 'seed': seed, 'anchor': anchor, 'runs': runs}
        write_json(folder/'result.json', record)
        rows.append(record)
        print(f"{index+1}/24 {task['id']} "+str([(m,runs[m]['found'],runs[m]['model_calls']) for m in METHODS]), flush=True)
    report(output, rows)


if __name__ == '__main__':
    main()
