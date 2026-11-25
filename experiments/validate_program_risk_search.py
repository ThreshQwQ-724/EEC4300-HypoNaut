"""Frozen paired test of addition-first risk avoidance with program/Qwen controls."""
import argparse
from collections import Counter
from itertools import permutations
from pathlib import Path
import random
import re
from statistics import mean
from time import perf_counter

import requests

from common import HERE, digest
from evaluate_qwen_risk_trigger import risk
from run_four_groups import BASE_URL, MODEL, SEEDS
from run_local_causal import write_json
from run_partial_graph_pipeline import forbidden_edges, program_anchor
from run_qwen_rr_budget import MODEL_REFERENCE, read
from run_qwen_tight_budget import LIMIT, MENU, NODES, SLOTS, neighbors, priority, qwen_choice
from stream_causal import check_graph
from validate_graph_rr_budget import DATA

SELECTION_SEED = 1019126
PER_STRATUM = 6
METHODS = ('program_add', 'program_risk', 'qwen_add')
NAMES = dict(zip(METHODS, ('程序增边优先', '程序增边＋避险', '每槽 Qwen 选边')))


def select_tasks():
    used = set()
    for path in (HERE/'runs').glob('*/config.json'):
        if not path.parent.name.startswith('program_risk_validation'):
            used.update(re.findall(r'\bn[1-6]_\d{3}\b', path.read_text(encoding='utf-8')))
    rng = random.Random(SELECTION_SEED)
    tasks = read(DATA)
    picked = []
    for count in range(1, 7):
        pool = sorted((r for r in tasks if r['n_observations'] == count and r['id'] not in used
                       and r['n_ground_truths'] >= SLOTS), key=lambda r: r['id'])
        picked.extend(rng.sample(pool, PER_STRATUM))
    return picked


def evaluate(task, anchor, method, session, folder, seed):
    observations = task['observations']
    excluded = forbidden_edges(observations)
    allowed = tuple(e for e in permutations(NODES, 2) if e not in excluded)
    emitted, checked, closures, trace = [anchor], {anchor}, {}, []
    start = perf_counter()
    for slot in range(2, SLOTS+1):
        edits = neighbors(emitted, allowed, checked)
        menu = sorted(edits, key=lambda g: (edits[g]['action'] != 'add', priority(g, emitted)))[:MENU]
        if menu and edits[menu[0]]['action'] == 'add':
            menu = [g for g in menu if edits[g]['action'] == 'add']
        first = menu[0] if menu else None
        chosen, model_call, valid_choice = first, None, None
        probes = []
        closures_before = len(closures)
        if method == 'program_risk' and first is not None:
            for graph in menu:
                reason = risk(edits[graph], observations, closures)
                probes.append({'graph': graph, 'reason': reason})
                if reason is None:
                    chosen = graph
                    break
        elif method == 'qwen_add' and len(menu) > 1:
            chosen, model_call, valid_choice = qwen_choice(
                session, folder/f'qwen_{slot:02}.json', seed, observations, emitted, menu, edits)
        valid = False
        if chosen is not None:
            assert chosen not in checked
            checked.add(chosen)
            valid = check_graph({'nodes': NODES, 'edges': chosen}, observations,
                                {'nodes': NODES, 'max_edges': LIMIT})
            if valid:
                emitted.append(chosen)
        if method == 'program_risk' and probes and probes[-1]['reason'] is None:
            assert valid
        trace.append({'slot': slot, 'first': first, 'chosen': chosen, 'valid': valid, 'probes': probes,
                      'new_closures': len(closures)-closures_before, 'model_call': model_call is not None,
                      'valid_choice': valid_choice,
                      'prompt_tokens': model_call['response'].get('prompt_eval_count', 0) if model_call else 0,
                      'output_tokens': model_call['response'].get('eval_count', 0) if model_call else 0})
    assert len(emitted) == len(set(emitted)) <= min(SLOTS, task['n_ground_truths'])
    return {'found': len(emitted), 'rr': len(emitted)/task['n_ground_truths'],
            'checks': len(checked)-1, 'risk_probes': sum(len(t['probes']) for t in trace),
            'parent_closures': len(closures), 'switched': sum(t['chosen'] != t['first'] for t in trace),
            **{key: sum(t[key] for t in trace) for key in ('model_call','prompt_tokens','output_tokens')},
            'seconds': perf_counter()-start, 'trace': trace}


def summarize(output, records):
    def aggregate(cases, method):
        rows = [r['runs'][method] for r in cases]
        sums = ('found','checks','risk_probes','parent_closures','switched','model_call',
                'prompt_tokens','output_tokens','seconds')
        return {'tasks': len(rows), 'rr': mean(r['rr'] for r in rows),
                'full': sum(r['found'] == SLOTS for r in rows),
                **{key: sum(r[key] for r in rows) for key in sums}}

    overall = {m: aggregate(records, m) for m in METHODS}
    strata = {str(n): {m: aggregate([r for r in records if r['observations'] == n], m)
                      for m in METHODS} for n in range(1, 7)}
    paired = {}
    for left, right in [('program_risk','program_add'), ('program_risk','qwen_add')]:
        diffs = [r['runs'][left]['found']-r['runs'][right]['found'] for r in records]
        paired[left+'_vs_'+right] = {'win_tie_loss': [sum(d>0 for d in diffs), sum(d==0 for d in diffs),
                                                       sum(d<0 for d in diffs)],
                                       'rr_delta': overall[left]['rr']-overall[right]['rr']}
    summary = {'overall': overall, 'strata': strata, 'paired': paired,
               'ceiling_rr': mean(SLOTS/r['truth'] for r in records)}
    write_json(output/'summary.json', summary)
    lines = ['# 增边优先＋程序避险独立验证', '',
             '按 1–6 次干预各抽 6 道新题，排除先前推理实验配置中出现的题号。每题至少有 8 个合法图，每题配一个采样种子。三组共用合法程序首图及观测剪枝。首图占一槽，其余 7 槽每槽最多一次完整图验证，失败占槽。', '',
             '冻结上一轮的程序避险规则。在相同增边优先菜单里，优先验证不会成环或引入禁止路径的增边。删边保留风险标记，菜单无无风险项时使用首选。可达关系按已验证父图缓存，风险判断次数和闭包构造次数另行计数。每槽 Qwen 对照在菜单至少两项时调用模型。', '',
             '| 方法 | 平均 RR | 不同合法图 | 满 8 图 | 完整检查 | 风险判断 | 父图闭包 | Qwen 调用 | 输出 token | 秒数 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m, row in overall.items():
        lines.append(f"| {NAMES[m]} | {row['rr']:.4%} | {row['found']} | {row['full']}/{len(records)} | "
                     f"{row['checks']} | {row['risk_probes']} | {row['parent_closures']} | "
                     f"{row['model_call']} | {row['output_tokens']} | {row['seconds']:.3f} |")
    lines += ['', f"同题 8 槽的平均 RR 上限为 {summary['ceiling_rr']:.4%}。"
              f"程序避险相对普通程序的逐题胜平负为 {paired['program_risk_vs_program_add']['win_tie_loss']}，"
              f"相对每槽 Qwen 为 {paired['program_risk_vs_qwen_add']['win_tie_loss']}。", '',
              '| 干预次数 | 程序找到 | 程序避险找到 | Qwen 找到 |',
              '|---:|---:|---:|---:|']
    for n, block in strata.items():
        lines.append(f"| {n} | "+' | '.join(str(block[m]['found']) for m in METHODS)+' |')
    lines += ['', 'RR 按题宏平均，答案数只用于分层与评分。相同完整验证上限下，风险预判与模型成本另行统计。耗时含本地请求和搜索开销。样本为同一六节点数据集且每题一个种子。']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(summary, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE/'runs/program_risk_validation_20261113')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    tasks = select_tasks()
    assert Counter(r['n_observations'] for r in tasks) == {n: PER_STRATUM for n in range(1, 7)}
    if args.verify_only:
        totals = Counter()
        for task in tasks:
            anchor = program_anchor(NODES, LIMIT, task['observations'])
            assert anchor is not None
            for m in METHODS[:2]:
                totals[m] += evaluate(task, anchor, m, None, None, 0)['found']
        print(len(tasks), totals)
        return
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.trust_env = False
    response = session.get(BASE_URL+'/api/tags', timeout=10)
    response.raise_for_status()
    model = next(m for m in response.json()['models'] if m['name'] == MODEL)
    assert model['digest'] == read(MODEL_REFERENCE)['model']['digest']
    response = session.get(BASE_URL+'/api/version', timeout=10)
    response.raise_for_status()
    config = {'model': model['digest'], 'ollama': response.json(), 'cases': [t['id'] for t in tasks],
              'seed_schedule': SEEDS, 'selection_seed': SELECTION_SEED, 'slots': SLOTS,
              'checks_per_slot': 1, 'menu': MENU, 'data_sha': digest(DATA),
              'code_hashes': {name: digest(HERE/name) for name in ('validate_program_risk_search.py',
                    'evaluate_qwen_risk_trigger.py', 'run_qwen_tight_budget.py', 'run_qwen_guided_edits.py',
                    'run_partial_graph_pipeline.py', 'stream_causal.py')}}
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
        anchor = program_anchor(NODES, LIMIT, task['observations'])
        assert anchor is not None and check_graph({'nodes': NODES, 'edges': anchor}, task['observations'],
                                                   {'nodes': NODES, 'max_edges': LIMIT})
        seed = SEEDS[i % len(SEEDS)]
        runs = {m: evaluate(task, anchor, m, session, folder, seed) for m in METHODS}
        record = {'item': task['id'], 'observations': task['n_observations'], 'truth': task['n_ground_truths'],
                  'anchor': anchor, 'seed': seed, 'runs': runs}
        write_json(folder/'result.json', record)
        records.append(record)
        print(f'{i+1}/{len(tasks)} {task["id"]} ' + str([runs[m]['found'] for m in METHODS]), flush=True)
    assert len(list(output.glob('*/qwen_*.json'))) == sum(r['runs']['qwen_add']['model_call'] for r in records)
    summarize(output, records)


if __name__ == '__main__':
    main()
