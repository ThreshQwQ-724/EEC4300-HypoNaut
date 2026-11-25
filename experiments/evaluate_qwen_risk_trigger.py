"""Exploratory pre-check trigger on the frozen selective-Qwen cases."""
import argparse
from itertools import permutations
from pathlib import Path
from statistics import mean
from time import perf_counter

import networkx as nx
import requests

from common import HERE, digest
from run_four_groups import BASE_URL, MODEL
from run_local_causal import write_json
from run_partial_graph_pipeline import forbidden_edges
from run_qwen_rr_budget import read
from run_qwen_tight_budget import LIMIT, MENU, NODES, SLOTS, neighbors, priority, qwen_choice
from stream_causal import check_graph
from validate_graph_rr_budget import DATA

SOURCE = HERE/'runs/selective_qwen_search_20261113'


def risk(edit, obs, closures):
    if edit['action'] == 'remove':
        return 'removal_may_break_required_path'
    parent = edit['parent']
    if parent not in closures:
        graph = nx.DiGraph(parent)
        graph.add_nodes_from(NODES)
        closures[parent] = {n: nx.descendants(graph, n) | {n} for n in NODES}
    reach = closures[parent]
    a, b = edit['edge']
    if a in reach[b]:
        return 'cycle'
    for row in obs:
        if a in reach[row['perturbed_node']] and any(
                row['effects'][target] == 0 for target in reach[b]):
            return 'forbidden_path'
    return None


def run(task, base, folder, session, method):
    obs = task['observations']
    anchor = tuple(map(tuple, base['anchor']))
    blocked = forbidden_edges(obs)
    allowed = tuple(e for e in permutations(NODES, 2) if e not in blocked)
    checked, emitted, trace, closures = {anchor}, [anchor], [], {}
    started = perf_counter()
    for slot in range(2, SLOTS+1):
        edits = neighbors(emitted, allowed, checked)
        ordered = sorted(edits, key=lambda g: (edits[g]['action'] != 'add', priority(g, emitted)))
        menu = ordered[:MENU]
        if menu and edits[menu[0]]['action'] == 'add':
            menu = [g for g in menu if edits[g]['action'] == 'add']
        first = menu[0] if menu else None
        reason = risk(edits[first], obs, closures) if first is not None else None
        candidate, call, parsed = first, None, None
        if reason and len(menu) > 1:
            if method == 'risk_qwen':
                candidate, call, parsed = qwen_choice(
                    session, folder/f'risk_{slot:02}.json', base['seed'], obs, emitted, menu, edits)
            else:
                candidate = next((g for g in menu if risk(edits[g], obs, closures) is None), first)
        valid = False
        if candidate is not None:
            checked.add(candidate)
            valid = check_graph({'nodes': NODES, 'edges': candidate}, obs, {'nodes': NODES, 'max_edges': LIMIT})
            if valid:
                emitted.append(candidate)
        trace.append({'slot': slot, 'candidate': candidate, 'valid': valid, 'reason': reason,
                      'model_call': call is not None, 'valid_choice': parsed,
                      'tokens': call['response'].get('eval_count', 0) if call else 0})
    assert len(set(emitted)) == len(emitted) <= min(SLOTS, base['truth'])
    return {'found': len(emitted), 'rr': len(emitted)/base['truth'],
            'checks': sum(t['candidate'] is not None for t in trace),
            'model_calls': sum(t['model_call'] for t in trace), 'tokens': sum(t['tokens'] for t in trace),
            'seconds': perf_counter()-started, 'trace': trace}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=SOURCE/'risk_followup')
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    config = read(SOURCE/'config.json')
    assert digest(DATA) == config['data_sha']
    session = requests.Session()
    session.trust_env = False
    response = session.get(BASE_URL+'/api/tags', timeout=10)
    response.raise_for_status()
    assert next(m for m in response.json()['models'] if m['name'] == MODEL)['digest'] == config['model']
    manifest = {'source_config_sha': digest(SOURCE/'config.json'), 'runner_sha': digest(__file__),
                'choice_helper_sha': digest(HERE/'run_qwen_tight_budget.py')}
    if (output/'config.json').exists():
        assert read(output/'config.json') == manifest
    else:
        write_json(output/'config.json', manifest)
    tasks = {r['id']: r for r in read(DATA)}
    records = []
    for item in config['cases']:
        base = read(SOURCE/item/'result.json')
        folder = output/item
        folder.mkdir(exist_ok=True)
        for method in ('risk_qwen', 'risk_program'):
            base['runs'][method] = run(tasks[item], base, folder, session, method)
        records.append(base)
        write_json(folder/'result.json', base)
    names = {'program_add': '程序增边优先', 'qwen_add': '每槽 Qwen',
             'selective_qwen': '失败后调用 Qwen', 'risk_qwen': '风险出现时调用 Qwen',
             'risk_program': '程序避开高风险候选'}
    summary = {}
    lines = ['# 选择性 Qwen 的风险触发补测', '',
             '复用主实验 24 题。首选为增边时，在合法父图的缓存可达关系上检查新增边是否会成环或引入禁止路径。删边统一标记为可能破坏已有路径。风险 Qwen 组在首选有风险、菜单有多个候选时询问 Qwen。风险程序组在同一菜单里优先挑无风险项。每槽仍至多一次完整图验证。风险判断和模型成本另算。', '',
             '| 策略 | RR | 不同合法图 | 满 8 图 | 完整图检查 | Qwen 调用 | 输出 token | 累计秒数 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for method, label in names.items():
        runs = [r['runs'][method] for r in records]
        d = {k: sum(r[k] for r in runs) for k in ('found', 'checks', 'model_calls', 'tokens', 'seconds')}
        d.update(rr=mean(r['rr'] for r in runs), full=sum(r['found'] == SLOTS for r in runs))
        summary[method] = d
        lines.append(f"| {label} | {d['rr']:.4%} | {d['found']} | {d['full']}/24 | {d['checks']} | "
                     f"{d['model_calls']} | {d['tokens']} | {d['seconds']:.3f} |")
    for other in ('program_add', 'qwen_add', 'risk_program'):
        ds = [r['runs']['risk_qwen']['found']-r['runs'][other]['found'] for r in records]
        summary['risk_qwen_vs_'+other] = [sum(x > 0 for x in ds), sum(x == 0 for x in ds), sum(x < 0 for x in ds)]
    lines += ['', '触发规则在查看主实验后加入，本表属于同题探索性对照。程序风险判断利用图结构和题面观测，仍对提交候选进行完整验证。']
    assert len(list(output.glob('*/risk_*.json'))) == summary['risk_qwen']['model_calls']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    write_json(output/'summary.json', summary)
    print(summary, flush=True)


if __name__ == '__main__':
    main()
