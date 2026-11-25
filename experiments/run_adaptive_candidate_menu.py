"""Expand the ranked menu only when the previous batch has no safe edit."""
import argparse
from itertools import permutations
from pathlib import Path
import random
import re
from statistics import mean
from time import perf_counter

from common import HERE, digest
from run_formal_causal_comparison import (
    DATA, LIMIT, MENU, NODES, SLOTS, deletion_safe, forbidden_edges, neighbors,
    priority, program_anchor, read, risk, valid,
)
from run_local_causal import write_json

MAX_MENU = 32
SELECTION_SEED = 1021126
POLICIES = {'fixed_8': MENU, 'adaptive_32': MAX_MENU, 'full': None}
FORMAL = HERE/'runs/formal_causal_comparison_20261113'


def select_tasks(tasks, output):
    used = set()
    for path in (HERE/'runs').glob('*/config.json'):
        if path.parent.resolve() != output.resolve():
            used.update(re.findall(r'\bn[1-6]_\d{3}\b', path.read_text(encoding='utf-8')))
    rng = random.Random(SELECTION_SEED)
    return [task for n in range(1, 7) for task in rng.sample(sorted(
        (row for row in tasks if row['n_observations'] == n and row['id'] not in used
         and (row['n_ground_truths'] >= SLOTS if n < 6 else row['n_ground_truths'] > 1)),
        key=lambda row: row['id']), 6)]


def ordered_edits(emitted, checked, allowed):
    edits = neighbors(emitted, allowed, checked)
    ordered = sorted(edits, key=lambda g: (edits[g]['action'] != 'add', priority(g, emitted)))
    if ordered and edits[ordered[0]]['action'] == 'add':
        ordered = [g for g in ordered if edits[g]['action'] == 'add']
    return edits, ordered


def choose(ordered, edits, observations, closures, cap):
    """Scan consecutive windows without rechecking an earlier window."""
    end = len(ordered) if cap is None else min(len(ordered), cap)
    traversals = 0
    for start in range(0, end, MENU):
        for index in range(start, min(start+MENU, end)):
            candidate = ordered[index]
            edit = edits[candidate]
            if edit['action'] == 'add':
                safe = risk(edit, observations, closures) is None
            else:
                safe, cost = deletion_safe(edit, candidate, observations)
                traversals += cost
            if safe:
                return candidate, index+1, index+1, traversals
    # Preserve the prior one-check fallback when the bounded scan fails.
    return (ordered[0] if ordered else None), None, end, traversals


def solve(task, anchor, policy, cap):
    observations = task['observations']
    blocked = forbidden_edges(observations)
    allowed = tuple(edge for edge in permutations(NODES, 2) if edge not in blocked)
    emitted, checked, closures, trace = [anchor], {anchor}, {}, []
    started = perf_counter()
    for slot in range(2, SLOTS+1):
        edits, ordered = ordered_edits(emitted, checked, allowed)
        candidate, rank, probes, traversals = choose(ordered, edits, observations, closures, cap)
        accepted = candidate is not None and valid(candidate, observations)
        assert rank is None or accepted
        if candidate is not None:
            assert candidate not in checked
            checked.add(candidate)
        if accepted:
            emitted.append(candidate)
        trace.append({'slot': slot, 'menu_size': len(ordered), 'chosen': candidate, 'valid': accepted,
                      'safe_rank': rank, 'probes': probes, 'deletion_traversals': traversals,
                      'expanded': probes > MENU, 'cap_hit': rank is None and probes < len(ordered),
                      'action': edits[candidate]['action'] if candidate is not None else None})
    elapsed = perf_counter()-started
    ceiling = min(SLOTS, task['n_ground_truths'])
    assert len(set(emitted)) == len(emitted) <= ceiling
    assert all(valid(graph, observations) for graph in emitted)
    assert len(checked)-1 <= SLOTS-1
    assert cap is None or all(step['probes'] <= cap for step in trace)
    return {'policy': policy, 'found': len(emitted), 'rr': len(emitted)/task['n_ground_truths'],
            'ceiling': ceiling, 'graphs': emitted, 'checks': len(checked)-1, 'closures': len(closures),
            'seconds': elapsed, 'trace': trace,
            **{key: sum(step[key] for step in trace)
               for key in ('probes', 'deletion_traversals', 'expanded', 'cap_hit')}}


def aggregate(rows):
    output = {}
    for policy in POLICIES:
        runs = [row['runs'][policy] for row in rows]
        output[policy] = {
            'tasks': len(rows), 'rr': mean(run['rr'] for run in runs),
            'ceiling_hits': sum(run['found'] == run['ceiling'] for run in runs),
            **{key: sum(run[key] for run in runs) for key in
               ('found', 'checks', 'probes', 'deletion_traversals', 'expanded', 'cap_hit', 'closures', 'seconds')},
        }
    for other in ('fixed_8', 'full'):
        diffs = [row['runs']['adaptive_32']['found']-row['runs'][other]['found'] for row in rows]
        output[f'adaptive_vs_{other}'] = [sum(d > 0 for d in diffs), sum(d == 0 for d in diffs), sum(d < 0 for d in diffs)]
    output['rr_ceiling'] = mean(min(SLOTS, row['truth'])/row['truth'] for row in rows)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE/'runs/adaptive_candidate_menu_20261113')
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    path = output/'config.json'
    all_tasks = read(DATA)
    lookup = {t['id']: t for t in all_tasks}
    selected = [lookup[key] for key in read(path)['cases']] if path.exists() else select_tasks(all_tasks, output)
    assert len(selected) == len({t['id'] for t in selected}) == 36
    formal_cases = read(FORMAL/'config.json')['cases']
    assert not set(formal_cases).intersection(t['id'] for t in selected)
    config = {'cases': [t['id'] for t in selected], 'selection_seed': SELECTION_SEED,
              'formal_replay': formal_cases, 'slots': SLOTS, 'batch': MENU, 'policies': POLICIES,
              'data_sha': digest(DATA), 'code_hashes': {name: digest(HERE/name) for name in (
                  'run_adaptive_candidate_menu.py', 'run_formal_causal_comparison.py',
                  'run_qwen_tight_budget.py', 'evaluate_qwen_risk_trigger.py',
                  'run_partial_graph_pipeline.py', 'stream_causal.py')}}
    if path.exists():
        assert read(path) == config
    else:
        write_json(path, config)
    records = {}
    for label, tasks in [('new', selected), ('replay', [lookup[key] for key in formal_cases])]:
        rows = []
        for task in tasks:
            anchor = program_anchor(NODES, LIMIT, task['observations'])
            assert anchor is not None and valid(anchor, task['observations'])
            runs = {name: solve(task, anchor, name, cap) for name, cap in POLICIES.items()}
            if label == 'replay':
                old = read(FORMAL/task['id']/'result.json')['runs']['delete_risk']
                fixed = runs['fixed_8']
                assert [list(map(list, graph)) for graph in fixed['graphs']] == old['graphs']
                assert fixed['probes'] == old['probes'] and fixed['deletion_traversals'] == old['deletion_traversals']
            rows.append({'item': task['id'], 'observations': task['n_observations'],
                         'truth': task['n_ground_truths'], 'runs': runs})
        records[label] = rows
    summary = {label: aggregate(rows) for label, rows in records.items()}
    write_json(output/'results.json', {'summary': summary, 'records': records})
    lines = ['# 按需扩大候选菜单验证', '',
             '对照固定 8 项、按需扩大至 32 项、无菜单上限。每槽从同一增边优先排序开始，每批扫描 8 个候选。找到首个通过预判的候选就停止。上一批均失败时才继续下一批。没有通过预判的候选时，沿用原来的首选回退。每槽最多执行一次完整图验证。三组都在找到新图后更新搜索状态。', '',
             '新题按 1–6 次干预各取六道，排除此前配置中出现的模型实验题号。完整观测层剩余未用题都少于 8 个解，因此从有至少 2 个解的题中抽取。达到上限指找到 min(8, 合法图总数) 个合法图。题目来自此前全量程序诊断覆盖过的同一数据集。另重放原正式 24 题复核修复位置。', '',
             '| 题集 | 方法 | 平均 RR | 合法图 | 达到上限 | 完整检查 | 预判 | 删边遍历 | 扩展槽 | 耗时（秒） |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    labels = {'fixed_8': '固定8项', 'adaptive_32': '按需，最多32项', 'full': '不限菜单'}
    for section, table in summary.items():
        for name in POLICIES:
            row = table[name]
            lines.append(f"| {section} | {labels[name]} | {row['rr']:.4%} | {row['found']} | "
                         f"{row['ceiling_hits']}/{row['tasks']} | {row['checks']} | {row['probes']} | "
                         f"{row['deletion_traversals']} | {row['expanded']} | {row['seconds']:.4f} |")
    lines += ['', '不限菜单组也在首个通过预判的候选处停止。若合法项在第 32 位以内，按需组与不限菜单组的检查成本可以完全相同。检查数不含共同首图，预判和删边遍历另计。所有输出都核对无环、边数、题面观测和去重。', '',
              '| 题目 | 题集 | 合法解数 | 固定8 | 按需32 | 不限 |',
              '|---|---|---:|---:|---:|---:|']
    for section, rows in records.items():
        for row in rows:
            counts = [row['runs'][name]['found'] for name in POLICIES]
            if len(set(counts)) > 1:
                lines.append(f"| {row['item']} | {section} | {row['truth']} | " + ' | '.join(map(str, counts)) + ' |')
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(output.resolve())
    print(summary)


if __name__ == '__main__':
    main()
