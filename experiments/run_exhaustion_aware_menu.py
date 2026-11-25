"""Stop exact-complete searches and cache rejected local candidates."""
import argparse
from itertools import permutations
from pathlib import Path
from statistics import mean

from common import HERE, digest
from graph_completion import complete_observation_count
from run_adaptive_candidate_menu import MAX_MENU, ordered_edits, choose
from run_formal_causal_comparison import (
    DATA, LIMIT, MENU, NODES, SLOTS, deletion_safe, forbidden_edges, neighbors,
    priority, program_anchor, read, risk, valid,
)
from run_local_causal import write_json

REFERENCE = HERE/'runs/adaptive_candidate_menu_20261113'


def solve_menu(observations, anchor, *, slots=SLOTS, cap=MAX_MENU):
    """Return verified graphs and a certificate when this search is complete."""
    if not valid(anchor, observations):
        raise ValueError('The initial graph must satisfy the observations')
    exact = complete_observation_count(NODES, LIMIT, observations)
    blocked = forbidden_edges(observations)
    allowed = tuple(edge for edge in permutations(NODES, 2) if edge not in blocked)
    emitted, checked, closures, trace = [anchor], {anchor}, {}, []
    completion = 'unproven'
    for slot in range(2, slots+1):
        if len(emitted) == exact:
            break
        edits, ordered = ordered_edits(emitted, checked, allowed)
        traversals = probes = 0
        selected = None
        inspected = set()
        while probes < cap:
            # Prefer additions. Once those are ruled out, continue with deletions.
            if not ordered:
                edits, ordered = ordered_edits(emitted, checked | inspected, allowed)
            if not ordered:
                break
            batch, ordered = ordered[:MENU], ordered[MENU:]
            for candidate in batch[:cap-probes]:
                probes += 1
                edit = edits[candidate]
                if edit['action'] == 'add':
                    safe = risk(edit, observations, closures) is None
                else:
                    safe, count = deletion_safe(edit, candidate, observations)
                    traversals += count
                if safe:
                    selected = candidate
                    break
                # Both prechecks are exact for a single edit of a validated parent.
                inspected.add(candidate)
            if selected is not None:
                break
        checked.update(inspected)
        if selected is not None:
            assert valid(selected, observations)
            checked.add(selected)
            emitted.append(selected)
        trace.append({'slot': slot, 'probes': probes, 'deletion_traversals': traversals,
                      'chosen': selected, 'rejected': len(inspected)})
        if selected is None and not neighbors(emitted, allowed, checked):
            completion = 'single_edge_component_exhausted'
            break
    if len(emitted) == exact:
        completion = 'all_legal_graphs_found'
    return {'graphs': emitted, 'found': len(emitted), 'completion': completion, 'exact_count': exact,
            'checks': len(emitted)-1, 'probes': sum(t['probes'] for t in trace),
            'deletion_traversals': sum(t['deletion_traversals'] for t in trace), 'trace': trace}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE/'runs/exhaustion_aware_menu_20261113')
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    previous = read(REFERENCE/'results.json')['records']
    tasks = {r['id']: r for r in read(DATA)}
    reports = {}
    summaries = {}
    for label, group in previous.items():
        records = []
        for row in group:
            task = tasks[row['item']]
            obs, truth = task['observations'], task['n_ground_truths']
            anchor = program_anchor(NODES, LIMIT, obs)
            result = solve_menu(obs, anchor)
            assert len(set(result['graphs'])) == result['found'] <= min(SLOTS, truth)
            assert all(valid(g, obs) for g in result['graphs'])
            if result['exact_count'] is not None:
                assert result['exact_count'] == truth
            if result['completion'] == 'all_legal_graphs_found':
                assert result['found'] == truth
            if result['completion'] == 'single_edge_component_exhausted':
                blocked = forbidden_edges(obs)
                allowed = tuple(e for e in permutations(NODES, 2) if e not in blocked)
                assert not any(valid(g, obs) for g in neighbors(result['graphs'], allowed, set(result['graphs'])))
            records.append({'item': row['item'], 'truth': truth, 'before': row['runs']['adaptive_32'],
                            'after': result})
        reports[label] = records
        summaries[label] = {}
        for name in ('before', 'after'):
            summaries[label][name] = {
                'rr': mean(r[name]['found']/r['truth'] for r in records),
                'ceiling_hits': sum(r[name]['found'] == min(SLOTS, r['truth']) for r in records),
                **{k: sum(r[name][k] for r in records) for k in
                   ('found', 'checks', 'probes', 'deletion_traversals')}}
        assert all(r['after']['found'] >= r['before']['found'] for r in records)
    write_json(output/'results.json', {'summary': summaries, 'records': reports,
        'data_sha': digest(DATA), 'comparison_sha': digest(REFERENCE/'results.json'),
        'code_hashes': {name: digest(HERE/name) for name in ('graph_completion.py',
            'run_exhaustion_aware_menu.py', 'run_adaptive_candidate_menu.py',
            'run_formal_causal_comparison.py', 'run_partial_graph_pipeline.py')}})
    lines = ['# 候选搜索停机验证', '',
             '完整观测从可达关系构造传递约简。约简中的边必须存在，其余已知可达边均为可选边。'
             '在边数限制内计数组合，找到同样数量的互异合法图即可确认全局找齐。答案数只用于回放评分。', '',
             '部分观测缓存经精确单边预判排除的候选。若已提交图的增删邻居都已排除，'
             '只证明当前单边连通分量搜尽。剩余未验证候选或提交预算用尽时保留 unproven。'
             '当增边候选均被排除时，可在同一预判上限内切到删边候选。', '',
             '每槽最多预判 32 个候选，最多执行一次完整验证。前后对照复用上次同一批 60 题。', '',
             '| 题集 | 方法 | RR | 合法图 | 达到上限 | 完整检查 | 预判 | 删边遍历 |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for label, block in summaries.items():
        for method, row in block.items():
            lines.append(f"| {label} | {method} | {row['rr']:.4%} | {row['found']} | {row['ceiling_hits']} | "
                         f"{row['checks']} | {row['probes']} | {row['deletion_traversals']} |")
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(output.resolve(), summaries)


if __name__ == '__main__':
    main()
