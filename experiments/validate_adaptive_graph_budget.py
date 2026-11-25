"""Compare fixed and adaptive search on the frozen six-node observation set."""
import argparse
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

from common import HERE, digest
from run_local_causal import write_json
from solve_partial_graph import CHECK_BATCH_SIZE, DEFAULT_CHECKS_PER_SLOT, solve
from stream_causal import check_graph

DATA = HERE/'runs/pruned_graph_validation_20261113_151627/observations.json'
POLICIES = {
    'fixed_4': {'adaptive': False, 'checks_per_slot': 4},
    'fixed_8': {'adaptive': False, 'checks_per_slot': 8},
    'adaptive_8': {'adaptive': True, 'checks_per_slot': 8},
    'adaptive_default': {'adaptive': True, 'checks_per_slot': DEFAULT_CHECKS_PER_SLOT},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE/'runs/adaptive_graph_budget_20261113')
    args = parser.parse_args()
    tasks = json.loads(DATA.read_text(encoding='utf-8'))
    assert len(tasks) == len({task['id'] for task in tasks}) == 510
    metadata = {'nodes': list('ABCDEF'), 'max_edges': 10}
    rows = []
    for task in tasks:
        for name, policy in POLICIES.items():
            started = perf_counter()
            result = solve(metadata['nodes'], metadata['max_edges'], task['observations'], **policy)
            elapsed = perf_counter()-started
            graphs = [row['edges'] for row in result['proposals']]
            checks = result['neighbor_checks_by_slot']
            cap = min(8, task['n_ground_truths'])
            assert len(set(graphs)) == len(graphs) <= cap
            assert all(check_graph({'nodes': metadata['nodes'], 'edges': edges},
                                   task['observations'], metadata) for edges in graphs)
            assert result['checks'] == 1+sum(checks)
            assert len(checks) <= 7 and max(checks, default=0) <= policy['checks_per_slot']
            rows.append({'item': task['id'], 'method': name, 'observations': task['n_observations'],
                         'graphs': len(graphs), 'cap': cap, 'rr': len(graphs)/task['n_ground_truths'],
                         'checks': result['checks'], 'slot_checks': checks, 'seconds': elapsed})
    summaries = {}
    for name in POLICIES:
        subset = [row for row in rows if row['method'] == name]
        summaries[name] = {'cap_hits': sum(row['graphs'] == row['cap'] for row in subset),
                           'graphs': sum(row['graphs'] for row in subset),
                           'rr': mean(row['rr'] for row in subset),
                           'checks': mean(row['checks'] for row in subset),
                           'seconds': sum(row['seconds'] for row in subset),
                           'tasks_over_8_in_a_slot': sum(max(row['slot_checks'], default=0)>8 for row in subset)}
    before, after = summaries['fixed_8'], summaries['adaptive_default']
    assert after['cap_hits'] == before['cap_hits'] == len(tasks)
    assert after['graphs'] == before['graphs'] and after['rr'] == before['rr']
    assert after['checks'] < before['checks']
    reduction = 1 - after['checks']/before['checks']
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output/'results.json', {'input_sha': digest(DATA), 'policies': POLICIES,
        'code_hashes': {name: digest(HERE/name) for name in ('solve_partial_graph.py',
            'run_partial_graph_pipeline.py', 'validate_adaptive_graph_budget.py')},
        'batch': CHECK_BATCH_SIZE, 'summary': summaries, 'records': rows})
    lines = ['# 按需检查预算对照', '',
             f"同一批 510 道六节点题上，按需策略保留固定 8 次方案的覆盖率，平均检查减少 {reduction:.2%}。", '',
             '所有组开启观测剪枝，每题最多提交 8 个不同合法图。固定组每槽尝试用满预算。按需组先使用池中已验证的未提交候选，池空时每批检查最多 4 个新邻居，直到池非空、邻居耗尽或达到每槽上限。检查数包含一次首图校验。所有输出均独立核对无环、边数限制、观测一致性及去重。RR 分母来自先前数据审计。', '',
             '| 策略 | 达到覆盖上限 | 不同合法图总数 | 平均 RR | 平均检查 | 本次累计求解秒数 |',
             '|---|---:|---:|---:|---:|---:|']
    labels = {'fixed_4': '固定 4 次', 'fixed_8': '固定 8 次', 'adaptive_8': '按需，上限 8 次',
              'adaptive_default': f'按需，上限 {DEFAULT_CHECKS_PER_SLOT} 次'}
    for name, d in summaries.items():
        lines.append(f"| {labels[name]} | {d['cap_hits']}/510 | {d['graphs']} | {d['rr']:.4%} | "
                     f"{d['checks']:.2f} | {d['seconds']:.3f} |")
    misses = [r['item'] for r in rows if r['method']=='adaptive_8' and r['graphs']<r['cap']]
    lines += ['', f'按需上限 8 次时，{len(misses)} 道题各少一个图（{", ".join(misses)}）。'
              f'因此在这批开发题上把默认上限调到 {DEFAULT_CHECKS_PER_SLOT}。'
              f'实测只有 {after["tasks_over_8_in_a_slot"]} 道题在某一槽超过 8 次。'
              '该上限是在当前题集上选定的，结果适用于这批六节点、无噪声观测。时间为本机一次测量。', '',
              'solve_partial_graph.py 默认使用按需预算。可传 --fixed-budget --checks-per-slot 8 切回固定 8 次。'
              '输出的 neighbor_checks_by_slot 记录首图之后每个搜索槽的实际检查数。'
              '运行 validate_adaptive_graph_budget.py 可复现对照。']
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(args.output.resolve())
    print(json.dumps(summaries))


if __name__ == '__main__':
    main()
