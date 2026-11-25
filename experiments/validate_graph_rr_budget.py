"""Measure coverage and unobserved intervention signatures at nested budgets."""
import argparse
import json
import os
from pathlib import Path
from statistics import mean
from time import perf_counter

import networkx as nx

from common import HERE, digest
from run_local_causal import write_json
from solve_partial_graph import solve
from stream_causal import check_graph

BUDGETS = (8, 16, 32, 64, 128)
DATA = HERE/'runs/pruned_graph_validation_20261113_151627/observations.json'


def profile(task, result, nodes, max_edges):
    graphs = [p['edges'] for p in result['proposals']]
    slots = result['proposal_slots']
    slot_checks = result['neighbor_checks_by_slot']
    assert len(set(graphs)) == len(graphs) == len(slots)
    assert slots == sorted(set(slots)) and slots[0] == 1 and slots[-1] <= BUDGETS[-1]
    observed = {row['perturbed_node'] for row in task['observations']}
    unknown = [node for node in nodes if node not in observed]
    signatures = []
    for edges in graphs:
        assert check_graph({'nodes': nodes, 'edges': edges}, task['observations'],
                           {'nodes': nodes, 'max_edges': max_edges})
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        signatures.append(tuple(tuple(sorted(nx.descendants(graph, source))) for source in unknown))
    rows = []
    truth = task['n_ground_truths']
    for budget in BUDGETS:
        found = sum(slot <= budget for slot in slots)
        ceiling = min(budget, truth)
        assert found <= ceiling
        rows.append({'budget': budget, 'found': found, 'ceiling': ceiling, 'rr': found/truth,
                     'rr_ceiling': ceiling/truth, 'checks': 1+sum(slot_checks[:budget-1]),
                     'signatures': len(set(signatures[:found])) if unknown else None})
    return rows


def summarize(records):
    sections = {}
    for label, cases in [('all', records),
                         ('partial', [r for r in records if r['observations'] < 6]),
                         ('low', [r for r in records if r['observations'] <= 3])]:
        stats = []
        for i, budget in enumerate(BUDGETS):
            rows = [r['budgets'][i] for r in cases]
            predictions = [r['signatures'] for r in rows if r['signatures'] is not None]
            stats.append({'budget': budget, 'tasks': len(rows), 'graphs': sum(r['found'] for r in rows),
                          'capacity': sum(r['ceiling'] for r in rows),
                          'ceiling_hits': sum(r['found'] == r['ceiling'] for r in rows),
                          'rr': mean(r['rr'] for r in rows), 'rr_ceiling': mean(r['rr_ceiling'] for r in rows),
                          'mean_checks': mean(r['checks'] for r in rows),
                          'mean_signatures': mean(predictions) if predictions else None})
        sections[label] = stats
    return sections


def plot(output, summary):
    os.environ.setdefault('MPLCONFIGDIR', str(HERE/'.mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for ax, key, label in zip(axes, ('all', 'low'), ('All 510 tasks', '1–3 interventions (210 tasks)')):
        rows = summary[key]
        ax.plot(BUDGETS, [100*r['rr_ceiling'] for r in rows], '--', color='#c06928', label='Budget ceiling')
        ax.plot(BUDGETS, [100*r['rr'] for r in rows], 'o-', color='#287590', label='Measured RR')
        ax.set(xscale='log', xlabel='Submission slots per task', ylabel='Mean recovery rate (%)', title=label)
        ax.set_xticks(BUDGETS, labels=[str(b) for b in BUDGETS])
        ax.grid(axis='y', alpha=.25)
        ax.legend()
    fig.savefig(output/'rr_budget.png', dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE/'runs/graph_rr_budget_20261113')
    args = parser.parse_args()
    tasks = json.loads(DATA.read_text(encoding='utf-8'))
    assert len(tasks) == len({r['id'] for r in tasks}) == 510
    args.output.mkdir(parents=True, exist_ok=True)
    nodes = list('ABCDEF')
    records, sensitivity = [], []
    elapsed = 0
    for task in tasks:
        started = perf_counter()
        result = solve(nodes, 10, task['observations'], slots=BUDGETS[-1])
        elapsed += perf_counter()-started
        budgets = profile(task, result, nodes, 10)
        prefix = solve(nodes, 10, task['observations'], slots=8)
        assert prefix['proposals'] == result['proposals'][:budgets[0]['found']]
        assert prefix['neighbor_checks_by_slot'] == result['neighbor_checks_by_slot'][:7]
        record = {'item': task['id'], 'truth': task['n_ground_truths'],
                  'observations': task['n_observations'], 'budgets': budgets}
        records.append(record)
        if budgets[-1]['found'] < budgets[-1]['ceiling']:
            extra = solve(nodes, 10, task['observations'], slots=BUDGETS[-1], checks_per_slot=48)
            final = profile(task, extra, nodes, 10)[-1]
            sensitivity.append({'item': task['id'], 'default': budgets[-1], 'checks_cap_48': final})
    summary = summarize(records)
    write_json(args.output/'results.json', {'input_sha': digest(DATA), 'summary': summary,
              'records': records, 'sensitivity': sensitivity, 'default_solver_seconds': elapsed,
              'source_hashes': {name: digest(HERE/name) for name in (
                  'solve_partial_graph.py', 'run_partial_graph_pipeline.py', 'validate_graph_rr_budget.py')}})
    plot(args.output, summary)
    lines = ['# RR 与候选提交预算', '',
             '在现有 510 道六节点题上，每题用同一轮搜索从 8 槽推进到 128 槽。检查上限沿用默认的每槽 12 次，池空时按每批 4 次追加。中间结果按提交槽位截取，包括未成功提交的槽位。全部 510 道题的前 8 槽与独立运行的 8 槽结果一致。', '',
             '每个输出均核对去重、无环、边数和题面一致性。RR 按题取宏平均，分母来自源数据的答案数审计。理论上限为逐题 min(槽位数, 合法图数) / 合法图数。', '',
             '| 槽位 | 全体 RR | 理论上限 | 达到上限 | 不同合法图 | 平均检查 | 低观测组 RR | 低观测组平均预测种类 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for full, low in zip(summary['all'], summary['low']):
        lines.append(f"| {full['budget']} | {full['rr']:.4%} | {full['rr_ceiling']:.4%} | "
                     f"{full['ceiling_hits']}/510 | {full['graphs']:,} | {full['mean_checks']:.2f} | "
                     f"{low['rr']:.4%} | {low['mean_signatures']:.2f} |")
    recovered = sum(r['checks_cap_48']['found'] == r['checks_cap_48']['ceiling'] for r in sensitivity)
    lines += ['', '![RR 与提交预算](rr_budget.png)', '',
              '低观测组包含只有 1–3 次干预的 210 题。预测种类按每张图对所有未观测节点的干预结果合并去重，统计找到的预测签名数。此处未枚举所有可能预测，因此报告数量而非预测覆盖率。', '',
              f'128 槽下有 {len(sensitivity)} 道题未达到上限。仅对这些题进行诊断补测，将单槽检查上限从 12 放宽到 48 后，{recovered} 道达到上限。此补测按已知评分选择题目，用于定位缺口。逐题成本和结果保存在 results.json。', '',
              '测试复用已有六节点数据。生成和剪枝只读取观测，答案数用于评分。求解入口默认仍是 8 槽，显式传入 --slots 128 可获得扩展结果。']
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'all': summary['all'], 'low': summary['low'], 'sensitivity_recovered': recovered}))


if __name__ == '__main__':
    main()
