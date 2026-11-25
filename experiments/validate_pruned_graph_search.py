"""Test the production solver on the 510 six-node tasks."""
import argparse
from datetime import datetime
import json
import mmap
from pathlib import Path
from statistics import mean

from common import DATA, HERE, RESULTS, digest
from run_local_causal import write_json
from solve_partial_graph import solve
from stream_causal import check_graph

BUDGETS = (1, 2, 4, 8, 24)
REFERENCE_BUDGET = 8


def load_observations(source, audited):
    """Read the small header of each known task, skipping its huge answer array."""
    tasks = []
    with source.open('rb') as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
        cursor = 0
        for task in audited:
            marker = f'"observation_set_id": "{task["id"]}"'.encode('ascii')
            start = data.find(marker, cursor)
            if start < 0:
                raise ValueError(f'Task not found: {task["id"]}')
            begin = data.find(b'[', data.find(b'"observations"', start, start+512), start+512)
            end = data.find(b'"ground_truth_graphs"', start, start+16384)
            if not start < begin < end:
                raise ValueError(f'Unexpected layout: {task["id"]}')
            observations = json.loads(data[begin:end].strip().rstrip(b','))
            assert len(observations) == task['n_observations']
            assert len({row['perturbed_node'] for row in observations}) == len(observations)
            tasks.append({**task, 'observations': observations})
            cursor = end
    return tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or HERE/'runs'/('pruned_graph_validation_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True, exist_ok=True)
    source = DATA/'causal/datasets/node06/n6_all_observations.json'
    audit = json.loads((RESULTS/'causal_audit.json').read_text(encoding='utf-8'))
    assert source.stat().st_size == audit['size_bytes'] and not audit['errors']
    metadata = audit['metadata']
    inputs = load_observations(source, audit['instances'])
    assert len(inputs) == 510
    records = []
    for task in inputs:
        truth = task['n_ground_truths']
        ceiling = min(8, truth)
        for budget in BUDGETS:
            for prune in (False, True):
                solution = solve(metadata['nodes'], metadata['max_edges'], task['observations'],
                                 checks_per_slot=budget, prune=prune, adaptive=False)
                graphs = [row['edges'] for row in solution['proposals']]
                assert len(set(graphs)) == len(graphs) <= ceiling
                assert all(check_graph({'nodes': metadata['nodes'], 'edges': edges},
                                       task['observations'], metadata) for edges in graphs)
                assert solution['checks'] <= 1+7*budget
                records.append({'id': task['id'], 'observations': task['n_observations'], 'truth': truth,
                                'ceiling': ceiling, 'budget': budget, 'prune': prune, 'found': len(graphs),
                                'checks': solution['checks'], 'rr': len(graphs)/truth})
    summary = []
    for count in (1, 2, 3, 4, 5, 6, 'low', 'all'):
        for budget in BUDGETS:
            for prune in (False, True):
                group = [r for r in records if (count == 'all' or count == r['observations']
                         or (count == 'low' and r['observations'] <= 3))
                         and r['budget'] == budget and r['prune'] == prune]
                summary.append({'observations': count, 'budget': budget, 'prune': prune,
                                'tasks': len(group), 'ceiling_hits': sum(r['found'] == r['ceiling'] for r in group),
                                'distinct': sum(r['found'] for r in group), 'mean_rr': mean(r['rr'] for r in group),
                                'mean_checks': mean(r['checks'] for r in group)})
    write_json(output/'observations.json', inputs)
    write_json(output/'validation.json', {'previous_source_sha': audit['source_sha256'],
               'input_sha': digest(output/'observations.json'),
               'code_hashes': {name: digest(HERE/name) for name in ('solve_partial_graph.py',
                   'run_partial_graph_pipeline.py', 'validate_pruned_graph_search.py')},
               'reference_budget': REFERENCE_BUDGET, 'summary': summary, 'records': records})
    old = next(r for r in summary if r['observations'] == 'all' and r['budget'] == 24 and not r['prune'])
    new = next(r for r in summary if r['observations'] == 'all' and r['budget'] == REFERENCE_BUDGET and r['prune'])
    reduction = 1 - new['mean_checks']/old['mean_checks']
    report = ['# 六节点因果图的观测剪枝验证', '',
              f"固定 8 次剪枝策略有 {new['ceiling_hits']}/{new['tasks']} 道题达到 8 槽覆盖上限，旧设置为 "
              f"{old['ceiling_hits']}/{old['tasks']}。每题平均检查从 {old['mean_checks']:.2f} 次降到 "
              f"{new['mean_checks']:.2f} 次，减少 {reduction:.2%}。", '',
              '覆盖数据集全部 510 道六节点题，其中 210 道为 1–3 次干预。每题最多输出 8 个不同合法图。固定首图构造及多样性排序，只调整观测剪枝与每槽检查上限。每个输出均独立核对无环、边数、观测和去重。RR 分母来自此前全文件审计的逐题答案数。', '',
              '若观测要求 A 到达 B，则所有合法图都排除 B→A。若要求 A 不到达 B，且根据正观测必有 A 到达 U 和 V 到达 B，那么 U→V 也可排除。两端允许与 A 或 B 相同，因此该规则也涵盖直接禁止的边。剪枝在首轮扩展之前完成，剩余候选继续进行完整验证。', '',
              '检查次数包含首图。平均 RR 先按题计算，再对题取宏平均。达到上限表示找到了 min(8, 合法图总数) 个图。', '',
              '| 干预次数 | 每槽检查上限 | 剪枝 | 达到 8 槽覆盖上限 | 不同合法图总数 | 平均 RR | 平均检查次数 |',
              '|---|---:|---|---:|---:|---:|---:|']
    for row in summary:
        report.append(f"| {row['observations']} | {row['budget']} | {'是' if row['prune'] else '否'} | "
                      f"{row['ceiling_hits']}/{row['tasks']} | {row['distinct']} | {row['mean_rr']:.4%} | "
                      f"{row['mean_checks']:.2f} |")
    report += ['', '运行 validate_pruned_graph_search.py 可复现固定预算对照。独立入口 solve_partial_graph.py 可用 --fixed-budget --no-pruning --checks-per-slot 24 重现无剪枝旧设置。全部任务采用六节点、无噪声干预观测。RR 分母来自已有全文件审计，当前运行独立验证所输出图的合法性。']
    (output/'REPORT.md').write_text('\n'.join(report)+'\n', encoding='utf-8')
    print(output.resolve())
    print(json.dumps([row for row in summary if row['observations'] in ('low', 'all')]))


if __name__ == '__main__':
    main()
