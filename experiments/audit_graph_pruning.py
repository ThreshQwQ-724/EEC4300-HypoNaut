"""Audit observation-based edge pruning on the six-node 4–6 observation blocks."""
import argparse
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import re
from statistics import mean
from unittest.mock import patch

from common import DATA, HERE, digest
import run_partial_graph_pipeline as pipeline
from run_local_causal import write_json


def forbidden_edges(observations):
    """A negative effect forbids a direct edge; a positive effect forbids its reverse."""
    return {(target, row['perturbed_node']) if effect else (row['perturbed_node'], target)
            for row in observations for target, effect in row['effects'].items()
            if target != row['perturbed_node']}


def evaluate(nodes, observations, anchor, budget, prune):
    frontier = pipeline.Frontier(nodes, 10, observations, anchor)
    if prune:
        blocked = forbidden_edges(observations)
        assert not blocked.intersection(anchor)
        frontier.edges = tuple(edge for edge in frontier.edges if edge not in blocked)
        frontier.queue[0] = (anchor, iter(frontier.edges))
    checks = 1
    with patch.object(pipeline, 'CHECKS_PER_SLOT', budget):
        for _ in range(7):
            checks += frontier.expand()
            choices = frontier.shortlist()
            if choices:
                frontier.emit(choices[0])
    assert len(frontier.emitted) == len(set(frontier.emitted))
    return frontier.emitted, checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or HERE/'runs'/('graph_pruning_audit_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True, exist_ok=True)
    source = DATA/'causal/datasets/node06/n6_all_observations.json'
    with source.open('rb') as stream:
        stream.seek(-96 * 1024**2, 2)
        tail = stream.read()
    nodes = list('ABCDEF')
    records = []
    blocks = {}
    for count in (4, 5, 6):
        marker, = re.finditer(rb'^    "'+str(count).encode()+rb'": (\[)', tail, re.M)
        raw = tail[marker.start(1):].decode('utf-8')
        tasks, end = json.JSONDecoder().raw_decode(raw)
        blocks[count] = sha256(raw[:end].encode('utf-8')).hexdigest()
        assert len(tasks) == 100
        for task in tasks:
            obs = task['observations']
            truth = {tuple(sorted(map(tuple, graph['edges']))) for graph in task['ground_truth_graphs']}
            assert len(obs) == count and len(truth) == task['n_compatible_graphs']
            anchor = pipeline.program_anchor(nodes, 10, obs)
            assert anchor in truth
            cap = min(8, len(truth))
            for budget in (8, 16, 24):
                for prune in (False, True):
                    graphs, checks = evaluate(nodes, obs, anchor, budget, prune)
                    assert set(graphs) <= truth and len(graphs) <= cap
                    records.append(dict(item=task['observation_set_id'], observations=count, gt=len(truth),
                                        budget=budget, prune=prune, checks=checks, ceiling=cap,
                                        found=len(graphs), rr=len(graphs)/len(truth)))
    summary = []
    for count in (4, 5, 6, 'all'):
        for budget in (8, 16, 24):
            for prune in (False, True):
                selected = [row for row in records if (count == 'all' or row['observations'] == count)
                            and row['budget'] == budget and row['prune'] == prune]
                summary.append(dict(observations=count, budget=budget, prune=prune, tasks=len(selected),
                                    ceiling_hits=sum(row['found'] == row['ceiling'] for row in selected),
                                    rr=mean(row['rr'] for row in selected),
                                    mean_checks=mean(row['checks'] for row in selected)))
    write_json(output/'audit.json', {'blocks_sha256': blocks, 'source_bytes': source.stat().st_size,
                                    'source_hashes': {name: digest(HERE/name) for name in (
                                        'audit_graph_pruning.py', 'run_partial_graph_pipeline.py')},
                                    'summary': summary, 'records': records})
    before = next(row for row in summary if row['observations'] == 'all' and row['budget'] == 24 and not row['prune'])
    after = next(row for row in summary if row['observations'] == 'all' and row['budget'] == 8 and row['prune'])
    lines = [
        '# 六节点因果图优化诊断', '',
        '复查组员数据中 4、5、6 次干预的全部 300 道六节点题。这是开发诊断，其中包含昨天用过的题。每题最多提交 8 个不同合法图，邻居检查预算在每槽 8、16、24 三档间变化。首图和所有提交均与标准图集合核对，候选生成只读取观测。', '',
        f"原方案（每槽 24 次检查）有 {before['ceiling_hits']}/300 题达到覆盖上限，平均每题检查 {before['mean_checks']:.2f} 次。剪枝方案（每槽 8 次检查）有 {after['ceiling_hits']}/300 题达到上限，平均每题检查 {after['mean_checks']:.2f} 次。检查次数含首图校验。", '',
        '若干预 A 时 B 不受影响，A→B 违反观测。若 B 受影响，B→A 会与必需的 A 到 B 路径形成环。搜索可跳过这两类边，最终仍验证无环、边数上限及全部观测。', '',
        '| 观测数 | 每槽检查上限 | 剪枝 | 达到覆盖上限 | 平均 RR | 每题平均检查 |',
        '|---|---:|---|---:|---:|---:|',
    ]
    for row in summary:
        lines.append(f"| {row['observations']} | {row['budget']} | {'是' if row['prune'] else '否'} | "
                     f"{row['ceiling_hits']}/{row['tasks']} | {row['rr']:.2%} | {row['mean_checks']:.2f} |")
    lines += [
        '', '### 下一轮验证重点', '',
        '优先把安全剪枝接入默认搜索，在 1–3 次干预题与更紧的检查预算上复核。除 RR 外，报告达到覆盖上限的题数、不同合法图数量和边集距离。覆盖上限是 min(8, 合法图总数)，只用于评分。', '',
        '本数据设定的六节点首图有结构保证。把每条正观测写成一条可达约束，得到的约束图是任一真实解可达关系的子图，因而无环。传递约简保留全部正观测，也不会产生真实解里不存在的路径。约简图的无向版本没有三角形，否则三角形的一条边可由另外两条边替代。六节点无三角形图最多 9 条边，满足数据集的 10 边上限。这个推导适用于一致、无噪声的观测。', '',
        '如要检验 Qwen 的额外作用，可在固定首图、检查次数和 8 个提交槽位的条件下，比对更紧预算中的程序排序与 Qwen 排序。若引入噪声观测或更严格的边数约束，应独立报告改动后的任务设定。'
    ]
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(output.resolve())
    print(json.dumps({'original_24': before, 'pruned_8': after}))


if __name__ == '__main__':
    main()
