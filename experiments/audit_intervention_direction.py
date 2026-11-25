"""Compare prediction diversity and selection of one more intervention."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from math import log2
from pathlib import Path
import re
from statistics import mean
from unittest.mock import patch

import networkx as nx

from common import DATA, HERE, digest
from run_local_causal import write_json
from run_partial_graph_pipeline import Frontier, edge_distance
from solve_partial_graph import solve
from stream_causal import check_graph
from validate_graph_rr_budget import DATA as OBSERVATIONS

NODES = list('ABCDEF')
METADATA = {'nodes': NODES, 'max_edges': 10}


def signature(edges, sources):
    graph = nx.DiGraph(edges)
    graph.add_nodes_from(NODES)
    return frozenset((a, b) for a in sources for b in nx.descendants(graph, a))


def prediction_shortlist(frontier):
    unknown = sorted(set(frontier.nodes)-{o['perturbed_node'] for o in frontier.observations})
    cache = frontier.__dict__.setdefault('_prediction_audit', {})

    def features(graph):
        if graph not in cache:
            cache[graph] = signature(graph, unknown)
        return cache[graph]

    seen = [features(graph) for graph in frontier.emitted]

    def rank(graph):
        current = features(graph)
        prediction_distances = [len(current ^ old) for old in seen]
        edge_distances = [edge_distance(graph, old) for old in frontier.emitted]
        return (-int(current not in seen), -min(prediction_distances), -sum(prediction_distances),
                -min(edge_distances), -sum(edge_distances), graph)

    return sorted(frontier.pool, key=rank)[:8]


def closure_bits(edges):
    reach = [0]*len(NODES)
    index = {node: i for i, node in enumerate(NODES)}
    for a, b in edges:
        reach[index[a]] |= 1 << index[b]
    for k in range(len(NODES)):
        for i in range(len(NODES)):
            if reach[i] & (1 << k):
                reach[i] |= reach[k]
    return reach


def entropy(counts):
    n = sum(counts.values())
    return -sum((value/n)*log2(value/n) for value in counts.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE/'runs/intervention_direction_20261113')
    args = parser.parse_args()
    tasks = json.loads(OBSERVATIONS.read_text(encoding='utf-8'))
    by_id = {r['id']: r for r in tasks}
    assert len(by_id) == 510
    program_graphs, diversity = {}, []
    for task in tasks:
        obs = task['observations']
        unknown = sorted(set(NODES)-{o['perturbed_node'] for o in obs})
        baseline = solve(NODES, 10, obs)
        with patch.object(Frontier, 'shortlist', prediction_shortlist):
            prediction = solve(NODES, 10, obs)
        counts = []
        for run in (baseline, prediction):
            graphs = [row['edges'] for row in run['proposals']]
            assert len(graphs) == len(set(graphs)) == min(8, task['n_ground_truths'])
            assert all(check_graph({'nodes': NODES, 'edges': graph}, obs, METADATA) for graph in graphs)
            counts.append(len({signature(graph, unknown) for graph in graphs}))
        program_graphs[task['id']] = [row['edges'] for row in baseline['proposals']]
        if unknown:
            diversity.append({'item': task['id'], 'observations': len(obs),
                              'edge_signatures': counts[0], 'prediction_signatures': counts[1],
                              'edge_checks': baseline['checks'], 'prediction_checks': prediction['checks']})

    source = DATA/'causal/datasets/node06/n6_all_observations.json'
    with source.open('rb') as stream:
        stream.seek(-96 * 1024**2, 2)
        tail = stream.read()
    active, source_blocks = [], {}
    for n in (4, 5):
        marker, = re.finditer(rb'^    "'+str(n).encode()+rb'": (\[)', tail, re.M)
        block = tail[marker.start(1):].decode('utf-8')
        rows, end = json.JSONDecoder().raw_decode(block)
        source_blocks[n] = sha256(block[:end].encode('utf-8')).hexdigest()
        assert len(rows) == 100
        for task in rows:
            item = task['observation_set_id']
            obs = task['observations']
            assert obs == by_id[item]['observations']
            unknown = sorted(set(NODES)-{o['perturbed_node'] for o in obs})
            proposed = [closure_bits(graph) for graph in program_graphs[item]]
            truth = [closure_bits(row['edges']) for row in task['ground_truth_graphs']]
            assert len(truth) == by_id[item]['n_ground_truths']
            sample = {a: Counter(graph[NODES.index(a)] for graph in proposed) for a in unknown}
            exact = {a: Counter(graph[NODES.index(a)] for graph in truth) for a in unknown}
            numerator = {a: sum(v*v for v in exact[a].values()) for a in unknown}
            remaining = {a: value/len(truth)**2 for a, value in numerator.items()}
            choice = max(unknown, key=lambda a: entropy(sample[a]))
            active.append({'item': item, 'observations': n, 'choice': choice,
                           'candidate_entropy': {a: entropy(sample[a]) for a in unknown},
                           'remaining_by_node': remaining, 'selected': remaining[choice],
                           'random': mean(remaining.values()), 'oracle': min(remaining.values()),
                           'choice_matters': len(set(numerator.values())) > 1})
    summary = {}
    for n in (4, 5):
        selected = [r for r in active if r['observations'] == n]
        summary[n] = {key: mean(r[key] for r in selected) for key in ('selected', 'random', 'oracle')}
        summary[n]['tasks'] = len(selected)
        summary[n]['choice_matters'] = sum(r['choice_matters'] for r in selected)
    gains = [r['prediction_signatures']-r['edge_signatures'] for r in diversity]
    diversity_summary = {k: mean(r[k] for r in diversity) for k in (
        'edge_signatures', 'prediction_signatures', 'edge_checks', 'prediction_checks')}
    diversity_summary['win_tie_loss'] = [sum(d>0 for d in gains), sum(d==0 for d in gains), sum(d<0 for d in gains)]
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output/'results.json', {'diversity_summary': diversity_summary,
        'intervention_summary': summary, 'diversity': diversity, 'interventions': active,
        'observations_sha': digest(OBSERVATIONS), 'source_blocks_sha': source_blocks,
        'code_sha': digest(__file__), 'solver_sha': digest(HERE/'solve_partial_graph.py')})
    lines = ['# 下一步优化方向的离线诊断', '',
             '使用现有六节点数据。候选生成只读取已知观测。所有输出图核验合法、互异，并达到各题 8 槽覆盖上限。'
             '本次为方向筛选，题目与先前程序评测重叠。', '',
             '第一项只改程序候选的排序。优先选择未观测干预预测签名不同的图，然后比较预测差异和边集差异。'
             f'410 道部分观测题的题均预测种类从 {diversity_summary["edge_signatures"]:.4f} 变为 '
             f'{diversity_summary["prediction_signatures"]:.4f}，逐题胜平负为 {diversity_summary["win_tie_loss"]}。', '',
             '第二项用原始 8 槽程序候选选择一个未观测干预。对每个可选节点，按候选图预测的完整后代集合分组，'
             '选择候选预测熵最大的节点，熵相同则按节点字母顺序。真值集合只用于事后计算干预结果的分组。'
             '若每个合法图作为隐藏真图的概率相等，则某干预的期望剩余比例为各输出分组比例的平方和。'
             '随机基线对未观测节点均匀选择，参考上限由真值集合挑出使该比例最低的节点。', '',
             '| 已观测干预数 | 题数 | 选择会影响剩余量的题 | 随机剩余 | 8 图分歧选择剩余 | 最优参考剩余 |',
             '|---:|---:|---:|---:|---:|---:|']
    for n, row in summary.items():
        lines.append(f"| {n} | {row['tasks']} | {row['choice_matters']} | {row['random']:.2%} | "
                     f"{row['selected']:.2%} | {row['oracle']:.2%} |")
    lines += ['', '下一轮优先检验低观测题上的干预选择。将已有观测固定后追加一次模拟干预，比较随机、'
              '程序候选预测分歧和 Qwen 选择。所有组共享同样的候选池、可选节点和额外观测预算。'
              '这是新增观测的扩展实验，HypoSpace 原始 RR 继续按固定观测单独报告。当前未测试真实干预，也未调用 Qwen。']
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(args.output.resolve(), diversity_summary, summary)


if __name__ == '__main__':
    main()
