"""Frozen, paired validation of the slot-five one-edge repair policy."""
import argparse
import csv
from datetime import datetime
from hashlib import sha256
from itertools import combinations, permutations
import json
from pathlib import Path
import random
import re
from statistics import mean
import time

import networkx as nx
import requests

from common import DATA, HERE, digest, load_upstream
from evaluate_early_graph_repair import REPAIR_SLOT, repair_one_edge
from run_feedback_study import graph_line
from run_four_groups import BASE_URL, MODEL, SEEDS
from run_local_causal import write_json
from run_verified_graph_search import branch, generate_partial_baseline, read, replay_score
from stream_causal import check_graph
from evaluate_full_observation_reference import compatible_edge_sets

SELECTION_SEED = 1001126
METHODS = ('baseline', 'search', 'early_repair')
PROJECT = HERE/'runs/qwen3_project_evaluation_20261112_145049'
DEVELOPMENT = HERE/'runs/verified_graph_search_20261112_161255'


def observation_key(observations, nodes):
    return tuple(sorted((obs['perturbed_node'], tuple(obs['effects'][node] for node in nodes))
                        for obs in observations))


def fresh_datasets():
    rng = random.Random(SELECTION_SEED)
    original = read(HERE/'data/causal_n4_full_observations.json')
    project = read(PROJECT/'config.json')
    previous = read(DEVELOPMENT/'partial_test.json')['datasets']
    used4 = set(project['n4_excluded']) | {row['parent_id'] for row in previous}
    used4.update(item for group, item, *_ in project['cases'] if group == 'n4')
    used6 = {item for group, item, *_ in project['cases'] if group == 'n6'}
    nodes = original['metadata']['nodes']
    used_keys = {observation_key(row['observations'], nodes) for row in previous}
    possible = tuple(permutations(nodes, 2))
    dags = []
    for size in range(7):
        for edges in combinations(possible, size):
            graph = nx.DiGraph()
            graph.add_nodes_from(nodes)
            graph.add_edges_from(edges)
            if nx.is_directed_acyclic_graph(graph):
                reachable = {a: nx.descendants(graph, a) for a in nodes}
                dags.append((edges, {a: tuple(int(b in reachable[a]) for b in nodes) for a in nodes}))
    assert len(dags) == 543
    partial = []
    for n_obs in (2, 3):
        pool = [(row['observation_set_id'], obs) for row in original['datasets']
                if row['observation_set_id'] not in used4 for obs in combinations(row['observations'], n_obs)]
        rng.shuffle(pool)
        for parent, obs in pool:
            key = observation_key(obs, nodes)
            if key in used_keys or parent in used4:
                continue
            solutions = [edges for edges, effects in dags if all(effects[a] == values for a, values in key)]
            if len(solutions) < 4:
                continue
            partial.append({'observation_set_id': f'{parent}_obs{n_obs}', 'parent_id': parent,
                            'observations': list(obs), 'n_observations': n_obs,
                            'ground_truth_graphs': [{'nodes': nodes, 'edges': [list(e) for e in edges]}
                                                    for edges in solutions]})
            used4.add(parent)
            used_keys.add(key)
            if sum(row['n_observations'] == n_obs for row in partial) == 2:
                break
    assert len(partial) == 4
    path = DATA/'causal/datasets/node06/n6_all_observations.json'
    with path.open('rb') as stream:
        stream.seek(-16 * 1024**2, 2)
        tail = stream.read()
    marker, = re.finditer(rb'^    "6": (\[)', tail, re.M)
    block = tail[marker.start(1):].decode('utf-8')
    full_rows, end = json.JSONDecoder().raw_decode(block)
    assert len(full_rows) == 100
    assert sha256(block[:end].encode('utf-8')).hexdigest() == project['n6_exact_block_sha']
    meta6 = read(PROJECT/'n6_test.json')['metadata']
    full = []
    for low, high in ((2, 8), (9, 219)):
        pool = [row for row in full_rows if row['observation_set_id'] not in used6
                and low <= len(row['ground_truth_graphs']) <= high]
        full.extend(rng.sample(sorted(pool, key=lambda row: row['observation_set_id']), 2))
    for row in full:
        truth = {tuple(sorted(map(tuple, g['edges']))) for g in row['ground_truth_graphs']}
        exact = set(compatible_edge_sets(meta6['nodes'], meta6['max_edges'], row['observations']))
        assert exact == truth and len(exact) == len(row['ground_truth_graphs'])
        assert all(check_graph(g, row['observations'], meta6) for g in row['ground_truth_graphs'])
    return {'partial_n4': {'metadata': {**original['metadata'], 'max_edges': 6, 'total_observation_sets': 4},
                           'datasets': partial},
            'complete_n6': {'metadata': {**meta6, 'total_observation_sets': 4}, 'datasets': full}}


def evaluate(benchmark, instance, raw_calls):
    baseline, rows = replay_score(benchmark, instance, raw_calls)
    observations = instance['observations']
    searched, trace = branch(benchmark, observations, raw_calls)
    search, search_rows = replay_score(benchmark, instance, searched)
    prefix = rows[:REPAIR_SLOT-1]
    attempted = not any(row['valid'] for row in prefix)
    started = time.perf_counter()
    repaired, edit, checks = (repair_one_edge(benchmark, observations, prefix) if attempted
                              else (None, None, 0))
    elapsed = time.perf_counter()-started
    if repaired is None:
        early, early_rows, early_trace = search, search_rows, trace
    else:
        inputs = list(raw_calls)
        inputs[REPAIR_SLOT-1] = {'response': {'message': {'content': graph_line(repaired)},
                                             'done_reason': 'one_edge_repair'}, 'seconds': elapsed}
        proposals, early_trace = branch(benchmark, observations, inputs)
        early_trace[REPAIR_SLOT-1] = {'source': 'one_edge_repair', 'edit': edit, 'checks': checks}
        early, early_rows = replay_score(benchmark, instance, proposals)
        assert early_rows[REPAIR_SLOT-1]['new_valid']
    for selected, audit in ((search_rows, trace), (early_rows, early_trace)):
        assert len(selected) == len(audit) == 8
        for row, step in zip(selected, audit):
            if step['source'] in ('new_local_graph', 'one_edge_repair'):
                assert row['valid'] and row['new_valid']
    return {'scores': dict(baseline=baseline, search=search, early_repair=early),
            'rows': dict(baseline=rows, search=search_rows, early_repair=early_rows),
            'traces': dict(search=trace, early_repair=early_trace),
            'attempted': attempted, 'repaired': repaired is not None, 'repair_checks': checks,
            'repair_seconds': elapsed}


def report(output, records):
    metric_keys = ('recovery_rate', 'valid_rate', 'novelty_rate')
    summary = {}
    for group in ('all', 'partial_n4', 'complete_n6'):
        subset = [r for r in records if group == 'all' or r['group'] == group]
        summary[group] = {'tasks': len(subset), 'attempts': sum(r['attempted'] for r in subset),
                          'successes': sum(r['repaired'] for r in subset)}
        for method in METHODS:
            summary[group][method] = {m: mean(r['scores'][method][m] for r in subset) for m in metric_keys}
            summary[group][method]['distinct'] = sum(r['scores'][method]['n_recovered_gts'] for r in subset)
            summary[group][method]['model_calls'] = sum(8 if method == 'baseline' else
                sum(t['source'] == 'model_prefix' for t in r['traces'][method]) for r in subset)
        summary[group]['paired_early_vs_search'] = {}
        for m in metric_keys:
            differences = [r['scores']['early_repair'][m]-r['scores']['search'][m] for r in subset]
            summary[group]['paired_early_vs_search'][m] = [sum(d > 0 for d in differences),
                                                            sum(d == 0 for d in differences),
                                                            sum(d < 0 for d in differences)]
    write_json(output/'summary.json', summary)
    fields = ['group','item','seed','gt','method','query','answer','source','valid','new_valid','cumulative_recovered']
    with (output/'all_slots.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for r in records:
            for method in METHODS:
                for i, row in enumerate(r['rows'][method]):
                    source = 'model' if method == 'baseline' else r['traces'][method][i]['source']
                    writer.writerow({**{k: r[k] for k in fields[:4]}, **row, 'method': method, 'source': source})
    labels = {'all': '全部', 'partial_n4': '四节点部分观测', 'complete_n6': '六节点完整观测',
              'baseline': '原策略', 'search': '首个合法图后搜索', 'early_repair': '第5槽修复加搜索'}
    lines = ['# 提前单边修复的独立题集复核', '',
             '固定抽样种子 1001126，从既有实验未使用的题中选取 4 道四节点部分观测题和 4 道六节点完整观测题。四节点各有两题保留 2 条、3 条观测，每题至少有 4 个合法图。六节点各有两题落在 2–8、9–219 个合法图区间。每题一个固定种子。', '',
             '重新生成 64 个模型响应，并用相同前缀对照三个策略。原策略在前七槽都没找到合法图时，于第八槽使用思考。搜索组在首个合法图后使用已验证的单边邻居。提前修复组沿用开发阶段的规则：前四槽全错才在第 5 槽尝试一次增边或删边，修复成功后切换搜索。三组每题各提交 8 槽。确定性分支复用新生成的模型前缀，表中模型调用为每个策略需要的次数。', '',
             '| 题集 | 方法 | RR | VR | NR | 不同合法图总数 | 模型调用 |',
             '|---|---|---:|---:|---:|---:|---:|']
    for group, s in summary.items():
        for method in METHODS:
            d = s[method]
            lines.append(f"| {labels[group]} | {labels[method]} | {d['recovery_rate']:.2%} | {d['valid_rate']:.2%} | "
                         f"{d['novelty_rate']:.2%} | {d['distinct']} | {d['model_calls']} |")
    overall = summary['all']
    lines += ['', f"第 5 槽修复成功 {overall['successes']}/{overall['attempts']} 次。相对只做搜索，"
              + '、'.join(f"{label} 胜/平/负为 "+'/'.join(map(str, overall['paired_early_vs_search'][metric]))
                          for label, metric in zip(('RR','VR','NR'), metric_keys))+'。', '',
              '| 题目 | 合法图总数 | 原策略找到 | 首图后搜索找到 | 修复加搜索找到 | 修复成功 |',
              '|---|---:|---:|---:|---:|---|']
    for r in records:
        found = ' | '.join(str(r['scores'][m]['n_recovered_gts']) for m in METHODS)
        lines.append(f"| {r['item']} | {r['gt']} | {found} | {'是' if r['repaired'] else '否'} |")
    lines += ['', '官方评测与独立评分一致。全部选题和模型设置保存在 config.json，原始响应保存在分题目录，逐槽评分在 all_slots.csv。验证只涉及八题和每题一个采样种子。']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    datasets = fresh_datasets()
    cases = {group: [(row['observation_set_id'], len(row['ground_truth_graphs'])) for row in dataset['datasets']]
             for group, dataset in datasets.items()}
    if args.verify_only:
        print(json.dumps(cases))
        return
    output = args.output or HERE/'runs'/('early_graph_repair_validation_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.trust_env = False
    response = session.get(BASE_URL+'/api/tags', timeout=10)
    response.raise_for_status()
    model = next(row for row in response.json()['models'] if row['name'] == MODEL)
    assert model['digest'] == read(DEVELOPMENT/'config.json')['model']['digest']
    version = session.get(BASE_URL+'/api/version', timeout=10)
    version.raise_for_status()
    module = load_upstream('causal', 'run_causal_benchmark.py')
    config = {'selection_seed': SELECTION_SEED, 'repair_slot': REPAIR_SLOT, 'slots': 8,
              'model': model['digest'], 'ollama': version.json(), 'seeds': SEEDS,
              'cases': {key: [list(c) for c in values] for key, values in cases.items()},
              'source_hashes': {name: digest(HERE/name) for name in ('validate_early_graph_repair.py',
                  'evaluate_early_graph_repair.py','run_verified_graph_search.py')},
              'evaluator_sha': digest(module.__file__)}
    if (output/'config.json').exists():
        assert read(output/'config.json') == config
    else:
        write_json(output/'config.json', config)
    print('OUTPUT_DIR='+str(output.resolve()), flush=True)
    records = []
    for group, dataset in datasets.items():
        file = output/f'{group}_test.json'
        write_json(file, dataset)
        benchmark = module.CausalBenchmarkEnhanced(str(file))
        for index, instance in enumerate(benchmark.all_observation_sets):
            item = instance['observation_set_id']
            seed = SEEDS[index % len(SEEDS)]
            folder = output/group/item
            folder.mkdir(parents=True, exist_ok=True)
            calls = generate_partial_baseline(benchmark, instance, seed, folder, session)
            result = evaluate(benchmark, instance, calls)
            record = {'group': group, 'item': item, 'seed': seed,
                      'gt': len(instance['ground_truth_graphs']), **result}
            write_json(folder/'result.json', record)
            records.append(record)
            counts = [result['scores'][m]['n_recovered_gts'] for m in METHODS]
            print(f"{len(records)}/8 {item} {counts} of {record['gt']} repair={result['repaired']}", flush=True)
    assert len(list(output.glob('*/*/call_*.json'))) == 64
    report(output, records)


if __name__ == '__main__':
    main()
