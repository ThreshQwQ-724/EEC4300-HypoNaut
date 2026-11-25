"""Compare model and program starts, then isolate ranking from a shared start."""
import argparse
from collections import deque
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
from run_constraint_feedback import OPTIONS
from run_feedback_study import graph_line, observations_check
from run_four_groups import BASE_URL, MODEL, SEEDS
from run_local_causal import write_json
from run_verified_graph_search import generate_partial_baseline, read, replay_score
from stream_causal import check_graph
from validate_early_graph_repair import evaluate

PROJECT = HERE/'runs/qwen3_project_evaluation_20261112_145049'
TEST_SEED = 1002126
SLOTS = 8
CHECKS_PER_SLOT = 24
SHORTLIST = 8
METHODS = {
    'current': '现有提前修复＋单边搜索',
    'qwen_start_diverse': '沿用 Qwen 流程起点＋多起点多样性排序',
    'program_fifo': '程序首图＋按发现顺序',
    'program_diverse': '程序首图＋多样性排序',
    'program_qwen': '同一程序首图＋Qwen 选边',
}


def selected_dataset():
    source = DATA/'causal/datasets/node06/n6_all_observations.json'
    with source.open('rb') as stream:
        stream.seek(-16 * 1024**2, 2)
        tail = stream.read()
    marker, = re.finditer(rb'^    "5": (\[)', tail, re.M)
    block = tail[marker.start(1):].decode('utf-8')
    pool, end = json.JSONDecoder().raw_decode(block)
    assert len(pool) == 100
    meta = read(PROJECT/'n6_test.json')['metadata']
    rng = random.Random(TEST_SEED)
    selected = []
    for low, high in ((4, 8), (9, 32), (33, 1020)):
        matching = [r for r in pool if low <= len(r['ground_truth_graphs']) <= high]
        selected.extend(rng.sample(sorted(matching, key=lambda r: r['observation_set_id']), 2))
    for row in selected:
        truth = row['ground_truth_graphs']
        assert row['n_observations'] == 5 and len(truth) == row['n_compatible_graphs']
        assert all(check_graph(g, row['observations'], meta) for g in truth)
        assert len({tuple(sorted(map(tuple, g['edges']))) for g in truth}) == len(truth)
    return {'metadata': {**meta, 'total_observation_sets': len(selected)}, 'datasets': selected}, sha256(
        block[:end].encode('utf-8')).hexdigest()


def program_anchor(nodes, max_edges, observations):
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from((row['perturbed_node'], target)
                         for row in observations for target, required in row['effects'].items() if required)
    if not nx.is_directed_acyclic_graph(graph):
        return None
    edges = tuple(sorted(nx.transitive_reduction(graph).edges))
    if len(edges) > max_edges or observations_check(nodes, observations, edges):
        return None
    return edges


def edge_distance(left, right):
    return len(set(left) ^ set(right))


def forbidden_edges(observations):
    """Exclude edges that force a prohibited path or reverse a required path."""
    required = nx.DiGraph()
    for row in observations:
        source = row['perturbed_node']
        required.add_nodes_from(row['effects'])
        required.add_edges_from((source, target) for target, effect in row['effects'].items() if effect)
    downstream = {node: nx.descendants(required, node) | {node} for node in required}
    upstream = {node: nx.ancestors(required, node) | {node} for node in required}
    blocked = {(target, source) for source, targets in downstream.items() for target in targets
               if target != source}
    for row in observations:
        for target, effect in row['effects'].items():
            if not effect:
                blocked.update((a, b) for a in downstream[row['perturbed_node']]
                               for b in upstream[target] if a != b)
    return blocked


class Frontier:
    """Expand every emitted graph and retain its unsubmitted legal neighbors."""

    def __init__(self, nodes, max_edges, observations, anchor, *, prune=False):
        self.nodes, self.max_edges, self.observations = nodes, max_edges, observations
        blocked = forbidden_edges(observations) if prune else set()
        self.edges = tuple(edge for edge in permutations(sorted(nodes), 2) if edge not in blocked)
        self.seen = {anchor}
        self.emitted = [anchor]
        self.queue = deque([(anchor, iter(self.edges))])
        self.pool = {}

    def expand(self, budget=None):
        if budget is None:
            budget = CHECKS_PER_SLOT
        checks = 0
        while self.queue and checks < budget:
            parent, edges = self.queue[0]
            edge = next(edges, None)
            if edge is None:
                self.queue.popleft()
                continue
            candidate = tuple(sorted(set(parent) ^ {edge}))
            if candidate in self.seen or len(candidate) > self.max_edges:
                continue
            self.seen.add(candidate)
            checks += 1
            graph = nx.DiGraph()
            graph.add_nodes_from(self.nodes)
            graph.add_edges_from(candidate)
            if nx.is_directed_acyclic_graph(graph) and not observations_check(
                    self.nodes, self.observations, candidate):
                self.pool[candidate] = {'parent': parent, 'edge': edge,
                                        'action': 'remove' if edge in parent else 'add'}
        return checks

    def shortlist(self):
        def diversity(edges):
            distances = [edge_distance(edges, old) for old in self.emitted]
            return (-min(distances), -sum(distances), edges)
        return sorted(self.pool, key=diversity)[:SHORTLIST]

    def emit(self, candidate):
        edit = self.pool.pop(candidate)
        self.register(candidate)
        return edit

    def register(self, candidate):
        """Expand an already verified graph without checking it again."""
        self.seen.add(candidate)
        self.emitted.append(candidate)
        self.queue.append((candidate, iter(self.edges)))


def as_call(answer, seconds=0.0, response=None):
    response = response or {}
    return {'seconds': seconds, 'response': {'message': {'content': answer},
            'prompt_eval_count': response.get('prompt_eval_count', 0), 'eval_count': response.get('eval_count', 0),
            'done_reason': response.get('done_reason', 'program')}}


def rank_with_qwen(session, folder, slot, seed, observations, emitted, shortlist):
    candidates = sorted(shortlist)
    choices = '\n'.join(f'{i}. {graph_line(edges)}' for i, edges in enumerate(candidates, 1))
    known = '\n'.join(graph_line(edges) for edges in emitted)
    evidence = '\n'.join(row['string'] for row in observations)
    prompt = (f'Select the candidate that best expands the diversity of the already submitted DAGs '
              f'and is a promising starting point for further valid edge edits. Every candidate below '
              f'has already been validated against all observations. Answer only with one integer from '
              f'1 to {len(candidates)}.\nObservations:\n{evidence}\nSubmitted:\n{known}\nCandidates:\n{choices}')
    payload = {'model': MODEL, 'think': False, 'stream': False, 'keep_alive': '10m',
               'options': {**OPTIONS, 'seed': seed, 'num_predict': 32},
               'messages': [{'role': 'user', 'content': prompt}]}
    path = folder/f'rank_{slot:02d}.json'
    if path.exists():
        call = read(path)
        assert call['request'] == payload
    else:
        started = time.perf_counter()
        response = session.post(BASE_URL+'/api/chat', json=payload, timeout=(10, 180))
        response.raise_for_status()
        call = {'request': payload, 'response': response.json(), 'seconds': time.perf_counter()-started}
        write_json(path, call)
    raw = call['response']['message']['content'].strip()
    match = re.fullmatch(r'\d+', raw)
    if match and 1 <= int(raw) <= len(candidates):
        selected = candidates[int(raw)-1]
    else:
        selected = shortlist[0]
    return selected, call, bool(match and 1 <= int(raw) <= len(candidates))


def grow(benchmark, instance, prefix, anchor, ranker, session, folder, seed):
    observations = instance['observations']
    frontier = Frontier(benchmark.nodes, benchmark.max_edges, observations, anchor)
    calls = list(prefix)
    trace = []
    for slot in range(len(prefix)+1, SLOTS+1):
        started = time.perf_counter()
        checks = frontier.expand()
        shortlist = frontier.shortlist()
        ranking = None
        rank_valid = None
        edit = None
        if not shortlist:
            selected = anchor
        else:
            if ranker == 'fifo':
                selected = next(iter(frontier.pool))
            elif ranker == 'qwen' and len(shortlist) > 1:
                selected, ranking, rank_valid = rank_with_qwen(
                    session, folder, slot, seed, observations, frontier.emitted, shortlist)
            else:
                selected = shortlist[0]
            edit = frontier.emit(selected)
        calls.append(as_call(graph_line(selected), time.perf_counter()-started,
                             ranking['response'] if ranking else None))
        trace.append({'slot': slot, 'checks': checks, 'edit': edit,
                      'llm_ranked': ranking is not None, 'rank_valid': rank_valid,
                      'shortlist_size': len(shortlist), 'selected': selected,
                      'differs_from_diversity': bool(shortlist and selected != shortlist[0])})
    return calls, trace


def record(benchmark, instance, calls, trace, model_calls, checks):
    official, rows = replay_score(benchmark, instance, calls)
    valid = {}
    for row in rows:
        if row['new_valid']:
            parsed = benchmark.parse_llm_response(row['answer'])
            edges = tuple(sorted(parsed.edges))
            valid[edges] = None
    distances = [edge_distance(a, b) for a, b in combinations(valid, 2)]
    for step in trace:
        if step['edit']:
            assert rows[step['slot']-1]['new_valid']
    return {'scores': official, 'rows': rows, 'trace': trace, 'checks': checks,
            'model_calls': model_calls, 'output_tokens': sum(r['output_tokens'] for r in rows),
            'seconds': sum(r['seconds'] for r in rows),
            'edge_distance': mean(distances) if distances else None}


def evaluate_item(benchmark, instance, folder, session, seed):
    baseline = generate_partial_baseline(benchmark, instance, seed, folder, session)
    current = evaluate(benchmark, instance, baseline)
    early_rows = current['rows']['early_repair']
    early_trace = current['traces']['early_repair']
    scores = current['scores']['early_repair']
    methods = {'current': {'scores': scores, 'rows': early_rows, 'trace': early_trace,
                           'checks': sum(t.get('checks', 0) for t in early_trace) +
                                     (current['repair_checks'] if current['attempted'] and not current['repaired'] else 0),
                           'model_calls': sum(t['source']=='model_prefix' for t in early_trace),
                           'output_tokens': sum(r['output_tokens'] for r in early_rows),
                           'seconds': sum(r['seconds'] for r in early_rows) +
                                      (current['repair_seconds'] if current['attempted'] and not current['repaired'] else 0),
                           'edge_distance': None}}
    first = next((i for i, row in enumerate(early_rows) if row['valid']), None)
    if first is None:
        methods['qwen_start_diverse'] = methods['current']
    else:
        prefix = baseline[:first+1]
        repaired = early_trace[first]['source'] == 'one_edge_repair'
        if repaired:
            prefix[first] = as_call(early_rows[first]['answer'], current['repair_seconds'])
        anchor = tuple(sorted(benchmark.parse_llm_response(early_rows[first]['answer']).edges))
        calls, trace = grow(benchmark, instance, prefix, anchor, 'diverse', session, folder, seed)
        if current['attempted'] and not repaired:
            calls[-1] = {**calls[-1], 'seconds': calls[-1]['seconds'] + current['repair_seconds']}
        methods['qwen_start_diverse'] = record(benchmark, instance, calls, trace,
                                               len(prefix)-int(repaired), current['repair_checks'] +
                                               sum(t['checks'] for t in trace))
    started = time.perf_counter()
    anchor = program_anchor(benchmark.nodes, benchmark.max_edges, instance['observations'])
    elapsed = time.perf_counter()-started
    for ranker in ('fifo', 'diverse', 'qwen'):
        if anchor is None:
            calls, trace = [as_call('', elapsed)] + [as_call('') for _ in range(SLOTS-1)], []
        else:
            prefix = [as_call(graph_line(anchor), elapsed)]
            calls, trace = grow(benchmark, instance, prefix, anchor, ranker, session, folder, seed)
        methods['program_'+ranker] = record(benchmark, instance, calls, trace,
                                            sum(t['llm_ranked'] for t in trace),
                                            int(anchor is not None)+sum(t['checks'] for t in trace))
    assert methods['program_diverse']['rows'][0]['answer'] == methods['program_qwen']['rows'][0]['answer']
    return methods, None if first is None else first+1


def report(output, records):
    totals = {}
    for method in METHODS:
        rows = [r['methods'][method] for r in records]
        stats = {k: mean(row['scores'][k] for row in rows) for k in ('valid_rate','novelty_rate','recovery_rate')}
        stats.update({k: sum(row[k] for row in rows) for k in ('checks','model_calls','output_tokens','seconds')})
        stats['distinct'] = sum(row['scores']['n_recovered_gts'] for row in rows)
        distances = [row['edge_distance'] for row in rows if row['edge_distance'] is not None]
        stats['edge_distance'] = mean(distances) if distances else None
        totals[method] = stats
    capacity = sum(min(SLOTS, r['gt']) for r in records)
    ceiling = mean(min(SLOTS, r['gt'])/r['gt'] for r in records)
    comparisons = []
    for r in records:
        left = r['methods']['program_diverse']['scores']
        right = r['methods']['program_qwen']['scores']
        comparisons.append(right['n_recovered_gts'] - left['n_recovered_gts'])
    summary = {'methods': totals, 'capacity_distinct': capacity, 'rr_ceiling': ceiling,
               'paired_qwen_vs_diverse': [sum(d>0 for d in comparisons), sum(d==0 for d in comparisons),
                                         sum(d<0 for d in comparisons)]}
    write_json(output/'summary.json', summary)
    fields = ['item','seed','gt','method','query','answer','valid','new_valid','cumulative_recovered',
              'checks','edit','llm_ranked','rank_valid']
    with (output/'all_slots.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for r in records:
            for name, result in r['methods'].items():
                trace = {t['slot']: t for t in result['trace'] if 'slot' in t}
                for row in result['rows']:
                    writer.writerow({**{k:r[k] for k in fields[:3]}, **row, 'method': name,
                                     **trace.get(row['query'], {})})
    lines = ['# 六节点部分观测的图生成与选边对照', '',
             '固定抽样种子 1002126，抽取组员数据中五次干预的六道六节点题，按合法图数 4–8、9–32、33–1020 各两题，每题一个种子。每种方法提交 8 个候选。真值仅用于评分和选题分层。', '',
             '程序从已知的正可达关系构造 DAG 并做传递约简，再验证全部观测和边数上限。模型起点沿用已有的提前修复流程。搜索把所有已提交的合法图加入队列，逐个尝试单边增删，每槽至多检查 24 个新邻居，未提交的合法图留在候选池。多样性排序优先选与已提交图的最小边集距离更大的候选。Qwen 从同一规则产生的至多 8 个合法候选中选择，回答无效时采用程序排序。', '',
             '| 方法 | RR | VR | NR | 不同合法图总数 | 程序检查 | 模型调用 | 输出 token | 请求与搜索秒数 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, d in totals.items():
        lines.append(f"| {METHODS[name]} | {d['recovery_rate']:.2%} | {d['valid_rate']:.2%} | {d['novelty_rate']:.2%} | "
                     f"{d['distinct']} | {d['checks']} | {d['model_calls']} | {d['output_tokens']} | {d['seconds']:.2f} |")
    lines += ['', f'8 槽覆盖上限为 {capacity} 个不同合法图，宏平均 RR 上限为 {ceiling:.2%}。'
              '相同程序起点下，Qwen 选边相对程序多样性排序的逐题 RR 胜/平/负为 '
              + '/'.join(map(str, summary['paired_qwen_vs_diverse'])) + '。', '',
              '| 题目 | 合法图总数 | 现有流程首图槽位 | 现有 | Qwen 流程起点＋多样性 | 程序顺序 | 程序多样性 | Qwen 选边 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in records:
        counts = ' | '.join(str(r['methods'][name]['scores']['n_recovered_gts']) for name in METHODS)
        lines.append(f"| {r['item']} | {r['gt']} | {r['first_current'] or '无'} | {counts} |")
    lines += ['', '程序检查不计入提交槽位。三个程序起点分支共享起点、搜索规则和检查预算，排序可能改变后续候选池。现有流程和模型起点多样性组复用相同模型前缀。表内模型调用和时间按每种策略单独使用时计算，实验额外生成的对照后缀保存在原始记录中。不同合法图的平均边集距离和逐槽选边均已存档。', '',
              '本轮为六题、每题一个种子的构造验证。官方评分与独立可达性检查一致。用 run_partial_graph_pipeline.py --output 指向当前目录可复用已保存的响应。']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    dataset, block_hash = selected_dataset()
    cases = [[r['observation_set_id'], len(r['ground_truth_graphs'])] for r in dataset['datasets']]
    if args.verify_only:
        print(json.dumps(cases))
        return
    output = args.output or HERE/'runs'/('partial_graph_pipeline_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.trust_env = False
    response = session.get(BASE_URL+'/api/tags', timeout=10)
    response.raise_for_status()
    model = next(row for row in response.json()['models'] if row['name'] == MODEL)
    assert model['digest'] == read(PROJECT/'config.json')['model']['digest']
    version = session.get(BASE_URL+'/api/version', timeout=10)
    version.raise_for_status()
    mod = load_upstream('causal', 'run_causal_benchmark.py')
    config = dict(cases=cases, selection_seed=TEST_SEED, seeds=SEEDS, slots=SLOTS, checks_per_slot=CHECKS_PER_SLOT,
                  shortlist=SHORTLIST, model=model['digest'], ollama=version.json(), data_block_sha=block_hash,
                  runner_sha=digest(__file__), evaluator_sha=digest(mod.__file__),
                  helpers={name: digest(HERE/name) for name in ('run_verified_graph_search.py',
                    'validate_early_graph_repair.py', 'evaluate_early_graph_repair.py')})
    path = output/'config.json'
    if path.exists():
        assert read(path) == config
    else:
        write_json(path, config)
    print('OUTPUT_DIR='+str(output.resolve()), flush=True)
    data = output/'test.json'
    write_json(data, dataset)
    benchmark = mod.CausalBenchmarkEnhanced(str(data))
    records = []
    for i, instance in enumerate(benchmark.all_observation_sets):
        item = instance['observation_set_id']
        seed = SEEDS[i % len(SEEDS)]
        folder = output/item
        folder.mkdir(exist_ok=True)
        methods, first = evaluate_item(benchmark, instance, folder, session, seed)
        record_data = dict(item=item, seed=seed, gt=len(instance['ground_truth_graphs']),
                           first_current=first, methods=methods)
        write_json(folder/'result.json', record_data)
        records.append(record_data)
        print(f"{len(records)}/6 {item} first={first} " +
              str([methods[name]['scores']['n_recovered_gts'] for name in METHODS]), flush=True)
    report(output, records)


if __name__ == '__main__':
    main()
