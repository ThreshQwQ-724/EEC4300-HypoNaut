"""Replay solver outputs in the official scorer and audit against the raw answer file."""
import argparse
from collections import Counter
from pathlib import Path
from statistics import mean
from time import perf_counter

from common import COMMIT, DATA, HERE, Replay, digest, load_upstream
import ijson
from run_feedback_study import graph_line
from run_local_causal import write_json
from run_qwen_rr_budget import read
from solve_partial_graph import solve
from stream_causal import HashReader


RAW = DATA/'causal/datasets/node06/n6_all_observations.json'
FOUR_NODE = HERE/'data/causal_n4_full_observations.json'
METHODS = {'adaptive': '按需单边搜索', 'constructive': '约束构造＋回退'}
SLOTS = 8


def read_object(events):
    builder = ijson.common.ObjectBuilder()
    builder.event('start_map', None)
    depth = 1
    while depth:
        _, event, value = next(events)
        builder.event(event, value)
        depth += (event in ('start_map', 'start_array')) - (event in ('end_map', 'end_array'))
    return builder.value


def solve_case(row, nodes, limit):
    output = {}
    for method in METHODS:
        started = perf_counter()
        result = solve(nodes, limit, row['observations'], slots=SLOTS, constructive=method == 'constructive')
        elapsed = perf_counter()-started
        slots = result.get('proposal_slots', [])
        graphs = [proposal['edges'] for proposal in result['proposals']]
        assert len(slots) == len(graphs) == len(set(graphs)) and slots == sorted(set(slots))
        output[method] = {'graphs': graphs, 'slots': slots, 'checks': result['checks'],
                          'fallback': result.get('fallback_used', False), 'seconds': elapsed}
    return output


def score_case(benchmark, row, outputs, gt_count, listed_count):
    for result in outputs.values():
        responses = ['']*SLOTS
        for slot, graph in zip(result['slots'], result['graphs'], strict=True):
            responses[slot-1] = graph_line(graph)
        replay = Replay(responses)
        metrics = benchmark.evaluate_single_observation_set(replay, row, SLOTS, False, max_retries=1)
        assert replay.calls == SLOTS
        assert metrics['n_valid'] == metrics['n_unique_all'] == len(result['graphs'])
        assert metrics['n_recovered_gts'] == len(result['graphs']), row['observation_set_id']
        result.update(found=metrics['n_recovered_gts'], vr=metrics['valid_rate'], nr=metrics['novelty_rate'],
                      rr=metrics['n_recovered_gts']/gt_count)
    return {'id': row['observation_set_id'], 'observations': len(row['observations']),
            'unique_truth': gt_count, 'listed_truth': listed_count, 'methods': outputs}


def raw_cases(module, solve_row=solve_case):
    benchmark = module.CausalBenchmarkEnhanced()
    with RAW.open('rb') as stream:
        reader = HashReader(stream)
        events = iter(ijson.parse(reader))
        for prefix, event, _ in events:
            if prefix == 'metadata' and event == 'start_map':
                metadata = read_object(events)
                nodes = benchmark.nodes = metadata['nodes']
                limit = benchmark.max_edges = metadata['max_edges']
                bits = {(a,b): 1 << (i*len(nodes)+j) for i,a in enumerate(nodes) for j,b in enumerate(nodes)}
            if not (prefix.startswith('datasets_by_n_observations.') and prefix.count('.') == 2
                    and event == 'start_map'):
                continue
            row_builder = ijson.common.ObjectBuilder()
            row_builder.event('start_map', None)
            for field, kind, value in events:
                if field == prefix+'.ground_truth_graphs' and kind == 'start_array':
                    row = row_builder.value
                    outputs = solve_row(row, nodes, limit)
                    wanted = {sum(bits[e] for e in g): g for result in outputs.values() for g in result['graphs']}
                    masks, matches, entries = set(), set(), 0
                    graph_item = field+'.item'
                    edge_item = graph_item+'.edges.item.item'
                    mask = 0
                    source = None
                    for path, token, data in events:
                        if path == edge_item:
                            if source is None:
                                source = data
                            else:
                                mask |= bits[source, data]
                                source = None
                        elif path == graph_item and token == 'end_map':
                            assert source is None
                            entries += 1
                            masks.add(mask)
                            if mask in wanted:
                                matches.add(mask)
                            mask = 0
                        elif path == field and token == 'end_array':
                            break
                    row_builder.event('map_key', 'ground_truth_graphs')
                    row_builder.event('start_array', None)
                    row_builder.event('end_array', None)
                    row['ground_truth_graphs'] = [{'nodes': nodes, 'edges': wanted[key]} for key in sorted(matches)]
                else:
                    row_builder.event(kind, value)
                if field == prefix and kind == 'end_map':
                    break
            assert entries == row['n_compatible_graphs'] and masks
            yield score_case(benchmark, row, outputs, len(masks), entries)
        assert reader.bytes_read == RAW.stat().st_size
        yield {'raw_sha256': reader.hasher.hexdigest(), 'metadata': metadata}


def summarize(rows):
    return {method: {**{key: sum(r['methods'][method][key] for r in rows)
                       for key in ('found', 'checks', 'fallback', 'seconds')},
                     **{key: mean(r['methods'][method][key] for r in rows) for key in ('rr', 'vr', 'nr')},
                     'cap_hits': sum(r['methods'][method]['found'] == min(SLOTS, r['unique_truth']) for r in rows)}
            for method in METHODS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    module = load_upstream('causal', 'run_causal_benchmark.py')
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    four = module.CausalBenchmarkEnhanced(str(FOUR_NODE))
    four_rows = []
    for row in four.all_observation_sets:
        results = solve_case(row, four.nodes, four.max_edges)
        labels = row['ground_truth_graphs']
        count = len({tuple(sorted(map(tuple, g['edges']))) for g in labels})
        four_rows.append(score_case(four, row, results, count, len(labels)))
    assert len(four_rows) == 219
    print('Four-node generated HypoSpace cases scored: 219', flush=True)
    six_rows = []
    for row in raw_cases(module):
        if 'raw_sha256' in row:
            provenance = row
        else:
            six_rows.append(row)
            if len(six_rows) % 25 == 0:
                print(f'Raw six-node cases scored: {len(six_rows)}/510', flush=True)
    assert len(six_rows) == len({r['id'] for r in six_rows}) == provenance['metadata']['total_observation_sets'] == 510
    audit = read(HERE/'results/causal_audit.json')
    assert provenance['raw_sha256'] == audit['source_sha256']
    assert {r['id']:r['listed_truth'] for r in six_rows} == {r['id']:r['n_ground_truths'] for r in audit['instances']}
    cohorts = {'原始六节点 510 题': six_rows, '四节点完全观测 219 题': four_rows}
    summary = {label: summarize(rows) for label, rows in cohorts.items()}
    differences = {label: dict(Counter(r['methods']['constructive']['found']-r['methods']['adaptive']['found']
                                     for r in rows)) for label, rows in cohorts.items()}
    write_json(output/'results.json', {'upstream_commit': COMMIT, 'evaluator_sha': digest(module.__file__),
        'solver_sha': digest(HERE/'solve_partial_graph.py'), 'runner_sha': digest(__file__),
        'four_node_sha': digest(FOUR_NODE), **provenance, 'slots': SLOTS, 'cohorts': cohorts,
        'summary': summary, 'paired_found_deltas': differences})
    lines = ['# HypoSpace 构造式搜索原始数据复核', '',
             '每题固定 8 个提交槽。六节点组流式读取原始 HypoSpace 文件，直接从文件中的观测求解，'
             '再逐项扫描所有标准答案，用边集核对每张输出的图。'
             '四节点组使用工作区已有的 HypoSpace 生成器完全观测数据，直接传完整答案集给官方评分器。', '',
             '两个题集均复用原始评分器的单题评测入口。六节点组仅把原始答案中匹配到的图传给评分器，'
             '其余答案在流式扫描中用边集去重后计数。RR 用官方评测返回的命中数除以这个完整的去重计数。'
             '原始答案不进入求解过程。', '',
             '| 题集 | 方法 | 不同合法图 | RR | VR / NR | 达到提交上限 | 完整检查 | 求解秒数 |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for label, methods in summary.items():
        for method, d in methods.items():
            lines.append(f"| {label} | {METHODS[method]} | {d['found']:,} | {d['rr']:.4%} | {d['vr']:.2%} | "
                         f"{d['cap_hits']}/{len(cohorts[label])} | {d['checks']:,} | {d['seconds']:.4f} |")
    lines += ['', '时间仅含本轮求解，不含扫描原始数据和事后评分。完整检查含首图和回退检查。'
              '四节点组每题都是完全观测，只用于检验尺寸迁移。两种方法均无模型调用。', '']
    (output/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(summary, flush=True)


if __name__ == '__main__':
    main()
