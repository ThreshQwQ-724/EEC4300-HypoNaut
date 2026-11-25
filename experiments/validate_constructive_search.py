"""Audit constructive DAG search against the existing adaptive solver."""
import argparse
from pathlib import Path
from statistics import mean, median
from time import perf_counter

from common import HERE, digest, load_upstream
from run_feedback_study import graph_line
from run_formal_agent_debate import REFERENCE
from run_formal_causal_comparison import DATA, LIMIT, NODES, SLOTS, read
from run_local_causal import write_json
from solve_partial_graph import solve


METHODS = {'adaptive': '按需单边搜索', 'constructive': '约束构造＋按需回退'}
REPEATS = 7


def measure(task, constructive, benchmark):
    observations = task['observations']
    times, checked_times = [], []
    for _ in range(REPEATS):
        start = perf_counter()
        result = solve(NODES, LIMIT, observations, constructive=constructive)
        elapsed = perf_counter()-start
        graphs = [proposal['edges'] for proposal in result['proposals']]
        for graph in graphs:
            parsed = benchmark.parse_llm_response(graph_line(graph))
            assert parsed is not None and tuple(sorted(parsed.edges)) == graph
            assert benchmark.validate_hypothesis(parsed, observations)
        checked_times.append(perf_counter()-start)
        times.append(elapsed)
    found = len(graphs)
    assert found == len(set(graphs)) <= min(SLOTS, task['n_ground_truths'])
    slots = result['proposal_slots']
    checks = result['neighbor_checks_by_slot']
    assert slots == sorted(set(slots)) and len(slots) == found and slots[-1] <= SLOTS
    assert result['checks'] == 1+sum(checks)
    return {'found': found, 'rr': found/task['n_ground_truths'], 'vr': found/SLOTS, 'nr': found/SLOTS,
            'checks': result['checks'], 'seconds': median(times), 'verified_seconds': median(checked_times),
            'constructed': result.get('constructed', 0), 'orders': result.get('construction_orders', 0),
            'construction_candidates': result.get('construction_candidates', 0),
            'fallback_used': result.get('fallback_used', False), 'slots': slots,
            'neighbor_checks_by_slot': checks, 'graphs': graphs}


def report(output, cohorts, records, historical):
    lines = ['# HypoSpace 构造式搜索效率复核', '',
             '对合法首图取拓扑顺序。新增边只沿顺序向前，避免环路。'
             '对每条新增边 u→v，凡是观测中能到达 u 的干预源（包括 u 自身是干预源时），都必须允许到达 v。'
             '这条局部条件沿路径逐边传递。保留首图的所有边也保留了原有正观测。'
             '构造到提交上限即停止。部分观测最多枚举 32 个拓扑顺序，候选不足时继续用单边搜索。', '',
             '两组共享题目、默认边数限制和 8 槽预算。完整图检查数含一次首图合法性检查及搜索中做的候选检查。'
             '构造出的图由规则保证合法，构造次数单列。计时包括首图构造，逐题运行七次取中位数后相加。'
             '含复核耗时还计入对每张输出图调用原始评测的解析与验证。全程使用零次模型调用。', '',
             '| 题集 | 方法 | 合法图 | RR | VR / NR | 完整检查 | 构造枚举 | 拓扑顺序 | 回退题 | 求解秒数 | 含复核秒数 |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    summary = {}
    for label, cases in cohorts.items():
        block = {}
        for method in METHODS:
            selected = [records[case][method] for case in cases]
            stats = {key: sum(row[key] for row in selected) for key in (
                'found', 'checks', 'constructed', 'orders', 'construction_candidates',
                'fallback_used', 'seconds', 'verified_seconds')}
            stats.update({key: mean(row[key] for row in selected) for key in ('rr', 'vr', 'nr')})
            block[method] = stats
            lines.append(f"| {label} | {METHODS[method]} | {stats['found']} | {stats['rr']:.4%} | {stats['vr']:.2%} | "
                         f"{stats['checks']} | {stats['construction_candidates']} | {stats['orders']} | "
                         f"{stats['fallback_used']} | {stats['seconds']:.4f} | {stats['verified_seconds']:.4f} |")
        block['full_check_reduction'] = 1-block['constructive']['checks']/block['adaptive']['checks']
        summary[label] = block
    lines += ['', f"历史正式 24 题的双 Qwen 编号版提交 {historical['found']} 张图，模型调用 {historical['calls']} 次，"
                  f"运行 {historical['seconds']:.2f} 秒。历史耗时包含模型请求，且使用与本次不同的计时口径。", '',
              '510 题包含正式 24 题。合法解数量只用于评分和核对，构造与回退只使用题目观测。'
              'VR 和 NR 相同，因为只提交不同的合法图。', '', '```powershell',
              "& 'D:\\Anaconda\\python.exe' validate_constructive_search.py --output runs/constructive_search_20261113",
              '```', '']
    write_json(output/'summary.json', {'cohorts': summary, 'historical_dual_qwen': historical})
    (output/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = read(REFERENCE/'config.json')
    assert source['source_sha'] == digest(DATA)
    module = load_upstream('causal', 'run_causal_benchmark.py')
    assert source['evaluator_sha'] == digest(module.__file__)
    benchmark = module.CausalBenchmarkEnhanced(str(REFERENCE/'observations.json'))
    tasks = read(DATA)
    cohorts = {'正式 24 题': source['cases'], '完整 510 题': [task['id'] for task in tasks]}
    assert len(set(cohorts['完整 510 题'])) == len(tasks) == 510
    assert set(source['cases']) <= set(cohorts['完整 510 题']) and len(source['cases']) == 24
    history = HERE/'runs/numbered_agent_search_formal_20261113'
    assert read(history/'config.json')['cases'] == source['cases']
    historical = read(history/'summary.json')['methods']['dual']
    records = {}
    for index, task in enumerate(tasks):
        order = tuple(METHODS) if index % 2 == 0 else tuple(reversed(METHODS))
        row = {method: measure(task, method == 'constructive', benchmark) for method in order}
        assert all(r['found'] == min(SLOTS, task['n_ground_truths']) for r in row.values()), task['id']
        records[task['id']] = row
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'config.json', {'cohorts': cohorts, 'repeats': REPEATS, 'slots': SLOTS,
                                     'data_sha': digest(DATA), 'evaluator_sha': digest(module.__file__),
                                     'historical_sha': digest(history/'summary.json'),
                                     'code_sha': {name: digest(HERE/name) for name in (Path(__file__).name,
                                         'solve_partial_graph.py', 'run_partial_graph_pipeline.py')}})
    write_json(output/'results.json', records)
    report(output, cohorts, records, historical)
    print(output.resolve())


if __name__ == '__main__':
    main()
