"""Compare Qwen and program diversity ranking from a shared verified graph pool."""
import argparse
from itertools import combinations
from pathlib import Path
import random
import re
from statistics import mean
from time import perf_counter

import requests

from common import HERE, digest
from run_feedback_study import graph_line
from run_formal_agent_debate import REFERENCE, SECOND_AGENT_SEED_OFFSET
from run_formal_causal_comparison import DATA, LIMIT, MENU, NODES, SLOTS, program_anchor, read, valid
from run_four_groups import BASE_URL, MODEL
from run_local_causal import write_json
from run_partial_graph_pipeline import Frontier
from run_qwen_guided_edits import decode, request, schema, signature


POOL_LIMIT = 32
BUILD_BUDGET = 128
SELECTION_SEED = 11132029
METHODS = ('edge', 'diversity', 'agent_b')
LABELS = {'edge': '程序边集距离', 'diversity': '程序预测多样性', 'agent_b': 'Agent B 选编号'}


def make_pool(observations, anchor):
    frontier = Frontier(NODES, LIMIT, observations, anchor, prune=True)
    graphs = [anchor]
    probes = 0
    while len(graphs) < POOL_LIMIT:
        if frontier.queue and probes < BUILD_BUDGET:
            probes += frontier.expand(min(MENU, BUILD_BUDGET-probes))
        options = frontier.shortlist()
        if not options:
            if not frontier.queue or probes == BUILD_BUDGET:
                break
            continue
        graph = options[0]
        frontier.emit(graph)
        graphs.append(graph)
    assert all(valid(graph, observations) for graph in graphs)
    return graphs, probes


def distance(left, right):
    return sum(len(a ^ b) for a, b in zip(left, right, strict=True))


def choose(session, folder, slot, seed, unknown, predictions, history, menu):
    options = sorted(menu)

    def show(graph):
        return '; '.join(f'{node} reaches {", ".join(sorted(targets)) or "none"}'
                         for node, targets in zip(unknown, predictions[graph], strict=True))

    prompt = ('All candidate DAGs already satisfy the given observations and have different edge sets. '
              'Choose a candidate that adds a new joint prediction across the unobserved intervention nodes. '
              'Among candidates with a new joint prediction, maximize the minimum number of changed '
              'reachability bits relative to any already selected graph. Each source-target pair counts '
              'as one bit. Return only the JSON choice number.\nSelected predictions:\n'
              + '\n'.join(show(graph) for graph in history) + '\nCandidates:\n'
              + '\n'.join(f'{i}. {show(graph)}' for i, graph in enumerate(options, 1)))
    call = request(session, folder/f'B_{slot:02}.json', prompt, seed+SECOND_AGENT_SEED_OFFSET,
                   schema({'choice': {'type': 'integer', 'minimum': 1, 'maximum': len(options)}}))
    pick = decode(call).get('choice')
    parsed = type(pick) is int and 1 <= pick <= len(options)
    return (options[pick-1] if parsed else menu[0]), call, parsed


def run(graphs, predictions, unknown, method, session, folder, seed):
    start = perf_counter()
    emitted, remaining, trace = graphs[:1], set(graphs[1:]), []
    masks = {graph: set(graph) for graph in graphs}
    while remaining and len(emitted) < SLOTS:
        known = {predictions[graph] for graph in emitted}

        def rank(graph):
            edge_distances = [len(masks[graph] ^ masks[old]) for old in emitted]
            if method == 'edge':
                return -min(edge_distances), -sum(edge_distances), graph
            scores = [distance(predictions[graph], p) for p in known]
            return predictions[graph] in known, -min(scores), -sum(scores), -min(edge_distances), graph

        ordered = sorted(remaining, key=rank)
        program = ordered[0]
        selected, call, parsed = program, None, None
        if method == 'agent_b':
            # Offer one representative per prediction so identical choices do not crowd the menu.
            representatives = {}
            for graph in ordered:
                representatives.setdefault(predictions[graph], graph)
            menu = list(representatives.values())[:MENU]
            if len(menu) > 1:
                selected, call, parsed = choose(session, folder, len(emitted)+1, seed, unknown,
                                                 predictions, emitted, menu)
        trace.append({'selected': selected, 'program_choice': program,
                      'menu': menu if call else None, 'parsed': parsed,
                      'new_pattern': predictions[selected] not in known,
                      'program_new_pattern': predictions[program] not in known,
                      'nearest_distance': min(distance(predictions[selected], p) for p in known),
                      'program_nearest_distance': min(distance(predictions[program], p) for p in known),
                      'calls': int(call is not None), 'seconds': call['seconds'] if call else 0,
                      'tokens': call['response'].get('eval_count', 0) if call else 0})
        remaining.remove(selected)
        emitted.append(selected)
    assert len(emitted) == min(SLOTS, len(graphs)) == len(set(emitted))
    distances = [distance(predictions[a], predictions[b]) for a, b in combinations(emitted, 2)]
    return {'graphs': emitted, 'patterns': len({predictions[g] for g in emitted}),
            'pairwise_distance': mean(distances) if distances else 0, 'trace': trace,
            'calls': sum(s['calls'] for s in trace), 'tokens': sum(s['tokens'] for s in trace),
            'request_seconds': sum(s['seconds'] for s in trace), 'runner_seconds': perf_counter()-start}


def report(output, records, cohorts):
    lines = ['# Agent B 合法候选多样性对照', '',
             '每题先以程序构造首图，再用共享单边搜索预先收集至多 32 张合法图。'
             '三组用同一首图和同一候选池，各选至多 8 张不同合法图。'
             '程序边集距离组按边差异排序，程序预测组优先选新预测模式，再最大化与已提交图的最小可达性差异。'
             'B 从程序预测排序前 8 个不同模式的代表图中选编号，每槽至多调用一次。'
             '各组的已提交历史可随选择变化，候选池固定。', '',
             '预测模式是每个未观测干预节点的全部可达目标的组合。'
             '平均两图差异按未观测干预的可达性位计算，再按题平均。'
             '完全观测题没有未观测干预，单独计入正式 RR、VR、NR，不进入多样性均值。', '',
             '| 题集 | 方法 | 部分观测题 | 平均模式数 | 候选池内模式上限 | 平均两图差异 | 合法图 | RR | VR / NR | B 调用 | 请求秒数 |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    summary = {}
    for cohort, cases in cohorts.items():
        rows = [records[case] for case in cases]
        partial = [row for row in rows if row['unknown']]
        ceiling = mean(row['pattern_cap'] for row in partial)
        block = {}
        for method in METHODS:
            runs = [row['runs'][method] for row in rows]
            result = {'patterns': mean(row['runs'][method]['patterns'] for row in partial),
                      'distance': mean(row['runs'][method]['pairwise_distance'] for row in partial),
                      'found': sum(len(run['graphs']) for run in runs),
                      'rr': mean(len(run['graphs'])/row['truth'] for row, run in zip(rows, runs, strict=True)),
                      'vr': mean(len(run['graphs'])/SLOTS for run in runs),
                      'calls': sum(run['calls'] for run in runs),
                      'request_seconds': sum(run['request_seconds'] for run in runs)}
            block[method] = result
            lines.append(f"| {cohort} | {LABELS[method]} | {len(partial)} | {result['patterns']:.3f} | {ceiling:.3f} | "
                         f"{result['distance']:.3f} | {result['found']} | {result['rr']:.4%} | {result['vr']:.2%} | "
                         f"{result['calls']} | {result['request_seconds']:.2f} |")
        choices = [step for row in partial for step in row['runs']['agent_b']['trace'] if step['calls']]
        gaps = [row['runs']['agent_b']['patterns']-row['runs']['diversity']['patterns'] for row in partial]
        summary[cohort] = block | {'pattern_cap': ceiling, 'checks': sum(row['checks'] for row in rows),
                                   'pattern_wtl': [sum(g > 0 for g in gaps), sum(g == 0 for g in gaps), sum(g < 0 for g in gaps)],
                                   'different_choices': sum(s['selected'] != s['program_choice'] for s in choices),
                                   'lost_new_pattern': sum(s['program_new_pattern'] and not s['new_pattern'] for s in choices),
                                   'lower_nearest_distance': sum(s['nearest_distance'] < s['program_nearest_distance'] for s in choices)}
    for cohort, block in summary.items():
        lines += ['', f"{cohort}，B 相对程序预测排序的逐题模式数胜／平／负为 {' / '.join(map(str, block['pattern_wtl']))}。"
                  f"固定在 B 当时的选择状态，B 有 {block['lost_new_pattern']} 次选到旧模式而程序首选是新模式。"
                  f"共享建池执行 {block['checks']} 次探查（首图另计）。"]
    lines += ['', '独立样本排除了已有实验配置中的题号，程序全量诊断此前覆盖过这批题。'
              '模式上限只针对固定候选池。各组均只提交经验证且不同的图，因此 VR 和 NR 相同。', '']
    write_json(output/'summary.json', summary)
    (output/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    source = read(REFERENCE/'config.json')
    assert source['source_sha'] == digest(DATA)
    tasks = {task['id']: task for task in read(DATA)}
    path = output/'config.json'
    if path.exists():
        independent = read(path)['cohorts']['独立 16 题']
    else:
        seen = {case for f in (HERE/'runs').glob('*/config.json') if read(f).get('all_tasks') is not True
                for case in re.findall(r'\bn[1-6]_\d{3}\b', f.read_text(encoding='utf-8'))}
        rng = random.Random(SELECTION_SEED)
        independent = [task['id'] for count in (2, 3, 4, 5) for task in rng.sample(sorted(
            (t for t in tasks.values() if t['id'] not in seen and t['n_observations'] == count
             and t['n_ground_truths'] >= SLOTS), key=lambda task: task['id']), 4)]
    cohorts = {'正式 24 题': source['cases'], '独立 16 题': independent}
    cases = source['cases']+independent
    assert len(cases) == len(set(cases)) == 40
    config = {'cohorts': cohorts, 'cases': cases, 'data_sha': digest(DATA), 'model': MODEL,
              'model_digest': source['model'], 'slots': SLOTS, 'menu': MENU, 'pool_limit': POOL_LIMIT,
              'build_budget': BUILD_BUDGET, 'selection_seed': SELECTION_SEED,
              'code_sha': {name: digest(HERE/name) for name in (Path(__file__).name,
                  'run_partial_graph_pipeline.py', 'run_qwen_guided_edits.py')}}
    if path.exists() and read(path) != config:
        raise ValueError('Output directory contains a different experiment configuration')
    write_json(path, config)
    records = {}
    with requests.Session() as session:
        session.trust_env = False
        tags = session.get(BASE_URL+'/api/tags', timeout=10)
        tags.raise_for_status()
        assert next(row for row in tags.json()['models'] if row['name'] == MODEL)['digest'] == source['model']
        for i, case in enumerate(cases):
            folder = output/case
            folder.mkdir(exist_ok=True)
            result = folder/'result.json'
            if result.exists():
                row = read(result)
            else:
                task = tasks[case]
                obs = task['observations']
                anchor = program_anchor(NODES, LIMIT, obs)
                assert anchor is not None
                unknown = sorted(set(NODES)-{o['perturbed_node'] for o in obs})
                graphs, checks = make_pool(obs, anchor)
                predictions = {g: tuple(map(frozenset, signature(g, NODES, unknown))) for g in graphs}
                seed = source['seeds'][i % len(source['seeds'])]
                runs = {method: run(graphs, predictions, unknown, method, session, folder, seed) for method in METHODS}
                assert len(graphs) <= task['n_ground_truths']
                row = {'id': case, 'seed': seed, 'truth': task['n_ground_truths'], 'unknown': unknown,
                       'pool': graphs, 'checks': checks, 'runs': runs,
                       'pattern_cap': min(SLOTS, len(set(predictions.values())))}
                tmp = result.with_suffix('.tmp')
                write_json(tmp, row)
                tmp.replace(result)
            records[case] = row
            print(f"{i+1}/{len(cases)} {case} " + str([row['runs'][m]['patterns'] for m in METHODS]), flush=True)
    report(output, records, cohorts)


if __name__ == '__main__':
    main()
