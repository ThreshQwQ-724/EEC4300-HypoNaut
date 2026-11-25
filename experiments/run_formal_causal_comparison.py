"""Paired causal benchmark, with a separate accounting of search and Qwen work."""
import argparse
from itertools import permutations
import json
from pathlib import Path
import random
import re
from statistics import mean
from time import perf_counter

import networkx as nx
import requests

from common import HERE, digest, load_upstream
from evaluate_qwen_risk_trigger import risk
from run_constraint_feedback import OPTIONS
from run_four_groups import BASE_URL, MODEL, SEEDS, score_calls
from run_local_causal import write_json
from run_partial_graph_pipeline import forbidden_edges, program_anchor
from run_qwen_rr_budget import MODEL_REFERENCE, read
from run_qwen_tight_budget import LIMIT, MENU, NODES, SLOTS, neighbors, priority, qwen_choice
from solve_partial_graph import solve
from stream_causal import check_graph
from validate_graph_rr_budget import DATA
from validate_program_risk_search import evaluate

SEED = 1020126
METHODS = {
    'qwen_direct': 'Qwen 官方提示',
    'fixed_unpruned': '固定24次，无剪枝',
    'fixed_pruned': '固定8次，观测剪枝',
    'adaptive': '剪枝＋按需检查',
    'program_add': '程序增边优先',
    'program_risk': '增边＋程序避险',
    'delete_risk': '增边＋避险＋删边预判',
    'qwen_add': '增边＋Qwen 选边',
}
META = {'nodes': NODES, 'max_edges': LIMIT}


def select_tasks():
    excluded = set()
    for file in (HERE/'runs').glob('*/config.json'):
        if not file.parent.name.startswith('formal_causal_comparison'):
            excluded.update(re.findall(r'\bn[1-6]_\d{3}\b', file.read_text(encoding='utf-8')))
    rng = random.Random(SEED)
    tasks = read(DATA)
    return [task for n in range(1, 7) for task in rng.sample(sorted(
        (r for r in tasks if r['n_observations'] == n and r['id'] not in excluded
         and r['n_ground_truths'] >= SLOTS), key=lambda r: r['id']), 4)]


def valid(graph, observations):
    return check_graph({'nodes': NODES, 'edges': graph}, observations, META)


def deletion_safe(edit, candidate, observations):
    """Removing an edge cannot introduce cycles or forbidden reachability."""
    source, target = edit['edge']
    graph = nx.DiGraph(candidate)
    graph.add_nodes_from(NODES)
    checks = 1
    if nx.has_path(graph, source, target):
        return True, checks
    # Only observed positive reachability can be lost.
    for row in observations:
        required = {node for node, effect in row['effects'].items() if effect}
        if required:
            checks += 1
            if not required.issubset(nx.descendants(graph, row['perturbed_node'])):
                return False, checks
    return True, checks


def improved(task, anchor):
    obs = task['observations']
    blocked = forbidden_edges(obs)
    allowed = tuple(edge for edge in permutations(NODES, 2) if edge not in blocked)
    emitted, checked, closures, trace = [anchor], {anchor}, {}, []
    for slot in range(2, SLOTS+1):
        edits = neighbors(emitted, allowed, checked)
        menu = sorted(edits, key=lambda graph: (edits[graph]['action'] != 'add', priority(graph, emitted)))[:MENU]
        if menu and edits[menu[0]]['action'] == 'add':
            menu = [g for g in menu if edits[g]['action'] == 'add']
        chosen = menu[0] if menu else None
        probes, traversals = 0, 0
        safe = False
        for candidate in menu:
            probes += 1
            edit = edits[candidate]
            if edit['action'] == 'add':
                safe = risk(edit, obs, closures) is None
            else:
                safe, count = deletion_safe(edit, candidate, obs)
                traversals += count
            if safe:
                chosen = candidate
                break
        accepted = chosen is not None and valid(chosen, obs)
        assert not safe or accepted
        if chosen is not None:
            checked.add(chosen)
        if accepted:
            emitted.append(chosen)
        trace.append({'slot': slot, 'chosen': chosen, 'valid': accepted,
                      'action': edits[chosen]['action'] if chosen is not None else None,
                      'probes': probes, 'deletion_traversals': traversals})
    return emitted, len(checked)-1, trace, len(closures)


def direct(benchmark, task, seed, folder, session):
    obs, history, calls = task['observations'], [], []
    for slot in range(1, SLOTS+1):
        payload = {'model': MODEL, 'think': False, 'stream': False, 'keep_alive': '10m',
                   'options': {**OPTIONS, 'seed': seed},
                   'messages': [{'role': 'user', 'content': benchmark.create_prompt(obs, history)}]}
        path = folder/f'direct_{slot:02d}.json'
        if path.exists():
            call = read(path)
            assert call['request'] == payload
        else:
            started = perf_counter()
            response = session.post(BASE_URL+'/api/chat', json=payload, timeout=(10, 180))
            response.raise_for_status()
            call = {'request': payload, 'response': response.json(), 'seconds': perf_counter()-started}
            write_json(path, call)
        calls.append(call)
        parsed = benchmark.parse_llm_response(call['response']['message'].get('content', ''))
        if parsed is not None:
            history.append(parsed)
    graphs = {tuple(sorted(g.edges)) for g in history if benchmark.validate_hypothesis(g, obs)}
    rows, counts = score_calls(benchmark, {'observations': obs, 'ground_truth_graphs':
                                          [{'edges': g} for g in graphs]}, calls)
    assert counts['n_recovered_gts'] == len(graphs)
    return dict(found=len(graphs), vr=counts['n_valid']/SLOTS, nr=counts['n_unique_all']/SLOTS,
                checks=len(history), probes=0, closures=0, deletion_traversals=0, calls=len(calls),
                tokens=sum(r['output_tokens'] for r in rows), seconds=sum(r['seconds'] for r in rows),
                graphs=sorted(graphs), trace=rows)


def package(graphs, checks, seconds, **extra):
    return dict(found=len(graphs), vr=len(graphs)/SLOTS, nr=len(graphs)/SLOTS, graphs=graphs,
                checks=checks, seconds=seconds, probes=0, closures=0, deletion_traversals=0,
                calls=0, tokens=0) | extra


def run_case(benchmark, task, seed, folder, session):
    obs = task['observations']
    anchor = program_anchor(NODES, LIMIT, obs)
    assert anchor is not None and valid(anchor, obs)
    result = {'qwen_direct': direct(benchmark, task, seed, folder, session)}
    for name, budget, prune, adaptive in (
        ('fixed_unpruned', 24, False, False), ('fixed_pruned', 8, True, False), ('adaptive', 12, True, True)
    ):
        start = perf_counter()
        solved = solve(NODES, LIMIT, obs, anchor=anchor, checks_per_slot=budget, prune=prune, adaptive=adaptive)
        result[name] = package([row['edges'] for row in solved['proposals']],
                               solved['checks']-1, perf_counter()-start)
    for method in ('program_add', 'program_risk', 'qwen_add'):
        old = evaluate(task, anchor, method, session, folder, seed)
        graphs = [anchor] + [step['chosen'] for step in old['trace'] if step['valid']]
        result[method] = package(graphs, old['checks'], old['seconds'], probes=old['risk_probes'],
                                 closures=old['parent_closures'], calls=old['model_call'],
                                 tokens=old['output_tokens'], trace=old['trace'])
    start = perf_counter()
    graphs, checks, trace, closures = improved(task, anchor)
    result['delete_risk'] = package(graphs, checks, perf_counter()-start,
                                   probes=sum(row['probes'] for row in trace), closures=closures,
                                   deletion_traversals=sum(row['deletion_traversals'] for row in trace),
                                   trace=trace)
    for row in result.values():
        assert len(set(row['graphs'])) == row['found'] <= min(SLOTS, task['n_ground_truths'])
        assert all(valid(graph, obs) for graph in row['graphs'])
        row['rr'] = row['found']/task['n_ground_truths']
    return result


def report(output, records):
    summary = {}
    lines = ['# HypoSpace 六节点正式对照', '',
             '固定抽样种子 1020126。干预次数 1–6 各选四道题，每题至少 8 个合法图，排除先前模型实验配置中出现的题号。题目来自同一合成题库，其中程序全量诊断已经遍历过这批题。每题一个配对种子。标准答案数用于抽样和评分。', '',
             '每题 8 个提交槽。纯 Qwen 每槽生成一张图，使用官方历史提示和非思考模式，构成模型能力基线。其他方法共享程序首图，表中的检查数不含这次首图验证。固定预算与按需预算有不同的每槽上限。后四组共用每槽最多一次完整验证，风险预判、缓存闭包及删边可达性遍历另计。', '',
             '删边预判先找替代路径。如果原边起点仍能到达终点，原图的全部可达关系都保留。否则检查删边后是否仍满足正观测。删边不能引入新路径或环。通过预判的候选仍做完整验证。', '',
             '| 方法 | RR | VR | NR | 合法图数 | 满8图 | 完整检查 | 风险判断 | 父图闭包 | 删边遍历 | Qwen 调用 | 输出 token | 秒数 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for method, label in METHODS.items():
        runs = [r['runs'][method] for r in records]
        d = {key: mean(r[key] for r in runs) for key in ('rr', 'vr', 'nr')}
        d.update({key: sum(r[key] for r in runs) for key in (
            'found', 'checks', 'probes', 'closures', 'deletion_traversals', 'calls', 'tokens', 'seconds')})
        d['full'] = sum(r['found'] == SLOTS for r in runs)
        summary[method] = d
        lines.append(f"| {label} | {d['rr']:.4%} | {d['vr']:.2%} | {d['nr']:.2%} | {d['found']} | "
                     f"{d['full']}/{len(records)} | {d['checks']} | {d['probes']} | {d['closures']} | "
                     f"{d['deletion_traversals']} | {d['calls']} | {d['tokens']} | {d['seconds']:.3f} |")
    paired = {}
    for other in ('program_risk', 'program_add', 'qwen_add', 'adaptive'):
        delta = [r['runs']['delete_risk']['found']-r['runs'][other]['found'] for r in records]
        paired[other] = {'win_tie_loss': [sum(x>0 for x in delta), sum(x==0 for x in delta), sum(x<0 for x in delta)],
                         'rr_delta': summary['delete_risk']['rr']-summary[other]['rr']}
    failures = [{'id': r['id'], 'truth': r['truth'], 'found': r['runs']['delete_risk']['found'],
                 'failed_slots': [s['slot'] for s in r['runs']['delete_risk']['trace'] if not s['valid']]}
                for r in records if r['runs']['delete_risk']['found'] < SLOTS]
    ceiling = mean(SLOTS/r['truth'] for r in records)
    lines += ['', f'这批题 8 槽 RR 上限为 {ceiling:.4%}。删边预判相对上一版避险的逐题胜平负为 '
                  f"{paired['program_risk']['win_tie_loss']}。剩余失分题为 {len(failures)} 道。", '',
              '| 题目 | 干预次数 | 合法解数 | '+' | '.join(METHODS.values())+' |',
              '|---|---:|---:|'+'---:|'*len(METHODS)]
    for r in records:
        lines.append(f"| {r['id']} | {r['observations']} | {r['truth']:,} | "
                     +' | '.join(str(r['runs'][m]['found']) for m in METHODS)+' |')
    lines += ['', 'VR 为合法槽比例，NR 为不同可解析候选占槽比例。程序分支只提交通过验证的不同图，因此 VR 与 NR 相同。纯模型基线的解析失败不计入图验证，但占提交槽。RR 为每题合法覆盖比例的宏平均。运行时间含排序、检查与模型调用，不含共同首图构造。耗时为单次本机测量。', '',
              '完整逐槽记录、选题、代码哈希与原始模型调用均保存在运行目录。']
    write_json(output/'summary.json', {'methods': summary, 'ceiling_rr': ceiling, 'paired': paired, 'failures': failures})
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'methods': summary, 'ceiling_rr': ceiling, 'paired': paired, 'failures': failures}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE/'runs/formal_causal_comparison_20261113')
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    path = output/'config.json'
    all_tasks = {r['id']: r for r in read(DATA)}
    tasks = [all_tasks[key] for key in read(path)['cases']] if path.exists() else select_tasks()
    assert len(tasks) == len({r['id'] for r in tasks}) == 24
    session = requests.Session()
    session.trust_env = False
    response = session.get(BASE_URL+'/api/tags', timeout=10)
    response.raise_for_status()
    model = next(m for m in response.json()['models'] if m['name'] == MODEL)
    assert model['digest'] == read(MODEL_REFERENCE)['model']['digest']
    version = session.get(BASE_URL+'/api/version', timeout=10)
    version.raise_for_status()
    module = load_upstream('causal', 'run_causal_benchmark.py')
    test = output/'observations.json'
    write_json(test, {'metadata': {**META, 'total_observation_sets': len(tasks)},
                      'datasets': [{'observation_set_id': t['id'], 'observations': t['observations'],
                                    'ground_truth_graphs': []} for t in tasks]})
    benchmark = module.CausalBenchmarkEnhanced(str(test))
    config = dict(cases=[r['id'] for r in tasks], seed=SEED, seeds=SEEDS, methods=list(METHODS),
                  slots=SLOTS, menu=MENU, model=model['digest'], ollama=version.json(), options=OPTIONS,
                  source_sha=digest(DATA), evaluator_sha=digest(module.__file__),
                  code_hashes={name: digest(HERE/name) for name in (
                      'run_formal_causal_comparison.py','validate_program_risk_search.py',
                      'evaluate_qwen_risk_trigger.py','solve_partial_graph.py','run_qwen_tight_budget.py',
                      'run_partial_graph_pipeline.py','run_qwen_guided_edits.py')})
    if path.exists():
        assert read(path) == config
    else:
        write_json(path, config)
    print('OUTPUT_DIR='+str(output.resolve()), flush=True)
    records = []
    for i, task in enumerate(tasks):
        folder = output/task['id']
        folder.mkdir(exist_ok=True)
        seed = SEEDS[i % len(SEEDS)]
        result_file = folder/'result.json'
        if result_file.exists():
            row = read(result_file)
        else:
            runs = run_case(benchmark, task, seed, folder, session)
            row = {'id': task['id'], 'truth': task['n_ground_truths'], 'observations': task['n_observations'],
                   'seed': seed, 'runs': runs}
            write_json(result_file, row)
        records.append(row)
        print(f"{i+1}/{len(tasks)} {task['id']} " +
              str([row['runs'][m]['found'] for m in METHODS]), flush=True)
    assert len(list(output.glob('*/direct_*.json'))) == 24*SLOTS
    assert len(list(output.glob('*/qwen_*.json'))) == sum(r['runs']['qwen_add']['calls'] for r in records)
    report(output, records)


if __name__ == '__main__':
    main()
