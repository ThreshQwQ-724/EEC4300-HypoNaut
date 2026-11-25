"""Compare Qwen edge edits and candidate selection with a shared legal anchor."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time
from statistics import mean

import networkx as nx
import requests

from common import HERE, digest, load_upstream
from run_constraint_feedback import OPTIONS
from run_feedback_study import graph_line, observations_check
from run_four_groups import BASE_URL, MODEL, SEEDS
from run_local_causal import write_json
from run_partial_graph_pipeline import Frontier, forbidden_edges, program_anchor
from run_qwen_rr_budget import select, score, read, MODEL_REFERENCE, REFERENCE
from solve_partial_graph import CHECK_BATCH_SIZE, DEFAULT_CHECKS_PER_SLOT

SLOTS = 32
MILESTONES = (8, 16, 32)
PREVIOUS = HERE/'runs/qwen_rr_budget_20261113_160444'
METHODS = {'edits': 'Qwen 单边提议', 'rank': 'Qwen 已验证候选选图', 'program': '纯程序排序'}


def signature(edges, nodes, unknown):
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return tuple(tuple(sorted(nx.descendants(graph, node))) for node in unknown)


def request(session, path, prompt, seed, schema):
    payload = {'model': MODEL, 'think': False, 'stream': False, 'keep_alive': '10m',
               'options': {**OPTIONS, 'seed': seed, 'num_predict': 96},
               'format': schema, 'messages': [{'role': 'user', 'content': prompt}]}
    if path.exists():
        result = read(path)
        assert result['request'] == payload
        return result
    started = time.perf_counter()
    response = session.post(BASE_URL+'/api/chat', json=payload, timeout=(10, 180))
    response.raise_for_status()
    call = {'request': payload, 'response': response.json(), 'seconds': time.perf_counter()-started}
    write_json(path, call)
    return call


def decode(call):
    try:
        answer = json.loads(call['response']['message']['content'])
        return answer if isinstance(answer, dict) else {}
    except (KeyError, ValueError):
        return {}


def schema(properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties),
            'additionalProperties': False}


def choose_edit(session, path, seed, nodes, limit, observations, emitted, blocked, last_feedback):
    # IDs refer to up to eight recent valid graphs. A record of every accepted
    # graph prevents repetition even when the model only sees a short history.
    recent = emitted[-8:]
    parents = '\n'.join(f'{i}. {graph_line(edges)}' for i, edges in enumerate(recent, 1))
    evidence = '\n'.join(row['string'] for row in observations)
    banned = ', '.join(f'{a}->{b}' for a, b in sorted(blocked))
    known = '\n'.join(graph_line(edges) for edges in emitted)
    prompt = (f'Find a new DAG on {", ".join(nodes)} with at most {limit} edges. '
              'A perturbed node reaches exactly the nodes marked 1 via directed paths. '
              'Select a parent below and add one absent edge or remove one present edge. '
              'Preserve all observed effects, avoid cycles and never repeat a submitted graph. '
              f'Forbidden additions: {banned or "none"}.\nObservations:\n{evidence}'
              f'\nEligible parents:\n{parents}\nSubmitted graphs:\n{known}'
              f'\nPrevious attempt: {last_feedback}\nReturn parent, action, source, target.')
    call = request(session, path, prompt, seed, schema({
        'parent': {'type': 'integer', 'minimum': 1, 'maximum': len(recent)},
        'action': {'enum': ['add', 'remove']}, 'source': {'enum': nodes}, 'target': {'enum': nodes}}))
    answer = decode(call)
    parent_id = answer.get('parent')
    a, b, action = answer.get('source'), answer.get('target'), answer.get('action')
    if (type(parent_id) is not int or not 1 <= parent_id <= len(recent)
            or a not in nodes or b not in nodes or a == b or action not in ('add', 'remove')):
        return None, call, 'Invalid edit format or self-loop.', 0
    parent = set(recent[parent_id-1])
    edge = (a, b)
    if (edge in parent) == (action == 'add'):
        return None, call, f'{action} {a}->{b} does not change the parent graph.', 0
    if action == 'add' and edge in blocked:
        return None, call, f'{a}->{b} is forbidden by the observations.', 0
    candidate = tuple(sorted(parent ^ {edge}))
    if candidate in emitted:
        return None, call, f'{graph_line(candidate)} was already submitted.', 0
    graph = nx.DiGraph(candidate)
    if len(candidate) > limit or not nx.is_directed_acyclic_graph(graph):
        return None, call, f'{graph_line(candidate)} exceeds the edge limit or has a cycle.', 1
    failures = observations_check(nodes, observations, candidate)
    if failures:
        first = failures[0]
        return None, call, f'{a}->{b} edit violates {first["source"]} to {first["target"]} reachability.', 1
    return candidate, call, 'Accepted a new valid graph.', 1


def choose_rank(session, path, seed, nodes, unknown, observations, frontier, shortlist):
    # Lexicographic IDs avoid showing the program's preferred ranking.
    choices = sorted(shortlist)
    options = '\n'.join(f'{i}. {graph_line(edges)}' for i, edges in enumerate(choices, 1))
    seen = '\n'.join(graph_line(edges) for edges in frontier.emitted)
    obs = '\n'.join(row['string'] for row in observations)
    prompt = (f'Choose a new graph on {", ".join(nodes)} from the numbered list. All are verified '
              f'and unsubmitted. Prefer a graph predicting new downstream effects when perturbing '
              f'{", ".join(unknown) or "any node"}, and then edge differences from submitted graphs. '
              f'\nObservations:\n{obs}\nSubmitted:\n{seen}\nCandidates:\n{options}\nReturn choice.')
    call = request(session, path, prompt, seed, schema({
        'choice': {'type': 'integer', 'minimum': 1, 'maximum': len(choices)}}))
    index = decode(call).get('choice')
    if type(index) is int and 1 <= index <= len(choices):
        return choices[index-1], call, False
    return shortlist[0], call, True


def run(benchmark, task, anchor, method, session, folder, seed):
    nodes, limit, obs = benchmark.nodes, benchmark.max_edges, task['observations']
    unknown = sorted(set(nodes)-{row['perturbed_node'] for row in obs})
    frontier = Frontier(nodes, limit, obs, anchor, prune=True)
    emitted = frontier.emitted
    blocked = forbidden_edges(obs)
    feedback = 'No edit yet.'
    trace = []
    for slot in range(1, SLOTS+1):
        started = time.perf_counter()
        checks, call, fallback = 0, None, False
        if slot == 1:
            selected, checks = anchor, 1
        elif method == 'edits':
            selected, call, feedback, checks = choose_edit(
                session, folder/f'{method}_{slot:02}.json', seed, nodes, limit, obs, emitted, blocked, feedback)
            if selected is not None:
                emitted.append(selected)
        else:
            while not frontier.pool and frontier.queue and checks < DEFAULT_CHECKS_PER_SLOT:
                checks += frontier.expand(min(CHECK_BATCH_SIZE, DEFAULT_CHECKS_PER_SLOT-checks))
            shortlist = frontier.shortlist()
            selected = shortlist[0] if shortlist else None
            if method == 'rank' and len(shortlist) > 1:
                selected, call, fallback = choose_rank(session, folder/f'{method}_{slot:02}.json', seed,
                                                       nodes, unknown, obs, frontier, shortlist)
            if selected is not None:
                frontier.emit(selected)
        if selected is not None:
            parsed = benchmark.parse_llm_response(graph_line(selected))
            assert parsed is not None and benchmark.validate_hypothesis(parsed, obs)
        response = call['response'] if call else {}
        trace.append({'slot': slot, 'edges': selected, 'checks': checks,
                      'model_call': call is not None, 'fallback': fallback,
                      'feedback': feedback if method == 'edits' and slot > 1 else None,
                      'output_tokens': response.get('eval_count', 0),
                      'seconds': time.perf_counter()-started})
    assert len(emitted) == len(set(emitted)) == sum(row['edges'] is not None for row in trace)
    stats = []
    for budget in MILESTONES:
        rows = trace[:budget]
        graphs = [row['edges'] for row in rows if row['edges'] is not None]
        truth = task['n_ground_truths']
        assert len(graphs) <= min(budget, truth)
        stats.append(dict(budget=budget, found=len(graphs), rr=len(graphs)/truth, vr=len(graphs)/budget,
                          signatures=len({signature(edges, nodes, unknown) for edges in graphs}) if unknown else None,
                          **{k: sum(row[k] for row in rows) for k in ('checks','model_call','fallback','output_tokens','seconds')}))
    return {'budgets': stats, 'slots': trace}


def report(output, records):
    lines = ['# Qwen 合法图生成与选边改进对照', '',
             '沿用上轮的六道六节点题及配对种子，取旧 Qwen 的 32 槽结果作参考。三组新方法共享一个由观测构造的合法首图，首图占第一槽。Qwen 单边组每次返回一个增删操作，程序拒绝重复或不合法结果，拒绝仍占槽。Qwen 选图组从程序验证过的未提交候选中选一个，编号无效时用程序顺序。纯程序组采用同一候选池扩展规则和边集距离排序。', '',
             '两种候选池组采用观测剪枝、按需检查和每槽 12 次上限。选图改变后续扩展的图，因此后续池可能不同。三组只用观测验证，答案数量来自已审计数据。VR 按合法输出除以 32 槽，空槽计入分母。', '',
             '| 32 槽方法 | 平均 RR | VR | 不同合法图 | 模型调用 | 输出 token | 图检查 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    summary = {}
    for key in ('previous', *METHODS):
        rows = [r['methods'][key]['budgets'][-1] for r in records]
        d = {k: mean(r[k] for r in rows) for k in ('rr','vr')}
        d.update({k: sum(r[k] for r in rows) for k in ('found','model_call','output_tokens','checks','fallback')})
        partial = [r['signatures'] for r in rows if r.get('signatures') is not None]
        d['mean_signatures'] = mean(partial) if partial else None
        summary[key] = d
        label = METHODS.get(key, '上一轮自由生成')
        check_count = str(d['checks']) if key != 'previous' else '—'
        lines.append(f"| {label} | {d['rr']:.4%} | {d['vr']:.2%} | {d['found']} | "
                     f"{d['model_call']} | {d['output_tokens']:,} | {check_count} |")
    lines += ['', '| 题目 | 合法图总数 | 旧 Qwen | Qwen 单边 | Qwen 选图 | 纯程序 |',
              '|---|---:|---:|---:|---:|---:|']
    for r in records:
        values = ' | '.join(str(r['methods'][k]['budgets'][-1]['found']) for k in ('previous', *METHODS))
        lines.append(f"| {r['id']} | {r['truth']:,} | {values} |")
    lines += ['', '这是同批题的开发对照，每题一个种子。首图和候选合法性由程序保障，混合方法的分数包含这部分贡献。原始模型请求、拒绝原因和逐槽记录均保存在运行目录。']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    write_json(output/'summary.json', summary)
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    tasks = select()
    previous = read(PREVIOUS/'config.json')
    assert [r['id'] for r in tasks] == previous['cases']
    output = args.output or HERE/'runs'/('qwen_guided_edits_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.trust_env = False
    models = session.get(BASE_URL+'/api/tags', timeout=10)
    models.raise_for_status()
    model = next(m for m in models.json()['models'] if m['name'] == MODEL)
    assert model['digest'] == read(MODEL_REFERENCE)['model']['digest']
    module = load_upstream('causal', 'run_causal_benchmark.py')
    benchmark = module.CausalBenchmarkEnhanced(str(PREVIOUS/'test.json'))
    config = {'model': model['digest'], 'cases': previous['cases'], 'slots': SLOTS,
              'code_sha': digest(__file__), 'previous_summary_sha': digest(PREVIOUS/'summary.json'),
              'frontier_sha': digest(HERE/'run_partial_graph_pipeline.py'), 'seeds': SEEDS,
              'options': OPTIONS, 'checks_per_slot': DEFAULT_CHECKS_PER_SLOT}
    path = output/'config.json'
    if path.exists():
        assert read(path) == config
    else:
        write_json(path, config)
    print('OUTPUT_DIR='+str(output.resolve()), flush=True)
    records = []
    for index, task in enumerate(tasks):
        item = task['id']
        folder = output/item
        folder.mkdir(exist_ok=True)
        seed = SEEDS[index % len(SEEDS)]
        old = read(PREVIOUS/item/'result.json')
        assert old['seed'] == seed
        anchor = program_anchor(benchmark.nodes, benchmark.max_edges, task['observations'])
        assert anchor is not None and not observations_check(benchmark.nodes, task['observations'], anchor)
        methods = {'previous': {'budgets': [{**r, 'model_call': r['budget'], 'checks': 0, 'fallback': 0}
                    for r in old['budgets'] if r['budget'] in MILESTONES]}}
        for method in METHODS:
            methods[method] = run(benchmark, task, anchor, method, session, folder, seed)
        r = {'id': item, 'seed': seed, 'truth': task['n_ground_truths'], 'anchor': anchor, 'methods': methods}
        write_json(folder/'result.json', r)
        records.append(r)
        print(f'{item} ' + str([m['budgets'][-1]['found'] for m in methods.values()]), flush=True)
    report(output, records)


if __name__ == '__main__':
    main()
