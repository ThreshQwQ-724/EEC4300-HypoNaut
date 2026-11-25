"""Compare Qwen and program ordering before validating single-edge edits."""
import argparse
from datetime import datetime
from itertools import permutations
from pathlib import Path
import random
from statistics import mean
from time import perf_counter

import networkx as nx
import requests

from common import HERE, digest
from run_feedback_study import graph_line, observations_check
from run_four_groups import BASE_URL, MODEL, SEEDS
from run_local_causal import write_json
from run_partial_graph_pipeline import edge_distance, forbidden_edges, program_anchor
from run_qwen_guided_edits import decode, request, schema, signature
from run_qwen_rr_budget import MODEL_REFERENCE, read
from stream_causal import check_graph
from validate_graph_rr_budget import DATA

NODES = list('ABCDEF')
LIMIT = 10
SLOTS = 8
MENU = 8
BUDGETS = (1, 2)
SELECTION_SEED = 1014126
METHODS = ('lexical', 'diversity', 'qwen')
LABELS = {'lexical': '固定边集顺序', 'diversity': '程序多样性', 'qwen': 'Qwen 优先检查'}


def select_tasks():
    excluded = set()
    for path in (HERE/'runs').glob('*/config.json'):
        if path.parent.name.startswith('qwen_tight_budget_'):
            continue
        # Previous run manifests contain IDs in either lists or nested rows.
        import re
        excluded.update(re.findall(r'\bn[1-6]_\d{3}\b', path.read_text(encoding='utf-8')))
    rng = random.Random(SELECTION_SEED)
    tasks = read(DATA)
    selected = []
    for count in range(1, 7):
        pool = sorted((r for r in tasks if r['n_observations'] == count
                       and r['id'] not in excluded and r['n_ground_truths'] >= SLOTS), key=lambda r: r['id'])
        selected.extend(rng.sample(pool, 2))
    return selected


def neighbors(emitted, allowed, checked):
    edits = {}
    for parent in emitted:
        edges = set(parent)
        for edge in allowed:
            if edge not in edges and len(edges) == LIMIT:
                continue
            candidate = tuple(sorted(edges ^ {edge}))
            if candidate not in checked:
                edits.setdefault(candidate, {'parent': parent, 'edge': edge,
                                             'action': 'remove' if edge in edges else 'add'})
    return edits


def priority(candidate, emitted):
    distances = [edge_distance(candidate, previous) for previous in emitted]
    return -min(distances), -sum(distances), candidate


def qwen_choice(session, path, seed, observations, emitted, menu, edits):
    options = sorted(menu)
    lines = []
    for i, candidate in enumerate(options, 1):
        edit = edits[candidate]
        a, b = edit['edge']
        lines.append(f'{i}. {edit["action"]} {a}->{b}. Result: {graph_line(candidate)}')
    prompt = (
        'Choose which unverified DAG candidate to check first. Choose a graph that is acyclic, '
        'obeys every intervention and differs from already submitted graphs. When a node is '
        'perturbed, exactly its descendants have effect 1. Check directed paths, not just edges. '
        'Respond with the JSON choice number.\nObservations:\n'
        + '\n'.join(row['string'] for row in observations)
        + '\nSubmitted:\n' + '\n'.join(graph_line(g) for g in emitted)
        + '\nCandidates:\n' + '\n'.join(lines))
    call = request(session, path, prompt, seed, schema({
        'choice': {'type': 'integer', 'minimum': 1, 'maximum': len(options)}}))
    answer = decode(call).get('choice')
    valid = type(answer) is int and 1 <= answer <= len(options)
    return (options[answer-1] if valid else menu[0]), call, valid


def run(task, anchor, budget, method, session, folder, seed):
    obs = task['observations']
    blocked = forbidden_edges(obs)
    allowed = tuple(edge for edge in permutations(NODES, 2) if edge not in blocked)
    unknown = sorted(set(NODES)-{row['perturbed_node'] for row in obs})
    emitted, checked, verified, trace = [anchor], {anchor}, set(), []
    started = perf_counter()
    for slot in range(2, SLOTS+1):
        edits = neighbors(emitted, allowed, checked)
        ordered = sorted(edits, key=lambda candidate: priority(candidate, emitted))
        menu = ordered[:MENU]
        call, valid = None, None
        chosen = ordered[:budget]
        if method == 'lexical':
            chosen = sorted(edits)[:budget]
        elif method == 'qwen' and len(menu) > 1:
            first, call, valid = qwen_choice(session, folder/f'qwen_{budget}_{slot:02}.json',
                                            seed, obs, emitted, menu, edits)
            chosen = ([first] + [g for g in ordered if g != first])[:budget]
        checks = []
        for candidate in chosen:
            graph = nx.DiGraph(candidate)
            valid_graph = nx.is_directed_acyclic_graph(graph) and not observations_check(NODES, obs, candidate)
            checked.add(candidate)
            if valid_graph:
                verified.add(candidate)
            checks.append({'edges': candidate, 'valid': valid_graph, **edits[candidate]})
        submitted = None
        if verified:
            submitted = min(verified) if method == 'lexical' else min(
                verified, key=lambda candidate: priority(candidate, emitted))
            verified.remove(submitted)
            emitted.append(submitted)
        trace.append({'slot': slot, 'checks': checks, 'submitted': submitted, 'model_call': call is not None,
                      'valid_choice': valid, 'first_differs': bool(chosen and ordered and chosen[0] != ordered[0]),
                      'output_tokens': call['response'].get('eval_count', 0) if call else 0})
    elapsed = perf_counter()-started
    truth = task['n_ground_truths']
    assert len(emitted) == len(set(emitted)) <= min(SLOTS, truth)
    assert all(check_graph({'nodes': NODES, 'edges': edges}, obs, {'nodes': NODES, 'max_edges': LIMIT})
               for edges in emitted)
    assert all(len(row['checks']) <= budget for row in trace)
    return {'found': len(emitted), 'rr': len(emitted)/truth, 'vr': len(emitted)/SLOTS,
            'checks': sum(len(row['checks']) for row in trace),
            'valid_checks': sum(c['valid'] for row in trace for c in row['checks']),
            'model_calls': sum(row['model_call'] for row in trace),
            'valid_choices': sum(row['valid_choice'] is True for row in trace),
            'first_differs': sum(row['first_differs'] and row['model_call'] for row in trace),
            'output_tokens': sum(row['output_tokens'] for row in trace),
            'signatures': len({signature(edges, NODES, unknown) for edges in emitted}) if unknown else None,
            'seconds': elapsed, 'trace': trace}


def report(output, records):
    summary = {}
    lines = ['# 紧检查预算下的 Qwen 选边对照', '',
             '每个干预数抽两道至少有八个合法图的六节点题，共 12 题。排除之前推理实验的题目。'
             '每题一个固定种子，三组共享经验证的程序首图，首图占一槽。后续各槽最多检查 1 或 2 个邻居，'
             '从已提交图做单边增删，统一做观测剪枝和全轮去重。图检查数仅统计首图之后的检查。', '',
             '程序多样性组按与已提交图的最小边集距离排序，取优先候选。Qwen 从同一排序选出的至多 8 个未验证候选中指定先检查哪一个。'
             '预算为 2 时第二个候选沿用程序顺序。无效编号回退到程序选择。两组用相同规则从已验证池提交，'
             '已提交图不同可能导致下一槽的菜单不同。固定顺序组提供额外参考。', '',
             '| 每槽上限 | 方法 | 平均 RR | 合法图总数 | 满 8 图 | 实际检查 | 合法检查 | 模型调用 | 输出 token | 秒数 |',
             '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for budget in BUDGETS:
        block = {}
        for method in METHODS:
            rows = [r['runs'][str(budget)][method] for r in records]
            d = {key: sum(row[key] for row in rows) for key in (
                'found', 'checks', 'valid_checks', 'model_calls', 'valid_choices', 'first_differs', 'output_tokens', 'seconds')}
            d.update(rr=mean(r['rr'] for r in rows), vr=mean(r['vr'] for r in rows),
                     full=sum(r['found'] == SLOTS for r in rows),
                     signatures=mean(r['signatures'] for r in rows if r['signatures'] is not None))
            block[method] = d
            lines.append(f"| {budget} | {LABELS[method]} | {d['rr']:.4%} | {d['found']} | {d['full']}/12 | "
                         f"{d['checks']} | {d['valid_checks']} | {d['model_calls']} | {d['output_tokens']} | {d['seconds']:.2f} |")
        diffs = [r['runs'][str(budget)]['qwen']['found'] - r['runs'][str(budget)]['diversity']['found'] for r in records]
        block['paired_qwen_vs_diversity'] = [sum(d > 0 for d in diffs), sum(d == 0 for d in diffs), sum(d < 0 for d in diffs)]
        summary[str(budget)] = block
    lines += ['', '| 题目 | 合法图总数 | 上限 1，固定/多样性/Qwen | 上限 2，固定/多样性/Qwen |',
              '|---|---:|---|---|']
    for r in records:
        counts = [' / '.join(str(r['runs'][str(b)][m]['found']) for m in METHODS) for b in BUDGETS]
        lines.append(f"| {r['id']} | {r['truth']} | {counts[0]} | {counts[1]} |")
    lines += ['', '每题的输出图均独立核对无环、边数、观测一致性和去重。RR 分母来自数据审计。'
              '比较采用相同的图检查上限，模型的 token 和耗时另计。逐槽检查及模型原始响应已保存。']
    write_json(output/'summary.json', summary)
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(summary, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or HERE/'runs'/('qwen_tight_budget_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.trust_env = False
    response = session.get(BASE_URL+'/api/tags', timeout=10)
    response.raise_for_status()
    model = next(row for row in response.json()['models'] if row['name'] == MODEL)
    assert model['digest'] == read(MODEL_REFERENCE)['model']['digest']
    tasks = select_tasks()
    config = {'model': model['digest'], 'cases': [r['id'] for r in tasks], 'selection_seed': SELECTION_SEED,
              'seeds': SEEDS, 'budgets': list(BUDGETS), 'slots': SLOTS, 'menu': MENU, 'input_sha': digest(DATA),
              'code_hashes': {name: digest(HERE/name) for name in ('run_qwen_tight_budget.py',
                  'run_qwen_guided_edits.py', 'run_partial_graph_pipeline.py')}}
    path = output/'config.json'
    if path.exists():
        assert read(path) == config
    else:
        write_json(path, config)
    print('OUTPUT_DIR='+str(output.resolve()), flush=True)
    records = []
    for index, task in enumerate(tasks):
        folder = output/task['id']
        folder.mkdir(exist_ok=True)
        seed = SEEDS[index % len(SEEDS)]
        anchor = program_anchor(NODES, LIMIT, task['observations'])
        assert anchor is not None
        runs = {str(b): {method: run(task, anchor, b, method, session, folder, seed) for method in METHODS}
                for b in BUDGETS}
        record = {'id': task['id'], 'seed': seed, 'truth': task['n_ground_truths'],
                  'observations': task['observations'], 'anchor': anchor, 'runs': runs}
        write_json(folder/'result.json', record)
        records.append(record)
        print(f"{len(records)}/12 {task['id']} " +
              str([[runs[str(b)][m]['found'] for m in METHODS] for b in BUDGETS]), flush=True)
    report(output, records)


if __name__ == '__main__':
    main()
