"""Evaluate E-UoT-inspired DAG edits under a shared candidate-check budget."""
import argparse
from itertools import permutations
import json
from pathlib import Path
import random
import re
from statistics import mean
from time import perf_counter

import requests

from common import HERE, digest
from graph_completion import complete_observation_count
from run_constraint_feedback import OPTIONS
from run_feedback_study import graph_line
from run_formal_causal_comparison import (
    DATA, LIMIT, NODES, SLOTS, deletion_safe, forbidden_edges, program_anchor, read, risk, valid,
)
from run_four_groups import BASE_URL, MODEL, SEEDS
from run_local_causal import write_json
from run_qwen_guided_edits import decode, request, schema
from run_qwen_rr_budget import MODEL_REFERENCE


PAPER = 'https://arxiv.org/html/2511.20471v3'
SELECTION_SEED = 251120471
MENU_SIZE = 8
SHIFT_FIRST_SLOTS = (4, 7)
QWEN_STRATA = (2, 3, 4, 5)
LABELS = {'single': '程序单边', 'program': '程序 Deepen＋Shift', 'qwen': 'Qwen Deepen＋Shift'}


def make_edits(parent, allowed, shift):
    edges = set(parent)
    groups = {'deepen': {}, 'shift': {}}
    for edge in allowed:
        if edge in edges or len(edges) < LIMIT:
            graph = tuple(sorted(edges ^ {edge}))
            groups['deepen'][graph] = {'parent': parent, 'edge': edge,
                                       'action': 'remove' if edge in edges else 'add'}
    if shift:
        # Enumerate both directions of a direct-edge / two-edge-path rewrite.
        for a, b, c in permutations(NODES, 3):
            direct, path = {(a, c)}, {(a, b), (b, c)}
            if not (direct | path) <= allowed:
                continue
            for removed, added in ((direct, path), (path, direct)):
                if removed <= edges and not added & edges:
                    candidate = (edges - removed) | added
                    if len(candidate) <= LIMIT:
                        groups['shift'].setdefault(tuple(sorted(candidate)), {
                            'parent': parent, 'remove': sorted(removed), 'add': sorted(added)})
    return groups


def ranked_groups(emitted, frontier, checked, slot, mixed):
    # Edge-set distance encourages novel graphs. It requires no observation checks.
    masks = [set(edges) for edges in emitted]
    ranked = {}
    for operator in (('deepen', 'shift') if mixed else ('deepen',)):
        families = {}
        for parent in emitted:
            candidates = [(graph, edit) for graph, edit in frontier[parent][operator].items()
                          if graph not in checked]

            def priority(item):
                graph, edit = item
                edges = set(graph)
                distances = [len(edges ^ previous) for previous in masks]
                return edit.get('action') == 'remove', -min(distances), -sum(distances), graph

            candidates.sort(key=priority)
            if candidates:
                families[parent] = (priority(candidates[0]), candidates)
        ranked[operator] = [family for _, family in sorted(families.values(), key=lambda row: row[0])]
    order = ('shift', 'deepen') if mixed and slot in SHIFT_FIRST_SLOTS else tuple(ranked)
    # Put the best candidate of each parent/operator family in the model menu.
    menu = []
    for rank in range(len(emitted)):
        for operator in order:
            if rank < len(ranked[operator]):
                graph, edit = ranked[operator][rank][0]
                menu.append((operator, graph, edit))
    # A graph can be reachable from different parents. Check it once.
    seen = {entry[1] for entry in menu}
    fallback = menu.copy()
    for operator in order:
        for family in ranked[operator]:
            for graph, edit in family[1:]:
                if graph not in seen:
                    fallback.append((operator, graph, edit))
                    seen.add(graph)
    return menu[:MENU_SIZE], fallback


def qwen_select(session, path, seed, observations, emitted, menu):
    # Sort the model's menu independently so the program's preferred option is not always first.
    options = sorted(menu, key=lambda row: (row[2]['parent'], row[0], row[1]))
    lines = [f'{i}. {operator}: {graph_line(edit["parent"])} -> {graph_line(graph)}'
             for i, (operator, graph, edit) in enumerate(options, 1)]
    prompt = (
        f'Choose one unexplored edit family to check first for a DAG on {", ".join(NODES)} '
        f'with at most {LIMIT} edges. Each option shows its starting graph and first unverified '
        'candidate. A perturbed node must reach exactly the nodes marked 1 via directed paths. '
        'Prefer an acyclic candidate consistent with every observation and different from all submitted '
        'graphs. Deepen changes one edge. Shift replaces a direct edge with a two-edge path or vice '
        'versa. Return the JSON choice number.\nObservations:\n'
        + '\n'.join(row['string'] for row in observations)
        + '\nSubmitted:\n' + '\n'.join(map(graph_line, emitted))
        + '\nEdit families:\n' + '\n'.join(lines))
    call = request(session, path, prompt, seed, schema({
        'choice': {'type': 'integer', 'minimum': 1, 'maximum': len(options)}}))
    choice = decode(call).get('choice')
    parsed = type(choice) is int and 1 <= choice <= len(options)
    return (options[choice-1] if parsed else menu[0]), parsed, call


def run(observations, anchor, budget, method, session=None, folder=None, seed=None):
    started = perf_counter()
    allowed = set(permutations(NODES, 2)) - forbidden_edges(observations)
    exact = complete_observation_count(NODES, LIMIT, observations)
    mixed = method != 'single'
    emitted, checked, frontier, closures, trace = [anchor], {anchor}, {}, {}, []
    generated = 0
    for slot in range(2, SLOTS+1):
        if len(emitted) == exact:
            break
        parent = emitted[-1]
        if parent not in frontier:
            frontier[parent] = make_edits(parent, allowed, mixed)
            generated += sum(map(len, frontier[parent].values()))
        menu, ordered = ranked_groups(emitted, frontier, checked, slot, mixed)
        if not menu:
            break
        call = None
        parsed = None
        first = menu[0]
        if method == 'qwen' and len({row[1] for row in menu}) > 1:
            first, parsed, call = qwen_select(session, folder/f'qwen_{budget}_{slot:02}.json',
                                              seed, observations, emitted, menu)
        steps = []
        deletion_traversals = 0
        for operator, candidate, edit in (first, *ordered):
            if candidate in checked:
                continue
            checked.add(candidate)
            if operator == 'shift':
                full_check = True
            elif edit['action'] == 'add':
                full_check = risk(edit, observations, closures) is None
            else:
                full_check, traversals = deletion_safe(edit, candidate, observations)
                deletion_traversals += traversals
            accepted = full_check and valid(candidate, observations)
            assert operator == 'shift' or accepted == full_check
            steps.append({'operator': operator, 'graph': candidate, 'edit': edit,
                          'full_check': full_check, 'accepted': accepted})
            if accepted:
                emitted.append(candidate)
                break
            if len(steps) == budget:
                break
        trace.append({'slot': slot, 'steps': steps, 'menu': menu if call else None,
                      'model_choice_changed_graph': first[1] != menu[0][1], 'valid_choice': parsed,
                      'deletion_traversals': deletion_traversals,
                      'model_calls': int(call is not None),
                      'model_seconds': call['seconds'] if call else 0,
                      'input_tokens': call['response'].get('prompt_eval_count', 0) if call else 0,
                      'output_tokens': call['response'].get('eval_count', 0) if call else 0})
    runner_seconds = perf_counter()-started
    steps = [step for row in trace for step in row['steps']]
    assert len(steps) == len(checked)-1 and all(len(row['steps']) <= budget for row in trace)
    assert len(emitted) == len(set(emitted)) <= SLOTS
    assert all(valid(graph, observations) for graph in emitted)
    return {'graphs': emitted, 'found': len(emitted), 'probes': len(steps),
            'full_checks': sum(step['full_check'] for step in steps),
            'shift_probes': sum(step['operator'] == 'shift' for step in steps),
            'shift_found': sum(step['operator'] == 'shift' and step['accepted'] for step in steps),
            'generated': generated, 'parent_closures': len(closures), 'runner_seconds': runner_seconds,
            'changed_choices': sum(row['model_choice_changed_graph'] for row in trace),
            'parsed_choices': sum(row['valid_choice'] is True for row in trace), 'trace': trace,
            **{key: sum(row[key] for row in trace) for key in (
                'model_calls', 'model_seconds', 'input_tokens', 'output_tokens', 'deletion_traversals')}}


def select_tasks(tasks, count):
    used = set()
    for path in (HERE/'runs').glob('*/config.json'):
        if not path.parent.name.startswith('exploratory_uot_'):
            used.update(re.findall(r'\bn[1-6]_\d{3}\b', path.read_text(encoding='utf-8')))
    rng = random.Random(SELECTION_SEED)
    return [task for n in QWEN_STRATA for task in rng.sample(sorted(
        (task for task in tasks if task['n_observations'] == n and task['id'] not in used
         and task['n_ground_truths'] >= SLOTS), key=lambda task: task['id']), count)]


def report(output, records, budgets, methods):
    summary = {}
    lines = ['# HypoSpace Exploratory UoT 对照', '', f'方法来源 [{PAPER}]({PAPER})。', '',
             f'{len(records)} 道六节点题，每题 {SLOTS} 个提交槽，各策略共享观测、首图和验证器。'
             '每槽至多检查指定数量的未验证候选，找到第一个新合法图即提交。'
             '每个候选算一次探查；单边沿用程序避险，联合改动直接验证，因此同时列出完整验证、删边遍历与耗时。'
             '首图校验与最后独立复核不计入搜索成本。答案数只用于抽样、评分与结果审计。', '',
             'Deepen 为单边增删，Shift 为直接边与两跳路径的双向替换。两种混合策略共享邻域和程序顺序。'
             '程序在第 4、7 槽优先尝试 Shift，其余槽优先 Deepen。两类操作从所有已提交图扩展。'
             'Qwen3:8B 通过结构化 JSON 选择最多 8 个父图/操作家族的代表候选，关闭思考。'
             '无效编号使用程序首选，剩余检查沿用程序顺序。', '',
             '| 每槽探查上限 | 方法 | RR | VR | NR | 合法图 | 达到覆盖上限 | 探查 | 完整验证 | Shift 成功/尝试 | 候选生成 | 删边遍历 | 模型调用 | 输入/输出 token | 运行秒数 | 请求秒数 |',
             '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for budget in budgets:
        key = str(budget)
        block = {}
        for method in methods:
            rows = [record['runs'][key][method] for record in records]
            totals = {name: sum(row[name] for row in rows) for name in (
                'found', 'probes', 'full_checks', 'shift_probes', 'shift_found', 'generated',
                'model_calls', 'input_tokens', 'output_tokens', 'parsed_choices', 'changed_choices',
                'runner_seconds', 'model_seconds', 'parent_closures', 'deletion_traversals')}
            totals.update(rr=mean(row['rr'] for row in rows), vr=mean(row['vr'] for row in rows),
                          nr=mean(row['nr'] for row in rows),
                          cap_hits=sum(row['found'] == min(SLOTS, record['truth'])
                                       for row, record in zip(rows, records)))
            block[method] = totals
            d = totals
            lines.append(f"| {budget} | {LABELS[method]} | {d['rr']:.4%} | {d['vr']:.2%} | {d['nr']:.2%} | "
                         f"{d['found']} | {d['cap_hits']}/{len(rows)} | {d['probes']} | {d['full_checks']} | "
                         f"{d['shift_found']}/{d['shift_probes']} | {d['generated']} | {d['deletion_traversals']} | "
                         f"{d['model_calls']} | {d['input_tokens']}/{d['output_tokens']} | "
                         f"{d['runner_seconds']:.3f} | {d['model_seconds']:.3f} |")
        comparisons = {}
        for left, right in zip(methods[1:], methods):
            changes = [record['runs'][key][left]['found']-record['runs'][key][right]['found']
                       for record in records]
            comparisons[f'{left}_vs_{right}'] = [sum(d > 0 for d in changes), sum(d == 0 for d in changes),
                                                 sum(d < 0 for d in changes)]
        summary[key] = block | {'paired_wins_ties_losses': comparisons}
    lines += ['', '逐题合法图数的胜 / 平 / 负', '']
    for budget, block in summary.items():
        for pair, (wins, ties, losses) in block['paired_wins_ties_losses'].items():
            left, right = pair.split('_vs_')
            lines.append(f'- 每槽 {budget} 查，{LABELS[left]}对比{LABELS[right]}为 {wins} / {ties} / {losses}。')
    lines += ['', 'RR 按题宏平均。各组只提交已验证且不同的图，因而 VR 与 NR 相同。'
              '运行秒数为本次执行时间，模型请求秒数来自原始请求记录，重跑使用缓存时两列口径不同。'
              '逐题检查、候选菜单与模型请求见同目录 JSON。']
    if 'qwen' in methods:
        lines += ['', 'Qwen 样本按 2–5 次干预分层随机抽取，每题至少有 8 个合法解，并排除既有运行配置中的题号。'
                  '1 次与 6 次干预层满足条件的题号都已出现在既有配置中，故本次 Qwen 对照未纳入这两层。']
    write_json(output/'summary.json', summary)
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(summary, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--per-stratum', type=int, default=6)
    parser.add_argument('--budget', type=int, nargs='+', default=[2])
    parser.add_argument('--program-only', action='store_true')
    parser.add_argument('--all-tasks', action='store_true')
    args = parser.parse_args()
    if min(args.budget) < 1 or len(set(args.budget)) != len(args.budget) or args.per_stratum < 1:
        parser.error('Use distinct positive budgets and a positive sample count')
    if args.all_tasks and not args.program_only:
        parser.error('--all-tasks requires --program-only')
    methods = ('single', 'program') if args.program_only else tuple(LABELS)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    all_tasks = read(DATA)
    config_path = output/'config.json'
    if config_path.exists():
        by_id = {task['id']: task for task in all_tasks}
        tasks = [by_id[case] for case in read(config_path)['cases']]
    else:
        tasks = all_tasks if args.all_tasks else select_tasks(all_tasks, args.per_stratum)
    session = None
    model = None
    if not args.program_only:
        session = requests.Session()
        session.trust_env = False
        response = session.get(BASE_URL+'/api/tags', timeout=10)
        response.raise_for_status()
        model = next(row for row in response.json()['models'] if row['name'] == MODEL)
        assert model['digest'] == read(MODEL_REFERENCE)['model']['digest']
    config = {'paper': PAPER, 'cases': [task['id'] for task in tasks], 'budgets': args.budget,
              'methods': methods, 'selection_seed': SELECTION_SEED, 'model': model, 'seeds': SEEDS,
              'slots': SLOTS, 'menu': MENU_SIZE, 'shift_first_slots': SHIFT_FIRST_SLOTS,
              'sample_strata': QWEN_STRATA if not args.all_tasks else None,
              'options': {**OPTIONS, 'num_predict': 96, 'think': False}, 'data_sha': digest(DATA),
              'code_sha': {name: digest(HERE/name) for name in (
                  Path(__file__).name, 'run_partial_graph_pipeline.py', 'evaluate_qwen_risk_trigger.py',
                  'run_formal_causal_comparison.py', 'graph_completion.py', 'run_qwen_guided_edits.py',
                  'stream_causal.py')}, 'per_stratum': args.per_stratum, 'all_tasks': args.all_tasks}
    if config_path.exists():
        if read(config_path) != json.loads(json.dumps(config)):
            raise ValueError('Output directory contains a different experiment configuration')
    else:
        write_json(config_path, config)
    records = []
    for i, task in enumerate(tasks):
        folder = output/task['id']
        folder.mkdir(exist_ok=True)
        record_path = folder/'result.json'
        if record_path.exists():
            record = read(record_path)
        else:
            observations, truth = task['observations'], task['n_ground_truths']
            seed = SEEDS[i % len(SEEDS)]
            anchor = program_anchor(NODES, LIMIT, observations)
            assert anchor is not None and valid(anchor, observations)
            record = {'id': task['id'], 'truth': truth, 'anchor': anchor, 'seed': seed, 'runs': {}}
            for budget in args.budget:
                results = {}
                for method in methods:
                    row = run(observations, anchor, budget, method, session, folder, seed)
                    assert row['found'] <= min(SLOTS, truth)
                    row.update(rr=row['found']/truth, vr=row['found']/SLOTS, nr=row['found']/SLOTS)
                    results[method] = row
                record['runs'][str(budget)] = results
            temporary = record_path.with_suffix('.tmp')
            write_json(temporary, record)
            temporary.replace(record_path)
        records.append(record)
        if not args.all_tasks or (i+1) % 50 == 0:
            counts = [[result['found'] for result in row.values()] for row in record['runs'].values()]
            print(f'{i+1}/{len(tasks)} {task["id"]} {counts}', flush=True)
    report(output, records, args.budget, methods)


if __name__ == '__main__':
    main()
