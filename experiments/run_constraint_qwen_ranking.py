"""Compare one-probe Qwen ranking with program ranking on a shared candidate menu."""
import argparse
from itertools import permutations
from pathlib import Path
import random
import re
from statistics import mean

import requests

from common import HERE, digest
from graph_completion import complete_observation_count
from run_exploratory_uot import make_edits, ranked_groups
from run_feedback_study import graph_line
from run_formal_agent_debate import REFERENCE
from run_formal_causal_comparison import DATA, LIMIT, MENU, NODES, SLOTS, forbidden_edges, program_anchor, read, valid
from run_four_groups import BASE_URL, MODEL
from run_local_causal import write_json
from run_numbered_agent_search import run as risk_rank
from run_qwen_guided_edits import decode, request, schema


SEED = 20261113
METHODS = {'program': '程序局部规则', 'qwen': '约束提示＋Qwen', 'risk': '既有程序避险'}


def two_hop(edit):
    a, b = edit['edge']
    parent = set(edit['parent'])
    return any((a, m) in parent and (m, b) in parent for m in NODES)


def local_contradiction(graph, edit, positive_sources):
    a, b = edit['edge']
    if edit['action'] == 'add':
        return (b, a) in graph
    return a in positive_sources and not any(source == a for source, _ in graph)


def select_qwen(session, path, seed, menu, edits, observations):
    # Stable but shuffled IDs keep the program's first choice out of a fixed position.
    options = sorted(menu)
    random.Random(seed).shuffle(options)
    lines = []
    for i, graph in enumerate(options, 1):
        edit = edits[graph]
        a, b = edit['edge']
        lines.append(f'{i}. {edit["action"]} {a}->{b}; {graph_line(graph)}')
    constraints = []
    for obs in observations:
        source = obs['perturbed_node']
        targets = ''.join(n for n in NODES if obs['effects'][n]) or 'none'
        constraints.append(f'{source} must reach exactly {targets}')
    prompt = (
        'Select one unverified graph by number. Every node named in a required reachability set '
        'must have a directed path from its source. Other nodes must not be reachable. '
        'A path can pass through intermediate nodes. Reject cycles. For deletions, look for '
        'an alternative path before dropping a required effect. For additions, follow the new '
        'edge downstream and check that it does not introduce a forbidden effect or a cycle. '
        'Choose a graph satisfying ALL these constraints. Every candidate is distinct and within '
        'the edge limit. Return only the JSON choice number.\n'
        + '\n'.join(constraints) + '\nUnverified candidates:\n' + '\n'.join(lines))
    call = request(session, path, prompt, seed, schema({
        'choice': {'type': 'integer', 'minimum': 1, 'maximum': len(options)}}))
    answer = decode(call).get('choice')
    parsed = type(answer) is int and 1 <= answer <= len(options)
    return (options[answer-1] if parsed else menu[0]), call, parsed


def solve_case(task, anchor, method, session, folder, seed, *, local_filter=False):
    observations = task['observations']
    exact = complete_observation_count(NODES, LIMIT, observations)
    allowed = set(permutations(NODES, 2))-forbidden_edges(observations)
    positive_sources = {obs['perturbed_node'] for obs in observations if any(obs['effects'].values())}
    emitted, checked, frontier, trace = [anchor], {anchor}, {}, []
    rejected = set()
    for slot in range(2, SLOTS+1):
        if len(emitted) == exact:
            break
        latest = emitted[-1]
        if latest not in frontier:
            frontier[latest] = make_edits(latest, allowed, False)
        _, entries = ranked_groups(emitted, frontier, checked, slot, False)
        menu = []
        for _, graph, edit in entries:
            if graph in menu or graph in rejected:
                continue
            if local_filter and local_contradiction(graph, edit, positive_sources):
                rejected.add(graph)
                continue
            menu.append(graph)
            if len(menu) == MENU:
                break
        if not menu:
            break
        edits = {graph: edit for _, graph, edit in entries}
        # Either adding or removing an edge with an existing two-hop path
        # preserves reachability for a valid parent.
        program = next((graph for graph in menu if two_hop(edits[graph])), menu[0])
        picked, call, parsed = program, None, None
        if method == 'qwen' and len(menu) > 1 and not two_hop(edits[program]):
            picked, call, parsed = select_qwen(session, folder/f'Qwen_{slot:02}.json',
                                              seed+slot, menu, edits, observations)
        accepted = valid(picked, observations)
        checked.add(picked)
        if accepted:
            emitted.append(picked)
        trace.append({'slot': slot, 'menu': menu, 'picked': picked, 'program': program,
                      'accepted': accepted, 'parsed': parsed, 'call': call is not None,
                      'seconds': call['seconds'] if call else 0,
                      'tokens': call['response'].get('eval_count', 0) if call else 0})
    assert len(emitted) == len(set(emitted)) <= min(SLOTS, task['n_ground_truths'])
    return {'graphs': emitted, 'found': len(emitted), 'probes': len(trace),
            'calls': sum(s['call'] for s in trace), 'seconds': sum(s['seconds'] for s in trace),
            'tokens': sum(s['tokens'] for s in trace), 'local_rejections': sorted(rejected), 'trace': trace}


def summarize(records):
    stats = {}
    for method in METHODS:
        rows = [r['runs'][method] for r in records]
        stats[method] = {'found': sum(r['found'] for r in rows),
                         'rr': mean(r['found']/record['truth'] for r, record in zip(rows, records, strict=True)),
                         'vr': mean(r['found']/SLOTS for r in rows),
                         **{key: sum(r.get(key, 0) for r in rows) for key in ('probes', 'calls', 'seconds', 'tokens')}}
    same_state = [s for r in records for s in r['runs']['qwen']['trace'] if s['call']]
    comparison = {'calls': len(same_state), 'qwen_correct': sum(s['accepted'] for s in same_state),
                  'program_correct': sum(s['program_accepted'] for s in same_state),
                  'rescues': sum(s['accepted'] and not s['program_accepted'] for s in same_state),
                  'losses': sum(not s['accepted'] and s['program_accepted'] for s in same_state)}
    deltas = [r['runs']['qwen']['found']-r['runs']['program']['found'] for r in records]
    return {'methods': stats, 'same_state': comparison,
            'paired_wtl': [sum(d > 0 for d in deltas), sum(d == 0 for d in deltas), sum(d < 0 for d in deltas)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--local-filter', action='store_true')
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    source = read(REFERENCE/'config.json')
    assert source['source_sha'] == digest(DATA)
    tasks = {task['id']: task for task in read(DATA)}
    config_path = output/'config.json'
    if config_path.exists():
        cases = read(config_path)['cases']
    else:
        used = set()
        for path in (HERE/'runs').glob('*/config.json'):
            previous_cases = set(re.findall(r'\bn[1-6]_\d{3}\b', path.read_text(encoding='utf-8')))
            if previous_cases != tasks.keys():
                used.update(previous_cases)
        rng = random.Random(SEED + int(args.local_filter))
        cases = [case for n in (2, 3, 4, 5) for case in rng.sample(sorted(
            task['id'] for task in tasks.values() if task['id'] not in used
            and task['n_observations'] == n and task['n_ground_truths'] >= SLOTS), 6)]
    config = {'cases': cases, 'seed': SEED, 'model': MODEL, 'model_digest': source['model'],
              'slots': SLOTS, 'probes_per_slot': 1, 'menu': MENU, 'data_sha': digest(DATA),
              'local_filter': args.local_filter,
              'code_sha': {name: digest(HERE/name) for name in (Path(__file__).name,
                  'run_numbered_agent_search.py', 'run_exploratory_uot.py', 'run_qwen_guided_edits.py')}}
    if config_path.exists() and read(config_path) != config:
        raise ValueError('Output directory contains a different experiment configuration')
    write_json(config_path, config)
    records = []
    with requests.Session() as session:
        session.trust_env = False
        tags = session.get(BASE_URL+'/api/tags', timeout=10)
        tags.raise_for_status()
        assert next(row for row in tags.json()['models'] if row['name'] == MODEL)['digest'] == source['model']
        for i, case in enumerate(cases, 1):
            task = tasks[case]
            folder = output/case
            folder.mkdir(exist_ok=True)
            path = folder/'result.json'
            if path.exists():
                record = read(path)
            else:
                obs = task['observations']
                anchor = program_anchor(NODES, LIMIT, obs)
                assert anchor is not None and valid(anchor, obs)
                seed = source['seeds'][(i-1) % len(source['seeds'])]
                runs = {method: solve_case(task, anchor, method, session, folder, seed,
                                           local_filter=args.local_filter)
                        for method in ('program', 'qwen')}
                runs['risk'] = risk_rank(task, anchor, 1, 'program', None, None, None)
                for step in runs['qwen']['trace']:
                    step['program_accepted'] = valid(step['program'], obs)
                for result in runs.values():
                    graphs = result['graphs']
                    assert len(graphs) == len(set(graphs)) and all(valid(graph, obs) for graph in graphs)
                    assert all(not valid(graph, obs) for graph in result.get('local_rejections', ()))
                record = {'id': case, 'truth': task['n_ground_truths'], 'runs': runs}
                temporary = path.with_suffix('.tmp')
                write_json(temporary, record)
                temporary.replace(path)
            records.append(record)
            print(f'{i}/{len(cases)} {case} {[record["runs"][m]["found"] for m in METHODS]}', flush=True)
    summary = summarize(records)
    lines = ['# HypoSpace 约束辅助 Qwen 排序', '',
             '从先前抽样实验配置未使用的 2–5 次干预题各抽 6 道，每题 8 槽，首图共享。'
             '后续每槽只能验证一个候选。程序局部规则与 Qwen 共享增删边候选生成和菜单。'
             '当候选含可保留可达性的两跳见证时，两组都选该候选。'
             '其他槽位由程序取首项，或让 Qwen 根据显式正负可达性约束选择编号。'
             '只验证被选中的图，反事实复核在整题搜索结束后进行。', '',
             '| 方法 | 不同合法图 | RR | VR / NR | 探查 | Qwen 调用 | 输出 token | 请求秒数 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for method, d in summary['methods'].items():
        lines.append(f"| {METHODS[method]} | {d['found']} | {d['rr']:.4%} | {d['vr']:.2%} | "
                     f"{d['probes']} | {d['calls']} | {d['tokens']} | {d['seconds'] if method == 'qwen' else 0:.2f} |")
    d = summary['same_state']
    if args.local_filter:
        lines += ['', '程序与 Qwen 都在生成菜单时排除两类可直接判错的候选。'
                  '新增反向边会形成二元环；删掉必有正效应的干预源的最后一条出边会丢失必要路径。'
                  '这两类检查只读边列表，另外记录筛除数，未计入完整图探查。', '',
                  '程序、Qwen 分支累计筛除数分别为 '
                  + '、'.join(str(sum(len(r['runs'][method]['local_rejections']) for r in records))
                             for method in ('program', 'qwen')) + '。']
    lines += ['', f"同一搜索状态下，Qwen 选对 {d['qwen_correct']}/{d['calls']}，程序首选正确 "
              f"{d['program_correct']}/{d['calls']}。Qwen 救回 {d['rescues']} 槽，选错 {d['losses']} 个程序本可通过的槽。"
              f"整题合法图数胜／平／负为 {' / '.join(map(str, summary['paired_wtl']))}。", '',
              '既有程序避险沿用原来的排序，作为更强的参考基线。'
              '本测试限制每槽一次探查。正式流程可使用更宽预算或直接构造。', '']
    write_json(output/'summary.json', summary)
    (output/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
