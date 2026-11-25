"""Test E-UoT-inspired Qwen graph edits where constructive search has coverage headroom."""
import argparse
from itertools import permutations
from pathlib import Path
from statistics import mean
from time import perf_counter

import requests

from common import HERE, digest, load_upstream
from run_exploratory_uot import PAPER, make_edits, ranked_groups
from run_feedback_study import graph_line
from run_formal_agent_debate import REFERENCE, SECOND_AGENT_SEED_OFFSET
from run_formal_causal_comparison import DATA, LIMIT, NODES, forbidden_edges, program_anchor, read
from run_four_groups import BASE_URL, MODEL
from run_local_causal import write_json
from run_qwen_guided_edits import decode, request, schema
from solve_partial_graph import constructive_prefix, solve


SLOTS = 16
METHODS = {'single': '程序单边', 'shift': '程序单边＋两跳 Shift',
           'qwen': 'Qwen 提案', 'debate': '双 Qwen Debate'}


def candidate_order(emitted, frontier, checked, slot, mixed):
    _, ordered = ranked_groups(emitted, frontier, checked, slot, mixed)
    return list(dict.fromkeys(graph for _, graph, _ in ordered))


def apply_edit(answer, parent, allowed):
    """Check edit syntax and edge membership before spending a graph probe."""
    edits = {}
    for name in ('remove', 'add'):
        values = answer.get(name)
        if not isinstance(values, list) or len(values) > 2:
            return None
        if any(not isinstance(v, str) or len(v) != 4 or v[1:3] != '->' for v in values):
            return None
        edges = {(v[0], v[-1]) for v in values}
        if len(edges) != len(values):
            return None
        edits[name] = edges
    original = set(parent)
    removed, added = edits['remove'], edits['add']
    if not (removed or added) or not removed <= original or added & original or not added <= allowed:
        return None
    graph = (original - removed) | added
    return tuple(sorted(graph)) if len(graph) <= LIMIT else None


def propose(session, folder, slot, seed, parent, allowed, observations, seen, debate):
    edge_label = lambda e: f'{e[0]}->{e[1]}'
    fmt = schema({name: {'type': 'array', 'maxItems': 2, 'uniqueItems': True,
                         'items': {'enum': values or ['unavailable']}}
                  for name, values in (
                      ('remove', [edge_label(e) for e in parent]),
                      ('add', [edge_label(e) for e in sorted(allowed-set(parent))]))})
    evidence = '\n'.join(f"{o['perturbed_node']} reaches exactly "
                         + (','.join(n for n in NODES if o['effects'][n]) or 'none') for o in observations)
    prompt = (
        'Find one new DAG by editing the parent graph. Nodes are A–F. An intervention reaches '
        'its descendants along directed paths, including indirect ones. Match every reachability '
        f'constraint below, avoid cycles and use at most {LIMIT} edges. The result must differ '
        'from every previously checked or submitted graph listed below. Use a single-edge '
        'refinement (Deepen) or change the mechanism with coordinated edits (Shift). For example, '
        'replace a direct edge with a two-edge path or rewire a required route while preserving its '
        'observed effects. You may remove up to two existing edges and add up to two new edges. '
        'Return JSON arrays remove and add with strings like "A->B". Use [] for an unused operation.\n'
        f'Parent: {graph_line(parent)}\nConstraints:\n{evidence}\nPreviously seen:\n'
        + '\n'.join(graph_line(g) for g in sorted(seen)))
    a = request(session, folder/f'{slot:02}_A.json', prompt, seed, fmt)
    calls = [a]
    draft = decode(a)
    revision = None
    if debate:
        critique = (
            '\n\nYou are Agent B auditing Agent A. Its proposal was '
            + a['response']['message'].get('content', '')
            + '\nApply the proposal mentally to the same parent. Check for duplicate edge sets, cycles, '
              'required paths that disappear, and forbidden paths that appear. Return the same edits '
              'if the proposal passes, or a corrected edit with a new graph. Use the same JSON schema.')
        b = request(session, folder/f'{slot:02}_B.json', prompt+critique, seed+SECOND_AGENT_SEED_OFFSET, fmt)
        calls.append(b)
        revision = apply_edit(decode(b), parent, allowed)
    return apply_edit(draft, parent, allowed), revision, calls


def run(task, anchor, prefix, method, benchmark, session, folder, seed):
    obs = task['observations']
    allowed = set(permutations(NODES, 2))-forbidden_edges(obs)
    emitted = list(prefix)
    checked = set(emitted)
    mixed = method == 'shift'
    frontier = {graph: make_edits(graph, allowed, mixed) for graph in emitted}
    trace = []
    start = perf_counter()
    for slot in range(len(prefix)+1, SLOTS+1):
        order = candidate_order(emitted, frontier, checked, slot, mixed)
        first = order[0] if order else None
        graph, origin = first, 'program'
        draft = revision = None
        calls = []
        parent = emitted[-1]
        if method in ('qwen', 'debate'):
            draft, revision, calls = propose(session, folder, slot, seed+slot, parent, allowed,
                                             obs, checked, method == 'debate')
            for choice, role in ((revision, 'B'), (draft, 'A')):
                if choice is not None and choice not in checked:
                    graph, origin = choice, role
                    break
        accepted = False
        outside = graph is not None and min(len(set(graph)^set(old)) for old in emitted) > 1
        if graph is not None:
            assert graph not in checked
            checked.add(graph)
            parsed = benchmark.parse_llm_response(graph_line(graph))
            accepted = parsed is not None and benchmark.validate_hypothesis(parsed, obs)
            if accepted:
                emitted.append(graph)
                frontier[graph] = make_edits(graph, allowed, mixed)
        trace.append({'slot': slot, 'graph': graph, 'accepted': accepted, 'origin': origin,
                      'draft': draft, 'revision': revision, 'outside_single': outside,
                      'changed_by_b': origin == 'B' and revision != draft,
                      'parent': parent, 'program_first': first, 'calls': len(calls),
                      'input_tokens': sum(c['response'].get('prompt_eval_count', 0) for c in calls),
                      'tokens': sum(c['response'].get('eval_count', 0) for c in calls),
                      'request_seconds': sum(c['seconds'] for c in calls)})
    assert len(emitted) == len(set(emitted)) <= min(SLOTS, task['n_ground_truths'])
    return {'found': len(emitted), 'rr': len(emitted)/task['n_ground_truths'], 'vr': len(emitted)/SLOTS,
            'probes': sum(t['graph'] is not None for t in trace),
            'outside_found': sum(t['outside_single'] and t['accepted'] for t in trace),
            'b_changes': sum(t['changed_by_b'] for t in trace),
            'calls': sum(t['calls'] for t in trace), 'tokens': sum(t['tokens'] for t in trace),
            'request_seconds': sum(t['request_seconds'] for t in trace), 'seconds': perf_counter()-start,
            'graphs': emitted, 'trace': trace}


def report(output, records):
    summary = {}
    for method in METHODS:
        rows = [r['runs'][method] for r in records]
        summary[method] = {**{key: sum(r[key] for r in rows) for key in (
            'found', 'probes', 'outside_found', 'b_changes', 'calls', 'tokens', 'request_seconds')},
            **{key: mean(r[key] for r in rows) for key in ('rr', 'vr')}}
    comparisons = {}
    for method in ('single', 'shift', 'qwen'):
        gaps = [r['runs']['debate']['found']-r['runs'][method]['found'] for r in records]
        comparisons[method] = [sum(d > 0 for d in gaps), sum(d == 0 for d in gaps), sum(d < 0 for d in gaps)]
    lines = ['# HypoSpace E-UoT × 双 Qwen Debate', '',
             '从原始六节点题的审计观测提取中，选出构造器加每槽一次旧版回退仍未达到 16 槽覆盖上限的全部题。'
             '因此本轮是针对基线失分的压力测试。合法解数用于样本筛选与评分。', '',
             '各组共享约束构造前缀，随后每槽最多完整验证一个候选。'
             '程序单边组沿用已有排序，程序 Shift 组同时枚举直接边与两跳路径替换。'
             'Qwen 可以对最新合法图自由组合至多两条删边及两条增边。'
             'Debate 的 B 看到 A 的编辑，审查后可保留或改写。'
             '程序先检查格式、静态边限制和重复。若 B 的改动不可用，尝试 A，再按程序单边排序补位。'
             '只有最终选中的图进行可达性验证，因此未验证的 A 不会免费充当备用图。', '',
             '| 方法 | 合法图 | RR | VR / NR | 后续完整探查 | 单边邻域外命中 | 模型调用 | 请求秒数 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for method, d in summary.items():
        lines.append(f"| {METHODS[method]} | {d['found']} | {d['rr']:.4%} | {d['vr']:.2%} | "
                     f"{d['probes']} | {d['outside_found']} | {d['calls']} | {d['request_seconds']:.2f} |")
    lines += ['', f"共享构造前缀共 {sum(r['prefix_count'] for r in records)} 张图。"
              '单边邻域外指与此前每张已提交图至少相差两条边。VR 与 NR 相同，空槽保留在分母。', '',
              '| 题目 | 合法解数 | 构造前缀 | 程序单边 | 程序 Shift | Qwen | Debate |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for row in records:
        lines.append(f"| {row['id']} | {row['truth']} | {row['prefix_count']} | "
                     + ' | '.join(str(row['runs'][m]['found']) for m in METHODS)+' |')
    lines += ['', f'方法参考 [E-UoT]({PAPER})。', '']
    write_json(output/'summary.json', {'methods': summary, 'debate_wtl': comparisons})
    (output/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    reference = read(REFERENCE/'config.json')
    assert digest(DATA) == reference['source_sha']
    upstream = load_upstream('causal', 'run_causal_benchmark.py')
    assert digest(upstream.__file__) == reference['evaluator_sha']
    benchmark = upstream.CausalBenchmarkEnhanced(str(REFERENCE/'observations.json'))
    cases = []
    for task in read(DATA):
        obs = task['observations']
        old = solve(NODES, LIMIT, obs, slots=SLOTS, checks_per_slot=1, constructive=True)
        if len(old['proposals']) < min(SLOTS, task['n_ground_truths']):
            cases.append(task)
    config = {'cases': [task['id'] for task in cases], 'slots': SLOTS, 'model': MODEL,
              'digest': reference['model'], 'data_sha': digest(DATA), 'runner_sha': digest(__file__),
              'solver_sha': digest(HERE/'solve_partial_graph.py'), 'evaluator_sha': digest(upstream.__file__)}
    path = output/'config.json'
    if path.exists() and read(path) != config:
        raise ValueError('Output directory contains a different experiment configuration')
    write_json(path, config)
    records = []
    with requests.Session() as session:
        session.trust_env = False
        tags = session.get(BASE_URL+'/api/tags', timeout=10)
        tags.raise_for_status()
        assert next(m for m in tags.json()['models'] if m['name'] == MODEL)['digest'] == config['digest']
        for index, task in enumerate(cases):
            case = task['id']
            folder = output/case
            folder.mkdir(exist_ok=True)
            path = folder/'result.json'
            if path.exists():
                row = read(path)
            else:
                obs = task['observations']
                anchor = program_anchor(NODES, LIMIT, obs)
                prefix, _, _ = constructive_prefix(NODES, LIMIT, obs, anchor, SLOTS)
                seed = reference['seeds'][index % len(reference['seeds'])]
                runs = {}
                for method in METHODS:
                    calls = folder/method
                    calls.mkdir(exist_ok=True)
                    runs[method] = run(task, anchor, prefix, method, benchmark, session, calls, seed)
                row = {'id': case, 'truth': task['n_ground_truths'], 'prefix_count': len(prefix), 'runs': runs}
                temporary = path.with_suffix('.tmp')
                write_json(temporary, row)
                temporary.replace(path)
            records.append(row)
            print(f"{index+1}/{len(cases)} {case} {[row['runs'][m]['found'] for m in METHODS]}", flush=True)
    report(output, records)


if __name__ == '__main__':
    main()
