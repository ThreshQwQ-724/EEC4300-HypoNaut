"""Test independent Qwen proposals after bounded single-edge search fails."""
import argparse
from itertools import permutations
from pathlib import Path

import requests

from common import HERE, digest, load_upstream
from run_uot_debate_menu import SOURCE, make_menu
from run_uot_debate_rescue import (
    BASE_URL, DATA, LIMIT, MODEL, NODES, REFERENCE, SECOND_AGENT_SEED_OFFSET,
    apply_edit, decode, forbidden_edges, graph_line, make_edits, program_anchor,
    read, request, schema, write_json,
)


ROLES = {
    'a': 'Focus on a route close to the current mechanism. Try a different intermediary.',
    'repeat': 'Focus on a route close to the current mechanism. Try a different intermediary.',
    'b': 'Explore a different mechanism. Consider changing which node first relays an observed effect, '
         'or splitting or merging routes while preserving the same observations.',
}


def verified(graph, benchmark, observations):
    parsed = benchmark.parse_llm_response(graph_line(graph))
    return parsed is not None and benchmark.validate_hypothesis(parsed, observations)


def exhaust(task, benchmark):
    obs = task['observations']
    allowed = set(permutations(NODES, 2)) - forbidden_edges(obs)
    anchor = program_anchor(NODES, LIMIT, obs)
    assert anchor is not None and verified(anchor, benchmark, obs)
    emitted, checked = [anchor], {anchor}
    for parent in emitted:
        for candidate in sorted(make_edits(parent, allowed, False)['deepen']):
            if candidate not in checked:
                checked.add(candidate)
                if verified(candidate, benchmark, obs):
                    emitted.append(candidate)
    assert len(emitted) <= task['n_ground_truths']
    return {'id': task['id'], 'truth': task['n_ground_truths'], 'found': len(emitted),
            'probes': len(checked)-1}


def ask(session, folder, role, seed, parent, allowed, excluded, observations):
    fmt = schema({name: {'type': 'array', 'minItems': 1, 'maxItems': 2, 'uniqueItems': True,
                         'items': {'enum': ['->'.join(edge) for edge in sorted(edges)] or ['unavailable']}}
                  for name, edges in (('remove', parent), ('add', allowed-set(parent)))})
    prompt = (
        'Propose one new DAG by reconnecting a path in the supplied parent. '
        'Remove one or two existing edges and add one or two absent edges in a single operation. '
        'For each intervention, exactly the nodes marked 1 must be reachable along directed paths. '
        f'Keep the graph acyclic and at most {LIMIT} edges. The final edge set must not appear in the '
        'excluded list and must differ by at least two edges from every submitted graph. '
        + ROLES[role] + ' Return JSON arrays remove and add, using strings such as "A->B".\n'
        + 'Parent: ' + graph_line(parent)
        + '\nObservations:\n' + '\n'.join(row['string'] for row in observations)
        + '\nExcluded graphs:\n' + '\n'.join(map(graph_line, sorted(excluded))))
    call = request(session, folder/f'{role}.json', prompt, seed, fmt)
    answer = decode(call)
    candidate = apply_edit(answer, parent, allowed)
    if not answer.get('remove') or not answer.get('add'):
        candidate = None
    return candidate, {'seconds': call['seconds'], 'tokens': call['response'].get('eval_count', 0)}


def path_rank(graph, observations, edit):
    children = {node: set() for node in NODES}
    for source, target in graph:
        children[source].add(target)
    errors = 0
    for row in observations:
        direct = children[row['perturbed_node']]
        reachable = direct | {target for middle in direct for target in children[middle]}
        errors += len(reachable ^ {node for node in NODES if row['effects'][node]})
    return errors, sum(map(len, edit)), graph


def test_case(task, baseline, benchmark, session, output, seed):
    obs = task['observations']
    allowed = set(permutations(NODES, 2)) - forbidden_edges(obs)
    emitted = [tuple(map(tuple, g)) for g in baseline['graphs'][:len(baseline['graphs'])-
               sum(step['accepted'] for step in baseline['trace'])]]
    checked = set(emitted)
    rows = []
    for step in baseline['trace']:
        if step['graph'] is not None:
            graph = tuple(map(tuple, step['graph']))
            checked.add(graph)
            if step['accepted']:
                emitted.append(graph)
                continue
        candidates = make_menu(emitted, checked, allowed, expanded=True)
        if not candidates:
            continue
        folder = output/task['id']/str(step['slot'])
        folder.mkdir(parents=True, exist_ok=True)
        parent = emitted[-1]
        far = max(emitted, key=lambda graph: (len(set(graph)^set(parent)), graph))
        drafts = {}
        for role in ROLES:
            # The two second proposals share seed and parent to isolate the role prompt.
            # Neither sees A's proposed edit or its validation outcome.
            offset = 0 if role == 'a' else SECOND_AGENT_SEED_OFFSET
            choice, cost = ask(session, folder, role, seed+step['slot']+offset,
                               parent if role == 'a' else far, allowed, checked, obs)
            drafts[role] = {'graph': choice, **cost}
        ordered = sorted(candidates, key=lambda graph: path_rank(graph, obs, candidates[graph]))
        # All legality checks below are retrospective. The two-probe scores count only
        # the distinct, structurally admissible proposals in each method's own list.
        proposals = {role: data['graph'] for role, data in drafts.items()}
        for role, graph in proposals.items():
            drafts[role]['admissible'] = graph in candidates
            drafts[role]['valid'] = graph in candidates and verified(graph, benchmark, obs)
        legal = {graph for graph in candidates if verified(graph, benchmark, obs)}
        methods = {'program': ordered[:2], 'repeat': [proposals['a'], proposals['repeat']],
                   'dual': [proposals['a'], proposals['b']]}
        scores = {}
        for name, graphs in methods.items():
            probes = set(graphs) & candidates.keys()
            scores[name] = {'probes': len(probes), 'found': len(probes & legal)}
        rows.append({'case': task['id'], 'slot': step['slot'], 'emitted': emitted.copy(),
                     'drafts': drafts, 'program_choices': ordered[:2], 'legal_candidates': sorted(legal),
                     'candidate_count': len(candidates), 'scores': scores})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    source = read(SOURCE/'config.json')
    reference = read(REFERENCE/'config.json')
    assert source['data_sha'] == digest(DATA) and source['digest'] == reference['model']
    upstream = load_upstream('causal', 'run_causal_benchmark.py')
    assert digest(upstream.__file__) == reference['evaluator_sha']
    benchmark = upstream.CausalBenchmarkEnhanced(str(REFERENCE/'observations.json'))
    tasks = {task['id']: task for task in read(DATA)}
    exhaustion = [exhaust(task, benchmark) for task in tasks.values() if task['n_ground_truths'] <= 64]
    print(f'Exhaustion audit complete: {len(exhaustion)} cases', flush=True)
    rows = []
    with requests.Session() as session:
        session.trust_env = False
        tags = session.get(BASE_URL+'/api/tags', timeout=10)
        tags.raise_for_status()
        assert next(item for item in tags.json()['models'] if item['name'] == MODEL)['digest'] == source['digest']
        for index, case in enumerate(source['cases']):
            prior = read(SOURCE/case/'result.json')
            baseline = prior['runs']['single']
            if baseline['found'] < min(source['slots'], prior['truth']):
                rows.extend(test_case(tasks[case], baseline, benchmark, session, output,
                                      reference['seeds'][index % len(reference['seeds'])]))
                print(f'{case}: {len(rows)} cumulative frozen states', flush=True)
    assert rows
    stats = {name: {key: sum(row['scores'][name][key] for row in rows) for key in ('found', 'probes')}
             for name in ('program', 'repeat', 'dual')}
    for method, role in (('repeat', 'repeat'), ('dual', 'b')):
        stats[method]['second_unique'] = sum(
            row['drafts'][role]['valid'] and row['drafts'][role]['graph'] != row['drafts']['a']['graph']
            for row in rows)
        stats[method]['seconds'] = sum(row['drafts'][stage]['seconds'] for row in rows for stage in ('a', role))
    solvable = sum(bool(row['legal_candidates']) for row in rows)
    covered = sum(row['found'] == row['truth'] for row in exhaustion)
    write_json(output/'results.json', {'runner_sha': digest(__file__), 'source_sha': digest(SOURCE/'config.json'),
                                      'model': MODEL, 'digest': source['digest'], 'exhaustion': exhaustion,
                                      'states_with_legal_rewire': solvable, 'methods': stats, 'states': rows})
    lines = ['# E-UoT 独立路径重接对照', '',
             f'在 {len(exhaustion)} 道合法解不超过 64 张的原始六节点题上，从程序首图完整遍历合法单边邻居。'
             f'其中 {covered} 道达到答案数。该诊断用于确认是否存在真实耗尽后的遗漏。', '',
             f'另外在旧版 16 槽压力测试中仍未达上限的题目上冻结 {len(rows)} 个失败状态。'
             f'其中 {solvable} 个状态在一到两删、一到两增的邻域里有合法候选。'
             '这一部分对应紧预算停滞。A 从最新父图出发，第二次采样和 B 均从距它最远的已提交父图出发，'
             '并共享种子。两者独立提案，只改变第二次的角色提示。'
             '每组至多验证两张不同候选，未用完的预算不补位。程序以两跳可达性误差排序。'
             '枚举全邻域验证仅用于事后诊断。重复命中同一张图的不同槽不能直接相加作为整题 RR。', '',
             '| 方法 | 合法命中 | 实际验证 | 第二次提案新增命中 | 请求秒数 |',
             '|---|---:|---:|---:|---:|']
    for name, label in (('program', '程序重接'), ('repeat', '单 Qwen 两次提案'), ('dual', '双 Agent 分工')):
        row = stats[name]
        lines.append(f"| {label} | {row['found']} | {row['probes']} | {row.get('second_unique', '—')} | "
                     f"{row.get('seconds', 0):.2f} |")
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(stats, flush=True)


if __name__ == '__main__':
    main()
