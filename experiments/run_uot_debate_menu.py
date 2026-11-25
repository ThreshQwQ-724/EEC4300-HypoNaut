"""Freeze failed single-edge states and compare ranking of unverified Shift candidates."""
import argparse
from itertools import combinations, permutations
from pathlib import Path
import random

import requests

from common import HERE, digest, load_upstream
from run_qwen_guided_edits import decode, request, schema
from run_uot_debate_rescue import DATA, LIMIT, MODEL, NODES, REFERENCE, SECOND_AGENT_SEED_OFFSET
from run_uot_debate_rescue import BASE_URL, forbidden_edges, graph_line, read, write_json


SOURCE = HERE/'runs/uot_debate_rescue_20261113'
MENU_SIZE = 8


def make_menu(emitted, checked, allowed, expanded=False):
    prior = [set(graph) for graph in emitted]
    candidates = {}
    for parent in reversed(prior):
        available = sorted(allowed-parent)
        sizes = ((1, 1), (1, 2), (2, 1), (2, 2)) if expanded else ((1, 1),)
        for remove_count, add_count in sizes:
            if len(parent)-remove_count+add_count > LIMIT:
                continue
            for removed in combinations(sorted(parent), remove_count):
                retained = parent - set(removed)
                for added in combinations(available, add_count):
                    edges = retained | set(added)
                    if any((b, a) in edges for a, b in added):
                        continue
                    candidate = tuple(sorted(edges))
                    if candidate in checked or candidate in candidates:
                        continue
                    if min(len(edges ^ graph) for graph in prior) >= 2:
                        candidates[candidate] = (removed, added)
    return candidates


def run(output, benchmark, session, task, source_row, seed, expanded, path_rank=False):
    case = task['id']
    obs = task['observations']
    result = source_row['runs']['single']
    emitted = [tuple(map(tuple, g)) for g in result['graphs'][:source_row['prefix_count']]]
    checked = set(emitted)
    allowed = set(permutations(NODES, 2))-forbidden_edges(obs)
    evidence = '\n'.join(row['string'] for row in obs)
    records = []
    for slot in result['trace']:
        if slot['graph'] is None:
            continue
        picked = tuple(map(tuple, slot['graph']))
        checked.add(picked)
        if slot['accepted']:
            emitted.append(picked)
            continue
        edits = make_menu(emitted, checked, allowed, expanded)
        if not edits:
            continue
        if path_rank:
            def priority(graph):
                children = {n: set() for n in NODES}
                for a, b in graph:
                    children[a].add(b)
                errors = 0
                for row in obs:
                    direct = children[row['perturbed_node']]
                    reachable = direct | {b for a in direct for b in children[a]}
                    errors += len(reachable ^ {n for n in NODES if row['effects'][n]})
                return errors, sum(map(len, edits[graph])), graph

            options = sorted(edits, key=priority)[:MENU_SIZE]
        elif expanded:
            positive_sources = {row['perturbed_node'] for row in obs if any(row['effects'].values())}
            options = sorted(edits, key=lambda g: (
                len(positive_sources - {a for a, _ in g}),
                -sum(row['effects'][b] for a, b in g for row in obs if row['perturbed_node'] == a),
                sum(map(len, edits[g])), g))[:MENU_SIZE]
        else:
            options = list(edits)[:MENU_SIZE]
        program = options[0]
        call_seed = seed+slot['slot']
        random.Random(call_seed).shuffle(options)
        lines = []
        for index, graph in enumerate(options, 1):
            removed, added = edits[graph]
            lines.append(f'{index}. remove {", ".join(map("->".join, removed))}, '
                         f'add {", ".join(map("->".join, added))}. {graph_line(graph)}')
        prompt = ('Choose the graph that satisfies all observations. A perturbed node reaches exactly the nodes '
                  'marked 1, via any directed path, and all graphs must be acyclic. Check deleted '
                  'and added edges. All options are new and have not been verified. Return a JSON choice number.\n'
                  'Observations:\n'+evidence+'\nCandidates:\n'+'\n'.join(lines))
        fmt = schema({'choice': {'type': 'integer', 'minimum': 1, 'maximum': len(options)}})
        folder = output/case
        folder.mkdir(exist_ok=True)
        a = request(session, folder/f'{slot["slot"]:02}_A.json', prompt, call_seed, fmt)
        draft = decode(a).get('choice')
        a_choice = options[draft-1] if type(draft) is int and 1 <= draft <= len(options) else program
        b_prompt = (prompt+f'\nAgent A selected option {options.index(a_choice)+1}. Independently audit '
                    'that choice for cycles and incorrect reachability. If it passes, retain it. If it '
                    'fails, choose a different option that satisfies all observations.')
        b = request(session, folder/f'{slot["slot"]:02}_B.json', b_prompt, call_seed+SECOND_AGENT_SEED_OFFSET, fmt)
        revision = decode(b).get('choice')
        b_choice = options[revision-1] if type(revision) is int and 1 <= revision <= len(options) else a_choice

        # Post-run diagnostics do not enter either decision. Each choice spends one graph check.
        legal = {}
        for graph in options:
            parsed = benchmark.parse_llm_response(graph_line(graph))
            legal[graph] = parsed is not None and benchmark.validate_hypothesis(parsed, obs)
        records.append({'case': case, 'slot': slot['slot'], 'menu': options,
                        'legal_options': [g for g in options if legal[g]], 'choices': {
                            label: {'graph': graph, 'valid': legal[graph]} for label, graph in (
                                ('program', program), ('qwen', a_choice), ('debate', b_choice))},
                        'request_seconds': a['seconds']+b['seconds']})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expanded', action='store_true')
    parser.add_argument('--path-rank', action='store_true')
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    reference = read(REFERENCE/'config.json')
    config = read(SOURCE/'config.json')
    assert config['data_sha'] == digest(DATA)
    module = load_upstream('causal', 'run_causal_benchmark.py')
    benchmark = module.CausalBenchmarkEnhanced(str(REFERENCE/'observations.json'))
    tasks = {task['id']: task for task in read(DATA)}
    rows = []
    with requests.Session() as session:
        session.trust_env = False
        tags = session.get(BASE_URL+'/api/tags', timeout=10)
        tags.raise_for_status()
        assert next(row for row in tags.json()['models'] if row['name'] == MODEL)['digest'] == config['digest']
        for index, case in enumerate(config['cases']):
            baseline = read(SOURCE/case/'result.json')
            rows.extend(run(output, benchmark, session, tasks[case], baseline,
                            reference['seeds'][index % len(reference['seeds'])],
                            args.expanded or args.path_rank, args.path_rank))
    assert rows
    totals = {method: sum(row['choices'][method]['valid'] for row in rows)
              for method in ('program', 'qwen', 'debate')}
    fixed = sum(not r['choices']['qwen']['valid'] and r['choices']['debate']['valid'] for r in rows)
    harmed = sum(r['choices']['qwen']['valid'] and not r['choices']['debate']['valid'] for r in rows)
    cases = len({row['case'] for row in rows})
    solvable = sum(bool(row['legal_options']) for row in rows)
    write_json(output/'results.json', {'source_config_sha': digest(SOURCE/'config.json'),
                                       'runner_sha': digest(__file__), 'methods': totals,
                                       'expanded': args.expanded or args.path_rank, 'path_rank': args.path_rank,
                                       'cases': cases, 'solvable': solvable,
                                       'fixed': fixed, 'harmed': harmed, 'rows': rows})
    setting = ('每组共享按局部出边和已知正效应排序的至多 8 个候选，每个候选同时删除与新增 1–2 条边。'
               if args.expanded else '各组共享最多 8 个未验证的一删一增候选。')
    if args.path_rank:
        setting = ('各组共享最多 8 个未验证的联合改边候选，按两跳内可达性与观测的差异排序。'
                   '该排序不检查更长路径或完整无环性，耗费相同的菜单构造工作。')
    lines = ['# E-UoT 双 Qwen 联合改边定点复核', '',
             f'从压力测试的程序单边分支冻结 {cases} 道题的 {len(rows)} 个失败槽。'
             + setting + '每个候选都与任一已提交图至少相差两条边。'
             '程序取菜单首项。A 从同一菜单选编号，B 审核 A 的选择并可改选。'
             '每组只评估最终选中的一张图。事后检查完整菜单，用于统计有解槽。', '',
             f'完整菜单中有合法图的槽位为 {solvable}/{len(rows)}。'
             f'程序、A 和 Debate 分别选中 {totals["program"]}、{totals["qwen"]} 和 {totals["debate"]} 张合法图。'
             f'B 救回 {fixed} 次 A 的失败，同时把 {harmed} 次 A 的正确选择改错。', '',
             f'A 和 B 各调用 {len(rows)} 次，总请求时间 {sum(r["request_seconds"] for r in rows):.2f} 秒。'
             '本轮固定在原先程序的失败状态，结果用于诊断选图，不能作为整题 RR。', '']
    (output/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(len(rows), cases, solvable, totals, fixed, harmed, flush=True)


if __name__ == '__main__':
    main()
