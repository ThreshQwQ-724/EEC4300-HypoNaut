"""Compare path-guided Qwen edits and a local heuristic on the same failed drafts."""
import argparse
from collections import Counter
from itertools import combinations, permutations
from pathlib import Path
from statistics import mean
from time import perf_counter

import requests

from common import HERE, digest, load_upstream
from run_feedback_study import graph_line
from run_independent_rewire import path_rank, verified
from run_single_qwen_feedback_rescue import SOURCE, feedback_for
from run_uot_debate_rescue import (
    BASE_URL, DATA, LIMIT, MODEL, NODES, REFERENCE, apply_edit, decode,
    forbidden_edges, read, request, schema, write_json,
)


FEEDBACK_RUN = HERE/'runs/single_qwen_feedback_rescue_20261113'
METHODS = {'reselect': 'Qwen 原菜单重选', 'program': '程序局部修复', 'qwen': 'Qwen 定向改边'}


def graph_key(edges):
    return tuple(map(tuple, edges))


def select_program(parent, allowed, excluded, obs):
    """Rank the shared edit space by local reachability, without a full DAG check."""
    old = set(parent)
    additions = sorted(allowed-old)
    first, best, considered = None, None, 0
    for remove_count in range(3):
        for add_count in range(3):
            if not (remove_count or add_count) or len(old)-remove_count+add_count > LIMIT:
                continue
            for removed in combinations(parent, remove_count):
                kept = old-set(removed)
                for added in combinations(additions, add_count):
                    graph = tuple(sorted(kept.union(added)))
                    if graph in excluded:
                        continue
                    considered += 1
                    rank = path_rank(graph, obs, (removed, added))
                    if best is None or rank < best:
                        first, best = graph, rank
    return first, considered


def select_qwen(session, path, seed, parent, allowed, excluded, obs, feedback):
    label = '->'.join
    fmt = schema({key: {'type': 'array', 'maxItems': 2, 'uniqueItems': True,
                        'items': {'enum': [label(edge) for edge in edges] or ['unavailable']}}
                  for key, edges in (('remove', parent), ('add', sorted(allowed-set(parent))))})
    evidence = '\n'.join(row['string'] for row in obs)
    prompt = (
        'Repair the rejected DAG using the exact verifier feedback below. Return one JSON object '
        'with arrays remove and add, each with zero to two edge strings such as "A->B". '
        'Remove only existing edges and add only absent edges allowed by the schema. '
        f'The result must be acyclic and have at most {LIMIT} edges. '
        'An intervention must reach exactly its nodes marked 1, along paths of any length. '
        'Break each forbidden path or cycle and restore missing required paths. Preserve the '
        'other observed effects. The repaired edge set must differ from every excluded graph. '
        'Return the edits only.\nRejected graph: ' + graph_line(parent)
        + '\nVerifier feedback:\n' + feedback + '\nObservations:\n' + evidence
        + '\nExcluded graphs:\n' + '\n'.join(map(graph_line, sorted(excluded))))
    call = request(session, path, prompt, seed, fmt)
    answer = decode(call)
    candidate = apply_edit(answer, parent, allowed)
    reason = 'inadmissible_edit' if candidate is None else 'excluded' if candidate in excluded else None
    return (candidate if reason is None else None), {
        'proposed': candidate, 'rejection': reason, 'seconds': call['seconds'],
        'tokens': call['response'].get('eval_count', 0)}


def run_case(session, folder, prior, feedback_row, observations, benchmark):
    case = prior['id']
    parent = graph_key(prior['A']['graph'])
    menu = {graph_key(g) for g in prior['menu']}
    prefix = {graph_key(g) for g in prior['prefix']}
    excluded = menu | prefix
    assert parent in menu and not verified(parent, benchmark, observations)
    feedback = feedback_for(parent, observations)
    assert feedback == feedback_row['diagnostic']
    allowed = set(permutations(NODES, 2))-forbidden_edges(observations)
    seed = read(SOURCE/case/'repeat.json')['request']['options']['seed']
    start = perf_counter()
    program, considered = select_program(parent, allowed, excluded, observations)
    program_seconds = perf_counter()-start
    model, detail = select_qwen(session, folder/f'{case}.json', seed, parent, allowed,
                                excluded, observations, feedback)
    choices = {'reselect': graph_key(feedback_row['picks']['feedback']), 'program': program, 'qwen': model}
    scores = {}
    for method, candidate in choices.items():
        if method != 'reselect' and candidate is not None:
            old, new = set(parent), set(candidate)
            assert candidate not in excluded and len(candidate) <= LIMIT
            assert len(old-new) <= 2 and len(new-old) <= 2 and new-old <= allowed
        hit = candidate is not None and verified(candidate, benchmark, observations)
        scores[method] = {'graph': candidate, 'probes': int(candidate is not None), 'hit': hit}
    assert scores['reselect']['hit'] == feedback_row['hits']['feedback']
    return {'id': case, 'truth': prior['truth'], 'prefix': len(prefix), 'parent': parent, 'feedback': feedback,
            'menu_has_legal': feedback_row['has_legal_retry'], 'scores': scores,
            'program_candidates_ranked': considered, 'program_seconds': program_seconds, 'qwen': detail}


def report(output, groups, provenance):
    lines = ['# HypoSpace 单 Qwen 错误路径定向修复', '',
             '从既有 75 道原始六节点题中复用全部首次失败状态。固定首图前缀、失败草稿和一次额外完整验证上限。'
             '程序与 Qwen 使用同一修复空间，最多删两条、加两条边。静态禁止边、已提交图和整个旧菜单排除。'
             '程序按两跳可达性差异排序，Qwen 收到验证器的具体路径反馈并自行组合改边。'
             '程序使用相同观测与失败图，直接以观测差异排序。原菜单重选沿用上轮带反馈的选择。', '',
             '所有最终候选均由原始评分器验证。事后使用旧菜单标签划分有解与无解状态。'
             '无效或重复的编辑占用提案机会，完整验证只用于可送验的图。', '',
             '| 提交上限 | 失败状态 | 方法 | 新合法图 | 旧菜单无解时新增 | 完整验证 | RR | VR / NR |',
             '|---:|---:|---|---:|---:|---:|---:|---:|']
    summary = {}
    for slots, rows in groups.items():
        block = {}
        for method, label in METHODS.items():
            hits = [r['scores'][method]['hit'] for r in rows]
            d = {'hits': sum(hits), 'probes': sum(r['scores'][method]['probes'] for r in rows),
                 'out_of_menu_hits': sum(hit for hit, r in zip(hits, rows) if not r['menu_has_legal']),
                 'rr': mean((r['prefix']+hit)/r['truth'] for hit, r in zip(hits, rows)),
                 'vr': mean((r['prefix']+hit)/int(slots) for hit, r in zip(hits, rows))}
            block[method] = d
            lines.append(f"| {slots} | {len(rows)} | {label} | {d['hits']} | {d['out_of_menu_hits']} | "
                         f"{d['probes']} | {d['rr']:.4%} | {d['vr']:.2%} |")
        gap = Counter(r['scores']['qwen']['hit']-r['scores']['program']['hit'] for r in rows)
        block.update(wtl=[gap[1], gap[0], gap[-1]], no_legal_menu=sum(not r['menu_has_legal'] for r in rows),
                     qwen_request_seconds=sum(r['qwen']['seconds'] for r in rows),
                     qwen_tokens=sum(r['qwen']['tokens'] for r in rows),
                     qwen_rejections=dict(Counter(r['qwen']['rejection'] for r in rows if r['qwen']['rejection'])),
                     program_seconds=sum(r['program_seconds'] for r in rows),
                     program_candidates_ranked=sum(r['program_candidates_ranked'] for r in rows))
        summary[slots] = block
    for slots, d in summary.items():
        lines += ['', f"- {slots} 槽组，Qwen 对程序修复胜／平／负为 {'／'.join(map(str, d['wtl']))}。"
                      f"旧菜单无解 {d['no_legal_menu']} 题。Qwen 累计请求 {d['qwen_request_seconds']:.2f} 秒，"
                      f"程序排序累计 {d['program_seconds']:.3f} 秒。"]
    lines += ['', 'RR、VR 和 NR 仅在首次失败的子集上统计，包含共享前缀。'
              '每题修复一次后停止。程序的局部排序工作单独记录。此前双 Agent 与单 Qwen 重选结果用于固定输入。', '']
    write_json(output/'results.json', {**provenance, 'summary': summary, 'groups': groups})
    (output/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(summary, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    reference = read(REFERENCE/'config.json')
    original = read(SOURCE/'results.json')
    feedback = read(FEEDBACK_RUN/'results.json')
    assert original['config']['data_sha'] == digest(DATA)
    assert feedback['source_sha'] == digest(SOURCE/'results.json')
    module = load_upstream('causal', 'run_causal_benchmark.py')
    assert digest(module.__file__) == reference['evaluator_sha']
    benchmark = module.CausalBenchmarkEnhanced(str(REFERENCE/'observations.json'))
    tasks = {task['id']: task for task in read(DATA)}
    groups = {}
    with requests.Session() as session:
        session.trust_env = False
        tags = session.get(BASE_URL+'/api/tags', timeout=10)
        tags.raise_for_status()
        assert next(r for r in tags.json()['models'] if r['name'] == MODEL)['digest'] == reference['model']
        for slots, rows in original['groups'].items():
            pairs = zip(rows, feedback['groups'][slots], strict=True)
            results = []
            for prior, f in pairs:
                assert prior['id'] == f['id']
                if not f['first_valid']:
                    results.append(run_case(session, output, prior, f, tasks[prior['id']]['observations'], benchmark))
            groups[slots] = results
            print(f'{slots} slots: {len(results)} failed drafts', flush=True)
    report(output, groups, {'runner_sha': digest(__file__), 'data_sha': digest(DATA),
                            'model_digest': reference['model'], 'feedback_sha': digest(FEEDBACK_RUN/'results.json')})


if __name__ == '__main__':
    main()
