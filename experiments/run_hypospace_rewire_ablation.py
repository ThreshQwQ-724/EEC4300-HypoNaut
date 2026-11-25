"""Paired second-proposal ablation on original HypoSpace observation sets."""
import argparse
from itertools import permutations
from math import comb
from pathlib import Path
import random
from statistics import mean

import requests

from common import digest, load_upstream
from run_independent_rewire import path_rank, verified
from run_uot_debate_menu import MENU_SIZE, make_menu
from run_uot_debate_rescue import (
    BASE_URL, DATA, LIMIT, MODEL, NODES, REFERENCE, SECOND_AGENT_SEED_OFFSET,
    constructive_prefix, decode, forbidden_edges, graph_line, program_anchor, read, request, schema, write_json,
)


ROLE = ('\nAct as the path-rewiring specialist. For each required downstream node, trace a '
        'route from its intervention source after the edit. Check that the relay is still reachable, '
        'and reject routes into nodes marked 0 and any cycles.')
METHODS = {'program': '程序两次', 'repeat': 'A＋普通第二次', 'b': 'A＋路径重接 B'}


def states(tasks, slots, excluded):
    for task in tasks:
        if task['id'] in excluded or task['n_observations'] == len(NODES):
            continue
        obs = task['observations']
        anchor = program_anchor(NODES, LIMIT, obs)
        prefix, _, _ = constructive_prefix(NODES, LIMIT, obs, anchor, slots)
        if len(prefix) > slots-2:
            continue
        allowed = set(permutations(NODES, 2))-forbidden_edges(obs)
        edits = make_menu(list(prefix), set(prefix), allowed, expanded=True)
        if len(edits) >= 3:
            menu = sorted(edits, key=lambda graph: path_rank(graph, obs, edits[graph]))[:MENU_SIZE]
            yield task, prefix, menu, edits


def choose(session, path, seed, menu, edits, observations, suffix=''):
    options = list(menu)
    random.Random(seed).shuffle(options)
    lines = []
    for i, graph in enumerate(options, 1):
        removed, added = edits[graph]
        lines.append(f'{i}. Remove {", ".join(map("->".join, removed))}; '
                     f'add {", ".join(map("->".join, added))}. {graph_line(graph)}')
    prompt = ('Choose one unverified candidate graph. A perturbed node reaches exactly the nodes '
              'marked 1 through directed paths. Every candidate has a different edge set and meets '
              'the edge limit. Choose an acyclic graph satisfying all observations. Return only a '
              'JSON choice number.\nObservations:\n' + '\n'.join(row['string'] for row in observations)
              + '\nCandidates:\n' + '\n'.join(lines) + suffix)
    call = request(session, path, prompt, seed, schema({
        'choice': {'type': 'integer', 'minimum': 1, 'maximum': len(options)}}))
    pick = decode(call).get('choice')
    parsed = type(pick) is int and 1 <= pick <= len(options)
    return {'graph': options[pick-1] if parsed else menu[0], 'parsed': parsed,
            'seconds': call['seconds'], 'tokens': call['response'].get('eval_count', 0)}


def run(task, prefix, menu, edits, session, folder, seed, benchmark, slots):
    obs = task['observations']
    a = choose(session, folder/'A.json', seed, menu, edits, obs)
    remaining = [graph for graph in menu if graph != a['graph']]
    second = {}
    order = ('repeat', 'b') if seed % 2 else ('b', 'repeat')
    for method in order:
        second[method] = choose(session, folder/f'{method}.json', seed+SECOND_AGENT_SEED_OFFSET,
                                remaining, edits, obs, ROLE if method == 'b' else '')
    # Full-menu validity is computed only after choices are frozen.
    legal = {graph for graph in menu if verified(graph, benchmark, obs)}
    scores = {}
    choices = {'program': menu[:2], **{method: [a['graph'], pick['graph']] for method, pick in second.items()}}
    for method, graphs in choices.items():
        assert len(graphs) == len(set(graphs)) == 2
        hits = sum(graph in legal for graph in graphs)
        found = len(prefix)+hits
        assert found <= min(slots, task['n_ground_truths'])
        scores[method] = {'hits': hits, 'second_valid': graphs[1] in legal,
                          'rr': found/task['n_ground_truths'], 'vr': found/slots, 'nr': found/slots}
    return {'id': task['id'], 'truth': task['n_ground_truths'], 'prefix': list(prefix), 'menu': menu,
            'legal': sorted(legal), 'A': a, 'second': second, 'scores': scores}


def report(output, config, groups):
    lines = ['# HypoSpace 原始题 · 独立 B 配对消融', '',
             '从原始六节点题的审计观测提取中选题。只用观测筛选构造前缀尚余至少两槽、且能生成至少三个联合改边候选的部分观测题。'
             '16 槽组取全部符合条件的题，32 槽组排除已在 16 槽组出现的题。两组均为探索性原始题子集。', '',
             '程序根据两跳可达性误差生成未验证菜单，每图同时删、增各一到两条边，且与已提交图至少相差两条边。'
             '两模型组固定同一个 A。第二次选择用相同种子、相同编号顺序和去掉 A 后的同一菜单，'
             '两组均不接收 A 的验证结果，只改变 B 的路径重接提示。每题每组实际完整验证两张。'
             '程序对照按同一排序取前两项。完整菜单标签仅在选择冻结后核对。', '',
             '| 提交上限 | 题数 | 方法 | 新合法图 | 完整探查 | RR | VR / NR | 请求秒数 |',
             '|---:|---:|---|---:|---:|---:|---:|---:|']
    summary = {}
    notes = []
    for slots, rows in groups.items():
        stats = {}
        for method, label in METHODS.items():
            scores = [r['scores'][method] for r in rows]
            d = {'hits': sum(s['hits'] for s in scores), 'probes': 2*len(rows),
                 'rr': mean(s['rr'] for s in scores), 'vr': mean(s['vr'] for s in scores),
                 'seconds': sum(r['A']['seconds']+r['second'][method]['seconds'] for r in rows)
                 if method != 'program' else 0}
            stats[method] = d
            lines.append(f"| {slots} | {len(rows)} | {label} | {d['hits']} | {d['probes']} | "
                         f"{d['rr']:.4%} | {d['vr']:.2%} | {d['seconds']:.2f} |")
        wins = sum(r['scores']['b']['hits'] > r['scores']['repeat']['hits'] for r in rows)
        losses = sum(r['scores']['b']['hits'] < r['scores']['repeat']['hits'] for r in rows)
        ties = len(rows)-wins-losses
        discordant = wins+losses
        p = min(1., 2*sum(comb(discordant,k) for k in range(min(wins,losses)+1))/2**discordant)
        stats.update(wtl=[wins,ties,losses], exact_p=p,
                     a_hits=sum(r['A']['graph'] in r['legal'] for r in rows),
                     b_rescues=sum(r['A']['graph'] not in r['legal'] and r['scores']['b']['second_valid'] for r in rows),
                     repeat_rescues=sum(r['A']['graph'] not in r['legal'] and r['scores']['repeat']['second_valid'] for r in rows),
                     solvable=sum(bool(r['legal']) for r in rows))
        summary[slots] = stats
        notes.append(f"- {slots} 槽组，B 对普通第二次的合法图胜／平／负为 {wins}／{ties}／{losses}，"
                     f"双侧精确配对检验 p={p:.4f}。A 未命中时，B 补中 {stats['b_rescues']} 题，"
                     f"普通第二次补中 {stats['repeat_rescues']} 题。菜单有解 {stats['solvable']} 题。")
    lines += ['', *notes, '', 'RR 按题宏平均，包括共享前缀。两次探查后即停，剩余槽位留空。'
              '同一题的所有方法共用候选生成工作。该实验衡量紧预算下的补位，原始题的标准答案单边连通。', '']
    write_json(output/'results.json', {'config': config, 'summary': summary, 'groups': groups})
    (output/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(summary, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    source = read(REFERENCE/'config.json')
    assert source['source_sha'] == digest(DATA)
    upstream = load_upstream('causal', 'run_causal_benchmark.py')
    assert source['evaluator_sha'] == digest(upstream.__file__)
    benchmark = upstream.CausalBenchmarkEnhanced(str(REFERENCE/'observations.json'))
    tasks = read(DATA)
    first = list(states(tasks, 16, set()))
    second = list(states(tasks, 32, {task['id'] for task, *_ in first}))
    config = {'cases': {str(slots): [task['id'] for task,*_ in group]
                        for slots, group in ((16,first),(32,second))},
              'data_sha': digest(DATA), 'runner_sha': digest(__file__), 'model_digest': source['model'],
              'evaluator_sha': digest(upstream.__file__), 'menu_size': MENU_SIZE, 'probes_per_case': 2}
    config_path = output/'config.json'
    if config_path.exists() and read(config_path) != config:
        raise ValueError('Output directory contains a different experiment configuration')
    write_json(config_path, config)
    groups = {}
    with requests.Session() as session:
        session.trust_env = False
        tags = session.get(BASE_URL+'/api/tags', timeout=10)
        tags.raise_for_status()
        assert next(item for item in tags.json()['models'] if item['name'] == MODEL)['digest'] == source['model']
        for slots, group in ((16, first), (32, second)):
            rows = []
            for index, (task, prefix, menu, edits) in enumerate(group, 1):
                folder = output/task['id']
                folder.mkdir(exist_ok=True)
                seed = source['seeds'][(index-1)%len(source['seeds'])]+index
                rows.append(run(task, prefix, menu, edits, session, folder, seed, benchmark, slots))
                if index % 8 == 0 or index == len(group):
                    print(f'{slots} slots: {index}/{len(group)}', flush=True)
            groups[slots] = rows
    report(output, config, groups)


if __name__ == '__main__':
    main()
