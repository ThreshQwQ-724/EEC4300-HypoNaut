"""Test verifier feedback for a single Qwen retry on frozen HypoSpace states."""
import argparse
from collections import Counter
from pathlib import Path
import random
from statistics import mean

import networkx as nx
import requests

from common import HERE, digest, load_upstream
from run_feedback_study import graph_line, observations_check
from run_hypospace_rewire_ablation import verified
from run_uot_debate_rescue import (
    BASE_URL, DATA, MODEL, NODES, REFERENCE, decode, read, request, write_json,
)


SOURCE = HERE/'runs/hypospace_rewire_ablation_20261113'
LABELS = {'repeat': '无反馈重选', 'feedback': '单 Qwen 验证反馈重选',
          'program': '程序补位（固定同一首次选择）'}


def feedback_for(edges, observations):
    graph = nx.DiGraph()
    graph.add_nodes_from(NODES)
    graph.add_edges_from(edges)
    if not nx.is_directed_acyclic_graph(graph):
        cycle = nx.find_cycle(graph)
        path = ' -> '.join([cycle[0][0], *(b for _, b in cycle)])
        return f'The rejected graph contains this cycle: {path}.'
    lines = []
    for error in observations_check(NODES, observations, edges):
        a, b = error['source'], error['target']
        if error['predicted']:
            path = ' -> '.join(nx.shortest_path(graph, a, b))
            lines.append(f'Forbidden path: {path}. The observation requires {b} to be unreachable from {a}.')
        else:
            lines.append(f'Missing required path from {a} to {b}.')
    assert lines
    return '\n'.join(lines)


def run_case(session, output, row, observations, benchmark):
    case = row['id']
    a = row['A']['graph']
    legal = row['legal']
    assert (a in legal) == verified(a, benchmark, observations)
    original = read(SOURCE/case/'repeat.json')
    payload = original['request']
    seed = payload['options']['seed']
    menu = [graph for graph in row['menu'] if graph != a]
    options = menu.copy()
    random.Random(seed).shuffle(options)
    assert all((graph in legal) == verified(graph, benchmark, observations) for graph in menu)
    repeat = row['second']['repeat']
    index = decode(original).get('choice')
    assert repeat['graph'] == (options[index-1] if repeat['parsed'] else menu[0])
    choice, seconds, called, diagnostic = repeat['graph'], repeat['seconds'], False, None
    if a not in legal:
        diagnostic = feedback_for(a, observations)
        suffix = ('\n\nYour first submission was rejected. ' + graph_line(a)
                  + '\nExact verifier feedback for that graph:\n' + diagnostic
                  + '\nUse this evidence to choose a different candidate satisfying all observations. '
                    'The feedback describes the rejected graph only. Return the JSON choice number.')
        call = request(session, output/f'{case}.json', payload['messages'][0]['content']+suffix,
                       seed, payload['format'])
        assert call['request']['options'] == payload['options']
        index = decode(call).get('choice')
        choice = options[index-1] if type(index) is int and 1 <= index <= len(options) else menu[0]
        called, seconds = True, call['seconds']
    picks = {'repeat': repeat['graph'], 'feedback': choice, 'program': menu[0]}
    assert all(graph in menu for graph in picks.values())
    return {'id': case, 'truth': row['truth'], 'prefix': len(row['prefix']), 'first_valid': a in legal,
            'diagnostic': diagnostic, 'called': called, 'picks': picks, 'has_legal_retry': any(g in legal for g in menu),
            'hits': {method: graph in legal for method, graph in picks.items()},
            'seconds': {'repeat': repeat['seconds'], 'feedback': seconds}}


def summarize(rows, slots):
    stats = {method: {'retry_hits': sum(r['hits'][method] for r in rows),
                      'rescues': sum(not r['first_valid'] and r['hits'][method] for r in rows),
                      'rr': mean((r['prefix']+r['first_valid']+r['hits'][method])/r['truth'] for r in rows),
                      'vr': mean((r['prefix']+r['first_valid']+r['hits'][method])/slots for r in rows),
                      'probes': 2*len(rows)} for method in LABELS}
    changes = Counter(r['hits']['feedback']-r['hits']['repeat'] for r in rows)
    stats.update(wtl=[changes[1], changes[0], changes[-1]], calls=sum(r['called'] for r in rows),
                 opportunities=sum(r['called'] and r['has_legal_retry'] for r in rows),
                 retry_seconds={method: sum(r['seconds'][method] for r in rows if r['called'])
                                for method in ('repeat', 'feedback')})
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    reference = read(REFERENCE/'config.json')
    frozen = read(SOURCE/'results.json')
    assert frozen['config']['data_sha'] == digest(DATA)
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
        for slots, rows in frozen['groups'].items():
            groups[slots] = [run_case(session, output, row, tasks[row['id']]['observations'], benchmark)
                             for row in rows]
            print(f'{slots} slots: {len(rows)} cases', flush=True)
    summary = {slots: summarize(rows, int(slots)) for slots, rows in groups.items()}
    write_json(output/'results.json', {'source_sha': digest(SOURCE/'results.json'),
               'runner_sha': digest(__file__), 'summary': summary, 'groups': groups})
    lines = ['# HypoSpace 单 Qwen 验证反馈重选', '',
             '复用双 Agent 消融的 75 道原始六节点题、构造前缀、未验证重接菜单、首次选择和无反馈重选。'
             '首次验证失败时追加具体环路或违反观测的路径，再用同一个 Qwen3:8B 重选。'
             '编号顺序、采样参数、种子和两个完整探查机会均与无反馈组一致。'
             '首次通过时两组使用同一份重选记录。程序补位对照固定同一个首次选择，第二次按程序排序取首项。', '',
             '| 提交上限 | 题数 | 方法 | 二次命中 | 首次失败后补中 | 完整探查 | RR | VR / NR |',
             '|---:|---:|---|---:|---:|---:|---:|---:|']
    for slots, stats in summary.items():
        for method, label in LABELS.items():
            d = stats[method]
            lines.append(f"| {slots} | {len(groups[slots])} | {label} | {d['retry_hits']} | {d['rescues']} | "
                         f"{d['probes']} | {d['rr']:.4%} | {d['vr']:.2%} |")
    for slots, d in summary.items():
        lines += ['', f"- {slots} 槽组，反馈对无反馈逐题胜／平／负为 {'／'.join(map(str, d['wtl']))}。"
                  f"首次失败 {d['calls']} 题，其中 {d['opportunities']} 题的剩余菜单有合法图。"
                  f"这批题第二次请求累计 {d['retry_seconds']['repeat']:.2f} → {d['retry_seconds']['feedback']:.2f} 秒。"]
    lines += ['', '本轮为同题配对复测，观察的是验证反馈对第二次选择的影响。RR 包含共享前缀。'
              '两次探查后停止。旧双 Agent 结果单独保留。', '']
    (output/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(summary, flush=True)


if __name__ == '__main__':
    main()
