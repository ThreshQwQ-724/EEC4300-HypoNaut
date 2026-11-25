"""Evaluate the single-Qwen path repair policy end to end on the frozen formal cases."""
import argparse
import hashlib
from itertools import permutations
from pathlib import Path
from statistics import mean
from time import perf_counter

import requests

from common import HERE, digest, load_upstream
from graph_completion import complete_observation_count
from run_exploratory_uot import make_edits, ranked_groups
from run_formal_agent_debate import REFERENCE, SECOND_AGENT_SEED_OFFSET
from run_hypospace_rewire_ablation import choose
from run_independent_rewire import verified
from run_numbered_agent_search import build_menu
from run_qwen_path_repair import select_program, select_qwen
from run_single_qwen_feedback_rescue import feedback_for
from run_uot_debate_rescue import (
    BASE_URL, DATA, LIMIT, MODEL, NODES, forbidden_edges, graph_line, program_anchor, read, write_json,
)


METHODS = {'reselect': '单 Qwen 路径反馈重选', 'program': '单 Qwen 首选＋程序修复',
           'repair': '单 Qwen 路径定向改边'}


def cache_path(folder, stage, seed, menu, emitted, checked):
    state = repr((stage, seed, menu, emitted, sorted(checked))).encode()
    return folder/f'{stage}_{hashlib.sha256(state).hexdigest()}.json'


def run(observations, anchor, slots, method, session, folder, seed, benchmark, *, prepare_menu=None):
    start = perf_counter()
    allowed = set(permutations(NODES, 2))-forbidden_edges(observations)
    exact = complete_observation_count(NODES, LIMIT, observations)
    emitted, checked, trace = [anchor], {anchor}, []
    frontier = {}
    for slot in range(2, slots+1):
        if len(emitted) == exact:
            break
        parent = emitted[-1]
        if parent not in frontier:
            frontier[parent] = make_edits(parent, allowed, False)
        _, entries = ranked_groups(emitted, frontier, checked, slot, False)
        if not entries:
            break
        a, b, edits = build_menu(entries, emitted)
        menu = a+b
        if prepare_menu is not None:
            menu = prepare_menu(menu, edits, observations)
        if not menu:
            break
        menu_edits = {g: ((edits[g]['edge'],), ()) if edits[g]['action'] == 'remove'
                      else ((), (edits[g]['edge'],)) for g in menu}
        call_seed = seed+slot
        first = choose(session, cache_path(folder, 'first', call_seed, menu, emitted, checked),
                       call_seed, menu, menu_edits, observations)
        first_graph = first['graph']
        first_valid = verified(first_graph, benchmark, observations)
        checked.add(first_graph)
        second = None
        diagnostic = None
        selected = first_graph if first_valid else None
        if not first_valid:
            diagnostic = feedback_for(first_graph, observations)
            retry_seed = call_seed+SECOND_AGENT_SEED_OFFSET
            excluded = checked | set(menu)
            if method == 'repair':
                graph, detail = select_qwen(
                    session, cache_path(folder, 'repair', retry_seed, menu, emitted, checked),
                    retry_seed, first_graph, allowed, excluded, observations, diagnostic)
                second = detail | {'graph': graph, 'model_call': True}
            elif method == 'program':
                graph, considered = select_program(first_graph, allowed, excluded, observations)
                second = {'graph': graph, 'candidates_ranked': considered, 'model_call': False}
            elif len(menu) > 1:
                suffix = ('\nYour first submission was rejected. '+graph_line(first_graph)
                          +'\nVerifier feedback:\n'+diagnostic
                          +'\nUse the feedback to select one remaining candidate. Return its JSON choice.')
                second = choose(
                    session, cache_path(folder, 'reselect', retry_seed, menu, emitted, checked),
                    retry_seed, [g for g in menu if g != first_graph], menu_edits, observations, suffix)
                second['model_call'] = True
            if second is not None:
                graph = second['graph']
                if graph is not None:
                    assert graph not in checked
                    assert method == 'reselect' or graph not in excluded
                    checked.add(graph)
                second['accepted'] = graph is not None and verified(graph, benchmark, observations)
                if second['accepted']:
                    selected = graph
        if selected is not None:
            emitted.append(selected)
        trace.append({'slot': slot, 'menu': menu, 'first': first | {'accepted': first_valid},
                      'feedback': diagnostic, 'second': second, 'submitted': selected})
    assert len(emitted) == len(set(emitted)) <= slots
    retries = [s['second'] for s in trace if s['second'] is not None]
    calls = [s['first'] for s in trace] + [r for r in retries if r['model_call']]
    return {'graphs': emitted, 'trace': trace, 'found': len(emitted),
            'checks': 1+len(trace)+sum(r['graph'] is not None for r in retries),
            'first_hits': sum(s['first']['accepted'] for s in trace),
            'rescues': sum(r['accepted'] for r in retries),
            'rejected_repairs': sum(r['graph'] is None for r in retries),
            'repair_attempts': len(retries), 'calls': len(calls),
            'tokens': sum(c['tokens'] for c in calls),
            'request_seconds': sum(c['seconds'] for c in calls),
            'runner_seconds': perf_counter()-start}


def report(output, records, slots, metadata):
    labels = {'qwen_direct': 'Qwen 官方提示（历史）', **METHODS,
              'dual': '双 Qwen 编号版（历史）', 'adaptive': '按需单边搜索（历史）'}
    lines = ['# Final Project 正式题 · 单 Qwen 定向改边端到端复测', '',
             f'复用正式因果评测的 {len(records)} 道六节点题和每题 {slots} 个提交槽。各干预次数四题，每题一个固定种子。'
             '新方法从共享合法首图开始。每槽用 Qwen 选一个未验证的单边候选，'
             '首次验证失败时再给一次修复机会。提交的图决定下一槽的搜索状态。', '',
             '三组首次选择规则相同。重选组收到错误路径并在原菜单内重选。程序修复和 Qwen 定向改边组使用同一个最多两删两增的编辑空间，'
             '排除旧菜单和已查图。程序按两跳可达误差排序，Qwen 根据错误路径输出结构化改边。'
             '每槽最多完整验证两张候选，无可送验的改边则保留空槽。标准答案数只用于评分。', '',
             '| 方法 | 合法图 | RR | VR | NR | 满槽题 | 完整图检查 | Qwen 调用 | 输出 token | 请求秒数 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    summary = {}
    for method, label in labels.items():
        rows = [r['runs'][method] for r in records]
        stats = {k: mean(r[k] for r in rows) for k in ('rr', 'vr', 'nr')}
        stats.update({k: sum(r[k] for r in rows) for k in ('found','checks','calls','tokens','request_seconds')})
        stats['full'] = sum(r['found'] == slots for r in rows)
        if method in METHODS:
            stats.update({k: sum(r[k] for r in rows) for k in ('first_hits','rescues','repair_attempts','rejected_repairs')})
        summary[method] = stats
        d = stats
        lines.append(f"| {label} | {d['found']} | {d['rr']:.4%} | {d['vr']:.2%} | {d['nr']:.2%} | "
                     f"{d['full']}/{len(records)} | {d['checks']} | {d['calls']} | {d['tokens']} | {d['request_seconds']:.2f} |")
    lines += ['', '新组的检查数包含首图。历史程序组和双 Qwen 组的检查数已补计共享首图。'
              '完整检查与程序的局部候选排序是两项成本。请求时间来自原始记录，配对分支共享同一请求时按各自流程计数。'
              'VR 为有效提交槽比例，NR 为不同可解析图占槽比例，RR 按题宏平均。', '',
              '| 题号 | 干预数 | 合法解数 | '+ ' | '.join(labels.values())+' |',
              '|---|---:|---:|'+'---:|'*len(labels)]
    for row in records:
        lines.append(f"| {row['id']} | {row['observations']} | {row['truth']:,} | "
                     +' | '.join(str(row['runs'][m]['found']) for m in labels)+' |')
    lines += ['', '## 修复与配对结果', '']
    for method in METHODS:
        d = summary[method]
        lines.append(f"- {labels[method]}在 {d['repair_attempts']} 次修复尝试中补中 {d['rescues']} 张图，"
                     f"{d['rejected_repairs']} 次没有可送验的修复图。")
    paired = {}
    for method in ('reselect', 'program', 'adaptive'):
        deltas = [r['runs']['repair']['found']-r['runs'][method]['found'] for r in records]
        wtl = [sum(d > 0 for d in deltas), sum(d == 0 for d in deltas), sum(d < 0 for d in deltas)]
        paired[method] = wtl
        lines.append(f"- Qwen 定向改边对{labels[method]}的逐题胜／平／负为 {'／'.join(map(str, wtl))}。")
    lines += ['', '三组首次选择因提交历史变化而可能分叉。本报告衡量完整流程效果。'
              '固定失败草稿的单步因果对照见上一轮 53 题实验。', '']
    write_json(output/'results.json', {'metadata': metadata, 'summary': summary, 'paired': paired, 'cases': records})
    (output/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    source = read(REFERENCE/'config.json')
    module = load_upstream('causal', 'run_causal_benchmark.py')
    assert source['source_sha'] == digest(DATA) and source['evaluator_sha'] == digest(module.__file__)
    benchmark = module.CausalBenchmarkEnhanced(str(REFERENCE/'observations.json'))
    tasks = {task['id']: task for task in read(DATA)}
    numbered = HERE/'runs/numbered_agent_search_formal_20261113'
    assert read(numbered/'config.json')['cases'] == source['cases']
    metadata = {'cases': source['cases'], 'slots': source['slots'], 'model': MODEL, 'digest': source['model'],
                'source_config_sha': digest(REFERENCE/'config.json'), 'data_sha': digest(DATA),
                'code_sha': {name: digest(HERE/name) for name in (Path(__file__).name, 'run_qwen_path_repair.py',
                    'run_hypospace_rewire_ablation.py', 'run_numbered_agent_search.py', 'run_qwen_guided_edits.py')}}
    config = output/'config.json'
    if config.exists() and read(config) != metadata:
        raise ValueError('Output directory contains a different experiment configuration')
    write_json(config, metadata)
    records = []
    with requests.Session() as session:
        session.trust_env = False
        tags = session.get(BASE_URL+'/api/tags', timeout=10)
        tags.raise_for_status()
        assert next(m for m in tags.json()['models'] if m['name'] == MODEL)['digest'] == source['model']
        for i, case in enumerate(source['cases'], 1):
            folder = output/case
            folder.mkdir(exist_ok=True)
            prior = read(REFERENCE/case/'result.json')
            truth = tasks[case]['n_ground_truths']
            obs = tasks[case]['observations']
            assert truth == prior['truth']
            anchor = program_anchor(NODES, LIMIT, obs)
            assert anchor is not None and verified(anchor, benchmark, obs)
            runs = {}
            for method, history in (
                ('qwen_direct', prior['runs']['qwen_direct']), ('adaptive', prior['runs']['adaptive']),
                ('dual', read(numbered/case/'dual_result.json')),
            ):
                checks = history['probes'] if method == 'dual' else history['checks']
                if method == 'dual':
                    checks = history['full_checks']
                runs[method] = {k: history[k] for k in ('rr','vr','nr','found','calls','tokens')}
                runs[method].update(checks=checks+int(method != 'qwen_direct'),
                                    request_seconds=history.get('request_seconds', history['seconds']) if history['calls'] else 0)
            for method in METHODS:
                path = folder/f'{method}_result.json'
                if path.exists():
                    result = read(path)
                else:
                    result = run(obs, anchor, source['slots'], method, session, folder, prior['seed'], benchmark)
                    tmp = path.with_suffix('.tmp')
                    write_json(tmp, result)
                    tmp.replace(path)
                graphs = result['graphs']
                assert len({tuple(map(tuple, g)) for g in graphs}) == len(graphs) == result['found']
                assert len(graphs) <= truth and all(verified(g, benchmark, obs) for g in graphs)
                result.update(rr=len(graphs)/truth, vr=len(graphs)/source['slots'], nr=len(graphs)/source['slots'])
                runs[method] = result
            records.append({'id':case, 'truth':truth, 'observations':len(obs), 'seed':prior['seed'], 'runs':runs})
            print(f'{i}/{len(source["cases"])} {case} '+str({m:runs[m]['found'] for m in METHODS}), flush=True)
    assert len(records) == len({r['id'] for r in records}) == 24
    report(output, records, source['slots'], metadata)


if __name__ == '__main__':
    main()
