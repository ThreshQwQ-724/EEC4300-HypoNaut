"""Compare independent Qwen proposals with and without assigned search roles."""
import argparse
from itertools import permutations
from pathlib import Path
from statistics import mean
from time import perf_counter

import networkx as nx
import requests

from common import HERE, digest, load_upstream
from run_constraint_feedback import extract_edge_graph
from run_exploratory_uot import make_edits, run as run_program
from run_feedback_study import graph_line, observations_check
from run_formal_agent_debate import REFERENCE, SECOND_AGENT_SEED_OFFSET, content, request
from run_formal_causal_comparison import DATA, LIMIT, NODES, SLOTS, forbidden_edges, program_anchor, read, valid
from run_four_groups import BASE_URL, MODEL
from run_local_causal import write_json
from run_partial_graph_pipeline import edge_distance


METHODS = {'single': '单 Qwen 多次采样', 'dual': '双 Qwen 分工探索'}
GUARDED = HERE/'runs/formal_agent_debate_guarded_20261113'
OUTPUT_TOKENS = 128
PROBES_PER_SLOT = 3


def diagnostics(graph, observations, answer):
    if graph is None:
        parsed = extract_edge_graph(answer, NODES)
        if parsed is not None and not nx.is_directed_acyclic_graph(parsed):
            return 'Break this directed cycle: ' + ', '.join(f'{a}->{b}' for a, b in nx.find_cycle(parsed))
        return f'Use a single Graph: line with at most {LIMIT} edges over A–F and no self-loops.'
    network = nx.DiGraph(graph)
    return '\n'.join(
        f'Block forbidden path {" -> ".join(nx.shortest_path(network, m["source"], m["target"]))}.'
        if m['predicted'] else f'Add a directed path from {m["source"]} to {m["target"]}.'
        for m in observations_check(NODES, observations, graph))


def solve(benchmark, session, task, anchor, seed, options, folder, method):
    start = perf_counter()
    obs = task['observations']
    history = [benchmark.parse_llm_response(graph_line(anchor))]
    emitted, pool, trace = [anchor], {}, []
    checked = {anchor: True}
    allowed = set(permutations(NODES, 2)) - forbidden_edges(obs)
    shift_cache = {}
    stalled = False
    for slot in range(2, SLOTS+1):
        events = []
        shift = False
        if not pool:
            base = benchmark.create_prompt(obs, history)
            latest = emitted[-1]
            far = max(emitted, key=lambda graph: (edge_distance(graph, latest), graph))
            if stalled and far not in shift_cache:
                shift_cache[far] = make_edits(far, allowed, True)['shift']
            shift = stalled and bool(shift_cache.get(far, {}).keys() - checked.keys()) if far in shift_cache else False
            roles = [
                '\n\nDeepen role. Start from the graph below and add or remove exactly one edge. Preserve all observed effects.\n'
                + graph_line(latest),
                '\n\nExplore a different starting graph below. '
                + ('Try a Shift: replace a direct edge A->C with A->B and B->C, or do the reverse. '
                   'Choose any distinct nodes. Apply the whole change together.' if shift else
                   'Add or remove exactly one edge while preserving all observed effects.') + '\n' + graph_line(far),
            ]

            def propose(stage, prompt, call_seed):
                call = request(session, folder/f'{method}_{slot:02d}_{stage}.json',
                               [{'role': 'user', 'content': prompt}], options, call_seed, OUTPUT_TOKENS)
                answer = content(call)
                parsed = benchmark.parse_llm_response(answer)
                key = tuple(sorted(parsed.edges)) if parsed is not None else None
                fresh_check = key is not None and key not in checked
                if fresh_check:
                    checked[key] = valid(key, obs)
                    assert checked[key] == benchmark.validate_hypothesis(parsed, obs)
                is_valid = key is not None and checked[key]
                new = is_valid and key not in emitted and key not in pool
                if new:
                    pool[key] = parsed
                events.append({'stage': stage, 'graph': key, 'valid': is_valid, 'new': new, 'checked': fresh_check,
                               'seconds': call['seconds'], 'output_tokens': call['response'].get('eval_count', 0),
                               'input_tokens': call['response'].get('prompt_eval_count', 0)})
                return answer, key, is_valid

            proposals = [propose(f'proposal_{index+1}', base + (role if method == 'dual' else ''),
                                 seed + index*SECOND_AGENT_SEED_OFFSET) for index, role in enumerate(roles)]
            if not pool:
                repair = []
                for index, (answer, key, is_valid) in enumerate(proposals, 1):
                    feedback = ('This legal graph duplicates a submitted graph. Change its edge set while keeping all effects.'
                                if is_valid else diagnostics(key, obs, answer))
                    repair.append(f'Draft {index}:\n{answer.strip() or "[empty]"}\nValidator feedback:\n{feedback}')
                propose('repair', base + '\n\nRepair one draft using the validator feedback.\n'
                        + '\n\n'.join(repair) + '\nReturn only one Graph: line.', seed+2*SECOND_AGENT_SEED_OFFSET)
        selected = next(iter(pool), None)
        if selected is not None:
            history.append(pool.pop(selected))
            emitted.append(selected)
        stalled = selected is None
        trace.append({'slot': slot, 'shift_prompt': method == 'dual' and shift, 'calls': events,
                      'submitted': selected, 'pool_size': len(pool)})
    events = [event for step in trace for event in step['calls']]
    assert all(len(step['calls']) <= PROBES_PER_SLOT for step in trace)
    assert len(emitted) == len(set(emitted)) <= min(SLOTS, task['n_ground_truths'])
    assert all(valid(graph, obs) for graph in emitted)
    return {'found': len(emitted), 'graphs': emitted, 'rr': len(emitted)/task['n_ground_truths'],
            'vr': len(emitted)/SLOTS, 'nr': len(emitted)/SLOTS, 'calls': len(events),
            'checks': sum(event['checked'] for event in events), 'seconds': perf_counter()-start,
            'request_seconds': sum(event['seconds'] for event in events),
            'tokens': sum(event['output_tokens'] for event in events),
            'input_tokens': sum(event['input_tokens'] for event in events),
            'repairs': sum(event['stage'] == 'repair' for event in events),
            'repair_success': sum(event['stage'] == 'repair' and event['new'] for event in events),
            'shift_prompts': sum(step['shift_prompt'] for step in trace),
            'shift_success': sum(step['shift_prompt'] and step['calls'][1]['new'] for step in trace),
            'trace': trace}


def report(output, records):
    names = {'guarded_debate': 'Debate＋校验（既有）', **METHODS, 'program': '程序单边搜索'}
    lines = ['# HypoSpace 正式对照 · 双 Agent 独立探索', '',
             '原正式 24 道六节点题，每题 8 槽。两个探索组与程序对照共享程序首图，首图占第一槽。'
             '探索组先独立生成两个候选，用不同种子，第二个候选看不到第一个。Ollama 串行执行请求。'
             '有新合法图则提交一张，剩余候选缓存。两次提案均无新合法图时，第三次调用使用程序诊断修复。'
             '合法性及去重以验证器为准；空槽计入 VR 和 NR。', '',
             '双 Agent 组增加角色提示，A 从最新合法图做单边编辑，B 从边集距离最远的已提交图出发。'
             '上一槽未找到图且存在未检查的联合编辑时，B 优先尝试 Shift。'
             '操作类型由提示引导，实际验收只检查合法性与去重。单 Qwen 两次使用相同的通用提示。'
             '程序组共用首图，沿用单边避险，每槽最多探查三个候选。既有 Debate 未使用程序首图，'
             '调用预算也不同，作为历史参照。', '',
             '| 方法 | RR | VR | NR | 不同合法图 | 调用 | 新候选探查／验证 | 输出 token | 秒数 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    summary = {}
    for method, label in names.items():
        rows = [row['runs'][method] for row in records]
        d = {key: mean(row[key] for row in rows) for key in ('rr', 'vr', 'nr')}
        d.update({key: sum(row.get(key, 0) for row in rows) for key in (
            'found', 'calls', 'checks', 'tokens', 'seconds', 'repairs', 'repair_success', 'shift_prompts', 'shift_success')})
        summary[method] = d
        lines.append(f"| {label} | {d['rr']:.4%} | {d['vr']:.2%} | {d['nr']:.2%} | {d['found']} | "
                     f"{d['calls']} | {d['checks']} | {d['tokens']} | {d['seconds']:.2f} |")
    deltas = [row['runs']['dual']['found']-row['runs']['single']['found'] for row in records]
    paired = [sum(d > 0 for d in deltas), sum(d == 0 for d in deltas), sum(d < 0 for d in deltas)]
    lines += ['', f'双 Agent 对比单 Qwen 的逐题合法图数胜／平／负为 {paired[0]} / {paired[1]} / {paired[2]}。', '',
              '| 题目 | 合法解数 | 单 Qwen | 双 Agent | 程序 |', '|---|---:|---:|---:|---:|']
    for row in records:
        values = ' | '.join(str(row['runs'][method]['found']) for method in ('single','dual','program'))
        lines.append(f"| {row['id']} | {row['truth']:,} | {values} |")
    lines += ['', 'RR 按题宏平均。两探索组每个空池槽至多三次请求；复用缓存图时无需请求。'
              '重复和解析失败不会重复进行完整验证。程序探查含单边风险判断，Debate 计提交图检查，'
              '各组检查计数见 JSON。历史 Debate 耗时为请求总时间，其余为本轮实测运行时间。', '']
    write_json(output/'summary.json', {'methods': summary, 'dual_vs_single': paired})
    (output/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    source = read(REFERENCE/'config.json')
    module = load_upstream('causal', 'run_causal_benchmark.py')
    assert digest(DATA) == source['source_sha'] and digest(module.__file__) == source['evaluator_sha']
    assert read(GUARDED/'config.json')['source_config_sha'] == digest(REFERENCE/'config.json')
    benchmark = module.CausalBenchmarkEnhanced(str(REFERENCE/'observations.json'))
    tasks = {task['id']: task for task in read(DATA)}
    with requests.Session() as session:
        session.trust_env = False
        tags = session.get(BASE_URL+'/api/tags', timeout=10)
        tags.raise_for_status()
        assert next(row for row in tags.json()['models'] if row['name'] == MODEL)['digest'] == source['model']
        config = {'cases': source['cases'], 'model': MODEL, 'model_digest': source['model'], 'slots': SLOTS,
                  'options': {**source['options'], 'num_predict': OUTPUT_TOKENS}, 'think': False,
                  'calls_per_slot': PROBES_PER_SLOT, 'source_sha': digest(REFERENCE/'config.json'),
                  'code_hashes': {name: digest(HERE/name) for name in (Path(__file__).name,
                      'run_exploratory_uot.py', 'run_formal_agent_debate.py', 'run_formal_causal_comparison.py')}}
        path = output/'config.json'
        if path.exists() and read(path) != config:
            raise ValueError('Output directory contains a different experiment configuration')
        write_json(path, config)
        records = []
        for index, case in enumerate(source['cases'], 1):
            folder = output/case
            folder.mkdir(exist_ok=True)
            task = tasks[case]
            truth, obs = task['n_ground_truths'], task['observations']
            anchor = program_anchor(NODES, LIMIT, obs)
            assert anchor is not None and valid(anchor, obs)
            seed = read(REFERENCE/case/'result.json')['seed']
            runs = {'guarded_debate': read(GUARDED/case/'guarded_debate_result.json')}
            order = tuple(METHODS) if index % 2 else tuple(reversed(METHODS))
            for method in order:
                path = folder/f'{method}_result.json'
                if path.exists():
                    runs[method] = read(path)
                else:
                    result = solve(benchmark, session, task, anchor, seed, source['options'], folder, method)
                    temporary = path.with_suffix('.tmp')
                    write_json(temporary, result)
                    temporary.replace(path)
                    runs[method] = result
            program = run_program(obs, anchor, PROBES_PER_SLOT, 'single')
            found = program['found']
            runs['program'] = program | {'rr': found/truth, 'vr': found/SLOTS, 'nr': found/SLOTS,
                                         'checks': program['probes'], 'seconds': program['runner_seconds']}
            record = {'id': case, 'truth': truth, 'seed': seed, 'anchor': anchor, 'runs': runs}
            write_json(folder/'result.json', record)
            records.append(record)
            print(f"{index}/24 {case} " + str({name: runs[name]['found'] for name in (*METHODS, 'program')}), flush=True)
        assert len(records) == 24
        report(output, records)


if __name__ == '__main__':
    main()
