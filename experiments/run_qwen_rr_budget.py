"""Run a paired Qwen-versus-search budget pilot on six causal tasks."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import random
from statistics import mean
from time import perf_counter

import requests

from common import HERE, digest, load_upstream
from run_constraint_feedback import OPTIONS, extra_feedback, extract_edge_graph
from run_feedback_study import make_feedback
from run_four_groups import BASE_URL, MODEL, SEEDS, score_calls
from run_local_causal import write_json
from run_thinking_trigger_study import THINKING_OPTIONS
from validate_graph_rr_budget import BUDGETS, DATA

SELECTION_SEED = 1013126
HISTORY_LIMIT = 8
RUNS = HERE/'runs'
REFERENCE = RUNS/'graph_rr_budget_20261113/results.json'
MODEL_REFERENCE = RUNS/'qwen3_project_evaluation_20261112_145049/config.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def select():
    tasks = read(DATA)
    # Exclude the six-node IDs already used by earlier Qwen experiments.
    excluded = {'n6_082', 'n6_026', 'n6_065', 'n6_086', 'n6_032', 'n6_051',
                'n6_077', 'n6_050', 'n6_076', 'n6_001',
                'n5_070', 'n5_050', 'n5_077', 'n5_074', 'n5_049', 'n5_025'}
    rng = random.Random(SELECTION_SEED)
    return [rng.choice(sorted((r for r in tasks if r['n_observations'] == count and r['id'] not in excluded),
                              key=lambda r: r['id'])) for count in range(1, 7)]


def run_task(benchmark, task, folder, seed, api):
    obs = task['observations']
    calls, parsed_history, history = [], [], []
    last_graph = None
    valid_seen = False
    for slot in range(1, BUDGETS[-1]+1):
        prompt = benchmark.create_prompt(obs, parsed_history[-HISTORY_LIMIT:])
        recent = history[-HISTORY_LIMIT:]
        prompt += make_feedback('combined', benchmark.nodes, obs, recent)
        if recent:
            feedback, _ = extra_feedback('diagnostics', benchmark.nodes, obs, recent, last_graph, None)
            prompt += feedback
        think = slot == 8 and not valid_seen
        options = {**OPTIONS, 'seed': seed, **(THINKING_OPTIONS if think else {})}
        payload = {'model': MODEL, 'think': think, 'stream': False, 'keep_alive': '10m',
                   'options': options, 'messages': [{'role': 'user', 'content': prompt}]}
        path = folder/f'call_{slot:03}.json'
        if path.exists():
            call = read(path)
            assert call['request'] == payload
        else:
            started = perf_counter()
            response = api('/api/chat', payload)
            call = {'request': payload, 'response': response, 'seconds': perf_counter()-started}
            write_json(path, call)
        calls.append(call)
        answer = call['response']['message'].get('content', '')
        parsed = benchmark.parse_llm_response(answer)
        if parsed is not None:
            parsed_history.append(parsed)
            valid_seen |= benchmark.validate_hypothesis(parsed, obs)
        history.append(list(parsed.edges) if parsed is not None else None)
        last_graph = extract_edge_graph(answer, benchmark.nodes)
        if slot in BUDGETS:
            print(f"{task['id']} {slot}/{BUDGETS[-1]}", flush=True)
    return calls


def score(benchmark, task, calls, reference):
    # The full source label set can be huge. Compatible outputs are checked
    # against observations, and the denominator comes from the audited set.
    obs = task['observations']
    found = set()
    for call in calls:
        candidate = benchmark.parse_llm_response(call['response']['message'].get('content', ''))
        if candidate is not None and benchmark.validate_hypothesis(candidate, obs):
            found.add(tuple(sorted(candidate.edges)))
    instance = {'observations': obs, 'ground_truth_graphs': [{'edges': graph} for graph in found]}
    rows, counts = score_calls(benchmark, instance, calls)
    assert counts['n_recovered_gts'] == len(found)
    assert all(row['valid'] == (graph is not None and benchmark.validate_hypothesis(graph, obs))
               for row in rows for graph in [benchmark.parse_llm_response(row['answer'])])
    truth = task['n_ground_truths']
    results = []
    for budget, program in zip(BUDGETS, reference['budgets'], strict=True):
        subset = rows[:budget]
        recovered = subset[-1]['cumulative_recovered']
        assert recovered <= min(budget, truth)
        results.append(dict(budget=budget, found=recovered, rr=recovered/truth,
                            vr=sum(row['valid'] for row in subset)/budget,
                            nr=sum(row['parsed'] and not row['duplicate'] for row in subset)/budget,
                            checks=program['checks'], program_found=program['found'],
                            program_rr=program['rr'], ceiling=program['ceiling'],
                            ceiling_rr=program['rr_ceiling'],
                            output_tokens=sum(row['output_tokens'] for row in subset),
                            seconds=sum(row['seconds'] for row in subset),
                            duplicates=sum(row['duplicate'] for row in subset)))
    return results, rows


def report(output, trials):
    summary = []
    for index, budget in enumerate(BUDGETS):
        rows = [trial['budgets'][index] for trial in trials]
        summary.append({'budget': budget, **{k: mean(row[k] for row in rows)
                        for k in ('rr','vr','nr','program_rr','ceiling_rr')},
                        **{k: sum(row[k] for row in rows)
                           for k in ('found','program_found','ceiling','output_tokens','seconds')}})
    write_json(output/'summary.json', {'summary': summary, 'tasks': trials})
    lines = ['# Qwen3 8B 的 128 槽对照', '',
             '六道六节点题按干预数 1–6 各固定抽取一道，每题一个采样种子。每槽向本机 Qwen3 8B 请求一张图，共 768 次调用。沿用观测与重复反馈、路径与环路诊断，前七答均无合法图时在第八槽启用思考。第九槽起继续普通生成，提示只保留最近八条历史，评分在全轮去重。所有候选图都来自模型答复。', '',
             '按题计算覆盖率 RR 后取宏平均。VR 是符合题面约束的回答比例，NR 是不同可解析图占候选槽的比例。程序值和预算上限取自同一题目的既有测试。因果图合法性由无环、边数上限和观测一致性确定，RR 分母来自全文件答案数审计。', '',
             '| 槽位 | Qwen RR | 程序 RR | 同题上限 | Qwen VR | Qwen NR | Qwen/程序/上限图数 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in summary:
        lines.append(f"| {row['budget']} | {row['rr']:.4%} | {row['program_rr']:.4%} | {row['ceiling_rr']:.4%} | "
                     f"{row['vr']:.2%} | {row['nr']:.2%} | {row['found']}/{row['program_found']}/{row['ceiling']} |")
    lines += ['', '| 题目 | 干预数 | 合法图总数 | Qwen 找到 | 程序找到 |',
              '|---|---:|---:|---:|---:|']
    for trial in trials:
        last = trial['budgets'][-1]
        lines.append(f"| {trial['id']} | {trial['n_observations']} | {trial['n_ground_truths']:,} | "
                     f"{last['found']} | {last['program_found']} |")
    lines += ['', f"累计模型请求耗时 {summary[-1]['seconds']:.1f} 秒，输出 {summary[-1]['output_tokens']:,} token。",
              '六题为分层小样本，按干预数等权。此前 49.39% 是 510 题按题等权的上限。逐题配置、完整请求与响应、验证记录均保存在本目录。']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    tasks = select()
    output = args.output or RUNS/('qwen_rr_budget_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.trust_env = False

    def api(route, payload=None):
        response = (session.get(BASE_URL+route, timeout=10) if payload is None
                    else session.post(BASE_URL+route, json=payload, timeout=(10, 300)))
        response.raise_for_status()
        return response.json()

    model = next(m for m in api('/api/tags')['models'] if m['name'] == MODEL)
    assert model['digest'] == read(MODEL_REFERENCE)['model']['digest']
    previous = read(REFERENCE)
    assert previous['input_sha'] == digest(DATA)
    program = {r['item']: r for r in previous['records']}
    module = load_upstream('causal', 'run_causal_benchmark.py')
    # Only task observations are exposed to the model.
    dataset = {'metadata': {'nodes': list('ABCDEF'), 'max_edges': 10, 'total_observation_sets': len(tasks)},
               'datasets': [{'observation_set_id': r['id'], 'observations': r['observations'],
                             'ground_truth_graphs': []} for r in tasks]}
    write_json(output/'test.json', dataset)
    benchmark = module.CausalBenchmarkEnhanced(str(output/'test.json'))
    config = {'model': model, 'version': api('/api/version'), 'seed_schedule': SEEDS,
              'selection_seed': SELECTION_SEED, 'cases': [r['id'] for r in tasks], 'budgets': list(BUDGETS),
              'ordinary_options': OPTIONS, 'thinking_options': THINKING_OPTIONS,
              'history_limit': HISTORY_LIMIT, 'input_sha': digest(DATA), 'reference_sha': digest(REFERENCE),
              'runner_sha': digest(__file__), 'official_sha': digest(module.__file__)}
    config_path = output/'config.json'
    if config_path.exists():
        assert read(config_path) == config
    else:
        write_json(config_path, config)
    print('OUTPUT_DIR='+str(output.resolve()), flush=True)
    trials = []
    for index, task in enumerate(tasks):
        folder = output/task['id']
        folder.mkdir(exist_ok=True)
        seed = SEEDS[index % len(SEEDS)]
        calls = run_task(benchmark, task, folder, seed, api)
        budgets, rows = score(benchmark, task, calls, program[task['id']])
        write_json(folder/'result.json', {'seed': seed, 'budgets': budgets, 'rows': rows})
        trials.append({**task, 'seed': seed, 'budgets': budgets})
    assert len(list(output.glob('*/call_*.json'))) == len(tasks)*BUDGETS[-1]
    report(output, trials)


if __name__ == '__main__':
    main()
