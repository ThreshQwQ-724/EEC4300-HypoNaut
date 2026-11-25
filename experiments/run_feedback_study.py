"""Paired feedback ablations and a separate thinking-budget diagnostic.

Feedback never receives ground truths. Every generation consumes one query.
Successful responses are journaled for replay/resume; transport errors abort.
"""
import argparse
import csv
import itertools
import json
import os
import random
import statistics
import time
from datetime import datetime
from pathlib import Path

import networkx as nx
import requests
from common import HERE, COMMIT, digest, load_upstream
from run_four_groups import score_calls, SEEDS, SHARED, MODEL, BASE_URL
from run_local_causal import write_json

GROUPS = ('baseline', 'duplicate', 'consistency', 'combined')


def observations_check(nodes, observations, edges):
    """Pure validator: only observations, nodes and a candidate edge set."""
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    mismatches = []
    for observation in observations:
        source = observation['perturbed_node']
        reachable = nx.descendants(graph, source)
        for target, expected in observation['effects'].items():
            predicted = int(target in reachable)
            if predicted != expected:
                mismatches.append({'source': source, 'target': target,
                                   'expected': expected, 'predicted': predicted})
    return mismatches


def graph_line(edges):
    return 'Graph: ' + (', '.join(f'{a}->{b}' for a, b in sorted(edges)) or 'No edges')


def make_feedback(group, nodes, observations, candidate_history):
    """No instance object or answer key enters this function."""
    if group == 'baseline' or not candidate_history:
        return ''
    last = candidate_history[-1]
    lines = ['\n\nExternal checker feedback (based only on observations and previous predictions):']
    if last is None:
        lines.append('The last response could not be parsed as a DAG. Return exactly one Graph: line with a valid DAG.')
    if group in ('consistency', 'combined') and last is not None:
        mismatches = observations_check(nodes, observations, last)
        if mismatches:
            lines.append('The last graph violates these observed effects:')
            for m in mismatches:
                requirement = 'must be reachable' if m['expected'] else 'must NOT be reachable'
                lines.append(f"- Perturb({m['source']}), node {m['target']}: expected {m['expected']}, predicted {m['predicted']}. "
                             f"{m['target']} {requirement} from {m['source']} by a directed path.")
            lines.append('Repair the violated reachability constraints, then check ALL observations before answering.')
        else:
            lines.append('The last graph satisfies all observed effects. Preserve agreement with ALL observations in the next graph.')
    if group in ('duplicate', 'combined'):
        unique = sorted({tuple(sorted(edges)) for edges in candidate_history if edges is not None})
        if last is not None:
            repeated = tuple(sorted(last)) in {tuple(sorted(e)) for e in candidate_history[:-1] if e is not None}
            lines.append('The last edge set is a DUPLICATE.' if repeated else 'The last edge set was new when submitted.')
        if unique:
            lines.append('Already proposed edge sets (do not submit any of them again if a new valid graph is possible):')
            lines.extend(graph_line(edges) for edges in unique)
        lines.append('Construct a different edge set: consider adding, removing, or reversing an edge. '
                     'Keep a DAG and satisfy all observations. Reordering edges does not make a new graph.')
    lines.append('Return only the next Graph: line. Do not output the checker analysis.')
    return '\n'.join(lines)


def verify_checker(benchmark, instances):
    """Exhaust all 543 four-node DAGs; compare observation checker and answer sets."""
    possible = [(a, b) for a in benchmark.nodes for b in benchmark.nodes if a != b]
    dags = []
    for bits in itertools.product((0, 1), repeat=len(possible)):
        edges = tuple(e for e, bit in zip(possible, bits) if bit)
        graph = nx.DiGraph()
        graph.add_nodes_from(benchmark.nodes)
        graph.add_edges_from(edges)
        if nx.is_directed_acyclic_graph(graph):
            dags.append(edges)
    assert len(dags) == 543
    checks = []
    for instance in instances:
        accepted = {tuple(sorted(e)) for e in dags if not observations_check(benchmark.nodes, instance['observations'], e)}
        truths = {tuple(sorted(tuple(e) for e in g['edges'])) for g in instance['ground_truth_graphs']}
        assert accepted == truths
        checks.append({'instance': instance['observation_set_id'], 'dags_checked': len(dags), 'valid': len(accepted)})
    # Edge order is immaterial for duplicate detection.
    feedback = make_feedback('duplicate', benchmark.nodes, instances[0]['observations'],
                             [[('A', 'B'), ('B', 'C')], [('B', 'C'), ('A', 'B')]])
    assert 'DUPLICATE' in feedback
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    mod = load_upstream('causal', 'run_causal_benchmark.py')
    data = HERE/'data/causal_n4_full_observations.json'
    benchmark = mod.CausalBenchmarkEnhanced(str(data))
    original = next(r for r in benchmark.all_observation_sets if r['observation_set_id'] == 'n4_219')
    instances = [original]
    rng = random.Random(20261112)
    for count in (2, 4, 8):
        instances.append(rng.choice([r for r in benchmark.all_observation_sets
                                    if len(r['ground_truth_graphs']) == count and r is not original]))
    verification = verify_checker(benchmark, instances)
    if args.verify_only:
        print(json.dumps(verification), flush=True)
        return
    output = args.output or HERE/'runs'/('qwen3_feedback_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.trust_env = False

    def api(route, payload=None):
        response = session.get(BASE_URL+route, timeout=10) if payload is None else session.post(
            BASE_URL+route, json=payload, timeout=(10, 300))
        response.raise_for_status()
        return response.json()

    config = {'model': next(m for m in api('/api/tags')['models'] if m['name'] == MODEL),
              'ollama_version': api('/api/version'), 'groups': GROUPS, 'seeds': SEEDS,
              'options': {**SHARED, 'temperature': .6, 'top_p': .95}, 'think': False,
              'queries_per_run': 8, 'instances': [r['observation_set_id'] for r in instances],
              'selection_seed': 20261112, 'dataset_sha256': digest(data), 'upstream_commit': COMMIT,
              'runner_sha256': digest(__file__), 'scoring_helper_sha256': digest(HERE/'run_four_groups.py'),
              'evaluator_sha256': digest(mod.__file__), 'retries': 0,
              'thinking_diagnostic': {'instance': 'n4_219', 'num_ctx': 16384,
                                      'limits': [2048, 4096, 8192], 'one_first_proposal_per_seed': True},
              'feedback_inputs': ['nodes', 'observations', 'prior model predictions'],
              'seed_policy': 'Same seed per call within each eight-query run, matching preceding baseline.'}
    # Normalize tuples so JSON round trips are stable.
    config = json.loads(json.dumps(config))
    if (output/'config.json').exists():
        assert json.loads((output/'config.json').read_text(encoding='utf-8')) == config
    else:
        write_json(output/'config.json', config)
    write_json(output/'instances.json', instances)
    write_json(output/'checker_verification.json', verification)
    write_json(output/'model_show.json', api('/api/show', {'model': MODEL}))
    print('OUTPUT_DIR='+str(output.resolve()), flush=True)

    def request(folder, index, prompt, options, think):
        folder.mkdir(parents=True, exist_ok=True)
        payload = {'model': MODEL, 'think': think, 'stream': False, 'keep_alive': '10m',
                   'messages': [{'role': 'user', 'content': prompt}], 'options': options}
        path = folder/f'call_{index:02d}.json'
        if path.exists():
            call = json.loads(path.read_text(encoding='utf-8'))
            assert call['request'] == payload
        else:
            write_json(folder/f'request_{index:02d}.json', payload)
            start = time.perf_counter()
            response = api('/api/chat', payload)
            call = {'request': payload, 'response': response, 'seconds': time.perf_counter()-start}
            write_json(path, call)
        return call

    results = []
    for instance in instances:
        for seed_index, seed in enumerate(SEEDS):
            # Rotate group order across seeds to reduce systematic timing effects.
            order = GROUPS[seed_index:]+GROUPS[:seed_index]
            for group in order:
                folder = output/'feedback'/f"{instance['observation_set_id']}_{seed}_{group}"
                calls, hypotheses, history = [], [], []
                for query in range(1, 9):
                    base_prompt = benchmark.create_prompt(instance['observations'], hypotheses)
                    feedback = make_feedback(group, benchmark.nodes, instance['observations'], history)
                    call = request(folder, query, base_prompt+feedback,
                                   {**config['options'], 'seed': seed}, False)
                    calls.append(call)
                    parsed = benchmark.parse_llm_response(call['response']['message'].get('content', ''))
                    history.append(list(parsed.edges) if parsed is not None else None)
                    if parsed is not None:
                        hypotheses.append(parsed)
                rows, counts = score_calls(benchmark, instance, calls)

                class Replay:
                    def __init__(self):
                        self.index = 0

                    def query_with_usage(self, prompt):
                        # Official evaluator rebuilds the same unaugmented history.
                        recorded = calls[self.index]
                        assert recorded['request']['messages'][0]['content'].startswith(prompt)
                        self.index += 1
                        response = recorded['response']
                        pin, pout = response.get('prompt_eval_count', 0), response.get('eval_count', 0)
                        return {'response': response['message'].get('content', ''),
                                'usage': {'prompt_tokens': pin, 'completion_tokens': pout, 'total_tokens': pin+pout}, 'cost': 0.0}

                official = benchmark.evaluate_single_observation_set(Replay(), instance, 8, False, max_retries=1)
                assert not official['errors']
                assert all(official[k] == v for k, v in counts.items())
                result = {'instance': instance['observation_set_id'], 'split': 'development' if instance is original else 'new_questions',
                          'seed': seed, 'group': group, 'official': official, 'details': rows,
                          'independent_scoring_matches': True}
                write_json(folder/'result.json', result)
                results.append(result)
                write_json(output/'progress.json', {'completed_feedback_runs': len(results), 'planned_feedback_runs': 48})
                print(f"FEEDBACK {len(results)}/48 {instance['observation_set_id']} {seed} {group}: "
                      f"VR={official['valid_rate']:.3f} NR={official['novelty_rate']:.3f} RR={official['recovery_rate']:.3f}", flush=True)
    write_json(output/'feedback_results.json', results)
    diagnostics = []
    for seed in SEEDS:
        for limit in (2048, 4096, 8192):
            folder = output/'thinking'/f'{seed}_{limit}'
            call = request(folder, 1, benchmark.create_prompt(original['observations'], []),
                           {**config['options'], 'num_ctx': 16384, 'num_predict': limit, 'seed': seed}, True)
            rows, counts = score_calls(benchmark, original, [call])
            diagnostic = {'seed': seed, 'limit': limit, **rows[0]}
            diagnostics.append(diagnostic)
            write_json(folder/'result.json', diagnostic)
            write_json(output/'thinking_results.json', diagnostics)
            print(f"THINKING {len(diagnostics)}/9 seed={seed} limit={limit}: valid={rows[0]['valid']} "
                  f"stop={rows[0]['done_reason']} tokens={rows[0]['output_tokens']} seconds={rows[0]['seconds']:.1f}", flush=True)
    write_json(output/'runtime_gpu.json', api('/api/ps'))
    summarize(output, results, diagnostics)


def summarize(output, results, diagnostics):
    aggregates = {}
    for split in ('development', 'new_questions'):
        aggregates[split] = {}
        for group in GROUPS:
            chosen = [r for r in results if r['split'] == split and r['group'] == group]
            rows = [d for r in chosen for d in r['details']]
            stats = {metric: statistics.mean(r['official'][metric] for r in chosen)
                     for metric in ('valid_rate', 'novelty_rate', 'recovery_rate')}
            stats.update({'calls': len(rows), 'duplicate_calls': sum(d['duplicate'] for d in rows),
                          'invalid_parsed': sum(d['parsed'] and not d['valid'] for d in rows),
                          'parse_failures': sum(not d['parsed'] for d in rows),
                          'prompt_tokens': sum(d['prompt_tokens'] for d in rows),
                          'output_tokens': sum(d['output_tokens'] for d in rows),
                          'seconds': sum(d['seconds'] for d in rows)})
            aggregates[split][group] = stats
    write_json(output/'summary.json', {'feedback': aggregates, 'thinking': diagnostics})
    fields = ['instance', 'split', 'group', 'seed', 'query', 'answer', 'parsed', 'valid', 'duplicate',
              'new_valid', 'cumulative_recovered', 'seconds', 'prompt_tokens', 'output_tokens', 'thinking_chars', 'done_reason']
    with (output/'all_calls.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for r in results:
            writer.writerows({**{k: r[k] for k in ('instance', 'split', 'group', 'seed')}, **d} for d in r['details'])
    report = ['# Qwen3 8B 反馈消融与思考预算实验', '',
              '## 实验设计', '',
              '反馈实验共 4 题 × 3 个种子 × 4 组 × 8 次调用 = 384 次。原题 n4_219 单独作为开发题；新题 n4_085、n4_182、n4_116 分别有 2、4、8 个合法图，使用选择种子 20261112 从同一官方四节点数据中预先抽取。新题只构成小规模迁移检查，结构难度仍接近开发题。', '',
              '四组为 baseline（B 原版）、duplicate（仅重复反馈）、consistency（仅观测一致性反馈）、combined（两者结合）。所有组都有官方原始历史和避免重复的提示；消融测试的是新增的显式反馈。第一轮无历史，因此不添加反馈。各组固定 qwen3:8b Q4_K_M，think=false，temperature=0.6，top_p=0.95，top_k=20，min_p=0，repeat_penalty=1，num_ctx=4096，num_predict=2048。三个种子为 33550336、33550337、33550338，轮内沿用固定种子以衔接上一轮实验。组执行顺序按种子轮换。', '',
              '每轮严格限制为 8 次生成，错误、重复、解析失败均消耗槽位；没有隐藏重试，也没有用正确答案替换模型输出。反馈只接收节点、观测与候选历史。验证器检查候选的可达关系，重复检测忽略边排列顺序。标准答案仅用于事后计分。反馈增加了输入 token，因此本实验是等调用次数比较，不能称为等总 token 比较。', '',
              '验证器已枚举全部 543 个四节点 DAG，在四道题上分别核对合法集合与标准答案完全一致。384 次生成还通过官方评测器回放和独立计分交叉检查。', '',
              'VR=有效提案次数/8；NR=不同可解析图数/8（可能包含错误图）；RR=不同合法图数/该题合法图总数。以下先按题目和种子计算，再取宏平均。', '']
    for split in aggregates:
        report += ['## '+('原开发题' if split == 'development' else '三道新题'), '',
                   '| 组别 | VR | NR | RR | 重复次数 | 输入 tokens | 输出 tokens | 总秒数 |',
                   '|---|---|---|---|---|---|---|---|']
        for group, s in aggregates[split].items():
            report.append(f"| {group} | {s['valid_rate']:.2%} | {s['novelty_rate']:.2%} | {s['recovery_rate']:.2%} | {s['duplicate_calls']}/{s['calls']} | {s['prompt_tokens']} | {s['output_tokens']} | {s['seconds']:.2f} |")
        report.append('')
    report += ['## 逐题逐种子结果', '', '| 题目 | 种子 | 组别 | VR | NR | RR |', '|---|---|---|---|---|---|']
    for r in results:
        s = r['official']
        report.append(f"| {r['instance']} | {r['seed']} | {r['group']} | {s['valid_rate']:.2%} | {s['novelty_rate']:.2%} | {s['recovery_rate']:.2%} |")
    report += ['', '## 思考预算诊断', '',
               '独立于反馈消融：只测原题的首个候选，每个上限各测三个种子，共 9 次请求。think=true，num_ctx统一为16384，num_predict分别为2048、4096、8192，其余采样参数同 B。扩大上下文是为了容纳最大输出，因此这里的2048组也重新运行；不能直接与之前num_ctx=4096的C组混用。没有历史、没有反馈。该诊断只衡量能否完成首答及其成本，不衡量8次生成的覆盖率。', '',
               '| 种子 | 输出上限 | 实际输出 tokens | 正确 | 结束原因 | 秒数 |', '|---|---|---|---|---|---|']
    for d in diagnostics:
        report.append(f"| {d['seed']} | {d['limit']} | {d['output_tokens']} | {d['valid']} | {d['done_reason']} | {d['seconds']:.2f} |")
    report += ['', '## 文件与限制', '',
               'config.json记录版本、模型摘要、参数和代码哈希；instances.json包含评测标准答案，未发送给模型；feedback和thinking目录保存逐次完整请求、响应和思考文本；all_calls.csv包含384次反馈实验明细；thinking_results.json包含9次预算诊断；summary.json为汇总。', '',
               '仅四节点、四题、三个种子，不能推断总体性能或统计显著性。同一轮调用具有历史依赖和固定种子，不能当成独立样本。RR依赖解空间大小，原题和新题绝对数值不直接横向比较。低解数题可能已经覆盖全部答案，此后重复本身不一定表示探索失败。思考与非思考实际消耗不同，耗时受加载、缓存和运行顺序影响。', '',
               '复跑：启动Ollama后执行 `D:\\Anaconda\\python.exe "Final Project/experiments/run_feedback_study.py"`。`--output`指向已有目录可重放已完成请求并继续；配置必须一致。`--verify-only`不调用模型，只验证反馈检查器。']
    (output/'REPORT.md').write_text('\n'.join(report)+'\n', encoding='utf-8')
    os.environ.setdefault('MPLCONFIGDIR', str(HERE/'.mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for ax, split, title in zip(axes, aggregates, ['Original development question', 'Three new questions (macro mean)']):
        x = np.arange(4)
        for j, (metric, label) in enumerate([('valid_rate', 'Validity'), ('novelty_rate', 'Uniqueness'), ('recovery_rate', 'Recovery')]):
            ax.bar(x+(j-1)*.24, [aggregates[split][g][metric]*100 for g in GROUPS], .22, label=label)
        ax.set_xticks(x, ['Baseline', 'Duplicate', 'Consistency', 'Combined'], rotation=15)
        ax.set_ylim(0, 115)
        ax.set_ylabel('Percent')
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.savefig(output/'comparison.png', dpi=170)
    plt.close(fig)
    print(json.dumps(aggregates, indent=2), flush=True)


if __name__ == '__main__':
    main()
