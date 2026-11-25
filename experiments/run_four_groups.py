"""Four paired Ollama configurations on the same HypoSpace instance.

Completed request files can be replayed with --output to resume an interrupted run.
"""
import argparse
import csv
import json
import os
import statistics
import subprocess
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import networkx as nx
import requests
from common import HERE, COMMIT, digest, load_upstream
from run_local_causal import write_json

GROUPS = {
    'A': {'think': False, 'temperature': 0.0, 'top_p': 0.95},
    'B': {'think': False, 'temperature': 0.6, 'top_p': 0.95},
    'C': {'think': True, 'temperature': 0.6, 'top_p': 0.95},
    'D': {'think': False, 'temperature': 0.7, 'top_p': 0.8},
}
SEEDS = [33550336, 33550337, 33550338]
SHARED = {'num_ctx': 4096, 'num_predict': 2048, 'top_k': 20, 'min_p': 0.0, 'repeat_penalty': 1.0}
MODEL = 'qwen3:8b'
BASE_URL = 'http://127.0.0.1:11434'


def score_calls(benchmark, instance, calls):
    truths = {tuple(sorted(tuple(e) for e in g['edges'])) for g in instance['ground_truth_graphs']}
    seen, recovered = set(), set()
    rows = []
    for index, call in enumerate(calls, 1):
        response = call['response']
        answer = response['message'].get('content', '')
        thinking = response['message'].get('thinking', '')
        parsed = benchmark.parse_llm_response(answer)
        row = {'query': index, 'answer': answer, 'thinking_chars': len(thinking),
               'parsed': parsed is not None, 'valid': False, 'duplicate': False,
               'new_valid': False, 'mismatches': [], 'seconds': call['seconds'],
               'prompt_tokens': response.get('prompt_eval_count', 0),
               'output_tokens': response.get('eval_count', 0),
               'done_reason': response.get('done_reason'),
               'load_seconds': response.get('load_duration', 0)/1e9}
        if parsed is not None:
            key = tuple(sorted(parsed.edges))
            graph = parsed.to_networkx()
            row['duplicate'] = key in seen
            seen.add(key)
            for observation in instance['observations']:
                affected = nx.descendants(graph, observation['perturbed_node'])
                actual = {node: int(node in affected) for node in benchmark.nodes}
                if actual != observation['effects']:
                    row['mismatches'].append({'perturb': observation['perturbed_node'],
                                              'expected': observation['effects'], 'actual': actual})
            row['valid'] = not row['mismatches']
            if row['valid'] and key in truths:
                row['new_valid'] = key not in recovered
                recovered.add(key)
        row['cumulative_recovered'] = len(recovered)
        rows.append(row)
    counts = {'n_valid': sum(r['valid'] for r in rows), 'n_unique_all': len(seen),
              'n_recovered_gts': len(recovered), 'parse_success_count': sum(r['parsed'] for r in rows)}
    return rows, counts


def summarize(output, results):
    aggregates = {}
    for group in GROUPS:
        selected = [r for r in results if r['group'] == group]
        rates = {metric: {'mean': statistics.mean(r['official'][metric] for r in selected),
                          'std': statistics.pstdev(r['official'][metric] for r in selected)}
                 for metric in ('valid_rate', 'novelty_rate', 'recovery_rate', 'parse_success_rate')}
        rows = [row for r in selected for row in r['details']]
        aggregates[group] = {**rates, 'calls': len(rows), 'seconds': sum(r['seconds'] for r in rows),
                             'prompt_tokens': sum(r['prompt_tokens'] for r in rows),
                             'output_tokens': sum(r['output_tokens'] for r in rows),
                             'truncated': sum(r['done_reason'] == 'length' for r in rows),
                             'empty_answers': sum(not r['answer'].strip() for r in rows),
                             'duplicate_calls': sum(r['duplicate'] for r in rows),
                             'invalid_parsed': sum(r['parsed'] and not r['valid'] for r in rows)}
    write_json(output/'summary.json', {'groups': aggregates, 'results': results,
                                     'interpretation': 'One development instance, three seeds. Standard deviations are across seeds, not confidence intervals.'})
    fields = ['group','seed','query','answer','parsed','valid','duplicate','new_valid','cumulative_recovered',
              'seconds','prompt_tokens','output_tokens','thinking_chars','done_reason']
    with (output/'all_calls.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for result in results:
            writer.writerows({'group': result['group'], 'seed': result['seed'], **row} for row in result['details'])
    group_rows = []
    for group, stats in aggregates.items():
        rates = ' | '.join(f'{stats[k]["mean"]:.2%} ± {stats[k]["std"]:.2%}' for k in ('valid_rate','novelty_rate','recovery_rate'))
        group_rows.append(f'| {group} | {rates} | {stats["output_tokens"]} | {stats["seconds"]:.2f} | {stats["truncated"]} |')
    seed_rows = '\n'.join(f'| {r["group"]} | {r["seed"]} | {r["official"]["valid_rate"]:.2%} | {r["official"]["novelty_rate"]:.2%} | {r["official"]["recovery_rate"]:.2%} |' for r in results)
    text = f'''# Qwen3 8B 四组参数对照实验

共完成 4 组 × 3 个种子 × 每题 8 次提案 = 96 次本地模型请求。使用与上一轮相同的 n4_219 题目，4 个节点、4 条干预观测、8 个合法图。该题来自官方生成器，属于开发用单题实验。

## 配置

| 组别 | think | temperature | top_p |
|---|---|---|---|
| A | false | 0 | 0.95 |
| B | false | 0.6 | 0.95 |
| C | true | 0.6 | 0.95 |
| D | false | 0.7 | 0.8 |

公共配置为 qwen3:8b、Q4_K_M、num_ctx=4096、num_predict=2048、top_k=20、min_p=0、repeat_penalty=1。三个种子为 33550336、33550337、33550338，每轮的 8 次请求使用相同的轮次种子。组间及种子间清空历史。每次请求均由官方 create_prompt 根据之前成功解析的候选重建用户消息，包括无效候选。无额外 system 提示。思考组仅将 message.content 送入评分器，完整 message.thinking 保存在原始调用文件。

A 组在共同的 2048 输出上限下重新运行，上一次 64 上限的结果单独保留。所有请求共享候选次数及输出上限，实际 token 消耗与耗时单独记录。每次失败或截断仍消耗一个槽位，不增加模型重试。组顺序为每个种子依次 A、B、C、D，耗时可能受到加载、缓存及运行顺序影响。

## 平均表现

均值和标准差均按三个种子计算，标准差不是置信区间。A 组采用温度 0，三个种子的运行不构成三个独立随机采样证据。

| 组别 | VR 均值 ± 标准差 | NR 均值 ± 标准差 | RR 均值 ± 标准差 | 输出 tokens 总计 | 请求耗时总计（秒） | 截断次数 |
|---|---|---|---|---|---|---|
{chr(10).join(group_rows)}

![对照结果](comparison.png)

## 每个种子的结果

| 组别 | 种子 | VR | NR | RR |
|---|---|---|---|---|
{seed_rows}

## 评分与完整记录

全部 96 次提案均使用固定版本官方因果图解析器和评测器，同时独立计算可达关系、不同边集和恢复的合法图进行核对。错误类型、重复次数、截断、空答案及逐次覆盖情况存入 summary.json 和 all_calls.csv。

config.json 保存公共配置、模型摘要、版本与脚本哈希。各 group_seed 子目录保存逐次 request、response、完整思考文本、耗时和官方结果。instance.json 包含题目和完整标准答案，标准答案不进入请求。runtime_gpu.json 保存运行时模型装载状态。

本结果限定于一题三个种子，不代表总体模型能力或统计显著性。B−A 对比温度影响，C−B 对比该共同采样配置下思考模式的影响，D 是另一套非思考配置。

## 复跑

启动 Ollama 后，在课程目录执行 `D:\\Anaconda\\python.exe "Final Project/experiments/run_four_groups.py"`，会生成新目录。传入 `--output` 指向已有目录可回放已完成的请求并继续缺失调用。配置和请求必须完全匹配才能复用。
'''
    (output/'REPORT.md').write_text(text, encoding='utf-8')
    os.environ.setdefault('MPLCONFIGDIR', str(HERE/'.mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(10,4), layout='constrained')
    x = np.arange(4)
    for index, (metric, label) in enumerate([('valid_rate','Validity'),('novelty_rate','Uniqueness'),('recovery_rate','Recovery')]):
        axes[0].bar(x+(index-1)*0.24,[aggregates[g][metric]['mean']*100 for g in GROUPS],0.22,
                    yerr=[aggregates[g][metric]['std']*100 for g in GROUPS],capsize=3,label=label)
    axes[0].set_xticks(x,list(GROUPS)); axes[0].set_ylim(0,112)
    axes[0].set_ylabel('Percent'); axes[0].legend(fontsize=8)
    axes[0].set_title('One instance, three seeds (mean +/- SD)')
    axes[1].bar(list(GROUPS),[aggregates[g]['seconds'] for g in GROUPS],color=['#777777','#377eb8','#4daf4a','#984ea3'])
    axes[1].set_ylabel('Total seconds for 24 requests'); axes[1].set_title('Measured inference time, including loading')
    fig.savefig(output/'comparison.png',dpi=170); plt.close(fig)
    print(json.dumps(aggregates, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    timestamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    output = args.output or HERE/'runs'/('qwen3_four_groups_'+timestamp)
    output.mkdir(parents=True, exist_ok=True)
    session = requests.Session(); session.trust_env = False

    def api(route, payload=None):
        if payload is None:
            response = session.get(BASE_URL+route,timeout=10)
        else:
            response = session.post(BASE_URL+route,json=payload,timeout=(10,300))
        response.raise_for_status()
        return response.json()

    mod = load_upstream('causal','run_causal_benchmark.py')
    data = HERE/'data/causal_n4_full_observations.json'
    benchmark = mod.CausalBenchmarkEnhanced(str(data))
    instance = next(row for row in benchmark.all_observation_sets if row['observation_set_id']=='n4_219')
    model = next(m for m in api('/api/tags')['models'] if m['name']==MODEL)
    config = {'groups':GROUPS,'shared_options':SHARED,'seeds':SEEDS,'queries_per_run':8,
              'model':model,'version':api('/api/version'),'dataset_sha256':digest(data),
              'instance_id':instance['observation_set_id'],'upstream_commit':COMMIT,
              'evaluator_sha256':digest(mod.__file__),'runner_sha256':digest(__file__),
              'history':'all previously parsed hypotheses; official prompt','retries':0,
              'timeout_seconds':300,'endpoint':BASE_URL+'/api/chat',
              'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip()}
    if (output/'config.json').exists():
        assert json.loads((output/'config.json').read_text(encoding='utf-8'))==config, 'Resume configuration differs'
    else:
        write_json(output/'config.json',config)
    write_json(output/'instance.json',instance)
    write_json(output/'model_show.json',api('/api/show',{'model':MODEL}))
    print('OUTPUT_DIR='+str(output.resolve()),flush=True)
    results=[]
    for seed in SEEDS:
        for group, params in GROUPS.items():
            folder=output/f'{group}_{seed}'; folder.mkdir(exist_ok=True)
            calls=[]

            class Adapter:
                def query_with_usage(self, prompt):
                    index=len(calls)+1
                    payload={'model':MODEL,'think':params['think'],'stream':False,'keep_alive':'10m',
                             'messages':[{'role':'user','content':prompt}],
                             'options':{**SHARED,'temperature':params['temperature'],'top_p':params['top_p'],'seed':seed}}
                    path=folder/f'call_{index:02d}.json'
                    if path.exists():
                        call=json.loads(path.read_text(encoding='utf-8'))
                        assert call['request']==payload,'Saved request differs'
                    else:
                        write_json(folder/f'request_{index:02d}.json',payload)
                        start=time.perf_counter()
                        response=api('/api/chat',payload)
                        call={'request':payload,'response':response,'seconds':time.perf_counter()-start}
                        write_json(path,call)
                    calls.append(call)
                    response=call['response']
                    content=response['message'].get('content','')
                    print(f'{group} seed={seed} {index}/8: {content!r}; stop={response.get("done_reason")}; output_tokens={response.get("eval_count")}; {call["seconds"]:.1f}s',flush=True)
                    prompt_tokens=response.get('prompt_eval_count',0)
                    output_tokens=response.get('eval_count',0)
                    return {'response':content,'usage':{'prompt_tokens':prompt_tokens,'completion_tokens':output_tokens,
                            'total_tokens':prompt_tokens+output_tokens},'cost':0.0}

            official=benchmark.evaluate_single_observation_set(Adapter(),instance,8,False,max_retries=1)
            if len(calls)!=8 or official['errors']:
                write_json(folder/'incomplete_result.json',official)
                raise RuntimeError('Transport/evaluator failure; resume completed calls using --output')
            rows, counts=score_calls(benchmark,instance,calls)
            assert all(official[key]==value for key,value in counts.items())
            result={'group':group,'seed':seed,'official':official,'details':rows,'independent_scoring_matches':True}
            write_json(folder/'result.json',result)
            results.append(result)
            write_json(output/'progress.json',{'completed_runs':len(results),'planned_runs':12,'results':results})
    write_json(output/'runtime_gpu.json',api('/api/ps'))
    summarize(output,results)


if __name__=='__main__':
    main()
