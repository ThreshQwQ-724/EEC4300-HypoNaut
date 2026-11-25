"""Run one pinned four-node HypoSpace instance against local Ollama."""
import itertools
import json
import random
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import networkx as nx
import requests
from common import HERE, COMMIT, digest, load_upstream

BASE_URL = 'http://127.0.0.1:11434'
MODEL = 'qwen3:8b'
SEED = 33550336
OPTIONS = {'temperature': 0, 'num_ctx': 4096, 'num_predict': 64, 'seed': SEED}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    started = datetime.now(timezone(timedelta(hours=8)))
    output = HERE / 'runs' / ('qwen3_8b_causal_n4_' + started.strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True)
    session = requests.Session()
    session.trust_env = False

    def api(route, payload=None):
        if payload is None:
            response = session.get(BASE_URL + route, timeout=10)
        else:
            response = session.post(BASE_URL + route, json=payload, timeout=(10, 180))
        response.raise_for_status()
        return response.json()

    dataset_path = HERE / 'data/causal_n4_full_observations.json'
    mod = load_upstream('causal', 'run_causal_benchmark.py')
    benchmark = mod.CausalBenchmarkEnhanced(str(dataset_path))
    candidates = [row for row in benchmark.all_observation_sets if row['n_compatible_graphs'] == 8]
    selected = random.Random(SEED).choice(candidates)
    write_json(output / 'instance.json', selected)
    # Independently enumerate all 4096 directed-edge subsets and compare valid DAGs.
    nodes = benchmark.nodes
    edges = list(itertools.permutations(nodes, 2))
    independent_gt = set()
    for mask in range(1 << len(edges)):
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edge for bit, edge in enumerate(edges) if mask & (1 << bit))
        if nx.is_directed_acyclic_graph(graph) and all(
            {node: int(node in nx.descendants(graph, obs['perturbed_node'])) for node in nodes} == obs['effects']
            for obs in selected['observations']
        ):
            independent_gt.add(tuple(sorted(graph.edges)))
    stored_gt = {tuple(sorted(tuple(edge) for edge in graph['edges'])) for graph in selected['ground_truth_graphs']}
    assert independent_gt == stored_gt and len(stored_gt) == 8

    model_info = next(model for model in api('/api/tags')['models'] if model['name'] == MODEL)
    show = api('/api/show', {'model': MODEL})
    write_json(output / 'model_show.json', show)
    config = {
        'started_at': started.isoformat(), 'model': MODEL, 'model_info': model_info,
        'ollama_version': api('/api/version'), 'endpoint': BASE_URL + '/api/chat',
        'think': False, 'stream': False, 'options': OPTIONS, 'keep_alive': '5m',
        'request_timeout_seconds': 180, 'api_retries': 0, 'max_retries_per_slot': 1,
        'n_instances': 1, 'n_queries': 8, 'node_count': 4, 'n_observations': 4,
        'max_edges': benchmark.max_edges, 'selection_seed': SEED,
        'selection_rule': 'Uniform choice among the 24 instances with exactly 8 ground truths',
        'instance_id': selected['observation_set_id'], 'dataset_path': str(dataset_path),
        'dataset_sha256': digest(dataset_path), 'upstream_commit': COMMIT,
        'evaluator_sha256': digest(Path(mod.__file__)), 'runner_sha256': digest(__file__),
        'prompt_policy': 'Official prompt with all previous successfully parsed predictions; one user message; no added system message',
        'independent_ground_truth_check': 'Exhaustive enumeration of 4096 edge subsets agrees with all 8 stored DAGs',
        'python': sys.version,
        'hardware': subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'], text=True).strip(),
        'unoverridden_model_parameters': show.get('parameters', ''),
    }
    write_json(output / 'config.json', config)
    print('OUTPUT_DIR=' + str(output), flush=True)
    logs = []

    class LocalModel:
        def query_with_usage(self, prompt):
            index = len(logs) + 1
            payload = {'model': MODEL, 'messages': [{'role': 'user', 'content': prompt}],
                       'stream': False, 'think': False, 'options': OPTIONS, 'keep_alive': '5m'}
            write_json(output / f'request_{index:02d}.json', payload)
            tick = time.perf_counter()
            try:
                response = api('/api/chat', payload)
            except Exception as error:
                log = {'query': index, 'request': payload, 'error': str(error), 'elapsed_seconds': time.perf_counter()-tick}
                logs.append(log)
                write_json(output / f'call_{index:02d}.json', log)
                raise
            log = {'query': index, 'request': payload, 'response': response, 'elapsed_seconds': time.perf_counter()-tick}
            logs.append(log)
            write_json(output / f'call_{index:02d}.json', log)
            content = response['message']['content']
            print(f'[{index}/8] {content!r} ({log["elapsed_seconds"]:.2f}s)', flush=True)
            prompt_tokens = response.get('prompt_eval_count', 0)
            completion_tokens = response.get('eval_count', 0)
            return {'response': content, 'usage': {'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens, 'total_tokens': prompt_tokens+completion_tokens}, 'cost': 0.0}

    tick = time.perf_counter()
    official = benchmark.evaluate_single_observation_set(LocalModel(), selected, 8, False, max_retries=1)
    duration = time.perf_counter() - tick
    write_json(output / 'official_result.json', official)
    write_json(output / 'runtime_gpu.json', api('/api/ps'))
    seen, recovered, valid_count, parse_count, details = set(), set(), 0, 0, []
    for log in logs:
        response = log.get('response', {})
        text = response.get('message', {}).get('content', '')
        parsed = benchmark.parse_llm_response(text)
        row = {'query': log['query'], 'answer': text, 'parsed': parsed is not None,
               'valid': False, 'duplicate': False, 'new_valid': False, 'mismatches': [],
               'stop_reason': response.get('done_reason'), 'seconds': log['elapsed_seconds'],
               'prompt_tokens': response.get('prompt_eval_count', 0), 'output_tokens': response.get('eval_count', 0)}
        if parsed is not None:
            parse_count += 1
            key = tuple(sorted(parsed.edges))
            row['duplicate'] = key in seen
            seen.add(key)
            graph = parsed.to_networkx()
            for obs in selected['observations']:
                actual = {node: int(node in nx.descendants(graph, obs['perturbed_node'])) for node in nodes}
                if actual != obs['effects']:
                    row['mismatches'].append({'perturb': obs['perturbed_node'], 'expected': obs['effects'], 'actual': actual})
            row['valid'] = not row['mismatches']
            valid_count += int(row['valid'])
            if row['valid'] and key in stored_gt:
                row['new_valid'] = key not in recovered
                recovered.add(key)
        row['cumulative_recovered'] = len(recovered)
        details.append(row)
    assert len(logs) == 8
    assert (official['n_valid'], official['n_unique_all'], official['n_recovered_gts'], official['parse_success_count']) == (
        valid_count, len(seen), len(recovered), parse_count)
    report = {'config': config, 'metrics': official, 'wall_seconds': duration, 'details': details,
              'independent_scoring_matches': True, 'api_requests': len(logs),
              'missing_ground_truths': [list(key) for key in sorted(stored_gt-recovered)],
              'scope': 'One actual local-LLM instance run; no aggregate model-performance inference.'}
    write_json(output / 'result.json', report)
    table = '\n'.join(f'| {r["query"]} | `{r["answer"].replace(chr(10)," ")}` | {r["parsed"]} | {r["valid"]} | {r["duplicate"]} | {r["cumulative_recovered"]}/8 | {r["prompt_tokens"]}/{r["output_tokens"]} | {r["seconds"]:.2f} |' for r in details)
    observations = '\n'.join(obs['string'] for obs in selected['observations'])
    params = '\n'.join(f'| {key} | `{value}` |' for key,value in {
        '模型': MODEL, '模型摘要':model_info['digest'], '量化':show.get('details',{}).get('quantization_level'),
        'Ollama':config['ollama_version'], 'GPU':config['hardware'], '节点数':4,'观测数':4,'边数上限':6,
        '题目ID':selected['observation_set_id'],'合法答案数':8,'请求数':8,'温度':0,'思考模式':False,
        '上下文 tokens':4096,'单次最大输出 tokens':64,'随机种子':SEED,'重试次数':0,
        '历史模式':'官方提示词包含前面所有成功解析的答案','API地址':config['endpoint']}.items())
    summary = f'''# Qwen3 8B 四节点 HypoSpace 实测

开始时间 {started.isoformat()}。本次沿用上次本地接口测试的温度 0、think=false、4096 上下文及 64 输出 token 上限，增加固定随机种子。组员的具体四节点题库尚未提供，因此使用固定版本官方生成器生成的题库。预先从含 8 个合法答案的 24 道题中按固定种子选定一道，预算为 8 次。单题结果仅用于验证实际评测链路。

## 参数

| 参数 | 值 |
|---|---|
{params}

模型内置、未由本次请求覆盖的参数如下，temperature 以请求中的 0 为准。

```text
{show.get('parameters','')}
```

## 题目

```text
{observations}
```

1 表示被干预节点的下游后代，0 表示不受影响，干预节点本身记为 0。必须包含 A→B、B→C、C→D，可自由选择加入 A→C、A→D、B→D，因此恰有 2³=8 个合法图。答案枚举只用于本地评分，发送给模型的请求可在 request_*.json 中核对。

## 结果

| 指标 | 数值 |
|---|---:|
| 解析成功率 | {official['parse_success_rate']:.2%} |
| Validity | {official['valid_rate']:.2%} |
| Uniqueness | {official['novelty_rate']:.2%} |
| Recovery | {official['recovery_rate']:.2%} |
| 有效提案次数 | {valid_count}/8 |
| 不同的可解析图 | {len(seen)} |
| 发现的不同合法图 | {len(recovered)}/8 |
| 输入 tokens 总数 | {official['token_usage']['prompt_tokens']} |
| 输出 tokens 总数 | {official['token_usage']['completion_tokens']} |
| 请求总墙钟时间（含首次加载） | {duration:.2f} 秒 |
| 云端 API 费用 | 0（本机电力成本未测量） |

| 次数 | 原始回答 | 解析成功 | 观测一致 | 重复 | 累积发现 | 输入/输出 tokens | 秒 |
|---|---|---|---|---|---|---|---|
{table}

官方评分与独立后代集合检查一致。每次请求均计入 8 次预算，解析失败、重复和不合法答案保留原始记录。评测采用官方未修改的因果图代码，提交 `{COMMIT}`。温度 0 与关闭思考是当前试跑配置，后续可作为需要比较的实验因素。

## 文件与复跑

`config.json` 保存配置和文件摘要，`model_show.json` 保存模型详情与模板，`instance.json` 保存完整题目和答案，`request_*.json` 与 `call_*.json` 保存逐次请求和返回，`official_result.json` 保存原版评测结果，`result.json` 保存独立检查与缺失答案，`runtime_gpu.json` 保存运行时 GPU 装载情况。

先启动 Ollama，再从课程目录执行如下命令。每次运行会产生新目录。

```powershell
& 'D:\\Anaconda\\python.exe' '.\\Final Project\\experiments\\run_local_causal.py'
```
'''
    (output / 'REPORT.md').write_text(summary, encoding='utf-8')
    print(json.dumps({'output':str(output),'VR':official['valid_rate'],'NR':official['novelty_rate'],
                      'RR':official['recovery_rate'],'tokens':official['token_usage'],'seconds':duration},ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
