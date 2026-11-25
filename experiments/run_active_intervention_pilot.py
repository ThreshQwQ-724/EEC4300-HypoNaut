"""Compare one-step intervention policies using shared eight-graph predictions."""
import argparse
from collections import Counter
import json
import mmap
from pathlib import Path
import random
import re
from statistics import mean

import requests

from common import DATA, HERE, RESULTS, digest
import ijson
from audit_intervention_direction import NODES, METADATA, entropy
from run_four_groups import BASE_URL, MODEL, SEEDS
from run_local_causal import write_json
from run_qwen_guided_edits import decode, request, schema
from run_qwen_rr_budget import MODEL_REFERENCE, read
from solve_partial_graph import solve
from stream_causal import check_graph
from validate_graph_rr_budget import DATA as OBSERVATIONS

SELECTION_SEED = 1022126
RUN_NAME = 'active_intervention_pilot_20261113'
SOURCE = DATA/'causal/datasets/node06/n6_all_observations.json'
INDEX = {node: i for i, node in enumerate(NODES)}


def select_tasks():
    used = set()
    for path in (HERE/'runs').glob('*/config.json'):
        if not path.parent.name.startswith('active_intervention_pilot'):
            used.update(re.findall(r'\bn[1-6]_\d{3}\b', path.read_text(encoding='utf-8')))
    rng = random.Random(SELECTION_SEED)
    tasks = read(OBSERVATIONS)
    return [row for n in (1, 2, 3) for row in rng.sample(sorted(
        (t for t in tasks if t['n_observations'] == n and t['id'] not in used),
        key=lambda t: t['id']), 4)]


def closure(edges):
    reach = [0]*len(NODES)
    for a, b in edges:
        reach[INDEX[a]] |= 1 << INDEX[b]
    for k in range(len(NODES)):
        bit = 1 << k
        for i in range(len(NODES)):
            if reach[i] & bit:
                reach[i] |= reach[k]
    return reach


def predict(graphs, nodes):
    counts = {node: Counter() for node in nodes}
    indices = [(node, INDEX[node]) for node in nodes]
    for edges in graphs:
        reach = closure(edges)
        for node, i in indices:
            counts[node][reach[i]] += 1
    return counts


def shown_counts(counts):
    def label(mask):
        return ''.join(node for node, i in INDEX.items() if mask & (1 << i)) or 'none'
    return {node: {label(mask): count for mask, count in sorted(outcomes.items())}
            for node, outcomes in counts.items()}


def qwen_select(task, counts, seed, path, session):
    predictions = shown_counts(counts)
    evidence = '\n'.join(row['string'] for row in task['observations'])
    prompt = (
        'Choose exactly one previously unobserved node for the next intervention. '
        'Your goal is to minimize the expected fraction of compatible DAGs remaining. '
        'The only evidence about unobserved interventions is below. Each entry groups eight '
        'candidate graphs by their predicted complete downstream node set and shows the '
        'group count. Prefer outcomes with balanced small groups. '
        'Respond with JSON containing the chosen node.\n'
        f'Known observations:\n{evidence}\n'
        f'Candidate prediction groups:\n{json.dumps(predictions, sort_keys=True)}')
    call = request(session, path, prompt, seed, schema({'node': {'type': 'string', 'enum': list(counts)}}))
    chosen = decode(call).get('node')
    if not isinstance(chosen, str) or chosen not in counts:
        return None, call
    return chosen, call


class ArrayReader:
    """Expose only the selected JSON answer array to the streaming parser."""
    def __init__(self, stream, begin, end):
        self.stream = stream
        self.remaining = end-begin
        self.stream.seek(begin)

    def read(self, size):
        block = self.stream.read(min(size, self.remaining))
        self.remaining -= len(block)
        return block


def exact_outcomes(tasks, audit):
    assert SOURCE.stat().st_size == audit['size_bytes'] and not audit['errors']
    source_order = {row['id']: i for i, row in enumerate(audit['instances'])}
    results = {}
    with SOURCE.open('rb') as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
        cursor = 0
        for task in sorted(tasks, key=lambda row: source_order[row['id']]):
            item = task['id']
            marker = f'"observation_set_id": "{item}"'.encode()
            start = mapped.find(marker, cursor)
            assert start >= 0
            obs_key = mapped.find(b'"observations"', start, start+16384)
            key = mapped.find(b'"ground_truth_graphs"', start, start+16384)
            begin = mapped.find(b'[', key, key+128)
            end = mapped.find(b'"n_compatible_graphs"', begin)
            assert start < obs_key < key < begin < end
            obs_begin = mapped.find(b'[', obs_key, key)
            assert read_observations(mapped[obs_begin:key]) == task['observations']
            closing = mapped.rfind(b']', begin, end)
            assert closing > begin and mapped[closing+1:end].strip() == b','
            unknown = sorted(set(NODES)-{row['perturbed_node'] for row in task['observations']})
            counts = predict((row['edges'] for row in ijson.items(
                ArrayReader(stream, begin, closing+1), 'item')), unknown)
            assert all(sum(c.values()) == task['n_ground_truths'] for c in counts.values())
            results[item] = counts
            cursor = end
            print(f'Exact scoring {len(results)}/{len(tasks)} {item}', flush=True)
    return results


def read_observations(raw):
    return json.loads(raw.strip().rstrip(b','))


def score(record, counts):
    total = record['compatible_graphs']
    remaining = {node: sum(n*n for n in groups.values())/(total*total) for node, groups in counts.items()}
    # An invalid model choice uses the declared uniform random policy.
    chosen = record['qwen_node']
    random_remaining = mean(remaining.values())
    record.update(exact_groups=shown_counts(counts), remaining_by_node=remaining,
                  random=random_remaining, entropy=remaining[record['entropy_node']],
                  qwen=remaining[chosen] if chosen is not None else random_remaining,
                  oracle=min(remaining.values()))
    assert record['oracle'] <= min(record[key] for key in ('random', 'entropy', 'qwen'))


def report(output, records):
    names = {'random': '均匀随机', 'entropy': '候选预测熵', 'qwen': 'Qwen 选节点', 'oracle': '完整解集最优参考'}
    summary = {}
    lines = ['# 主动干预选择试跑', '',
             '从此前模型实验未用过的六节点题中，按 1、2、3 次已知干预各抽 4 题。各组每题共享同样的 8 个合法候选图和一次额外观测机会。Qwen 和程序熵策略读取相同的候选预测分组。随机组对所有未观测节点均匀选择，取精确期望。', '',
             '模拟中每个合法图等可能成为隐藏真图。对某个干预节点，完整解集按该节点的后代集合分组。若分组大小为 n_i、合法图总数为 N，新增观测后期望剩余比例为 Σ(n_i/N)²。完整解集仅用于事后评分和最优参考。该指标独立于固定观测的 HypoSpace RR。', '',
             '| 已知干预数 | 题数 | 均匀随机剩余 | 候选预测熵剩余 | Qwen 剩余 | 最优参考剩余 |',
             '|---|---:|---:|---:|---:|---:|']
    for key in (1, 2, 3, 'all'):
        rows = [r for r in records if key == 'all' or r['observations'] == key]
        d = {m: mean(r[m] for r in rows) for m in names}
        d['tasks'] = len(rows)
        summary[str(key)] = d
        lines.append(f"| {key} | {len(rows)} | "+' | '.join(f'{d[m]:.2%}' for m in names)+' |')
    diffs = [r['entropy']-r['qwen'] for r in records]
    wins = [sum(d > 1e-12 for d in diffs), sum(abs(d) <= 1e-12 for d in diffs),
            sum(d < -1e-12 for d in diffs)]
    summary['qwen_vs_entropy_win_tie_loss'] = wins
    summary['model_calls'] = len(records)
    summary['invalid_choices'] = sum(r['qwen_node'] is None for r in records)
    for key in ('prompt_tokens', 'output_tokens', 'request_seconds'):
        summary[key] = sum(r[key] for r in records)
    lines += ['', f'Qwen 对候选预测熵的逐题胜、平、负为 {wins[0]}、{wins[1]}、{wins[2]}。'
              f'共 {len(records)} 次本地模型调用，{summary["invalid_choices"]} 次无效选择。'
              f'模型请求累计 {summary["request_seconds"]:.2f} 秒。无效选择按均匀随机策略评分。', '',
              '| 题目 | 合法图数 | 候选预测熵选点 | Qwen 选点 | 均匀随机剩余 | 候选预测熵剩余 | Qwen 剩余 | 最优剩余 |',
              '|---|---:|---|---|---:|---:|---:|---:|']
    for r in records:
        values = ' | '.join(f'{r[key]:.2%}' for key in names)
        lines.append(f"| {r['id']} | {r['compatible_graphs']:,} | {r['entropy_node']} | "
                     f"{r['qwen_node'] or '随机回退'} | {values} |")
    lines += ['', '样本为 12 题，每题一次 Qwen 选择。8 个候选由确定性搜索产生，候选分布可能与完整解集分布不同。假设是模拟的均匀隐藏图先验。']
    write_json(output/'results.json', {'summary': summary, 'records': records})
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE/'runs'/RUN_NAME)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    path = output/'config.json'
    lookup = {t['id']: t for t in read(OBSERVATIONS)}
    tasks = [lookup[key] for key in read(path)['cases']] if path.exists() else select_tasks()
    assert len(tasks) == len({t['id'] for t in tasks}) == 12
    session = requests.Session()
    session.trust_env = False
    response = session.get(BASE_URL+'/api/tags', timeout=10)
    response.raise_for_status()
    model = next(m for m in response.json()['models'] if m['name'] == MODEL)
    assert model['digest'] == read(MODEL_REFERENCE)['model']['digest']
    audit = read(RESULTS/'causal_audit.json')
    config = dict(cases=[t['id'] for t in tasks], selection_seed=SELECTION_SEED, seeds=SEEDS,
                  slots=8, model=model['digest'], observation_sha=digest(OBSERVATIONS),
                  previous_source_sha=audit['source_sha256'],
                  hashes={name: digest(HERE/name) for name in ('run_active_intervention_pilot.py',
                      'solve_partial_graph.py', 'run_partial_graph_pipeline.py', 'run_qwen_guided_edits.py')})
    if path.exists():
        assert read(path) == config
    else:
        write_json(path, config)
    print('OUTPUT_DIR='+str(output.resolve()), flush=True)
    records = []
    for i, task in enumerate(tasks):
        obs = task['observations']
        unknown = sorted(set(NODES)-{row['perturbed_node'] for row in obs})
        graphs = [p['edges'] for p in solve(NODES, 10, obs)['proposals']]
        assert len(graphs) == len(set(graphs)) == 8
        assert all(check_graph({'nodes': NODES, 'edges': g}, obs, METADATA) for g in graphs)
        counts = predict(graphs, unknown)
        assert all(sum(group.values()) == 8 for group in counts.values())
        selected = max(unknown, key=lambda node: entropy(counts[node]))
        seed = SEEDS[i % len(SEEDS)]
        qwen, call = qwen_select(task, counts, seed, output/f"{task['id']}_qwen.json", session)
        records.append(dict(id=task['id'], observations=len(obs), seed=seed,
                            compatible_graphs=task['n_ground_truths'], candidate_graphs=graphs,
                            candidate_groups=shown_counts(counts), entropy_node=selected, qwen_node=qwen,
                            prompt_tokens=call['response'].get('prompt_eval_count', 0),
                            output_tokens=call['response'].get('eval_count', 0), request_seconds=call['seconds']))
        print(f"Selection {i+1}/12 {task['id']} entropy={selected} qwen={qwen}", flush=True)
    assert len(list(output.glob('*_qwen.json'))) == 12
    exact = exact_outcomes(tasks, audit)
    for record in records:
        score(record, exact[record['id']])
    report(output, records)


if __name__ == '__main__':
    main()
