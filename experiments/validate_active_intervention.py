"""Validate observation selection and recovery after one simulated intervention."""
import argparse
from collections import Counter
import json
import mmap
from pathlib import Path
import random
import re
from statistics import mean

import requests

from run_active_intervention_pilot import (
    ArrayReader, HERE, INDEX, NODES, OBSERVATIONS, RESULTS, SOURCE, closure,
    digest, entropy, ijson, predict, qwen_select, read, write_json,
)
from common import load_upstream
from run_formal_causal_comparison import direct, valid
from run_four_groups import BASE_URL, MODEL, SEEDS
from run_qwen_rr_budget import MODEL_REFERENCE
from solve_partial_graph import solve

SELECTION_SEED = 1023126
SIMULATION_SEED = 1024126
SLOTS = 8
POLICIES = ('random', 'entropy', 'qwen', 'oracle')
MODEL_POLICIES = ('before', 'random', 'entropy')


def select_tasks(tasks):
    used = {item for path in (HERE/'runs').glob('*/config.json')
            for item in re.findall(r'\bn[1-6]_\d{3}\b', path.read_text(encoding='utf-8'))}
    rng = random.Random(SELECTION_SEED)
    return [row for n in (2, 3, 4) for row in rng.sample(sorted(
        (task for task in tasks if task['n_observations'] == n and task['id'] not in used
         and task['n_ground_truths'] >= SLOTS), key=lambda task: task['id']), 4)]


def new_observation(node, mask):
    effects = {target: int(bool(mask & (1 << INDEX[target]))) for target in NODES}
    return {'perturbed_node': node, 'effects': effects,
            'string': f'Perturb({node}) -> ' + ' '.join(f'{k}:{v}' for k, v in effects.items())}


def exact_distributions(tasks, hidden_indices, audit):
    """Read only selected answer arrays and retain counts, plus one sampled graph."""
    assert SOURCE.stat().st_size == audit['size_bytes'] and not audit['errors']
    order = {task['id']: i for i, task in enumerate(audit['instances'])}
    scored = {}
    with SOURCE.open('rb') as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
        cursor = 0
        for task in sorted(tasks, key=lambda t: order[t['id']]):
            item = task['id']
            marker = f'"observation_set_id": "{item}"'.encode()
            start = data.find(marker, cursor)
            obs_key = data.find(b'"observations"', start, start+16384)
            key = data.find(b'"ground_truth_graphs"', start, start+16384)
            begin = data.find(b'[', key, key+128)
            end = data.find(b'"n_compatible_graphs"', begin)
            assert 0 <= start < obs_key < key < begin < end
            obs_begin = data.find(b'[', obs_key, key)
            assert json.loads(data[obs_begin:key].strip().rstrip(b',')) == task['observations']
            closing = data.rfind(b']', begin, end)
            assert closing > begin and data[closing+1:end].strip() == b','
            unknown = sorted(set(NODES)-{o['perturbed_node'] for o in task['observations']})
            counts = {node: Counter() for node in unknown}
            hidden = None
            for i, row in enumerate(ijson.items(ArrayReader(stream, begin, closing+1), 'item')):
                reach = closure(row['edges'])
                for node in unknown:
                    counts[node][reach[INDEX[node]]] += 1
                if i == hidden_indices[item]:
                    hidden = row['edges']
            assert hidden is not None and valid(hidden, task['observations'])
            assert all(sum(c.values()) == task['n_ground_truths'] for c in counts.values())
            scored[item] = counts, hidden
            cursor = end
    return scored


def program_result(observations, denominator):
    result = solve(NODES, 10, observations)
    graphs = [p['edges'] for p in result['proposals']]
    assert len(set(graphs)) == len(graphs) <= min(SLOTS, denominator)
    assert all(valid(edges, observations) for edges in graphs)
    return {'found': len(graphs), 'rr': len(graphs)/denominator, 'checks': result['checks'],
            'graphs': graphs}


def evaluate_task(task, choices, distributions, hidden, benchmark, seed, folder, session):
    obs, total = task['observations'], task['n_ground_truths']
    before = program_result(obs, total)
    predictions = {}
    for node, counts in distributions.items():
        cases = []
        for outcome, size in sorted(counts.items()):
            simulated = new_observation(node, outcome)
            after = program_result(obs+[simulated], size)
            cases.append({'outcome': outcome, 'size': size, 'found': after['found'],
                          'rr': after['rr'], 'checks': after['checks']})
        predictions[node] = {
            'remaining': sum(c['size']**2 for c in cases)/total**2,
            'rr': sum(c['size']*c['rr'] for c in cases)/total,
            'found': sum(c['size']*c['found'] for c in cases)/total,
            'rr_ceiling': sum(min(SLOTS, c['size']) for c in cases)/total,
            'checks': sum(c['size']*c['checks'] for c in cases)/total,
            'outcomes': cases,
        }
    optimal = min(predictions, key=lambda node: predictions[node]['remaining'])
    metrics = ('remaining', 'rr', 'found', 'rr_ceiling', 'checks')
    uniform = {key: mean(p[key] for p in predictions.values()) for key in metrics}
    policies = {'random': uniform, 'entropy': predictions[choices['entropy']],
                'qwen': predictions[choices['qwen']] if choices['qwen'] else uniform,
                'oracle': predictions[optimal]}

    # Use one uniform hidden graph for all model arms. Identical new observations
    # share their model calls and both arms keep the original eight-slot budget.
    reach = closure(hidden)
    arms, cached = {}, {}
    for policy in MODEL_POLICIES:
        node = choices.get(policy)
        if node in cached:
            arms[policy] = cached[node]
            continue
        updated = list(obs)
        denominator = total
        if node is not None:
            outcome = reach[INDEX[node]]
            updated.append(new_observation(node, outcome))
            denominator = distributions[node][outcome]
        arm_dir = folder/('before' if node is None else f'intervene_{node}')
        arm_dir.mkdir(exist_ok=True)
        result = direct(benchmark, {'observations': updated}, seed, arm_dir, session)
        assert result['found'] <= min(SLOTS, denominator)
        assert all(valid(g, updated) for g in result['graphs'])
        arms[policy] = dict(result, node=node, denominator=denominator, rr=result['found']/denominator,
                            observations=updated)
        cached[node] = arms[policy]
    return {'item': task['id'], 'n_observations': len(obs), 'truth': total, 'seed': seed,
            'choices': dict(choices, oracle=optimal), 'program_before': before, 'program': policies,
            'interventions': predictions, 'hidden': hidden, 'model': arms}


def report(output, records, picks):
    program = {p: {k: mean(r['program'][p][k] for r in records) for k in
                   ('remaining', 'rr', 'found', 'rr_ceiling', 'checks')} for p in POLICIES}
    before_rr = mean(r['program_before']['rr'] for r in records)
    model = {p: {**{k: mean(r['model'][p][k] for r in records) for k in ('rr', 'vr', 'nr')},
                 'found': sum(r['model'][p]['found'] for r in records),
                 'solved': sum(r['model'][p]['found'] > 0 for r in records)} for p in MODEL_POLICIES}
    paired = {}
    for label, differences in {
        'entropy_vs_random_remaining': [r['program']['random']['remaining']-r['program']['entropy']['remaining']
                                       for r in records],
        'entropy_vs_qwen_remaining': [r['program']['qwen']['remaining']-r['program']['entropy']['remaining']
                                     for r in records],
        'model_entropy_vs_random_rr': [r['model']['entropy']['rr']-r['model']['random']['rr']
                                      for r in records],
    }.items():
        paired[label] = [sum(d > 1e-12 for d in differences), sum(abs(d) <= 1e-12 for d in differences),
                         sum(d < -1e-12 for d in differences)]
    requests = len(list(output.glob('*/*/direct_*.json')))
    assert requests == sum(len({arm['node'] for arm in r['model'].values()})*SLOTS for r in records)
    result = {'program': program, 'program_before_rr': before_rr, 'model': model, 'paired': paired,
              'selection_calls': len(picks), 'generation_calls': requests,
              'selection_seconds': sum(p['request_seconds'] for p in picks), 'records': records}
    write_json(output/'results.json', result)
    lines = ['# 主动干预后的覆盖率复核', '',
             '从此前未用于模型实验的六节点题中，按已有 2、3、4 次干预各抽 4 道。每题用同一组 8 张合法图估计未观测节点的预测分布。程序选预测熵最高的节点，Qwen 读取相同分组，随机策略对未观测节点均匀分布。', '',
             '完整答案集仅用于模拟与评分。隐藏图均匀分布。程序对每个节点的所有可能观测结果重新求解，按各结果的出现概率计算期望。最优参考选期望剩余比例最小的节点。', '',
             '| 选点策略 | 期望剩余比例 ↓ | 新观测后期望 RR ↑ | 题均合法图数 | 题均程序检查 |',
             '|---|---:|---:|---:|---:|']
    labels = dict(zip(POLICIES, ('均匀随机', '候选预测熵', 'Qwen 选点', '最优参考')))
    for p in POLICIES:
        d = program[p]
        lines.append(f"| {labels[p]} | {d['remaining']:.2%} | {d['rr']:.2%} | {d['found']:.2f} | {d['checks']:.2f} |")
    lines += ['', f'新增观测前，程序的平均 RR 为 {before_rr:.2%}。新增观测后的 RR 使用更新后的合法图总数。'
              '只有固定观测的结果可以和原始 HypoSpace RR 直接比较。', '',
              '每题另均匀抽取一张隐藏图，并为模型各条件共享该图。Qwen 使用相同官方历史提示，每条件八槽、非思考模式，新增观测的节点是均匀随机抽样或程序熵选择。相同节点复用模型响应。此处仅对每题一次模拟结果计分。', '',
              '| Qwen 生成条件 | 平均 RR | VR | NR | 不同合法图数 | 至少找出一图 |',
              '|---|---:|---:|---:|---:|---:|']
    for p, label in zip(MODEL_POLICIES, ('原观测', '随机追加', '预测熵追加')):
        d = model[p]
        lines.append(f"| {label} | {d['rr']:.2%} | {d['vr']:.2%} | {d['nr']:.2%} | "
                     f"{d['found']} | {d['solved']}/12 |")
    lines += ['', f'模型选点 {len(picks)} 次，生成 {requests} 次。每题一个模型种子和一张模拟隐藏图。'
              '程序期望与模型单次抽样采用各自的统计口径。逐题分组、预算、候选和请求均已保存。']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE/'runs/active_intervention_validation_20261113')
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    path = output/'config.json'
    tasks = read(OBSERVATIONS)
    lookup = {task['id']: task for task in tasks}
    tasks = [lookup[key] for key in read(path)['cases']] if path.exists() else select_tasks(tasks)
    assert len(tasks) == len({t['id'] for t in tasks}) == 12
    session = requests.Session()
    session.trust_env = False
    response = session.get(BASE_URL+'/api/tags', timeout=10)
    response.raise_for_status()
    model = next(row for row in response.json()['models'] if row['name'] == MODEL)
    assert model['digest'] == read(MODEL_REFERENCE)['model']['digest']
    audit = read(RESULTS/'causal_audit.json')
    module = load_upstream('causal', 'run_causal_benchmark.py')
    config = dict(cases=[t['id'] for t in tasks], selection_seed=SELECTION_SEED,
                  simulation_seed=SIMULATION_SEED, seeds=SEEDS, slots=SLOTS, model=model['digest'],
                  observations_sha=digest(OBSERVATIONS), source_audit_sha=audit['source_sha256'],
                  evaluator_sha=digest(module.__file__),
                  hashes={name: digest(HERE/name) for name in ('validate_active_intervention.py',
                          'run_active_intervention_pilot.py', 'run_formal_causal_comparison.py',
                          'solve_partial_graph.py', 'run_qwen_guided_edits.py')})
    if path.exists():
        assert read(path) == config
    else:
        write_json(path, config)
    test_path = output/'observations.json'
    write_json(test_path, {'metadata': {'nodes': NODES, 'max_edges': 10, 'total_observation_sets': len(tasks)},
               'datasets': [{'observation_set_id': t['id'], 'observations': t['observations'],
                             'ground_truth_graphs': []} for t in tasks]})
    benchmark = module.CausalBenchmarkEnhanced(str(test_path))
    print('OUTPUT_DIR='+str(output.resolve()), flush=True)
    rng = random.Random(SIMULATION_SEED)
    hidden_indices, picks = {}, []
    for i, task in enumerate(tasks):
        unknown = sorted(set(NODES)-{o['perturbed_node'] for o in task['observations']})
        graphs = program_result(task['observations'], task['n_ground_truths'])['graphs']
        assert len(graphs) == SLOTS
        counts = predict(graphs, unknown)
        seed = SEEDS[i % len(SEEDS)]
        folder = output/task['id']
        folder.mkdir(exist_ok=True)
        qwen, call = qwen_select(task, counts, seed, folder/'choose_node.json', session)
        picks.append({'item': task['id'], 'seed': seed, 'request_seconds': call['seconds'],
                      'choices': {'entropy': max(unknown, key=lambda node: entropy(counts[node])),
                                  'qwen': qwen, 'random': rng.choice(unknown)}})
        hidden_indices[task['id']] = rng.randrange(task['n_ground_truths'])
    assert len(list(output.glob('*/choose_node.json'))) == len(picks)
    write_json(output/'selection.json', picks)
    exact = exact_distributions(tasks, hidden_indices, audit)
    records = []
    for task, pick in zip(tasks, picks):
        folder = output/task['id']
        record = evaluate_task(task, pick['choices'], *exact[task['id']], benchmark, pick['seed'], folder, session)
        write_json(folder/'result.json', record)
        records.append(record)
        print(f"{len(records)}/12 {task['id']} "+str([record['model'][p]['found'] for p in MODEL_POLICIES]), flush=True)
    report(output, records, picks)


if __name__ == '__main__':
    main()
