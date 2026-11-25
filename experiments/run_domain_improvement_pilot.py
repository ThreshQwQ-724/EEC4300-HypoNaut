"""Paired small-sample improvements over the full Qwen evaluation."""
import argparse
from collections import Counter
from itertools import combinations, product
from pathlib import Path
import random
from statistics import mean

import requests

from common import DATA, HERE, digest, load_upstream
from run_all_domain_evaluation import METRICS, ROOT, SLOTS, atomic, pad, score, usage
from run_formal_agent_debate import REFERENCE
from run_four_groups import BASE_URL, MODEL
from run_qwen_guided_edits import decode, read, request, schema


OUTPUT = HERE/'runs/domain_improvement_pilot_20261114'
SEED = 11142026
DOMAINS = ('boolean', '3d', 'causal')
LABELS = {'boolean': 'Boolean', '3d': '3D', 'causal': '因果图'}


def pick(session, path, seed, prompt, ids):
    call = request(session, path, prompt, seed, schema({'choice': {'type': 'integer', 'enum': ids}}))
    choice = decode(call).get('choice')
    return (choice if type(choice) is int and choice in ids else None), call


def boolean(*, prune=True, ablation=False):
    from run_boolean_feedback_pilot import SOURCE
    from validated_evaluators import corrected_boolean_class
    module = load_upstream('boolean', 'boolean_benchmark.py')
    from boolean_dataset import BooleanDiscoveryGame
    original = module.BooleanBenchmarkRefined(str(SOURCE))
    corrected = corrected_boolean_class(module)(str(SOURCE))
    grammar = sorted(BooleanDiscoveryGame.generate_all_expressions(
        corrected.variables, corrected.operators, corrected.max_depth, corrected.mechanistic_opts),
        key=lambda expr: expr.formula)
    catalog = dict(enumerate(grammar, 1))
    assert len(catalog) == len({e.formula for e in grammar}) == 65

    def run(task, session, folder, seed):
        obs = [module.BooleanObservation(row['inputs'], row['output']) for row in task['observations']]
        eligible = set(catalog)
        answers, calls, trace = [], [], []
        feedback = 'No previous selection.'
        for slot in range(1, SLOTS+1):
            if not eligible:
                break
            options = sorted(eligible)
            prompt = ('Select one formula that matches all observed inputs and outputs. Return its JSON ID. '
                      'The menu may contain incompatible formulas.\nObservations:\n'
                      + '\n'.join(o.to_string() for o in obs) + '\nVerifier feedback: '+feedback
                      + '\nRemaining formulas:\n'
                      + '\n'.join(f'{i}. {catalog[i].formula}' for i in options))
            choice, call = pick(session, folder/f'{slot:02}.json', seed, prompt, options)
            calls.append(call)
            answers.append('Expression: '+catalog[choice].formula if choice is not None else '')
            eliminated = []
            counterexample = None
            if choice is not None:
                eligible.remove(choice)
                for o in obs:
                    actual = catalog[choice].evaluate(o.inputs)
                    if actual != o.output:
                        counterexample = {'inputs': o.inputs, 'expected': o.output, 'actual': actual}
                        eliminated = [i for i in sorted(eligible) if catalog[i].evaluate(o.inputs) != o.output]
                    if prune:
                        eligible.difference_update(eliminated)
                    else:
                        eliminated = []
                        break
                feedback = (f'At {counterexample["inputs"]} the last formula returned '
                            f'{counterexample["actual"]}; expected {counterexample["expected"]}. '
                            'The menu has been updated using this observation.' if counterexample else
                            'The last formula satisfied the observations and has been removed from the menu.')
                if ablation and counterexample:
                    feedback = (f'At {counterexample["inputs"]} the last formula returned '
                                f'{counterexample["actual"]}; expected {counterexample["expected"]}.')
            trace.append({'options': options, 'choice': choice, 'removed': eliminated, 'counterexample': counterexample})
        return pad(answers), usage(calls), trace

    return {t['observation_set_id']: t for t in original.all_observation_sets}, run, {
        'corrected': corrected, 'upstream': original}, lambda t: t['n_observations'], 3, SOURCE


def spatial(*, methods=('without_replacement', 'balanced')):
    from offline_3d import from_heights, to_response
    from validated_evaluators import corrected_3d_class
    source = DATA/'3d/datasets/3d_grid3_h3_tp3.json'
    module = load_upstream('3d', 'run_3d_benchmark.py')
    original = module.Benchmark3D(str(source))
    corrected = corrected_3d_class(module)(str(source))
    size, height = original.metadata['grid_size'], original.metadata['max_height']

    def stratum(task):
        cells = [divmod(i, size) for i, v in enumerate(task['observation']) if v == '1']
        return sum(abs(a-c)+abs(b-d) == 1 for (a,b),(c,d) in combinations(cells, 2))

    def run(task, session, folder, seed):
        obs = task['observation']
        cells = [(i//size+1, i%size+1) for i, v in enumerate(obs) if v == '1']
        catalog = dict(enumerate(product(range(1,height+1), repeat=len(cells)), 1))
        assert len(catalog) == 27
        results = {}
        for method in methods:
            available = set(catalog)
            counts = Counter()
            answers, calls, trace = [], [], []
            selected = []
            for slot in range(1, SLOTS+1):
                options = sorted(available)
                if method in ('balanced', 'maximin'):
                    least = min(counts[sum(catalog[i])] for i in options)
                    options = [i for i in options if counts[sum(catalog[i])] == least]
                if method == 'maximin' and selected:
                    distances = {i: min(sum(abs(a-b) for a,b in zip(catalog[i], old))
                                        for old in selected) for i in options}
                    best = max(distances.values())
                    options = [i for i in options if distances[i] == best]
                prompt = (f'The occupied columns are {cells} in row-column order. Select one height tuple '
                          'from the listed available tuples. Return only its JSON ID.\nCandidates:\n'
                          + '\n'.join(f'{i}. {catalog[i]}' for i in options))
                choice, call = pick(session, folder/f'{method}_{slot:02}.json', seed, prompt, options)
                calls.append(call)
                values = catalog[choice] if choice is not None else None
                answers.append(to_response(from_heights(obs, values), size) if values is not None else '')
                if values is not None:
                    available.remove(choice)
                    counts[sum(values)] += 1
                    selected.append(values)
                trace.append({'choice': choice, 'options': options, 'heights': values})
            results[method] = (answers, usage(calls), trace)
        return results

    return {str(t['observation_id']): t for t in original.all_observation_sets}, run, {
        'corrected': corrected, 'upstream': original}, stratum, 4, source


def causal():
    from run_constraint_qwen_ranking import local_contradiction
    from run_feedback_study import graph_line
    from run_final_qwen_path_repair import run as run_policy
    from run_formal_causal_comparison import DATA as source, LIMIT, NODES, program_anchor
    module = load_upstream('causal', 'run_causal_benchmark.py')
    benchmark = module.CausalBenchmarkEnhanced(str(REFERENCE/'observations.json'))

    def run(task, session, folder, seed):
        observations = task['observations']
        anchor = program_anchor(NODES, LIMIT, observations)
        filtered = []
        positives = {row['perturbed_node'] for row in observations if any(row['effects'].values())}

        def prepare(menu, edits, obs):
            available = []
            rejected = []
            # Retain the existing menu order. Refill from the same unverified frontier.
            for graph in dict.fromkeys([*menu, *edits]):
                if local_contradiction(graph, edits[graph], positives):
                    rejected.append(graph)
                else:
                    available.append(graph)
                if len(available) == len(menu):
                    break
            filtered.append(rejected)
            return available

        result = run_policy(observations, anchor, SLOTS, 'reselect', session, folder, seed, benchmark,
                            prepare_menu=prepare)
        for graph in {g for batch in filtered for g in batch}:
            parsed = benchmark.parse_llm_response(graph_line(graph))
            assert parsed is None or not benchmark.validate_hypothesis(parsed, observations)
        answers = pad([graph_line(anchor)] + [graph_line(t['submitted']) if t['submitted'] is not None else ''
                                             for t in result['trace']])
        cost = {'calls': result['calls'], 'seconds': result['request_seconds'], 'output_tokens': result['tokens'],
                'checks': result['checks'], 'rejected': sum(map(len, filtered)),
                'rejected_unique': len({g for batch in filtered for g in batch})}
        return answers, cost, {'slots': result['trace'], 'screened_out': filtered}

    return {t['id']: t for t in read(source)}, run, {'upstream': benchmark}, lambda t: t['n_observations'], 4, source


def score_causal(scorer, task, answers):
    observations = task['observations']
    keys = set()
    for answer in answers:
        parsed = scorer.parse_llm_response(answer)
        if parsed is not None and scorer.validate_hypothesis(parsed, observations):
            keys.add(tuple(sorted(parsed.edges)))
    instance = {'observation_set_id': task['id'], 'observations': observations,
                'ground_truth_graphs': [{'nodes': scorer.nodes, 'edges': key} for key in sorted(keys)]}
    scores = score(scorer, instance, answers)
    scores['recovery_rate'] = scores['n_recovered_gts']/task['n_ground_truths']
    assert scores['n_recovered_gts'] == sum(bool(a) for a in answers)
    return scores


def execute(domain, output):
    tasks, run, scorers, stratum, sample_size, source = {
        'boolean': boolean, '3d': spatial, 'causal': causal}[domain]()
    baseline_path = ROOT/domain/'results.json'
    baseline = {row['id']: row for row in read(baseline_path)}
    source_config = read(ROOT/domain/'config.json')
    assert set(tasks) == set(baseline) and source_config['source_sha'] == digest(source)
    strata = {s: sorted(k for k,t in tasks.items() if stratum(t) == s) for s in {stratum(t) for t in tasks.values()}}
    rng = random.Random(SEED)
    selected = [case for group in sorted(strata) for case in rng.sample(strata[group], sample_size)]
    output.mkdir(parents=True, exist_ok=True)
    config = {'domain': domain, 'cases': selected, 'sample_seed': SEED, 'slots': SLOTS, 'model': MODEL,
              'digest': source_config['digest'], 'baseline_sha': digest(baseline_path),
              'source_sha': digest(source), 'code_sha': {name: digest(HERE/name) for name in (Path(__file__).name,
                  'run_qwen_guided_edits.py', 'run_final_qwen_path_repair.py', 'run_constraint_qwen_ranking.py',
                  'run_all_domain_evaluation.py', 'validated_evaluators.py')}}
    config_path = output/'config.json'
    if config_path.exists() and read(config_path) != config:
        raise ValueError('Output directory has different settings')
    atomic(config_path, config)
    rows = []
    with requests.Session() as session:
        session.trust_env = False
        tags = session.get(BASE_URL+'/api/tags', timeout=10)
        tags.raise_for_status()
        assert next(m for m in tags.json()['models'] if m['name'] == MODEL)['digest'] == config['digest']
        for i, case in enumerate(selected, 1):
            task = tasks[case]
            folder = output/case
            folder.mkdir(exist_ok=True)
            result = folder/'result.json'
            if result.exists():
                row = read(result)
            else:
                prior = baseline[case]
                result_data = run(task, session, folder, prior['seed'])
                methods = result_data if domain == '3d' else {'improved': result_data}
                runs = {'baseline': prior['runs']['enhanced'], 'reference': prior['runs']['reference']}
                for name, (answers, cost, trace) in methods.items():
                    scores = {key: score_causal(scorer, task, answers) if domain == 'causal' else
                              score(scorer, task, answers) for key, scorer in scorers.items()}
                    runs[name] = {'answers': answers, 'scores': scores, 'cost': cost, 'trace': trace}
                row = {'id': case, 'stratum': stratum(task), 'truth': prior['truth'], 'seed': prior['seed'], 'runs': runs}
                atomic(result, row)
            rows.append(row)
            print(f'{domain} {i}/{len(selected)} {case}', flush=True)
    atomic(output/'results.json', rows)


def total_height_bins(answers, benchmark):
    return len({sum(int(layer.sum()) for layer in parsed.layers) for answer in answers
                if (parsed := benchmark.parse_llm_response(answer)) is not None})


def report(root):
    labels = {'baseline': '全量评测最佳 Qwen', 'reference': '程序参照',
              'without_replacement': '无放回编号', 'balanced': '无放回＋柱高总和分层'}
    improvements = {'boolean': '反例剪枝', 'causal': '局部剪枝＋路径反馈'}
    lines = ['# HypoSpace 三题型小样本改进验证', '',
             '每题 8 槽。复用全量评测的逐题 Qwen 基线与程序参照、题面、种子、评分口径。'
             '选题由固定种子分层抽样，不按基线分数筛选。'
             'Boolean 按观测数每层 3 题；3D 按相邻占据格对数每层 4 题；因果图按干预数每层 4 题。', '',
             'Boolean 在首次发现某条观测反例时，去除所有也违反该条观测的剩余公式。'
             '3D 分别测试无放回编号与额外按柱高总和均衡选择。'
             '因果图只预筛双向环和删光必有正效应节点出边的候选，再从同一邻域补齐菜单，沿用路径反馈重选。'
             '各增强组都由 Qwen 决定具体提交候选。', '',
             '| 题型 | 评分器 | 方法 | RR | VR | NR | 合法命中 | 调用 | 输出 token | 请求秒数 |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    details = {}
    for domain in DOMAINS:
        rows = read(root/domain/'results.json')
        config = read(root/domain/'config.json')
        assert [r['id'] for r in rows] == config['cases']
        primary = 'upstream' if domain == 'causal' else 'corrected'
        stats = {}
        for method in rows[0]['runs']:
            runs = [row['runs'][method] for row in rows]
            cost = {k: sum(r['cost'].get(k,0) for r in runs) for k in (
                'calls', 'output_tokens', 'seconds', 'checks', 'rejected', 'rejected_unique')}
            values = {}
            for scorer in runs[0]['scores']:
                scores = [r['scores'][scorer] for r in runs]
                d = {key: mean(s[key] for s in scores) for key in METRICS[:-1]}
                d['hits'] = sum(s['n_recovered_gts'] for s in scores)
                values[scorer] = d
                label = improvements[domain] if method == 'improved' else labels[method]
                lines.append(f"| {LABELS[domain]} | {scorer} | {label} | {d['recovery_rate']:.4%} | "
                             f"{d['valid_rate']:.2%} | {d['novelty_rate']:.2%} | {d['hits']} | "
                             f"{cost['calls']} | {cost['output_tokens']} | {cost['seconds']:.2f} |")
            deltas = [r['runs'][method]['scores'][primary]['n_recovered_gts']
                      -r['runs']['baseline']['scores'][primary]['n_recovered_gts'] for r in rows]
            stats[method] = {'scores': values, 'cost': cost,
                             'wtl': [sum(d > 0 for d in deltas), sum(d == 0 for d in deltas), sum(d < 0 for d in deltas)]}
        details[domain] = {'n': len(rows), 'cases': config['cases'], 'methods': stats}
        if domain == 'boolean':
            trace = [t for row in rows for t in row['runs']['improved']['trace']]
            details[domain]['pruning'] = {'counterexamples': sum(t['counterexample'] is not None for t in trace),
                                          'eliminated': sum(len(t['removed']) for t in trace)}
        if domain == '3d':
            _, _, scorers, _, _, _ = spatial()
            details[domain]['mean_height_totals'] = {
                method: mean(total_height_bins(r['runs'][method]['answers'], scorers['corrected']) for r in rows)
                for method in stats}
    lines += ['', '主口径为 Boolean 与 3D 的修正版和因果图原版。因果图以原版验证器和已审计的题目答案数计分。'
              '这是同一数据集上的小样本配对试验。逐题结果含候选和原始模型请求，summary.json 含胜平负与分层诊断。', '']
    atomic(root/'summary.json', details)
    (root/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('domain', choices=(*DOMAINS, 'summary'))
    parser.add_argument('--output', default=OUTPUT, type=Path)
    args = parser.parse_args()
    if args.domain == 'summary':
        report(args.output)
    else:
        execute(args.domain, args.output/args.domain)


if __name__ == '__main__':
    main()
