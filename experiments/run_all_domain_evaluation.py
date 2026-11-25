"""Full local HypoSpace comparison at a fixed eight-submission budget."""
import argparse
from pathlib import Path
from statistics import mean

import requests

from common import COMMIT, DATA, HERE, Replay, digest, load_upstream
from run_formal_agent_debate import REFERENCE, request as text_request
from run_four_groups import BASE_URL, MODEL, SEEDS
from run_local_causal import write_json
from run_qwen_guided_edits import decode, read, request, schema


SLOTS = 8
ROOT = HERE/'runs/all_domain_evaluation_20261114'
METRICS = ('valid_rate', 'novelty_rate', 'recovery_rate', 'parse_success_rate', 'n_recovered_gts')
LABELS = {'direct': 'Qwen 官方提示', 'enhanced': 'Qwen 约束＋验证反馈',
          'repair': 'Qwen 定向改边', 'reference': '程序参照'}
DOMAINS = {'boolean': ('Boolean', 80), '3d': ('3D', 84), 'causal': ('因果图', 510)}


def raw_generation(benchmark, observation, folder, seed, options, session, domain):
    answers, calls, history, hashes = [], [], [], set()
    for slot in range(1, SLOTS+1):
        prompt = benchmark.create_prompt(observation, history)
        call = text_request(session, folder/f'direct_{slot:02}.json',
                            [{'role': 'user', 'content': prompt}], options, seed, 256)
        answer = call['response']['message'].get('content', '')
        answers.append(answer)
        calls.append(call)
        parsed = benchmark.parse_llm_response(answer)
        if parsed is not None:
            if domain == '3d':
                parsed = parsed.normalize()
                key = parsed.get_hash()
                if key in hashes:
                    continue
                hashes.add(key)
            history.append(parsed)
    return answers, usage(calls)


def usage(calls):
    return {'calls': len(calls), 'input_tokens': sum(c['response'].get('prompt_eval_count', 0) for c in calls),
            'output_tokens': sum(c['response'].get('eval_count', 0) for c in calls),
            'seconds': sum(c['seconds'] for c in calls)}


def score(benchmark, task, answers):
    replay = Replay(answers)
    result = benchmark.evaluate_single_observation_set(replay, task, SLOTS, False, max_retries=1)
    assert replay.calls == SLOTS
    return {key: result[key] for key in METRICS}


def pack(answers, cost, scorers, task):
    assert len(answers) == SLOTS
    return {'answers': answers, 'cost': cost, 'scores': {name: score(b, task, answers) for name,b in scorers.items()}}


def pad(answers):
    return answers+['']*(SLOTS-len(answers))


def boolean_setup():
    from run_boolean_constrained_pilot import trial
    from run_boolean_feedback_pilot import SOURCE, grammar_reference
    from validated_evaluators import corrected_boolean_class
    module = load_upstream('boolean', 'boolean_benchmark.py')
    from boolean_dataset import BooleanDiscoveryGame
    original = module.BooleanBenchmarkRefined(str(SOURCE))
    corrected = corrected_boolean_class(module)(str(SOURCE))
    grammar = sorted(BooleanDiscoveryGame.generate_all_expressions(
        corrected.variables, corrected.operators, corrected.max_depth, corrected.mechanistic_opts), key=lambda e: e.formula)
    formulas = [e.formula for e in grammar]
    assert len(set(formulas)) == 65 and all(corrected._in_space(f) for f in formulas)

    def run(task, folder, seed, options, session):
        obs = [module.BooleanObservation(o['inputs'], o['output']) for o in task['observations']]
        scorers = {'corrected': corrected, 'upstream': original}
        direct, cost = raw_generation(original, obs, folder, seed, options, session, 'boolean')
        enhanced, _, calls = trial(corrected, module, task, formulas, 'grammar_feedback', seed, folder, session)
        reference = pad(['Expression: '+e.formula for e in grammar if all(e.evaluate(o.inputs) == o.output for o in obs)][:SLOTS])
        assert score(corrected, task, reference)['n_recovered_gts'] == min(SLOTS, len(task['ground_truth_expressions']))
        return {'direct': pack(direct, cost, scorers, task),
                'enhanced': pack(enhanced, usage(calls), scorers, task),
                'reference': pack(reference, usage([]), scorers, task)}

    tasks = {t['observation_set_id']: t for t in original.all_observation_sets}
    return tasks, run, SOURCE, module.__file__


def spatial_setup():
    from offline_3d import enumerate_physics, from_heights, to_response
    from validated_evaluators import corrected_3d_class
    source = DATA/'3d/datasets/3d_grid3_h3_tp3.json'
    module = load_upstream('3d', 'run_3d_benchmark.py')
    original = module.Benchmark3D(str(source))
    corrected = corrected_3d_class(module)(str(source))
    size, height = original.metadata['grid_size'], original.metadata['max_height']

    def run(task, folder, seed, options, session):
        obs = task['observation']
        # The safe parser supports history even if the upstream normalizer crashes on a zero voxel grid.
        direct, cost = raw_generation(corrected, obs, folder, seed, options, session, '3d')
        cells = [f'({i//size+1},{i%size+1})' for i,v in enumerate(obs) if v == '1']
        used, answers, calls = set(), [], []
        for slot in range(1, SLOTS+1):
            prompt = (f'The occupied columns in this top view are {", ".join(cells)}, in that order. '
                      f'Assign each of these columns an integer height from 1 to {height}. Other columns are empty. '
                      'Every column is filled from ground level. Return one JSON heights array. '
                      'Choose a different tuple each time. The verifier accepted the following previous tuples: '
                      + (', '.join(map(str, sorted(used))) or 'none') + '.')
            call = request(session, folder/f'heights_{slot:02}.json', prompt, seed,
                           schema({'heights': {'type': 'array', 'minItems': len(cells), 'maxItems': len(cells),
                                               'items': {'type': 'integer', 'minimum': 1, 'maximum': height}}}))
            values = decode(call).get('heights')
            valid = (isinstance(values, list) and len(values) == len(cells)
                     and all(type(v) is int and 1 <= v <= height for v in values))
            key = tuple(values) if valid else None
            answer = to_response(from_heights(obs, key), size) if valid else ''
            if valid:
                used.add(key)
            answers.append(answer)
            calls.append(call)
        reference = [to_response(g, size) for g in enumerate_physics(obs, height)[:SLOTS]]
        scorers = {'corrected': corrected, 'upstream': original}
        return {'direct': pack(direct, cost, scorers, task), 'enhanced': pack(answers, usage(calls), scorers, task),
                'reference': pack(reference, usage([]), scorers, task)}

    tasks = {str(t['observation_id']): t for t in original.all_observation_sets}
    return tasks, run, source, module.__file__


def causal_setup():
    from run_feedback_study import graph_line
    from run_final_qwen_path_repair import run as run_policy
    from run_formal_causal_comparison import DATA as source, LIMIT, NODES, program_anchor
    from solve_partial_graph import solve
    module = load_upstream('causal', 'run_causal_benchmark.py')
    benchmark = module.CausalBenchmarkEnhanced(str(REFERENCE/'observations.json'))

    def run(task, folder, seed, options, session):
        obs, truth = task['observations'], task['n_ground_truths']
        anchor = program_anchor(NODES, LIMIT, obs)
        direct, cost = raw_generation(benchmark, obs, folder, seed, options, session, 'causal')
        answers = {'direct': (direct, cost)}
        for name, method in (('enhanced', 'reselect'), ('repair', 'repair')):
            policy = run_policy(obs, anchor, SLOTS, method, session, folder, seed, benchmark)
            values = [graph_line(anchor)] + [graph_line(s['submitted']) if s['submitted'] is not None else ''
                                              for s in policy['trace']]
            answers[name] = (pad(values), {'calls': policy['calls'], 'output_tokens': policy['tokens'],
                                          'seconds': policy['request_seconds'], 'checks': policy['checks'],
                                          'rescues': policy['rescues'], 'repair_attempts': policy['repair_attempts']})
        reference = solve(NODES, LIMIT, obs, slots=SLOTS, constructive=True)
        values = ['']*SLOTS
        for slot, proposal in zip(reference['proposal_slots'], reference['proposals'], strict=True):
            values[slot-1] = graph_line(proposal['edges'])
        answers['reference'] = (values, usage([]) | {'checks': reference['checks']})
        # The raw file is separately scanned for actual answer membership after inference.
        accepted = set()
        for values, _ in answers.values():
            for value in values:
                g = benchmark.parse_llm_response(value)
                if g is not None and benchmark.validate_hypothesis(g, obs):
                    accepted.add(tuple(sorted(g.edges)))
        instance = {'observation_set_id': task['id'], 'observations': obs,
                    'ground_truth_graphs': [{'nodes': NODES, 'edges': g} for g in sorted(accepted)]}
        result = {name: pack(values, cost, {'upstream': benchmark}, instance) for name, (values,cost) in answers.items()}
        for method in result.values():
            metrics = method['scores']['upstream']
            metrics['recovery_rate'] = metrics['n_recovered_gts']/truth
        return result

    return {t['id']: t for t in read(source)}, run, source, module.__file__


def atomic(path, value):
    temporary = path.with_suffix('.tmp')
    write_json(temporary, value)
    temporary.replace(path)


def execute(domain, output, limit):
    tasks, run, source, scorer = {'boolean': boolean_setup, '3d': spatial_setup, 'causal': causal_setup}[domain]()
    assert len(tasks) == DOMAINS[domain][1]
    output.mkdir(parents=True, exist_ok=True)
    reference = read(REFERENCE/'config.json')
    cases = list(tasks)[:limit]
    metadata = {'domain': domain, 'cases': cases, 'slots': SLOTS, 'model': MODEL, 'digest': reference['model'],
                'source_sha': digest(source), 'upstream': COMMIT, 'scorer_sha': digest(scorer),
                'code_sha': {name: digest(HERE/name) for name in (Path(__file__).name, 'validated_evaluators.py',
                    'run_qwen_guided_edits.py', 'run_final_qwen_path_repair.py', 'run_boolean_constrained_pilot.py')},
                'options': reference['options'], 'direct_output_limit': 256, 'seeds': SEEDS}
    config = output/'config.json'
    if config.exists() and read(config) != metadata:
        raise ValueError(f'Output directory has a different configuration: {output}')
    atomic(config, metadata)
    rows = []
    with requests.Session() as session:
        session.trust_env = False
        tags = session.get(BASE_URL+'/api/tags', timeout=10)
        tags.raise_for_status()
        assert next(m for m in tags.json()['models'] if m['name'] == MODEL)['digest'] == metadata['digest']
        for i, case in enumerate(cases):
            task = tasks[case]
            folder = output/case
            folder.mkdir(exist_ok=True)
            result = folder/'result.json'
            if result.exists():
                row = read(result)
            else:
                seed = SEEDS[i % len(SEEDS)]
                truth = (task['n_ground_truths'] if domain == 'causal' else
                         len(task['ground_truth_expressions' if domain == 'boolean' else 'ground_truth_structures']))
                row = {'id': case, 'seed': seed, 'truth': truth, 'stratum': task.get('n_observations', 1),
                       'runs': run(task, folder, seed, reference['options'], session)}
                atomic(result, row)
            assert row['id'] == case
            rows.append(row)
            if (i+1) % 5 == 0 or i+1 == len(cases):
                print(f'{domain}: {i+1}/{len(cases)}', flush=True)
    atomic(output/'results.json', rows)


def summarize(root):
    summaries = {}
    lines = ['# HypoSpace 三题型全量评测', '',
             '本地 Qwen3:8B，关闭思考模式，每题固定 8 个提交槽。指标按题宏平均。'
             '各题型的增强版共享验证反馈这一思路，具体表示和规则按任务适配。'
             '调用与请求耗时按单个方法独立运行的口径统计，共享缓存可能减少本次实际请求。', '',
             'Boolean 使用公开文法编号和反例反馈；3D 使用受限柱高表示和已接受组合历史；'
             '因果图比较路径反馈重选和定向改边，首次失败可多用一次请求与验证。'
             '3D 的直接组沿用官方提示，由修正版解析器维护历史。'
             'Boolean 与 3D 对同一批答案分别用原版与修正版评分。', '',
             '| 题型 | 题数 | 评分器 | 方法 | RR | VR | NR | 命中 | Qwen 调用 | 输出 token | 请求秒数 |',
             '|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for domain, (label, count) in DOMAINS.items():
        rows = read(root/domain/'results.json')
        config = read(root/domain/'config.json')
        assert len(rows) == count and [r['id'] for r in rows] == config['cases']
        methods = rows[0]['runs']
        summary = {'tasks': count, 'ceiling': mean(min(SLOTS, r['truth'])/r['truth'] for r in rows), 'methods': {}}
        for method in methods:
            runs = [r['runs'][method] for r in rows]
            cost = {k: sum(r['cost'].get(k, 0) for r in runs) for k in
                    ('calls', 'output_tokens', 'seconds', 'checks', 'rescues', 'repair_attempts')}
            scores = {}
            for scorer in runs[0]['scores']:
                stats = {k: mean(r['scores'][scorer][k] for r in runs) for k in METRICS[:-1]}
                stats['hits'] = sum(r['scores'][scorer]['n_recovered_gts'] for r in runs)
                scores[scorer] = stats
                lines.append(f"| {label} | {count} | {scorer} | {LABELS[method]} | {stats['recovery_rate']:.4%} | "
                             f"{stats['valid_rate']:.2%} | {stats['novelty_rate']:.2%} | {stats['hits']:,} | "
                             f"{cost['calls']:,} | {cost['output_tokens']:,} | {cost['seconds']:.2f} |")
            summary['methods'][method] = {'scores': scores, 'cost': cost}
        scorer = 'upstream' if domain == 'causal' else 'corrected'
        summary['paired'] = {}
        for method in methods:
            if method in ('direct', 'reference'):
                continue
            delta = [r['runs'][method]['scores'][scorer]['n_recovered_gts']
                     - r['runs']['direct']['scores'][scorer]['n_recovered_gts'] for r in rows]
            summary['paired'][method] = [sum(d > 0 for d in delta), sum(d == 0 for d in delta), sum(d < 0 for d in delta)]
        summary['strata'] = {n: {'tasks': len(sub), **{
            m: {k: mean(r['runs'][m]['scores'][scorer][k] for r in sub) for k in METRICS[:-1]} for m in methods}}
            for n in sorted({r['stratum'] for r in rows}) if (sub := [r for r in rows if r['stratum'] == n])}
        summaries[domain] = summary
    lines += ['', '## 按题型的 8 槽 RR 上限与配对比较', '']
    for domain, (label, _) in DOMAINS.items():
        d = summaries[domain]
        lines.append(f"- {label} 的 8 槽平均 RR 上限为 {d['ceiling']:.4%}。")
        for method, wtl in d['paired'].items():
            lines.append(f"  {LABELS[method]}对官方提示的逐题命中数胜／平／负为 {'／'.join(map(str, wtl))}。")
    lines += ['', '分观测数量的指标保存在 summary.json，逐题答案与分数保存在各题型目录。', '']
    atomic(root/'summary.json', summaries)
    (root/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('domain', choices=(*DOMAINS, 'summary'))
    parser.add_argument('--output', type=Path, default=ROOT)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.domain == 'summary':
        summarize(args.output)
    else:
        execute(args.domain, args.output/args.domain, args.limit)


if __name__ == '__main__':
    main()
