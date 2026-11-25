"""Run paired follow-up experiments with Qwen3:8B and an eight-slot output budget."""
import argparse
from collections import defaultdict, deque
from itertools import combinations, permutations, product
from pathlib import Path
from random import Random
from statistics import mean

import requests

from audit_final_full_evaluation import audit_small
from common import HERE, digest, load_upstream
from graph_completion import complete_observation_count
from offline_3d import from_heights, to_response
from run_all_domain_evaluation import ROOT as BASE, SLOTS, atomic, boolean_setup, pad, pack, usage
from run_boolean_qwen_contribution import OUTPUT as BOOLEAN_BASE
from run_domain_improvement_pilot import causal, pick, score_causal, spatial
from run_exploratory_uot import make_edits, ranked_groups
from run_feedback_study import graph_line
from run_formal_causal_comparison import LIMIT, NODES, forbidden_edges, program_anchor
from run_four_groups import BASE_URL, MODEL
from run_independent_rewire import verified
from run_numbered_agent_search import build_menu
from run_qwen_guided_edits import read
from run_spatial_causal_followup import causal_pattern_fill, portfolios, prediction_metrics
from run_three_domain_optimization import height_patterns, rank_batch
from validate_constructive_hypospace import raw_cases

PRIOR = HERE / 'runs/current_best_rerun_20261115'
OUTPUT = HERE / 'runs/targeted_domain_optimization_20261115'
SEED = 20261115
CASES_PER_CAUSAL_STRATUM = 10


def boolean_group(task, catalog, groups, seed, folder, session, use_model):
    expected = tuple(int(obs['output']) for obs in task['observations'])
    remaining = dict(groups)
    calls, trace, answers = [], [], []
    observation_text = '\n'.join(f'{o["inputs"]} -> {o["output"]}' for o in task['observations'])
    for step in range(1, SLOTS + 1):
        if not remaining:
            break
        menu = sorted(remaining, key=lambda key: (-len(remaining[key]), remaining[key][0]))
        ids = {remaining[key][0]: key for key in menu}
        choice = next(iter(ids))
        if use_model:
            descriptions = '\n'.join(
                f'{i}. {catalog[i].formula} ({len(remaining[ids[i]])} formulas have the same outputs)'
                for i in ids
            )
            prompt = (
                'Choose the representative formula that matches every observation. Evaluate its logic. '
                'Each group shares its outputs on these inputs. Return only its JSON choice ID.\n'
                f'Observations:\n{observation_text}\nRepresentative formulas:\n{descriptions}'
            )
            choice, call = pick(session, folder / f'group_{step:02}.json', seed, prompt, list(ids))
            calls.append(call)
        fallback = choice is None
        choice = choice if choice is not None else next(iter(ids))
        key = ids[choice]
        failures = [i for i, (actual, target) in enumerate(zip(key, expected, strict=True)) if actual != target]
        trace.append({'menu': list(ids), 'choice': choice, 'fallback': fallback,
                      'counterexamples': failures, 'group_size': len(remaining[key])})
        if not failures:
            answers = [f'Expression: {catalog[i].formula}' for i in remaining[key][:SLOTS]]
            break
        remaining = {pattern: members for pattern, members in remaining.items()
                     if all(pattern[i] == expected[i] for i in failures)}
    return pad(answers), usage(calls) | {'checks': len(trace)}, trace


def boolean_rows(root, session, old):
    from validated_evaluators import corrected_boolean_class

    tasks, _, source, _ = boolean_setup()
    from boolean_dataset import BooleanDiscoveryGame

    module = load_upstream('boolean', 'boolean_benchmark.py')
    corrected = corrected_boolean_class(module)(str(source))
    scorers = {'corrected': corrected, 'upstream': module.BooleanBenchmarkRefined(str(source))}
    grammar = sorted(BooleanDiscoveryGame.generate_all_expressions(
        corrected.variables, corrected.operators, corrected.max_depth, corrected.mechanistic_opts),
        key=lambda e: e.formula)
    catalog = dict(enumerate(grammar, 1))
    assert len(catalog) == 65
    seeds = read(BOOLEAN_BASE / 'config.json')['seeds']
    for case, task in tasks.items():
        folder = root / case
        folder.mkdir(exist_ok=True)
        groups = defaultdict(list)
        for i, expr in catalog.items():
            groups[tuple(int(expr.evaluate(o['inputs'])) for o in task['observations'])].append(i)
        base = old[case]['runs']['ranked_qwen']
        assert len(base['trace']) <= SLOTS
        verified_answers = [answer for answer, step in zip(base['answers'], base['trace'])
                            if answer and not step['counterexamples']]
        runs = {'current': base, 'precheck_replay': pack(pad(verified_answers), base['cost'] | {
            'checks': sum(step['choice'] is not None for step in base['trace'])}, scorers, task)}
        for mode, use_model in [('group_qwen', True), ('group_program', False)]:
            answers, cost, trace = boolean_group(task, catalog, groups, seeds[case], folder, session, use_model)
            runs[mode] = pack(answers, cost, scorers, task) | {'trace': trace}
        yield {'id': case, 'stratum': task['n_observations'],
               'truth': len(task['ground_truth_expressions']), 'runs': runs}


def distances(plan):
    pairs = [sum(abs(x - y) for x, y in zip(a, b, strict=True)) for a, b in combinations(plan, 2)]
    return {'minimum': min(pairs), 'near_pairs': pairs.count(1), 'total': sum(pairs)}


def spatial_shortlist(geometries, current):
    catalog = list(product(range(1, 4), repeat=3))
    groups = [[h for h in catalog if sum(h) == total] for total in range(3, 10)]
    plans = [tuple(h for group in plan for h in group) for doubled in range(len(groups))
             for plan in product(*(combinations(g, 2 if i == doubled else 1) for i, g in enumerate(groups)))]
    measured = [(plan, distances(plan)) for plan in plans]
    assert len(measured) == 22680 and all(d['minimum'] == 1 for _, d in measured)
    out = {}
    for edges in geometries:
        best_pattern = max(height_patterns(plan, edges) for plan in current)
        best_distance = max(distances(plan)['total'] for plan in current)
        candidates = [(plan, d, height_patterns(plan, edges)) for plan, d in measured
                      if d['total'] >= best_distance]
        candidates = [item for item in candidates if item[2] >= best_pattern]
        out[edges] = sorted(candidates, key=lambda item: (item[1]['near_pairs'], -item[2], item[0]))[:8]
    return out


def spatial_rows(root, session, old):
    tasks, _, scorers, _, _, _ = spatial()
    seeds = {row['id']: row['seed'] for row in read(BASE / '3d' / 'results.json')}

    def adjacency(task):
        cells = [divmod(i, 3) for i, value in enumerate(task['observation']) if value == '1']
        return tuple((a, b) for a, b in combinations(range(3), 2)
                     if sum(abs(x - y) for x, y in zip(cells[a], cells[b], strict=True)) == 1)

    current = portfolios(list(product(range(1, 4), repeat=3)))
    shortlists = spatial_shortlist({adjacency(t) for t in tasks.values()}, current)
    for case, task in tasks.items():
        folder = root / case
        folder.mkdir(exist_ok=True)
        edges = adjacency(task)
        options = shortlists[edges]
        prompt = ('Choose an eight-structure plan with the fewest distance-one pairs; '
                  'break ties by most difference patterns. Return its JSON choice ID.\n'
                  + '\n'.join(f'{i}. {p}: close pairs={d["near_pairs"]}, patterns={n}'
                              for i, (p, d, n) in enumerate(options, 1)))
        choice, call = pick(session, folder / 'plan.json', seeds[case], prompt, list(range(1, len(options) + 1)))
        base = old[case]['runs']['shape_qwen']
        runs = {'current': base | {'distance': distances(current[base['plan'] - 1])}}
        for name, index, calls in [('distance_qwen', choice or 1, [call]), ('distance_program', 1, [])]:
            plan, distance, patterns = options[index - 1]
            answers = [to_response(from_heights(task['observation'], h), 3) for h in plan]
            runs[name] = pack(answers, usage(calls), scorers, task) | {
                'distance': distance, 'patterns': patterns, 'plan': index,
                'fallback': name == 'distance_qwen' and choice is None}
        yield {'id': case, 'runs': runs}


def first_batch(task, benchmark, session, folder, seed):
    """Keep the first Qwen ranking and use program order at each later queue refill."""
    obs = task['observations']
    allowed = set(permutations(NODES, 2)) - forbidden_edges(obs)
    anchor = program_anchor(NODES, LIMIT, obs)
    assert anchor is not None and verified(anchor, benchmark, obs)
    exact = complete_observation_count(NODES, LIMIT, obs)
    emitted, checked = [anchor], {anchor}
    queue, frontier, calls, answers, trace = deque(), {}, [], [graph_line(anchor)], []
    for slot in range(2, SLOTS + 1):
        if len(emitted) == exact:
            break
        latest = emitted[-1]
        if latest not in frontier:
            frontier[latest] = make_edits(latest, allowed, False)
        menu = []
        parsed = None
        if not queue:
            _, entries = ranked_groups(emitted, frontier, checked, slot, False)
            if not entries:
                break
            first, second, _ = build_menu(entries, emitted)
            menu = first + second
            ranked = menu[:4]
            if not calls:
                ranked, call, parsed = rank_batch(session, folder, slot, seed + slot, menu, obs)
                calls.append(call)
            queue.extend(ranked)
        probes, answer = [], ''
        for _ in range(min(2, len(queue))):
            graph = queue.popleft()
            assert graph not in checked
            checked.add(graph)
            accepted = verified(graph, benchmark, obs)
            probes.append({'graph': graph, 'accepted': accepted})
            if accepted:
                emitted.append(graph)
                answer = graph_line(graph)
                break
        answers.append(answer)
        trace.append({'slot': slot, 'menu': menu, 'parsed': parsed, 'probes': probes})
    return {'answers': pad(answers), 'cost': usage(calls) | {'checks': len(checked)}, 'trace': trace}


def causal_rows(root, session, old):
    tasks, _, scorers, _, _, _ = causal()
    benchmark = scorers['upstream']
    seeds = {row['id']: row['seed'] for row in read(BASE / 'causal' / 'results.json')}
    rng = Random(SEED)
    for stratum in range(1, 7):
        ids = sorted(case for case, row in old.items() if row['stratum'] == stratum)
        for case in rng.sample(ids, CASES_PER_CAUSAL_STRATUM):
            prior, task = old[case], tasks[case]
            folder = root / case
            folder.mkdir(exist_ok=True)
            run = first_batch(task, benchmark, session, folder, seeds[case])
            run['scores'] = {'upstream': score_causal(benchmark, task, run['answers'])}
            submitted, cost, _ = causal_pattern_fill(task, run, benchmark, SLOTS)
            unknown = sorted(set(benchmark.nodes) - {o['perturbed_node'] for o in task['observations']})
            run['filled'] = {'answers': submitted, 'cost': cost,
                             'scores': score_causal(benchmark, task, submitted),
                             'diversity': prediction_metrics(submitted, benchmark, unknown)}
            yield {'id': case, 'stratum': stratum, 'truth': prior['truth'], 'runs': {
                'current': prior['runs']['batch_qwen'], 'first_qwen': run,
                'program': prior['runs']['batch_program']}}


def audit_sample(rows):
    by_id = {row['id']: row for row in rows}
    module = load_upstream('causal', 'run_causal_benchmark.py')
    from run_formal_agent_debate import REFERENCE
    verifier = module.CausalBenchmarkEnhanced(str(REFERENCE / 'observations.json'))

    def submissions(task, nodes, limit):
        if task['observation_set_id'] not in by_id:
            return {}
        result = {}
        for name, run in by_id[task['observation_set_id']]['runs'].items():
            assert all(not a or a == b for a, b in zip(run['answers'], run['filled']['answers'], strict=True))
            for phase, trial in [('before', run), ('after', run['filled'])]:
                graphs, slots = [], []
                assert len(trial['answers']) == SLOTS
                for slot, text in enumerate(trial['answers'], 1):
                    if text:
                        graph = verifier.parse_llm_response(text)
                        assert graph is not None and verifier.validate_hypothesis(graph, task['observations'])
                        graphs.append(tuple(sorted(graph.edges)))
                        slots.append(slot)
                assert len(set(graphs)) == len(graphs)
                result[name, phase] = {'graphs': graphs, 'slots': slots}
        return result

    seen, count = set(), 0
    for result in raw_cases(module, solve_row=submissions):
        if 'raw_sha256' in result:
            assert seen == set(by_id)
            assert result['raw_sha256'] == read(HERE / 'results/causal_audit.json')['source_sha256']
            return {'score_replays': count, 'raw_sha256': result['raw_sha256']}
        case = result['id']
        if case not in by_id:
            continue
        seen.add(case)
        assert by_id[case]['truth'] == result['unique_truth'] == result['listed_truth']
        for (name, phase), scores in result['methods'].items():
            trial = by_id[case]['runs'][name]
            expected = trial['scores']['upstream'] if phase == 'before' else trial['filled']['scores']
            keys = ('n_recovered_gts', 'recovery_rate', 'valid_rate', 'novelty_rate')
            assert tuple(scores[k] for k in ('found', 'rr', 'vr', 'nr')) == tuple(expected[k] for k in keys)
            count += 1
    raise RuntimeError('Raw causal answer checksum missing')


def execute(domain, output):
    runners = {'boolean': boolean_rows, '3d': spatial_rows, 'causal': causal_rows}
    root = output / domain
    root.mkdir(parents=True, exist_ok=True)
    baseline = PRIOR / domain / 'results.json'
    old = {row['id']: row for row in read(baseline)}
    model_digest = read(PRIOR / domain / 'config.json')['digest']
    config = {'model': MODEL, 'digest': model_digest, 'slots': SLOTS, 'sample_seed': SEED,
              'baseline_sha': digest(baseline), 'runner_sha': digest(__file__)}
    config_path = root / 'config.json'
    if config_path.exists():
        assert read(config_path) == config
    else:
        atomic(config_path, config)
    with requests.Session() as session:
        session.trust_env = False
        tags = session.get(BASE_URL + '/api/tags', timeout=10)
        tags.raise_for_status()
        assert next(m['digest'] for m in tags.json()['models'] if m['name'] == MODEL) == model_digest
        rows = []
        for index, row in enumerate(runners[domain](root, session, old), 1):
            atomic(root / row['id'] / 'result.json', row)
            rows.append(row)
            if index % 10 == 0:
                print(f'{domain}: {index}', flush=True)
    expected = {'boolean': 80, '3d': 84, 'causal': 60}[domain]
    assert len(rows) == len({row['id'] for row in rows}) == expected
    atomic(root / 'results.json', rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('domain', choices=('boolean', '3d', 'causal'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--audit', action='store_true')
    args = parser.parse_args()
    if args.audit:
        root = args.output / args.domain
        source = root / 'results.json'
        rows = read(source)
        audit = audit_sample(rows) if args.domain == 'causal' else audit_small(args.domain, rows)
        atomic(root / 'audit.json', audit | {'results_sha': digest(source)})
        print(args.domain, audit, flush=True)
    else:
        execute(args.domain, args.output)


if __name__ == '__main__':
    main()
