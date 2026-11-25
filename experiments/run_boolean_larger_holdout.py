"""Frozen larger Boolean transfer study, with a bounded shared menu."""
import argparse
from collections import defaultdict
from itertools import product
from pathlib import Path
from random import Random
from statistics import mean
from time import perf_counter

import requests

from common import HERE, digest
from run_all_domain_evaluation import SLOTS, atomic
from run_qwen_guided_edits import BASE_URL, MODEL, decode, read, request, schema

OUT = HERE / 'runs/boolean_larger_holdout_v1'
MODEL_DIGEST = '500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41'
VARIABLES = ('a', 'b', 'c', 'd')
ROWS = tuple(product((0, 1), repeat=len(VARIABLES)))
ALL_BITS = (1 << len(ROWS)) - 1
PANEL_SEED, CASE_SEED = 11357, 719023
MODEL_SEEDS = (14591, 14592, 14593)
PANEL_SIZE, MENU_SIZE, CASES_PER_STRATUM = 256, 12, 12


def build_panel():
    rng = Random(PANEL_SEED)
    variables = [(name, sum(row[j] << i for i, row in enumerate(ROWS)))
                 for j, name in enumerate(VARIABLES)]

    def formula(depth):
        if not depth or rng.random() < 0.23:
            return rng.choice(variables)
        op = rng.choice(('AND', 'OR', 'XOR', 'NOT'))
        lhs, x = formula(depth - 1)
        if op == 'NOT':
            return f'NOT({lhs})', ALL_BITS ^ x
        rhs, y = formula(depth - 1)
        return f'{op}({lhs},{rhs})', {'AND': x & y, 'OR': x | y, 'XOR': x ^ y}[op]

    panel = {}
    while len(panel) < PANEL_SIZE:
        name, truth = formula(3)
        if truth not in (0, ALL_BITS):
            panel.setdefault(truth, name)
    return [{'id': i, 'formula': name, 'truth': truth}
            for i, (truth, name) in enumerate(panel.items(), 1)]


def build_tasks(panel):
    rng = Random(CASE_SEED)
    tasks = []
    for count in (4, 6):
        for index, target in enumerate(rng.sample(panel, CASES_PER_STRATUM), 1):
            positions = sorted(rng.sample(range(len(ROWS)), count))
            mask = sum(1 << i for i in positions)
            groups = defaultdict(list)
            for candidate in panel:
                groups[candidate['truth'] & mask].append(candidate)
            ranked = sorted(groups.items(), key=lambda x: (-len(x[1]), x[1][0]['id']))
            tasks.append({
                'id': f'{count}obs-{index:02d}',
                'positions': positions,
                'outputs': [(target['truth'] >> i) & 1 for i in positions],
                'target_id': target['id'],
                'group_count': len(groups),
                'groups': [{'id': i, 'size': len(members), 'formula': members[0]['formula'],
                            'signature': value, 'members': [c['id'] for c in members]}
                           for i, (value, members) in enumerate(ranked[:MENU_SIZE], 1)],
            })
    return tasks


def prompt_for(task):
    obs = '\n'.join(' '.join(f'{name}={bit}' for name, bit in zip(VARIABLES, ROWS[pos], strict=True))
                    + f' => {target}' for pos, target in zip(task['positions'], task['outputs'], strict=True))
    menu = '\n'.join(f'{g["id"]}: {g["formula"]} (group size {g["size"]})' for g in task['groups'])
    return ("Rank exactly two different representative IDs by consistency with ALL given observations. "
            "Operators AND, OR, XOR are binary, and NOT is unary. Group members share the representative's "
            "outputs on the observed rows. Return JSON with a ranked integer array named ids.\n"
            f"Observations:\n{obs}\nCandidate representatives:\n{menu}")


def verify(task, ranked, budget, expected_signature, good_ids):
    groups = {g['id']: g for g in task['groups']}
    for choice in ranked[:budget]:
        if groups[choice]['signature'] == expected_signature:
            found = groups[choice]['members'][:SLOTS]
            assert set(found).issubset(good_ids) and len(found) == len(set(found))
            return len(found) / len(good_ids)
    return 0.0


def run(output, replay):
    panel = build_panel()
    tasks = build_tasks(panel)
    output.mkdir(parents=True, exist_ok=True)
    plan = {'panel_seed': PANEL_SEED, 'case_seed': CASE_SEED, 'model_seeds': list(MODEL_SEEDS),
            'panel_size': PANEL_SIZE, 'menu_size': MENU_SIZE, 'slots': SLOTS, 'tasks': tasks, 'panel': panel,
            'model_digest': MODEL_DIGEST, 'runner_sha256': digest(__file__)}
    plan_path = output / 'protocol.json'
    if plan_path.exists():
        assert read(plan_path) == plan
    else:
        atomic(plan_path, plan)

    records = []
    with requests.Session() as session:
        session.trust_env = False
        if not replay:
            tags = session.get(BASE_URL + '/api/tags', timeout=15)
            tags.raise_for_status()
            assert any(x['name'] == MODEL and x['digest'] == MODEL_DIGEST for x in tags.json()['models'])
        for task in tasks:
            folder = output / task['id']
            folder.mkdir(exist_ok=True)
            mask = sum(1 << i for i in task['positions'])
            observed = sum(bit << i for i, bit in zip(task['positions'], task['outputs'], strict=True))
            # Independently score membership on the individual rows.
            good = {c['id'] for c in panel if all((c['truth'] >> i) & 1 == y
                    for i, y in zip(task['positions'], task['outputs'], strict=True))}
            assert task['target_id'] in good
            expected = [g['id'] for g in task['groups'] if g['signature'] == observed]
            assert all(g['signature'] & mask == g['signature'] for g in task['groups'])
            valid_ids = [g['id'] for g in task['groups']]
            selection_schema = schema({'ids': {'type': 'array', 'minItems': 2, 'maxItems': 2, 'uniqueItems': True,
                                               'items': {'type': 'integer', 'enum': valid_ids}}})
            model_runs = []
            for seed in MODEL_SEEDS:
                cache = folder / f'{seed}.json'
                if replay and not cache.is_file():
                    raise FileNotFoundError(cache)
                call = request(session, cache, prompt_for(task), seed, selection_schema)
                choices = decode(call).get('ids')
                valid = (isinstance(choices, list) and len(choices) == 2 and all(
                    type(i) is int and i in valid_ids for i in choices) and choices[0] != choices[1])
                ranking = choices if valid else []
                model_runs.append({'seed': seed, 'ranking': ranking, 'valid': valid,
                                   'rr': [verify(task, ranking, k, observed, good) for k in (1, 2)],
                                   'wall_seconds': call['seconds']})
            exact_start = perf_counter()
            exact = [g['id'] for g in task['groups'] if g['signature'] == observed][:1]
            exact_seconds = perf_counter() - exact_start
            assert exact == expected
            records.append({'id': task['id'], 'solutions': len(good), 'menu_has_target': bool(expected),
                            'ceiling': min(SLOTS, len(good)) / len(good), 'qwen': model_runs,
                            'size_rr': [verify(task, valid_ids, k, observed, good) for k in (1, 2)],
                            'exact_menu_rr': verify(task, exact, 1, observed, good),
                            'exact_seconds': exact_seconds})
            print(task['id'], flush=True)
    stats = {'cases': len(records), 'seeds': list(MODEL_SEEDS), 'menu_hits': sum(
        r['menu_has_target'] for r in records), 'median_groups': sorted(
        t['group_count'] for t in tasks)[len(tasks)//2], 'solution_range': [
            min(r['solutions'] for r in records), max(r['solutions'] for r in records)],
        'ceiling': mean(r['ceiling'] for r in records),
        'exact_menu_rr': mean(r['exact_menu_rr'] for r in records),
        'size_rr': [mean(r['size_rr'][i] for r in records) for i in range(2)],
        'qwen_rr_by_seed': [[mean(r['qwen'][j]['rr'][i] for r in records) for i in range(2)]
                            for j in range(len(MODEL_SEEDS))],
        'qwen_found_by_seed': [[sum(r['qwen'][j]['rr'][i] > 0 for r in records) for i in range(2)]
                               for j in range(len(MODEL_SEEDS))],
        'valid_replies': sum(t['valid'] for r in records for t in r['qwen']),
        'response_count': sum(len(r['qwen']) for r in records)}
    atomic(output / 'results.json', {'summary': stats, 'cases': records})
    print(stats)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--replay', action='store_true', help='Require and reuse all saved model replies')
    args = parser.parse_args()
    run(args.output, args.replay)
