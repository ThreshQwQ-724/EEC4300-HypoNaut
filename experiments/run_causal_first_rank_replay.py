"""Replay the first Qwen response per causal case and audit against raw labels."""
from collections import Counter
from statistics import mean

from common import HERE, digest
from run_all_domain_evaluation import ROOT as BASE, SLOTS, atomic
from run_domain_improvement_pilot import causal, score_causal
from run_qwen_guided_edits import read
from run_spatial_causal_followup import causal_pattern_fill, prediction_metrics
from run_targeted_domain_optimization import PRIOR, audit_sample, first_batch

OUTPUT = HERE / 'runs/comprehensive_evaluation_20261115' / 'causal'


def main():
    root = PRIOR / 'causal'
    old = read(root / 'results.json')
    seeds = {row['id']: row['seed'] for row in read(BASE / 'causal' / 'results.json')}
    tasks, _, scorers, _, _, _ = causal()
    benchmark = scorers['upstream']
    assert len(old) == len({row['id'] for row in old}) == len(tasks) == 510
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows, request_sha = [], {}
    for index, prior in enumerate(old, 1):
        case = prior['id']
        task = tasks[case]
        folder = root / case
        # A missing or changed cache causes a failure. Replay never sends an Ollama request.
        run = first_batch(task, benchmark, None, folder, seeds[case])
        if run['cost']['calls']:
            assert run['cost']['calls'] == 1
            request_sha[case] = digest(folder / 'batch_02.json')
        run['scores'] = {'upstream': score_causal(benchmark, task, run['answers'])}
        answers, cost, _ = causal_pattern_fill(task, run, benchmark, SLOTS)
        unknown = sorted(set(benchmark.nodes) - {o['perturbed_node'] for o in task['observations']})
        run['filled'] = {'answers': answers, 'cost': cost,
                         'scores': score_causal(benchmark, task, answers),
                         'diversity': prediction_metrics(answers, benchmark, unknown)}
        rows.append({'id': case, 'stratum': prior['stratum'], 'truth': prior['truth'], 'runs': {
            'current': prior['runs']['batch_qwen'], 'first_qwen': run, 'program': prior['runs']['batch_program']}})
        if index % 100 == 0:
            print(f'Causal replay {index}/510', flush=True)
    atomic(OUTPUT / 'config.json', {
        'baseline_sha': digest(root / 'results.json'), 'baseline_config_sha': digest(root / 'config.json'),
        'runner_sha': digest(__file__), 'reused_requests_sha': request_sha})
    atomic(OUTPUT / 'results.json', rows)
    summaries = {}
    for method in rows[0]['runs']:
        runs = [row['runs'][method] for row in rows]
        phases = {'before': [r['scores']['upstream'] for r in runs], 'after': [r['filled']['scores'] for r in runs]}
        data = {name: {key: mean(s[key] for s in scores)
                       for key in ('recovery_rate', 'valid_rate', 'novelty_rate')}
                for name, scores in phases.items()}
        for phase, scores in phases.items():
            data[phase]['hits'] = sum(s['n_recovered_gts'] for s in scores)
            data[phase]['at_cap'] = sum(s['n_recovered_gts'] == min(SLOTS, row['truth'])
                                        for row, s in zip(rows, scores, strict=True))
        data['cost'] = {key: sum(r['cost'][key] for r in runs) for key in ('calls', 'checks', 'input_tokens')}
        data['cost']['extra_checks'] = sum(r['filled']['cost']['extra_checks'] for r in runs)
        data['patterns'] = mean(r['filled']['diversity']['patterns']
                                for row, r in zip(rows, runs, strict=True) if row['stratum'] < 6)
        diff = Counter((d > 0) - (d < 0) for row, score in zip(rows, phases['before'], strict=True)
                       if (d := score['n_recovered_gts'] -
                           row['runs']['current']['scores']['upstream']['n_recovered_gts']) is not None)
        data['paired_wtl'] = [diff[k] for k in (1, 0, -1)]
        summaries[method] = data
    atomic(OUTPUT / 'summary.json', summaries)
    print('Scoring replay against original causal answers', flush=True)
    audit = audit_sample(rows)
    atomic(OUTPUT / 'audit.json', audit | {'results_sha': digest(OUTPUT / 'results.json')})
    print(audit, flush=True)


if __name__ == '__main__':
    main()
