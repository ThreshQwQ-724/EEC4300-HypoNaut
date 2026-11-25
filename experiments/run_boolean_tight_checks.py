"""Replay Boolean group selection at equal limits on representative checks."""
from collections import Counter
from random import Random
from statistics import mean

from common import HERE, digest, load_upstream
from run_all_domain_evaluation import SLOTS, atomic, boolean_setup, score
from run_qwen_guided_edits import read
from validated_evaluators import corrected_boolean_class

RUN = HERE / 'runs/targeted_domain_optimization_20261115/boolean'
OUTPUT = HERE / 'runs/boolean_tight_checks_20261115'
MODES = ('group_qwen', 'group_program')


def paired_interval(deltas):
    rng = Random(20261115)
    samples = sorted(mean(rng.choices(deltas, k=len(deltas))) for _ in range(10000))
    return [samples[250], samples[9749]]


def main():
    source = RUN / 'results.json'
    rows = read(source)
    benchmark_tasks, _, dataset, _ = boolean_setup()
    module = load_upstream('boolean', 'boolean_benchmark.py')
    scorers = {'corrected': corrected_boolean_class(module)(str(dataset)),
               'upstream': module.BooleanBenchmarkRefined(str(dataset))}
    assert len(rows) == 80 and {r['id'] for r in rows} == set(benchmark_tasks)
    result = {}
    for budget in (1, 2, 3, 4):
        per_case = []
        for row in rows:
            item = {'id': row['id'], 'observations': row['stratum'], 'modes': {}}
            for mode in MODES:
                run = row['runs'][mode]
                trace = run['trace']
                assert len(trace) == run['cost']['checks']
                assert all(t['counterexamples'] for t in trace[:-1]) and not trace[-1]['counterexamples']
                found = len(trace) <= budget
                answers = run['answers'] if found else [''] * SLOTS
                scores = {name: score(scorer, benchmark_tasks[row['id']], answers)
                          for name, scorer in scorers.items()}
                if found:
                    assert scores == run['scores']
                item['modes'][mode] = {'found': found, 'checks': min(budget, len(trace)), 'scores': scores}
            per_case.append(item)
        summary = {}
        for mode in MODES:
            trials = [r['modes'][mode] for r in per_case]
            summary[mode] = {'solved': sum(t['found'] for t in trials), 'checks': sum(t['checks'] for t in trials)}
            for key in scorers:
                values = [t['scores'][key] for t in trials]
                summary[mode][key] = {'rr': mean(v['recovery_rate'] for v in values),
                                      'hits': sum(v['n_recovered_gts'] for v in values)}
        deltas = [r['modes'][MODES[0]]['scores']['corrected']['recovery_rate'] -
                  r['modes'][MODES[1]]['scores']['corrected']['recovery_rate'] for r in per_case]
        directions = Counter((d > 0) - (d < 0) for d in deltas)
        result[budget] = {'summary': summary, 'corrected_rr_delta': mean(deltas),
                          'paired_ci': paired_interval(deltas),
                          'wtl': [directions[k] for k in (1, 0, -1)], 'cases': per_case}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    atomic(OUTPUT / 'results.json', {'baseline_sha256': digest(source), 'budgets': result})
    for budget, entry in result.items():
        print(budget, {k: v for k, v in entry.items() if k != 'cases'})


if __name__ == '__main__':
    main()
