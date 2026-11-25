"""Recompute headline scores from the archived per-task results."""
import hashlib
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / 'experiments' / 'runs'
TARGETED = RUNS / 'targeted_domain_optimization_20261115'
FULL = RUNS / 'comprehensive_evaluation_20261115'
TIGHT = RUNS / 'boolean_tight_checks_20261115'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def summarize(rows, method, scorer, *, filled=False):
    runs = [row['runs'][method] for row in rows]
    scores = [run['filled']['scores'] if filled else run['scores'][scorer] for run in runs]
    return (mean(score['recovery_rate'] for score in scores),
            sum(score['n_recovered_gts'] for score in scores),
            sum(run['cost']['calls'] for run in runs))


def main():
    manifest = read(ROOT / 'MANIFEST.sha256.json')
    for file, sha in manifest.items():
        assert hashlib.sha256((ROOT / file).read_bytes()).hexdigest() == sha, file
    inputs = [
        ('Boolean, original', TARGETED/'boolean', 80, 'current', 'upstream', False, (37.65, 254, 597)),
        ('Boolean, corrected', TARGETED/'boolean', 80, 'group_qwen', 'corrected', False, (75.02, 534, 178)),
        ('3D', TARGETED/'3d', 84, 'distance_qwen', 'upstream', False, (29.63, 672, 84)),
        ('Causal, before fill', FULL/'causal', 510, 'first_qwen', 'upstream', False, (26.98, 3363, 456)),
        ('Causal, after fill', FULL/'causal', 510, 'first_qwen', 'upstream', True, (27.49, 3437, 456)),
    ]
    for label, folder, count, mode, scorer, filled, expected in inputs:
        rows = read(folder / 'results.json')
        assert len(rows) == len({row['id'] for row in rows}) == count
        rr, hits, calls = summarize(rows, mode, scorer, filled=filled)
        assert (round(rr * 100, 2), hits, calls) == expected
        print(f'{label:24} RR {rr:7.2%}  hits {hits:5,d}  calls {calls:3,d}')
    ablation = read(TIGHT / 'results.json')
    assert hashlib.sha256((TARGETED/'boolean'/'results.json').read_bytes()).hexdigest() == ablation['baseline_sha256']
    for cap, row in ablation['budgets'].items():
        scores = []
        for mode in ('group_qwen', 'group_program'):
            rr = mean(case['modes'][mode]['scores']['corrected']['recovery_rate'] for case in row['cases'])
            assert rr == row['summary'][mode]['corrected']['rr']
            scores.append(rr)
        print(f'Boolean cap {cap}: {scores[0]:.2%} Qwen, {scores[1]:.2%} program')
    holdout = read(RUNS / 'boolean_larger_holdout_v1' / 'results.json')
    cases = holdout['cases']
    assert len(cases) == len({case['id'] for case in cases}) == 24
    summary = holdout['summary']
    assert sum(case['menu_has_target'] for case in cases) == summary['menu_hits'] == 15
    assert sum(len(case['qwen']) for case in cases) == summary['response_count'] == 72
    for index, seed in enumerate(summary['seeds']):
        trials = [case['qwen'][index] for case in cases]
        assert all(row['seed'] == seed and row['valid'] for row in trials)
        assert [mean(row['rr'][i] for row in trials) for i in (0,1)] == summary['qwen_rr_by_seed'][index]
    assert mean(case['exact_menu_rr'] for case in cases) == summary['exact_menu_rr']
    qwen = [mean(row[i] for row in summary['qwen_rr_by_seed']) for i in (0,1)]
    print(f'Independent Boolean: Qwen {qwen[0]:.2%} / {qwen[1]:.2%}, exact shortlist {summary["exact_menu_rr"]:.2%}')
    print(f'Checked {len(manifest)} included files.')


if __name__ == '__main__':
    main()
