"""Replay the final results and match every accepted causal graph against raw labels."""
import argparse
from pathlib import Path

from common import HERE, digest, load_upstream
from run_all_domain_evaluation import SLOTS, atomic, boolean_setup, score, spatial_setup
from run_formal_agent_debate import REFERENCE
from run_qwen_guided_edits import read
from validate_constructive_hypospace import raw_cases


ROOT = HERE/'runs/final_full_evaluation_20261114'
COUNTS = {'boolean':80, '3d':84, 'causal':510}


def audit_small(domain, rows):
    from validated_evaluators import corrected_3d_class, corrected_boolean_class
    setup = boolean_setup if domain == 'boolean' else spatial_setup
    tasks, _, source, _ = setup()
    if domain == 'boolean':
        mod = load_upstream(domain, 'boolean_benchmark.py')
        scorers = {'corrected':corrected_boolean_class(mod)(str(source)),
                   'upstream':mod.BooleanBenchmarkRefined(str(source))}
    else:
        mod = load_upstream(domain, 'run_3d_benchmark.py')
        scorers = {'corrected':corrected_3d_class(mod)(str(source)),
                   'upstream':mod.Benchmark3D(str(source))}
    assert {row['id'] for row in rows} == set(tasks)
    count = 0
    for row in rows:
        for run in row['runs'].values():
            assert len(run['answers']) == SLOTS
            for name, benchmark in scorers.items():
                assert score(benchmark, tasks[row['id']], run['answers']) == run['scores'][name]
                count += 1
    return {'score_replays': count}


def audit_causal(rows):
    by_id = {r['id']: r for r in rows}
    mod = load_upstream('causal', 'run_causal_benchmark.py')
    benchmark = mod.CausalBenchmarkEnhanced(str(REFERENCE/'observations.json'))
    assert digest(mod.__file__) == read(REFERENCE/'config.json')['evaluator_sha']
    seen, replays, accepted = set(), 0, 0

    def submissions(task, nodes, limit):
        del nodes, limit
        output = {}
        for name, run in by_id[task['observation_set_id']]['runs'].items():
            assert len(run['answers']) == len(run['filled']['answers']) == SLOTS
            assert all(not first or first == last for first,last in zip(
                run['answers'], run['filled']['answers'], strict=True))
            for phase, result in (('before', run), ('after', run['filled'])):
                graphs, slots = [], []
                for slot, answer in enumerate(result['answers'], 1):
                    if not answer:
                        continue
                    parsed = benchmark.parse_llm_response(answer)
                    assert parsed is not None and benchmark.validate_hypothesis(parsed, task['observations'])
                    graphs.append(tuple(sorted(parsed.edges)))
                    slots.append(slot)
                assert len(graphs) == len(set(graphs))
                output[name, phase] = {'graphs': graphs, 'slots': slots}
        return output

    for row in raw_cases(mod, solve_row=submissions):
        if 'raw_sha256' in row:
            assert seen == set(by_id)
            assert row['raw_sha256'] == read(HERE/'results/causal_audit.json')['source_sha256']
            return {'raw_sha256':row['raw_sha256'], 'score_replays':replays, 'matched_graphs':accepted}
        item = by_id[row['id']]
        seen.add(row['id'])
        assert row['unique_truth'] == row['listed_truth'] == item['truth']
        for (name, phase), result in row['methods'].items():
            run = item['runs'][name]
            stored = run['scores']['upstream'] if phase == 'before' else run['filled']['scores']
            assert (result['found'], result['rr'], result['vr'], result['nr']) == (
                stored['n_recovered_gts'], stored['recovery_rate'], stored['valid_rate'], stored['novelty_rate'])
            accepted += result['found']
            replays += 1
        if len(seen) % 100 == 0:
            print(f'Raw answer audit: {len(seen)}/510', flush=True)
    raise RuntimeError('Raw file checksum was not emitted')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('domain', choices=tuple(COUNTS))
    parser.add_argument('--output', type=Path, default=ROOT)
    args = parser.parse_args()
    folder = args.output/args.domain
    source = folder/'results.json'
    rows = read(source)
    assert len(rows) == len({r['id'] for r in rows}) == COUNTS[args.domain]
    cfg = read(folder/'config.json')
    assert cfg['full'] and cfg['runner_sha'] == digest(HERE/'run_three_domain_optimization.py')
    assert all(digest(HERE/name) == sha for name,sha in cfg['dependencies'].items())
    results = audit_causal(rows) if args.domain == 'causal' else audit_small(args.domain, rows)
    atomic(folder/'audit.json', {'cases':len(rows), 'results_sha':digest(source),
                                 'audit_sha':digest(__file__), **results})
    print(args.domain, results, flush=True)


if __name__ == '__main__':
    main()
