"""Check all saved results, including causal answer membership in the raw dataset."""
import argparse
from pathlib import Path

from common import HERE, digest, load_upstream
from run_all_domain_evaluation import DOMAINS, ROOT, SLOTS, atomic, boolean_setup, score, spatial_setup
from run_formal_agent_debate import REFERENCE
from run_qwen_guided_edits import read
from validate_constructive_hypospace import raw_cases
import sys


def load_rows(root, domain):
    rows = read(root/domain/'results.json')
    config = read(root/domain/'config.json')
    assert [row['id'] for row in rows] == config['cases']
    assert len({row['id'] for row in rows}) == DOMAINS[domain][1]
    assert config['slots'] == SLOTS
    for row in rows:
        for method in row['runs'].values():
            assert len(method['answers']) == SLOTS
            for values in method['scores'].values():
                assert values['n_recovered_gts'] <= min(row['truth'], SLOTS)
                assert abs(values['recovery_rate']-values['n_recovered_gts']/row['truth']) < 1e-12
    return rows


def causal_audit(root):
    rows = {row['id']: row for row in load_rows(root, 'causal')}
    module = load_upstream('causal', 'run_causal_benchmark.py')
    benchmark = module.CausalBenchmarkEnhanced(str(REFERENCE/'observations.json'))
    compared = 0

    def submissions(task, nodes, limit):
        del nodes, limit
        result = {}
        for name, method in rows[task['observation_set_id']]['runs'].items():
            graphs = set()
            for answer in method['answers']:
                parsed = benchmark.parse_llm_response(answer)
                if parsed is not None and benchmark.validate_hypothesis(parsed, task['observations']):
                    graphs.add(tuple(sorted(parsed.edges)))
            result[name] = {'graphs': sorted(graphs), 'slots': list(range(1, len(graphs)+1))}
        return result

    for audit in raw_cases(module, solve_row=submissions):
        if 'raw_sha256' in audit:
            assert audit['raw_sha256'] == read(HERE/'results/causal_audit.json')['source_sha256']
            return {'cases': len(rows), 'raw_sha256': audit['raw_sha256'], 'method_scores_verified': compared}
        case = rows[audit['id']]
        assert audit['unique_truth'] == audit['listed_truth'] == case['truth']
        for name, result in audit['methods'].items():
            saved = case['runs'][name]['scores']['upstream']
            assert saved['n_recovered_gts'] == result['found'] and saved['recovery_rate'] == result['rr']
            compared += 1
    raise RuntimeError('Raw causal stream ended without a checksum')


def smaller_audit(root, domain):
    load_rows(root, domain)
    setup = boolean_setup if domain == 'boolean' else spatial_setup
    tasks, _, _, _ = setup()
    from validated_evaluators import corrected_3d_class, corrected_boolean_class
    if domain == 'boolean':
        from run_boolean_feedback_pilot import SOURCE
        module = load_upstream(domain, 'boolean_benchmark.py')
        scorers = {'upstream': module.BooleanBenchmarkRefined(str(SOURCE)),
                   'corrected': corrected_boolean_class(module)(str(SOURCE))}
    else:
        module = load_upstream(domain, 'run_3d_benchmark.py')
        source = HERE.parent/'HypoSpace_datasets/3d/datasets/3d_grid3_h3_tp3.json'
        scorers = {'upstream': module.Benchmark3D(str(source)),
                   'corrected': corrected_3d_class(module)(str(source))}
    compared = 0
    for row in load_rows(root, domain):
        for run in row['runs'].values():
            for name, benchmark in scorers.items():
                assert score(benchmark, tasks[row['id']], run['answers']) == run['scores'][name]
                compared += 1
    return {'cases': len(tasks), 'method_scores_verified': compared}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT)
    parser.add_argument('domain', choices=tuple(DOMAINS))
    args = parser.parse_args()
    results = causal_audit(args.output) if args.domain == 'causal' else smaller_audit(args.output, args.domain)
    atomic(args.output/args.domain/'audit.json', {'runner_sha': digest(__file__), **results})
    print(args.domain, results, flush=True)


if __name__ == '__main__':
    main()
