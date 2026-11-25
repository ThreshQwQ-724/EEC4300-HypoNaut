"""Offline 3D controls, independent physics audit, and original evaluator parity.

No LLM, API, learned weights, or ground-truth-guided proposal generation.
"""
import itertools
import json
import random
import statistics
from collections import defaultdict
from common import DATA, COMMIT, RESULTS, Replay, digest, load_upstream, save
from validated_evaluators import corrected_3d_class


def normalized(layers):
    layers = tuple(layers)
    while layers and set(layers[-1]) <= {'0'}:
        layers = layers[:-1]
    return layers


def valid(layers, observation, grid_size, max_height):
    layers = normalized(layers)
    if not layers or not 1 <= len(layers) <= max_height:
        return False
    if any(len(s) != grid_size ** 2 or not set(s) <= {'0', '1'} for s in layers):
        return False
    if layers[0] != observation:
        return False
    return all(all(c != '1' or b == '1' for c,b in zip(cur, below))
               for below,cur in zip(layers,layers[1:]))


def from_heights(observation, heights):
    cells = [i for i,c in enumerate(observation) if c == '1']
    levels = []
    for z in range(1, max(heights)+1):
        row = ['0'] * len(observation)
        for cell,height in zip(cells, heights):
            row[cell] = str(int(height >= z))
        levels.append(''.join(row))
    return tuple(levels)


def enumerate_physics(observation, max_height):
    return [from_heights(observation, heights)
            for heights in itertools.product(range(1,max_height+1), repeat=observation.count('1'))]


def to_response(layers, grid_size):
    lines = ['Structure:']
    for i,layer in enumerate(layers,1):
        lines.append(f'Layer {i}:')
        lines.extend(' '.join(layer[j:j+grid_size]) for j in range(0,len(layer),grid_size))
    return '\n'.join(lines)


def score(proposals, gt, observation, grid_size, max_height):
    gt = {normalized(x) for x in gt}
    unique, recovered, n_valid, curve = set(), set(), 0, []
    for i,p in enumerate(proposals,1):
        key = normalized(p)
        unique.add(key)
        if valid(p,observation,grid_size,max_height):
            n_valid += 1
            if key in gt:
                recovered.add(key)
        curve.append({'queries': i, 'recovered': len(recovered), 'rr': len(recovered)/len(gt)})
    n = len(proposals)
    return {'n_queries':n, 'n_valid':n_valid, 'n_unique':len(unique), 'n_recovered_gts':len(recovered),
            'valid_rate':n_valid/n, 'novelty_rate':len(unique)/n,
            'recovery_rate':len(recovered)/len(gt), 'coverage_curve':curve}


def main():
    mod = load_upstream('3d','run_3d_benchmark.py')
    path = DATA/'3d/datasets/3d_grid3_h3_tp3.json'
    b = mod.Benchmark3D(str(path))
    corrected = corrected_3d_class(mod)(str(path))
    grid, height = b.metadata['grid_size'], b.metadata['max_height']
    errors, stored_hash_mismatches, total = [], [], 0
    for row in b.all_observation_sets:
        gt = [tuple(g['layers']) for g in row['ground_truth_structures']]
        exact = set(enumerate_physics(row['observation'],height))
        if set(gt) != exact or len(gt) != len(exact) or row['n_compatible_structures'] != len(exact):
            errors.append({'id':row['observation_id'],'error':'incomplete_or_duplicate_gt'})
        for g in row['ground_truth_structures']:
            total += 1
            response = to_response(g['layers'], grid)
            parsed = b.parse_llm_response(response)
            if not (valid(g['layers'],row['observation'],grid,height)
                    and parsed and b.validate_structure_matches_observations(parsed,row['observation'])):
                errors.append({'id':row['observation_id'],'error':'invalid_gt_or_roundtrip'})
            if parsed.get_hash() != g['hash']:
                stored_hash_mismatches.append({'id':row['observation_id'],'stored':g['hash'],'recomputed':parsed.get_hash()})
    assert not errors, errors[:3]
    assert len({r['observation'] for r in b.all_observation_sets}) == len(b.all_observation_sets)
    assert {r['observation'] for r in b.all_observation_sets} == {
        ''.join('1' if j in c else '0' for j in range(grid*grid))
        for c in itertools.combinations(range(grid*grid),b.metadata['max_blocks'])}

    # Behavioral counterexamples: upstream checks gravity/projection but omits max height.
    row = b.all_observation_sets[0]
    obs = row['observation']
    examples = {'valid':(obs,), 'too_tall':(obs,)*(height+1),
                'unsupported':('0'*len(obs),obs), 'wrong_projection':('0'*len(obs),)}
    edge_checks = []
    for label,layers in examples.items():
        parsed = b.parse_llm_response(to_response(layers,grid))
        upstream_valid = bool(parsed and b.validate_structure_matches_observations(parsed,obs))
        strict_valid = valid(layers,obs,grid,height)
        edge_checks.append({'case':label,'official_valid':upstream_valid,'strict_valid':strict_valid})
    assert edge_checks[0]['strict_valid']
    assert not any(x['strict_valid'] for x in edge_checks[1:])
    assert edge_checks[1]['official_valid'], 'Upstream changed: revisit documented max-height issue'
    # Verify duplicate and invalid handling through the real parser and evaluator.
    gt = [tuple(g['layers']) for g in row['ground_truth_structures']]
    zero_bug = None
    try:
        b.evaluate_single_observation_set(Replay([to_response(examples['wrong_projection'],grid)]),row,1,False,max_retries=1)
    except ValueError as e:
        zero_bug = str(e)
    assert zero_bug is not None, 'Upstream changed: revisit documented all-zero bug'
    for check in edge_checks:
        parsed = corrected.parse_llm_response(to_response(examples[check['case']],grid))
        check['corrected_valid'] = bool(parsed and corrected.validate_structure_matches_observations(parsed,obs))
        assert check['corrected_valid'] == check['strict_valid']
    fixture = Replay([to_response(gt[0],grid), to_response(gt[0],grid),
                      to_response(examples['wrong_projection'],grid), 'not a structure'])
    fixture_result = corrected.evaluate_single_observation_set(fixture,row,4,False,max_retries=1)
    assert fixture_result['n_valid'] == 2
    assert fixture_result['n_unique'] == 2
    assert fixture_result['n_recovered_gts'] == 1
    assert fixture.calls == 4

    subset = random.Random(33550336).sample(b.all_observation_sets,5)
    seeds = [33550336,33550337,33550338]
    runs=[]
    for seed in seeds:
        for row in subset:
            obs = row['observation']
            n = row['n_compatible_structures']
            for method in ('repeat_ground_layer','uniform_heights_iid','enumerate_heights'):
                # Only observation and public max-height constraint enter the generator.
                rng = random.Random(f'{seed}:{row["observation_id"]}:{method}')
                if method == 'repeat_ground_layer':
                    proposals = [(obs,)]*n
                elif method == 'uniform_heights_iid':
                    proposals = [from_heights(obs,[rng.randint(1,height) for _ in range(obs.count('1'))]) for _ in range(n)]
                else:
                    proposals = enumerate_physics(obs,height)[:n]
                responses = [to_response(p,grid) for p in proposals]
                adapter = Replay(responses)
                official = b.evaluate_single_observation_set(adapter,row,n,False,max_retries=1)
                fixed = corrected.evaluate_single_observation_set(Replay(responses),row,n,False,max_retries=1)
                measured = score(proposals,[g['layers'] for g in row['ground_truth_structures']],obs,grid,height)
                for key in ('valid_rate','novelty_rate','recovery_rate','n_valid','n_unique','n_recovered_gts'):
                    assert official[key] == measured[key], (method,key,official[key],measured[key])
                    assert fixed[key] == measured[key], (method,key,fixed[key],measured[key])
                assert adapter.calls == n
                runs.append({'method':method,'seed':seed,'observation_id':row['observation_id'],
                             **measured,'raw_responses':responses,'official_metrics':{
                                 k:official[k] for k in ('valid_rate','novelty_rate','recovery_rate','parse_success_rate')}})
    aggregate = {}
    for method in ('repeat_ground_layer','uniform_heights_iid','enumerate_heights'):
        rows = [r for r in runs if r['method']==method]
        aggregate[method] = {metric:{'mean':statistics.mean(r[metric] for r in rows),
                                    'std_across_instance_seed_pairs':statistics.pstdev(r[metric] for r in rows)}
                             for metric in ('valid_rate','novelty_rate','recovery_rate')}
    report={'kind':'OFFLINE_NON_LLM_CONTROLS','api_calls':0,'cost_usd':0,
            'source_sha256':digest(path),'upstream_commit':COMMIT,
            'n_instances':len(subset),'seeds':seeds,'queries_per_instance':27,
            'total_proposals':sum(r['n_queries'] for r in runs),
            'instance_ids':[r['observation_id'] for r in subset],
            'uniform_rr_expectation':1-(1-1/27)**27,
            'aggregate':aggregate,'runs':runs}
    save('offline_3d_baseline.json',report)
    save('3d_audit.json',{'source_sha256':digest(path),'metadata':b.metadata,
                        'observations_checked':len(b.all_observation_sets),'gt_entries_checked':total,
                        'completeness':'All 84 projections and all 27 height assignments per projection independently enumerated.',
                        'errors':errors,'stored_hash_mismatch_count':len(stored_hash_mismatches),
                        'stored_hash_mismatch_examples':stored_hash_mismatches[:3],
                        'validator_edge_checks':edge_checks,'metric_fixture_passed':True,
                        'upstream_zero_structure_crash':zero_bug,
                        'corrected_metric_fixture':{k:fixture_result[k] for k in ('valid_rate','novelty_rate','recovery_rate')},
                        'official_corrected_independent_parity':True})
    (RESULTS/'3d_prompt_example.txt').write_text(b.create_prompt(subset[0]['observation']),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='runs'},indent=2))


if __name__=='__main__':
    main()
