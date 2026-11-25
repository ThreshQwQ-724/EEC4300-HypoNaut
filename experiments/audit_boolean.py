"""Ground-truth roundtrip and metric audit, with no LLM/API calls."""
import ast
import json
from collections import Counter
from common import DATA, COMMIT, Replay, digest, load_upstream, save
from validated_evaluators import corrected_boolean_class


def main():
    mod=load_upstream('boolean','boolean_benchmark.py')
    path=DATA/'boolean/datasets/boolean_full_d2.json'
    b=mod.BooleanBenchmarkRefined(str(path))
    fixed=corrected_boolean_class(mod)(str(path))
    distinct={}
    rows=[]
    for row in b.all_observation_sets:
        for gt in row['ground_truth_expressions']:
            distinct[gt['mechanistic_key']]=gt
    problems=[]
    signatures=set()
    for stored_key,gt in distinct.items():
        expr=mod.BooleanExpression(gt['formula'],b.variables,b.operators)
        computed_key=b.get_expression_mechanistic_key(gt['formula'])
        stored_table={ast.literal_eval(k):v for k,v in gt['truth_table'].items()}
        signatures.add(tuple(sorted(stored_table.items())))
        record={'formula':gt['formula'],'stored_key':stored_key,'computed_key':repr(computed_key),
                'in_space':b._in_space(gt['formula']),
                'key_matches':computed_key==ast.literal_eval(stored_key),
                'truth_table_matches':expr.truth_table==stored_table,
                'sympy_depth':b._get_ast_depth(expr.sympy_expr)}
        if not all(record[k] for k in ('in_space','key_matches','truth_table_matches')):
            problems.append(record)
    for row in b.all_observation_sets:
        gt=row['ground_truth_expressions']
        obs=[mod.BooleanObservation(o['inputs'],o['output']) for o in row['observations']]
        adapter=Replay(['Expression: '+g['formula'] for g in gt])
        result=b.evaluate_single_observation_set(adapter,row,len(gt),False,max_retries=1)
        fixed_result=fixed.evaluate_single_observation_set(Replay(['Expression: '+g['formula'] for g in gt]),row,len(gt),False,max_retries=1)
        stored_keys=[g['mechanistic_key'] for g in gt]
        expected={key for key,g in distinct.items()
                  if all(g['truth_table'][str(tuple(o['inputs'][v] for v in sorted(b.variables)))] == o['output']
                         for o in row['observations'])}
        rows.append({'id':row['observation_set_id'],'n_observations':row['n_observations'],
                     'n_gt':len(gt),'stored_duplicate_keys':len(stored_keys)-len(set(stored_keys)),
                     'membership_matches_observations_within_supplied_universe':set(stored_keys)==expected,
                     'gt_valid_by_official':sum(b.validate_expression(g['formula'],obs)[0] for g in gt),
                     'oracle_replay_metrics':{k:result[k] for k in ('valid_rate','novelty_rate','recovery_rate')},
                     'corrected_oracle_replay_metrics':{k:fixed_result[k] for k in ('valid_rate','novelty_rate','recovery_rate')},
                     'calls':adapter.calls})
    report={'kind':'GT_ORACLE_REPLAY_DIAGNOSTIC_NOT_LLM','source_sha256':digest(path),
            'upstream_commit':COMMIT,'metadata':b.metadata,'observation_sets':len(rows),
            'gt_entries':sum(r['n_gt'] for r in rows),'distinct_stored_keys':len(distinct),
            'distinct_semantics':len(signatures),'problematic_distinct_expressions':len(problems),
            'regenerated_grammar_size':len(fixed.grammar_keys),
            'regenerated_grammar_matches_supplied_keys':fixed.grammar_keys=={ast.literal_eval(k) for k in distinct},
            'in_space_rejections':sum(not p['in_space'] for p in problems),
            'key_mismatches':sum(not p['key_matches'] for p in problems),
            'truth_table_mismatches':sum(not p['truth_table_matches'] for p in problems),
            'gt_entries_rejected_by_official':sum(r['n_gt']-r['gt_valid_by_official'] for r in rows),
            'instances_with_oracle_rr_below_one':sum(r['oracle_replay_metrics']['recovery_rate']<1 for r in rows),
            'corrected_instances_with_oracle_rr_below_one':sum(r['corrected_oracle_replay_metrics']['recovery_rate']<1 for r in rows),
            'scope':'Full supplied dataset consistency and official roundtrip; independent grammar completeness NOT established.',
            'problems':problems,'instances':rows}
    save('boolean_audit.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('problems','instances')},indent=2))
    print('Problem examples:',json.dumps(problems[:3],ensure_ascii=False,indent=2))
    assert all(all(v==1 for v in r['corrected_oracle_replay_metrics'].values()) for r in rows)
    assert all(r['membership_matches_observations_within_supplied_universe'] and not r['stored_duplicate_keys'] for r in rows)
    # Constraint regressions: depth, unknown variable/operator, and parse failures.
    assert fixed._in_space('NOT (NOT (x OR y))')
    assert not fixed._in_space('NOT (NOT (NOT x))')
    assert not fixed._in_space('z')
    assert not fixed._in_space('NOT ((x ^ y) OR y)')
    assert not fixed._in_space('NOT x AND NOT y AND x AND y')
    assert fixed.validate_expression('???',[mod.BooleanObservation({'x':0,'y':0},0)])[0] is False


if __name__=='__main__':
    main()
