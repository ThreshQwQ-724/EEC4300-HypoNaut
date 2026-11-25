"""Roundtrip a small complete subset through the untouched official causal evaluator."""
import json
from common import RESULTS, COMMIT, Replay, load_upstream, save


def main():
    mod=load_upstream('causal','run_causal_benchmark.py')
    b=mod.CausalBenchmarkEnhanced(str(RESULTS/'causal_small_subset.json'))
    rows=[]
    for row in b.all_observation_sets:
        responses=['Graph: '+(', '.join(f'{u}->{v}' for u,v in gt['edges']) or 'No edges')
                   for gt in row['ground_truth_graphs']]
        replay=Replay(responses)
        r=b.evaluate_single_observation_set(replay,row,len(responses),False,max_retries=1)
        metrics={k:r[k] for k in ('valid_rate','novelty_rate','recovery_rate','parse_success_rate')}
        assert all(v==1 for v in metrics.values()),(row['observation_set_id'],metrics)
        assert replay.calls==len(responses)
        rows.append({'id':row['observation_set_id'],'n_gt':len(responses),'metrics':metrics})
    regressions={'cycle':'Graph: A->B, B->A','self_loop':'Graph: A->A',
                 'unknown_node':'Graph: A->Z','bad_format':'This is not a graph.'}
    assert all(b.parse_llm_response(response) is None for response in regressions.values())
    report={'kind':'GT_ORACLE_REPLAY_DIAGNOSTIC_NOT_LLM','upstream_commit':COMMIT,
            'observations_checked':len(rows),'gt_entries_checked':sum(r['n_gt'] for r in rows),
            'all_oracle_metrics_equal_one':True,'invalid_structure_cases_checked':list(regressions),
            'subset_scope':'First 20 file-order instances with 2..32 GTs. Not representative of full distribution.',
            'instances':rows}
    save('causal_subset_audit.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='instances'},indent=2))


if __name__=='__main__':
    main()
