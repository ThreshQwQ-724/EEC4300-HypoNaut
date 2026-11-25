# Boolean representative checks under a fixed budget

The 80 paired cases use exactly the initial menus, group ordering, model replies and program decisions logged in the full Boolean experiment. For each run, we stop at 1, 2, 3 or 4 representative checks and rescore the submissions with both evaluators. A group that passes the check releases up to eight distinct formulas. The program ranks by group size and breaks ties by representative ID. This is a trace replay of the existing inference, with no new model requests.

| Maximum checks per task | Qwen solved cases | Program solved cases | Qwen corrected RR | Program corrected RR | Paired difference | 95% paired bootstrap interval |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 18 | 15 | 13.58% | 6.52% | +7.06 pp | −0.0004 to +14.69 pp |
| 2 | 51 | 47 | 40.88% | 36.12% | +4.76 pp | −7.51 to +17.03 pp |
| 3 | 73 | 79 | 66.27% | 73.77% | −7.50 pp | −15.00 to −1.25 pp |
| 4 | 80 | 80 | 75.02% | 75.02% | 0.00 pp | 0.00 to 0.00 pp |

The paired bootstrap resamples 80 tasks with replacement for 10,000 draws at each budget. These intervals describe uncertainty across the benchmark tasks, with model responses held fixed. At one check both methods recover exactly 120 formulas. The higher macro RR reflects which tasks supplied the recovered formulas. At two checks Qwen recovers 361 formulas versus 357, checking 142 groups versus 145. On the original evaluator, Qwen and program RR are respectively 6.59% and 0.44% at one check, 20.44% and 17.46% at two, 32.79% and 36.52% at three, and 37.35% each at four.

The workflow's main improvement comes from checking a representative once and submitting several legal formulas, together with constraint-based construction and deduplication. The tight-budget replay supports a possible Qwen ranking benefit relative to this size heuristic at small limits. Once complete Boolean signatures have been calculated, an exact program can match the target signature directly. Extending the evaluation to larger or held-out spaces under equal wall-clock or evaluation budgets would test a setting in which exact signatures are more costly.

From the experiments directory run `python run_boolean_tight_checks.py`. The replay scores every budget, task, policy and evaluator separately. The accompanying results.json includes the exact metrics, paired comparisons and the SHA-256 of the frozen input.
