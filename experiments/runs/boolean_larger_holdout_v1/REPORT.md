# Independent Boolean stress test

We froze this synthetic transfer protocol before the first model call. It uses 24 previously unused four-variable tasks and 256 depth-at-most-three formulas with distinct full truth tables. The candidates use AND, OR, XOR and NOT. Each task shows 4 or 6 observed input rows. Recovery is measured against compatible formulas in this finite panel, with up to 8 submissions. Reported RR is macro-averaged over all 24 tasks. The official 674-task results preserve their original task and scoring definitions.

The 24 targets and their observed rows come from deterministic random seeds. Every method receives the same shortlist of up to 12 groups ordered by group size. The Qwen prompt contains representative formulas and group sizes. The lookup program can compare their precomputed signatures with the observed outputs. All methods share the limits on representative verification. This measures verification count and does not equate preprocessing work, lookup time and model inference time.

| Method | RR at 1 check | RR at 2 checks |
|---|---:|---:|
| Qwen3:8B, mean over 3 seeds | 2.62% | 5.88% |
| Group size heuristic | 2.12% | 7.24% |
| Exact signature lookup on the shortlist | 34.63% | 34.63% |
| Full-pool reference ceiling | 70.99% | 70.99% |

Qwen seed results at 1 check are 3.32% / 1.23% / 3.32%. At 2 checks they are 5.59% / 6.44% / 5.59%. Qwen ranks two representatives per request. All 72 responses satisfy the schema. The compatible group is present on the shortlist for 15 of 24 tasks. The gap between exact shortlist lookup and the full-pool reference measures the shortlist restriction under this protocol. The groups contain between 3 and 32 compatible formulas per task in the full pool.

The earlier evidence for an advantage over a group size heuristic at tight budgets does not persist at two checks in this larger setting. An exact lookup remains the strongest shared-shortlist control. The pilot covers Boolean transfer only.

Run `python run_boolean_larger_holdout.py --replay` from `experiments/` to reproduce scores from the saved calls. Omit `--replay` and use a new `--output` folder for new inference. The JSON protocol contains the frozen task definitions, seeds and panel, and the saved requests contain the model prompts. An independent evaluator parsed every formula and checked all 4,096 truth-table entries, all ranking replies and both verification budgets.
