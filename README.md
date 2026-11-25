# HypoNaut · Structured hypothesis search on HypoSpace

EEC 4300 Machine Learning Final Project. The project uses local Qwen3:8B to choose or rank structured candidates in Boolean, 3D and causal tasks. Deterministic validation checks observations before final submission. All benchmark comparisons here use eight submission slots per problem.

- [PowerPoint](presentation/EEC4300_HypoNaut_1123_GitHub.pptx) and [speaker script](presentation/SPEAKER_SCRIPT.md)
- [674-task evaluation](experiments/runs/comprehensive_evaluation_20261115/REPORT.md)
- [Independent larger Boolean test](experiments/runs/boolean_larger_holdout_v1/REPORT.md)
- [Boolean equal-budget ablation](experiments/runs/boolean_tight_checks_20261115/REPORT.md)
- [Methods and earlier experiments](experiments/METHODS_OVERVIEW.md)
- [Project handoff](PROJECT_HANDOFF.md)

## Record the group video

The 1123-based slide deck contains 20 pages. Speaker 1 presents slides 1–7, Speaker 2 presents slides 8–12, and Speaker 3 presents slides 13–18 and 20. Slide 19 is an optional answer about the Boolean scorer. Each main section targets 3 minutes 20 seconds. The PowerPoint presenter notes and the Markdown speaker script use the same wording and page numbering.

## Check the reported results

From the archive root, run `python review_results.py`. This checks the included result files and recomputes the presentation's core RR, recovery counts and request totals using Python's standard library. It does not require Ollama or the benchmark data.

## Reproduce the scored experiments

Use Python 3.11 and a separate virtual environment. Run the commands from the repository root unless the instructions say otherwise.

```bash
python -m venv .venv
# Activate .venv for your platform, then:
python -m pip install -r experiments/requirements-offline.txt
git clone https://github.com/CTT-Pavilion/_HypoSpace.git upstream/_HypoSpace-c69e9318577b34b5b896996571aefd4ba6053f58
git -C upstream/_HypoSpace-c69e9318577b34b5b896996571aefd4ba6053f58 checkout c69e9318577b34b5b896996571aefd4ba6053f58
```

Use the original course-supplied benchmark files. If you received the separate HypoSpace_Dataset_Supplement.zip from the team, extract it so HypoSpace_datasets/ sits beside experiments/. The data supplement is approximately 238 MB compressed and 9.79 GB unpacked. The package includes a SHA-256 manifest and a Python verifier. Once the files are in place, run `python verify_hypospace_data.py`.

Expected paths

```text
HypoSpace_datasets/boolean/datasets/boolean_full_d2.json
HypoSpace_datasets/3d/datasets/3d_grid3_h3_tp3.json
HypoSpace_datasets/causal/datasets/node06/n6_all_observations.json
```

Run the integrity check before replay or use a separate copy for replay. The independent Boolean replay rewrites its result JSON with fresh local timing fields even when the scores match.

From `experiments/` run:

```bash
python run_boolean_tight_checks.py
python run_boolean_larger_holdout.py --replay
python run_targeted_domain_optimization.py boolean --audit
python run_targeted_domain_optimization.py 3d --audit
python run_causal_first_rank_replay.py
```

The larger Boolean transfer test replays 72 saved responses and uses its own synthetic data, which the script regenerates from the frozen seeds. Causal replay checks the saved Ollama first request against the reconstructed request, reuses its reply, and streams the original multi-gigabyte causal file to verify label membership. The SHA-256 of the pinned causal data is recorded in the audit. The original and corrected Boolean evaluators are reported separately. New local inference is optional: install Ollama, pull `qwen3:8b`, start the local service and run `python run_targeted_domain_optimization.py DOMAIN --output runs/new_run` with DOMAIN set to boolean or 3d for the full task suite, or causal for the 60-case live pilot. The 510-case causal validation command above reuses saved first replies. The runner checks the model digest. A different Ollama model build can give different results and should be tracked as a new experiment.

## Files for sharing

The ZIP includes team experiment code, selected result files and model-response traces used by the final analysis. It leaves out the original course data, upstream source snapshot, model weights and private environments. Older report links may reference full local logs outside this release. The original benchmark source and data remain subject to their respective terms. The team should choose a license for its own code before public distribution.

References: [Chen et al., HypoSpace (ICML 2026)](https://proceedings.mlr.press/v306/chen26bz.html), [pinned benchmark implementation](https://github.com/CTT-Pavilion/_HypoSpace/tree/c69e9318577b34b5b896996571aefd4ba6053f58), and [Universe of Thoughts](https://arxiv.org/abs/2511.20471).

## Version 3

The submission copy of the 1123 presentation corrects the model endpoint, distinguishes the candidate menu from the admissible answer set, and aligns slides and notes with the experimental reports. Table values come from the recorded experiments. The archive preserves the original experiment outputs and adds the historical inputs referenced by the handoff. This update concerns the presentation and reproducing the existing scores.
