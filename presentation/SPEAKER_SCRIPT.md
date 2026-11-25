# HypoNaut speaker script

Main recording 10:00. Follow the actual PowerPoint page numbers. Slide 19 is optional backup.

| Speaker | Pages | Time |
|---|---:|---:|
| 1 | 1–7 | 3:20 |
| 2 | 8–12 | 3:20 |
| 3 | 13–18 and 20 | 3:20 |

## Speaker 1

### Slide 1 · Title (25 sec)

Hello, we are Group 14. Our project, HypoNaut, uses a local language model to guide structured hypothesis search. We tested it on 674 HypoSpace problems. The goal is to recover several different explanations that all fit the same observations.

### Slide 2 · Benchmark introduction and experimental setup (5 sec)

I will start with the benchmark, scoring, and our experimental setup.

### Slide 3 · One observation set, many explanations (35 sec)

HypoSpace has 80 Boolean tasks, 84 three-dimensional tasks, and 510 causal-graph tasks. For Boolean tasks we look for formulas that match observed input-output pairs. For three-dimensional tasks we generate gravity-supported voxel structures that match the top view. For causal graphs we look for directed acyclic graphs consistent with intervention effects. In all three tasks, the same evidence can support multiple distinct solutions.

### Slide 4 · Evaluation metrics (40 sec)

Recovery Rate is the number of distinct reference answers found divided by the size of the reference answer set. Validity Rate is the share of eight slots used for valid hypotheses. Novelty Rate measures distinct outputs under the evaluator rules. If a case has M compatible answers, an eight-slot submission recovers at most the smaller of eight and M, divided by M. A case with 27 compatible answers therefore has a ceiling of about 29.63 percent.

### Slide 5 · Model choice (40 sec)

Our experiments use Qwen three, an eight-billion-parameter model, running locally through Ollama. The project code calls the Ollama chat API on the same machine. We use a fixed model build and record the requests and responses. This lets us compare different ways of organizing the same model's search. It also lets a reviewer re-score our saved outputs without rerunning inference. The baseline asks Qwen to propose answers directly.

### Slide 6 · Methodology (5 sec)

Next comes the structured search pipeline.

### Slide 7 · Candidate construction, selection and verification (50 sec)

First, we build a candidate menu from the public grammar, geometry rules, or graph constraints. The Boolean and causal menus can contain candidates that fail when checked against the observations. Second, we ask Qwen to choose an ID or rank a short list. Third, we verify the selected candidates, remove duplicates, and fill up to eight submission slots. The full reference set is reserved for scoring recovery. This distinction matters because the clock and verification budget limit which candidates we actually explore. I will hand over to Speaker Two for the task-specific experiments.

## Speaker 2

### Slide 8 · Experimental results (5 sec)

I'll walk through the Boolean, 3D, and causal-graph results.

### Slide 9 · Boolean formulas grouped by observed outputs (43 sec)

We group Boolean formulas that behave identically on the observed input rows. The evaluator still treats distinct legal formulas as separate answers. If a representative disagrees with an observation, we can rule out its group. If a group matches, we can submit several distinct formulas from it. Across 80 cases, this raised corrected recovery from 66.07 to 75.02 percent and reduced calls from 597 to 178. The corrected result recovers all 534 answers reachable with eight slots.

### Slide 10 · Boolean performance under tight verification budgets (42 sec)

This replay gives the model and program the same limit on representative checks. The program orders groups by size. Qwen has a higher mean recovery at one and two checks, although the paired intervals both include zero. The size heuristic is ahead at three checks, and both policies reach the corrected ceiling at four. The results show that the contribution of model ranking depends on the check budget and the control.

### Slide 11 · 3D plans maintain coverage and spread (50 sec)

In 3D, a structure is represented by the integer heights of three columns. Our refined candidate menu contains plans with eight structures per plan. Each plan covers all seven total-height levels. Qwen chooses a plan once for each case. All 84 cases return eight valid, distinct structures, so validity and novelty reach 100 percent and recovery reaches 29.63 percent. Compared with the earlier one-choice menu, the refined shortlist lowers the number of pairs one voxel apart from 3.10 to 2.00 per case.

### Slide 12 · Causal graph ranking and program fill (60 sec)

We start from a valid base graph, construct one-edge edits, and let Qwen rank the first queue. Program rules handle later queues and fill remaining slots. In the full replay of 510 cases, we reuse the stored first Qwen replies. Compared with using Qwen on every refill, pre-fill recovery rises from 26.29 to 26.98 percent and requests drop from 934 to 456. Program-only ranking reaches 27.09 percent before fill. After the program fill, all three tested strategies reach the same 27.49 percent average recovery and reach each case's eight-slot ceiling. I'll hand over to Speaker Three for the benchmark summary and the independent test.

## Speaker 3

### Slide 13 · Full benchmark results (45 sec)

The table reports the original scorer on all 674 tasks. Direct Qwen gives recovery of 0.16 percent for Boolean, 3.75 percent for 3D, and 6.23 percent for causal graphs. The project methods reach 37.65, 29.63, and 27.49 percent respectively. The original Boolean scorer gives the short-menu strategy 37.65 percent and the grouped strategy 37.35 percent. Under the grammar-corrected scorer, the grouped strategy reaches 75.02 percent. We report the two Boolean scoring conventions separately.

### Slide 14 · Independent test with a larger Boolean pool (55 sec)

We froze a separate synthetic test of 24 four-variable tasks with 256 distinct formulas before running Qwen. All three seeds use the same shortlists. With two representative checks, Qwen averages 5.88 percent recovery, the size heuristic reaches 7.24 percent, and exact lookup reaches 34.63 percent using precomputed signatures. The correct group was present on the shortlist for 15 cases. Qwen itself solved four, three, and four cases across the seeds at this budget. This separates candidate availability from the ability to rank the correct group.

### Slide 15 · Additional explorations and further discussions (5 sec)

We can now summarize what the ablations tell us.

### Slide 16 · What the ablations support (37 sec)

Grouping proposals and verifying their outputs recovers more distinct hypotheses within the submission budget. On the original Boolean tasks, Qwen's average ranking is better than the size heuristic at one and two checks, with uncertainty around both differences. At two checks on the new tasks, both program controls outperform Qwen. The full system improves on direct prompting across the benchmark. The matched program controls show why we should report system gains and model-ranking gains separately.

### Slide 17 · Limitations and future directions (35 sec)

The candidate generators use rules specific to each task, and the exact signatures in the independent test are cheap to compute. We would next test problems where candidate generation or signature computation costs more. Comparing the strongest applicable program with Qwen under the same end-to-end time budget would make the practical trade-off clearer. In this experiment we report verification counts, model calls, and recovery separately.

### Slide 18 · References and discussion (18 sec)

Our benchmark is HypoSpace by Chen and colleagues. The additional reference is Universe of Thoughts by Suzuki and Banaei-Kashani. The repository documents the pinned evaluator, saved responses, and commands for reproducing the reported scores.

### Slide 19 · Backup: interpreting the scorer discrepancy (optional backup)

If needed, explain that x AND NOR of x and y has depth two under the public grammar. The expanded syntax tree counts NOR as NOT followed by OR, giving depth three. The corrected evaluator counts NOR as a primitive operator and keeps the same depth limit. This accounts for the different Boolean scorer behavior.

### Slide 20 · Closing (5 sec)

Thank you for watching.
