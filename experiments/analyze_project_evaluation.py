"""Audit the frozen run and prepare its course-review report and figure."""
import argparse
import csv
from decimal import Decimal, ROUND_HALF_UP
import json
import os
from pathlib import Path
from statistics import mean

from common import HERE, digest
from run_local_causal import write_json

NAMES = {'b_baseline': 'B 基线', 'diagnostics': '仅诊断反馈', 'slot_8': '诊断＋末次思考'}
METHODS = tuple(NAMES)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def pct(rate):
    return f"{(Decimal(str(rate))*100).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)}%"


def audit(output, config, summary, results):
    assert len(results) == 54
    assert len(list(output.glob('*/*/*/call_*.json'))) == summary['actual_requests']
    with (output/'all_calls.csv').open(encoding='utf-8-sig') as stream:
        assert len(list(csv.DictReader(stream))) == 432
    for name, sha in config['source_hashes'].items():
        assert digest(HERE/name) == sha
    for size, sha in config['test_shas'].items():
        assert digest(output/f'{size}_test.json') == sha
    first_matches = 0
    for size, item, seed, gt in config['cases']:
        diagnostic = results[size, item, 'diagnostics']
        first_matches += (results[size, item, 'b_baseline']['rows'][0]['answer'] == diagnostic['rows'][0]['answer'])
        for method in METHODS:
            run = results[size, item, method]
            assert run['seed'] == seed and run['gt'] == gt
            assert len(run['rows']) == 8 and not run['official']['errors']
            assert run['triggered'] == (method == 'slot_8' and not any(row['valid'] for row in diagnostic['rows'][:7]))
            for row in run['rows']:
                slot = row['query']
                think = run['triggered'] and slot == 8
                shared = method == 'slot_8' and not think
                assert row['think'] == think and row['shared'] == shared
                file = f'call_{slot:02d}.json'
                source = 'diagnostics' if shared else method
                call = read(output/size/item/source/file)
                assert call['request']['think'] == think
                assert call['response']['message']['content'] == row['answer']
                assert call['response']['eval_count'] == row['output_tokens']
                assert call['response']['prompt_eval_count'] == row['prompt_tokens']
                options = {**config['ordinary_options'], 'seed': seed}
                if think:
                    options.update(config['thinking_overrides'])
                    control = read(output/size/item/'diagnostics'/file)
                    assert call['request']['messages'] == control['request']['messages']
                assert call['request']['options'] == options
    return first_matches


def compare(results, cases, left, right, metric):
    deltas = [results[size, item, left]['official'][metric] - results[size, item, right]['official'][metric]
              for size, item, _, _ in cases]
    return {'delta': mean(deltas), 'wins': sum(d > 0 for d in deltas), 'ties': sum(d == 0 for d in deltas),
            'losses': sum(d < 0 for d in deltas)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    output = args.run_dir
    summary, config = read(output/'summary.json'), read(output/'config.json')
    reference = read(output/'full_observation_reference.json')
    metrics = summary['metrics']
    cases = config['cases']
    results = {(r['size'], r['item'], r['method']): r for r in summary['results']}
    first_matches = audit(output, config, summary, results)
    subsets = {size: [c for c in cases if c[0] == size] for size in ('n4', 'n6')}
    subsets['all'] = cases
    stats = {}
    for size, subset in subsets.items():
        refs = [r for r in reference if size == 'all' or size == r['size']]
        assert len(refs) == len(subset) and all(r['exact_match_with_reference_set'] for r in refs)
        stats[size] = {
            'reference': {k: mean(r[k] for r in refs) for k in ('valid_rate', 'novelty_rate', 'recovery_rate')},
            'versus_baseline': compare(results, subset, 'slot_8', 'b_baseline', 'recovery_rate'),
            'versus_diagnostics': compare(results, subset, 'slot_8', 'diagnostics', 'recovery_rate'),
        }
    write_json(output/'verification_and_comparisons.json',
               {'first_answers_equal': first_matches, 'rows_verified': 432, 'sections': stats})
    all_base, all_diag, all_full = [metrics['all'][method] for method in METHODS]
    delta = stats['all']['versus_baseline']
    diag_delta = stats['all']['versus_diagnostics']
    report = [
        '# EEC 4300 Final Project 模拟评审：条件式思考补救', '',
        f"冻结策略在 18 道未参与推理调参的题上取得 **RR {pct(all_full['recovery_rate'])}**，"
        f"较 B 基线 {pct(all_base['recovery_rate'])} 改变 {delta['delta']*100:+.2f} 个百分点。"
        f"相对仅诊断反馈的配对差为 {diag_delta['delta']*100:+.2f} 个百分点。", '',
        '## 课程要求和实验设计', '',
        '课程文件要求开发能提升 HypoSpace 表现的 ML 方法，按创新性、重要性、完成质量评判。课程要求交 10 分钟小组视频和公开 GitHub 仓库，全员展示。课程文件未列出各维度的分值比例。本报告按三个维度提供评审证据。', '',
        '测试集为 12 道四节点题和 6 道六节点题，均为每个节点有一次干预的完整观测。四节点题从以往推理实验未用过的题中按合法图数 2、4、8 各抽四题。六节点题来自组员完整数据的六观测区段，按合法图数 1、2–8、9–219 各抽两题。抽样种子为 981126，每题使用一个配对种子。', '',
        '三个 LLM 策略均用本地 Qwen3 8B Q4_K_M，每题 8 个候选。B 基线使用官方历史提示，第二组追加观测、重复、具体路径与环路反馈。完整策略共享第二组的前 7 答，前 7 答都没有合法图时，把第 8 答改为 think=true、num_ctx=16384、num_predict=8192。触发只依赖观测和候选。普通参数为 temperature=0.60、top_p=0.95、top_k=20、min_p=0、repeat_penalty=1、num_ctx=4096、num_predict=2048。', '',
        'VR 为合法回答比例，NR 为不同可解析图比例，RR 为不同合法图占该题全部合法图的比例。先逐题计算，再取宏平均。全体均值按 12 道四节点题、6 道六节点题计算。比较固定候选预算，token 与耗时分别报告。', '',
        '## 官方指标', '',
        '| 题集 | 策略 | VR | NR | RR | 找到合法图的题数 | 思考触发 | 输出 token | 请求耗时（秒） |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for size, label in [('n4','四节点'), ('n6','六节点'), ('all','全体')]:
        for method, row in metrics[size].items():
            report.append(f"| {label} | {NAMES[method]} | {pct(row['valid_rate'])} | {pct(row['novelty_rate'])} | "
                          f"{pct(row['recovery_rate'])} | {row['solved']}/{row['tasks']} | {row['triggered']} | "
                          f"{row['output_tokens']:,} | {row['seconds']:.1f} |")
        r = stats[size]['reference']
        report.append(f"| {label} | 观测直接求解参考线 | {pct(r['valid_rate'])} | {pct(r['novelty_rate'])} | "
                      f"{pct(r['recovery_rate'])} | {len(subsets[size])}/{len(subsets[size])} | 0 | 0 | — |")
    report += ['', '![覆盖率图](coverage.png)', '',
               '完整策略对 B 基线的逐题 RR 胜、平、负为 '
               f"{delta['wins']}、{delta['ties']}、{delta['losses']}。"
               f"对仅诊断反馈为 {diag_delta['wins']}、{diag_delta['ties']}、{diag_delta['losses']}。"
               f"共触发 {all_full['triggered']} 次思考，其中 {all_full['thinking_valid']} 次输出满足观测的图，"
               f"{all_full['thinking_truncated']} 次达到输出上限。", '',
               '| 题目 | 节点数 | 合法图数 | B 基线 RR | 仅诊断 RR | 完整 RR | 触发 | 最后一次实际输出 token |',
               '|---|---:|---:|---:|---:|---:|---|---:|']
    for size, item, seed, gt in cases:
        runs = [results[size, item, m] for m in METHODS]
        report.append(f"| {item} | {size[1:]} | {gt} | " +
                      ' | '.join(f"{r['official']['recovery_rate']:.2%}" for r in runs) +
                      f" | {'是' if runs[-1]['triggered'] else '否'} | {runs[-1]['rows'][-1]['output_tokens']:,} |")
    report += ['', '## 模拟评审意见', '',
               f"- **创新性。** 当前贡献是把图可达性诊断与按失败触发的推理预算结合。固定 8 个候选的消融中，末次触发相对仅诊断改变 {diag_delta['delta']*100:+.2f} 个百分点 RR。完整观测可直接构建可达关系，用传递约简确定必需边，并枚举满足边数限制的可选边。该观测求解参考在 18 题上恢复的完整集合均与标准答案一致，因此当前证据支持对 LLM 推理流程的改进，项目的方法价值还需要在部分观测或其他 HypoSpace 任务中验证。",
               f"- **重要性。** 完整策略找到合法图的题数为 {all_full['solved']}/18，B 基线为 {all_base['solved']}/18。六节点 RR 从 {metrics['n6']['b_baseline']['recovery_rate']:.2%} 变为 {metrics['n6']['slot_8']['recovery_rate']:.2%}。一题有 120 个合法图、另一题有 63 个，8 次调用的 RR 上限分别是 6.67% 和 12.70%。评估需同时看覆盖率、解题数和成本。完整策略消耗 {all_full['output_tokens']:,} 个输出 token，仅诊断为 {all_diag['output_tokens']:,}。",
               f"- **完成质量。** 已冻结样本、模型、参数和候选预算，保留原始记录和代码哈希。54 轮共 432 个评分槽位，实际推理 {summary['actual_requests']} 次，其余槽位共享对照记录。{first_matches}/18 道题的 B 基线与诊断首答相同。官方评分与独立可达性评分一致。当前仅覆盖两种规模的因果图完整观测，每题一个种子；课程要求的视频和公开仓库属于后续交付。", '',
               '## 复现', '',
               '运行脚本为 run_project_evaluation.py。使用 --verify-only 可核验选题，使用 --output 指向本目录可重放已生成请求并续跑。运行 evaluate_full_observation_reference.py 可复现观测求解参考，运行 analyze_project_evaluation.py 可重建本报告。config.json、summary.json、all_calls.csv 与逐题目录保存配置、统计、逐槽结果及实际请求。']
    (output/'REPORT.md').write_text('\n'.join(report)+'\n', encoding='utf-8')
    os.environ.setdefault('MPLCONFIGDIR', str(HERE/'.mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9.3, 4.4), layout='constrained')
    names = ['Qwen B baseline', 'Diagnostic feedback', 'Feedback + slot 8 thinking', 'Exact observation reference']
    for i, (size, label, color) in enumerate([('n4','4 nodes (12 tasks)','#2b7294'), ('n6','6 nodes (6 tasks)','#da7837')]):
        values = [metrics[size][method]['recovery_rate']*100 for method in METHODS]
        values.append(stats[size]['reference']['recovery_rate']*100)
        bars = ax.barh([j+(i-.5)*.33 for j in range(4)], values, .31, label=label, color=color)
        ax.bar_label(bars, labels=[f'{v:.1f}%' for v in values], padding=3, fontsize=9)
    ax.set_yticks(range(4), names)
    ax.invert_yaxis()
    ax.set_xlim(0, 112)
    ax.set_xlabel('Mean recovery rate, 8 proposals per task')
    ax.legend(loc='upper right')
    ax.set_title('Held-out HypoSpace causal evaluation')
    fig.savefig(output/'coverage.png', dpi=170)
    plt.close(fig)
    print(f"Verified 432 scored slots and {summary['actual_requests']} actual requests.")


if __name__ == '__main__':
    main()
