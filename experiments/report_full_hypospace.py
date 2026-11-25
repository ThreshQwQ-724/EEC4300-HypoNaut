"""Audit causal submissions against the raw answers and summarize the full run."""
import argparse
from pathlib import Path
from statistics import mean

from common import HERE, digest, load_upstream
from run_full_hypospace import COUNTS, SLOTS, summarize
from run_local_causal import write_json
from run_qwen_rr_budget import read
from validate_constructive_hypospace import raw_cases


LABELS = {
    'boolean': ('Boolean', {'grammar': 'Qwen 文法编号', 'feedback': 'Qwen 文法编号＋反例反馈'}),
    '3d': ('3D', {'direct': 'Qwen 官方层格式提示', 'heights': 'Qwen 柱高编码', 'program': '程序枚举'}),
    'causal': ('因果图', {'reselect': 'Qwen 路径反馈重选', 'repair': 'Qwen 路径定向改边'}),
}


def load_results(root, domain):
    folder = root/domain
    config = read(folder/'config.json')
    assert config['slots'] == SLOTS and config['domain'] == domain
    assert len(set(config['cases'])) == COUNTS[domain]
    assert all(digest(HERE/name) == sha for name, sha in config['code_sha'].items())
    rows = [read(folder/case/'result.json') for case in config['cases']]
    assert [row['id'] for row in rows] == config['cases']
    assert read(folder/'summary.json')['methods'] == summarize(rows)
    return config, rows


def audit_causal(rows, output):
    index = {row['id']: row for row in rows}

    def proposals(task, nodes, limit):
        row = index[task['observation_set_id']]
        outputs = {}
        for name, run in row['runs'].items():
            slots = [1]+[step['slot'] for step in run['trace'] if step['submitted'] is not None]
            graphs = [tuple(map(tuple, graph)) for graph in run['graphs']]
            assert len(graphs) == len(set(graphs)) == len(slots) and slots == sorted(set(slots))
            outputs[name] = {'graphs': graphs, 'slots': slots}
        return outputs

    module = load_upstream('causal', 'run_causal_benchmark.py')
    prior = read(HERE/'results/causal_audit.json')
    provenance = None
    counts = {}
    for row in raw_cases(module, solve_row=proposals):
        if 'raw_sha256' in row:
            provenance = row
            continue
        expected = index[row['id']]
        assert row['unique_truth'] == row['listed_truth']
        for method, results in row['methods'].items():
            assert all(results[k] == expected['runs'][method][k] for k in ('rr','vr','nr','found'))
        counts[row['id']] = row['unique_truth']
        if len(counts) % 100 == 0:
            print(f'原始因果答案复核 {len(counts)}/{len(rows)}', flush=True)
    assert provenance and provenance['raw_sha256'] == prior['source_sha256']
    assert set(counts) == set(index)
    assert counts == {r['id']: r['n_ground_truths'] for r in prior['instances']}
    result = {'raw_sha256': provenance['raw_sha256'], 'cases': len(counts),
              'ground_truth_entries': sum(counts.values()),
              'checked_submission_count': sum(run['found'] for row in rows for run in row['runs'].values())}
    write_json(output/'causal_ground_truth_audit.json', result)
    return result


def write_report(output, cohorts, configs, audit):
    summaries = {domain: summarize(rows) for domain, rows in cohorts.items()}
    lines = ['# HypoSpace 三类题集全量测评', '',
             '覆盖本地 Boolean 80 题、3D 84 题、因果图 510 题。每题固定 8 个提交槽、一个配对种子。'
             '统一使用本地 Qwen3:8B、关闭思考模式，温度 0.60、top_p 0.95、上下文 4096。'
             '布尔与因果结构化输出上限为 96 token，3D 原始层格式为 256 token。'
             '因果验证失败时可有一次重试机会，调用数与提交槽数分别统计。', '',
             'Boolean 用公开文法生成未按题目筛选的候选编号，比较有无上一题内提案的验证反馈。'
             '3D 比较原始层格式与无重复柱高编码，同时报告程序枚举。'
             '因果图共享程序首图和候选生成规则，比较失败后在原菜单重选与菜单外的定向改边。', '',
             '| 题集 | 方法 | 不同合法答案 | RR | VR | NR | Qwen 调用 | 输出 token | 请求秒数 |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for domain, (title, methods) in LABELS.items():
        for method, label in methods.items():
            d = summaries[domain][method]
            lines.append(f"| {title} | {label} | {d['found']:,} | {d['rr']:.4%} | {d['vr']:.2%} | {d['nr']:.2%} | "
                         f"{d['calls']:,} | {d['tokens']:,} | {d['request_seconds']:.2f} |")
    lines += ['', 'VR、NR、RR 按题宏平均。不同合法答案数为逐题累加。'
              '请求秒数统计各流程内的本机请求，配对组共用的请求按各组分别计入。', '',
              '## 配对比较', '']
    comparisons = {}
    for domain, left, right in (('boolean','feedback','grammar'), ('3d','heights','direct'),
                               ('causal','repair','reselect')):
        gaps = [r['runs'][left]['found']-r['runs'][right]['found'] for r in cohorts[domain]]
        wtl = [sum(d>0 for d in gaps), sum(d==0 for d in gaps), sum(d<0 for d in gaps)]
        delta = {metric: summaries[domain][left][metric]-summaries[domain][right][metric]
                 for metric in ('rr','vr','nr')}
        title, labels = LABELS[domain]
        comparisons[domain] = {'wins_ties_losses': wtl, 'delta': delta}
        lines.append(f"- {title} 的{labels[left]}对{labels[right]}，合法答案数胜／平／负为 "
                     f"{wtl[0]}／{wtl[1]}／{wtl[2]}，RR 差 {100*delta['rr']:+.4f} 个百分点。")
    lines += ['', '## 因果图分层', '',
              '| 干预数 | 题数 | 重选 RR | 改边 RR | 重选 VR / NR | 改边 VR / NR |',
              '|---:|---:|---:|---:|---:|---:|']
    strata = {}
    for n in range(1,7):
        rows = [r for r in cohorts['causal'] if r['observations'] == n]
        d = strata[n] = summarize(rows)
        a, b = d['reselect'], d['repair']
        lines.append(f"| {n} | {len(rows)} | {a['rr']:.4%} | {b['rr']:.4%} | {a['vr']:.2%} | {b['vr']:.2%} |")
    causal_cost = {method: {key: sum(row['runs'][method][key] for row in cohorts['causal'])
                           for key in ('checks','rescues','repair_attempts','rejected_repairs')}
                   for method in LABELS['causal'][1]}
    lines += ['', '## 评分版本与覆盖范围', '',
              'Boolean 主表采用本地 NOR 深度修正评分器。同一批模型输出用固定版本原始评分器重评如下。', '',
              '| Boolean 方法 | 原始 RR | 原始 VR | 原始 NR |', '|---|---:|---:|---:|']
    upstream = {}
    for method, label in LABELS['boolean'][1].items():
        d = upstream[method] = {k: mean(r['runs'][method]['upstream'][k] for r in cohorts['boolean'])
                                for k in ('rr','vr','nr')}
        lines.append(f"| {label} | {d['rr']:.4%} | {d['vr']:.2%} | {d['nr']:.2%} |")
    lines += ['', '3D 使用补齐最大高度检查和空结构处理的评分器。'
              '柱高编码直接利用物理约束，可行的柱高序列均满足投影和重力；程序枚举提供对应参照。',
              f"因果图使用固定版本官方评分器，并扫过原始 JSON 的 {audit['ground_truth_entries']:,} 个答案，"
              f"核对两组所有 {audit['checked_submission_count']:,} 个合法提交的答案集成员资格。",
              '固定八槽实验用于比较同预算策略。数据文件含更多合法答案时，RR 受提交槽数限制。', '',
              '## 复现', '', '从 experiments 目录运行以下命令。', '',
              '```powershell', "foreach ($domain in @('boolean','3d','causal')) {",
              f"  & 'D:\\Anaconda\\python.exe' run_full_hypospace.py --domain $domain --output runs/{output.name}",
              '}', f"& 'D:\\Anaconda\\python.exe' report_full_hypospace.py --output runs/{output.name}",
              '```', '']
    write_json(output/'summary.json', {'cases': COUNTS, 'configs': configs, 'methods': summaries,
               'paired': comparisons, 'boolean_upstream': upstream, 'causal_by_interventions': strata,
               'causal_search_cost': causal_cost, 'causal_audit': audit})
    (output/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    output = parser.parse_args().output
    configs, cohorts = {}, {}
    for domain in COUNTS:
        configs[domain], cohorts[domain] = load_results(output, domain)
    audit = audit_causal(cohorts['causal'], output)
    write_report(output, cohorts, configs, audit)
    print(output/'REPORT.md', flush=True)


if __name__ == '__main__':
    main()
