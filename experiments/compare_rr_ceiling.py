"""Compare three-domain reruns with the per-case eight-submission RR ceilings."""
import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean

from common import HERE, digest
from run_all_domain_evaluation import ROOT as REFERENCE, SLOTS, atomic
from run_qwen_guided_edits import read


PREVIOUS = HERE / 'runs/final_full_evaluation_20261114'
METHODS = {
    'boolean': ('Boolean', 80, 'ranked_qwen', 'corrected', '短菜单＋全部反例反馈'),
    '3d': ('3D', 84, 'shape_qwen', 'corrected', '相邻柱高度差＋一次选方案'),
    'causal': ('因果图', 510, 'batch_qwen', 'upstream', 'Qwen 批量排序＋程序补位'),
}


def metrics(rows):
    rr = mean(score['recovery_rate'] for _, score in rows)
    ceiling = mean(min(SLOTS, truth) / truth for truth, _ in rows)
    missing = [min(SLOTS, truth) - score['n_recovered_gts'] for truth, score in rows]
    assert all(gap >= 0 for gap in missing)
    return {
        'cases': len(rows),
        'rr': rr,
        'vr': mean(score['valid_rate'] for _, score in rows),
        'nr': mean(score['novelty_rate'] for _, score in rows),
        'ceiling': ceiling,
        'gap_pp': (ceiling - rr) * 100,
        'hits': sum(score['n_recovered_gts'] for _, score in rows),
        'max_hits': sum(min(SLOTS, truth) for truth, _ in rows),
        'at_cap': missing.count(0),
    }


def calculate(output, domain, expected, method, scorer):
    truth = {r['id']: r['truth'] for r in read(REFERENCE / domain / 'results.json')}
    fresh = read(output / domain / 'results.json')
    previous = {r['id']: r['runs'][method] for r in read(PREVIOUS / domain / 'results.json')}
    assert len(fresh) == len(truth) == expected and {r['id'] for r in fresh} == set(truth) == set(previous)
    groups = defaultdict(list)
    final, prior, prefill = [], [], []
    calls = 0
    for row in fresh:
        case = row['id']
        actual = row['runs'][method]
        count = truth[case]
        assert row.get('truth', count) == count
        score = actual['scores'][scorer]
        old = previous[case]
        if domain == 'causal':
            prefill.append((count, score))
            score = actual['filled']['scores']
            old_score = old['filled']['scores']
        else:
            old_score = old['scores'][scorer]
        final.append((count, score))
        prior.append((count, old_score))
        calls += actual['cost']['calls']
        if 'stratum' in row:
            groups[row['stratum']].append((count, score))
    result = metrics(final) | {'prior_rr': metrics(prior)['rr'], 'calls': calls,
                               'strata': {k: metrics(v) for k, v in sorted(groups.items())}}
    if prefill:
        result['prefill'] = metrics(prefill)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    lines = [
        '# HypoSpace 全量重测与 8 槽理论上限', '',
        '本地 Qwen3:8B，关闭思考模式，沿用相同数据、种子和每题 8 个槽。'
        '在新目录重新请求当前三种方法。因果图同一批结果同时记录补位前后。',
        '',
        '每题标准答案数为 N，理论最多命中 min(8, N) 个不同答案。'
        '将每题 min(8, N) / N 宏平均作为 RR 上限。差距为理论 RR 减去实测 RR，单位为百分点。'
        'Boolean 和 3D 用修正版评分，因果图用原版评分。', '',
        '| 题型 | 方法 | 上轮 RR | 本轮 RR | VR / NR | RR 上限 | 差距（百分点） | 合法命中 / 上限 | 达上限题数 | Qwen 调用 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    summary = {}
    replay_count = 0
    for domain, (label, count, method, scorer, description) in METHODS.items():
        result = calculate(args.output, domain, count, method, scorer)
        summary[domain] = result
        folder = args.output / domain
        audit = read(folder / 'audit.json')
        assert audit['cases'] == count and audit['results_sha'] == digest(folder / 'results.json')
        replay_count += audit['score_replays']
        variants = [(description, result)]
        if 'prefill' in result:
            variants.insert(0, ('Qwen 批量排序（补位前）', result['prefill']))
        for name, m in variants:
            old = result['prior_rr'] if m is result else mean(
                r['runs'][method]['scores'][scorer]['recovery_rate']
                for r in read(PREVIOUS / domain / 'results.json'))
            lines.append(f"| {label} | {name} | {old:.4%} | {m['rr']:.4%} | "
                         f"{m['vr']:.2%} / {m['nr']:.2%} | {m['ceiling']:.4%} | "
                         f"{m['gap_pp']:.4f} | {m['hits']:,} / {m['max_hits']:,} | "
                         f"{m['at_cap']}/{count} | {result['calls']:,} |")
    lines += ['', '## Boolean 按观测数拆分', '',
              '| 观测数 | 题数 | RR | 上限 | 差距（百分点） | 合法命中 / 上限 |',
              '|---:|---:|---:|---:|---:|---:|']
    for n, m in summary['boolean']['strata'].items():
        lines.append(f"| {n} | {m['cases']} | {m['rr']:.4%} | {m['ceiling']:.4%} | "
                     f"{m['gap_pp']:.4f} | {m['hits']:,} / {m['max_hits']:,} |")
    lines += ['', f'三种题型共复核 {replay_count:,} 个评分结果。', '',
              '[全量对照结果](REPORT.md) · [机器可读汇总](ceiling_summary.json)', '']
    atomic(args.output / 'ceiling_summary.json', summary)
    (args.output / 'CEILING_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
