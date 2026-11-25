"""Audit the frozen formal test and diagnose the eight-item menu limit."""
import argparse
from pathlib import Path
from statistics import mean
from time import perf_counter
from unittest.mock import patch

import run_formal_causal_comparison as formal
from common import HERE, digest
from run_local_causal import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    output = parser.parse_args().run_dir
    config = formal.read(output/'config.json')
    assert all(digest(HERE/name) == expected for name, expected in config['code_hashes'].items())
    tasks = {task['id']: task for task in formal.read(formal.DATA)}
    assert len(config['cases']) == len(set(config['cases'])) == 24
    assert len(list(output.glob('*/direct_*.json'))) == 192
    assert len(list(output.glob('*/qwen_*.json'))) == 166
    results = []
    for item in config['cases']:
        task = tasks[item]
        obs = task['observations']
        record = formal.read(output/item/'result.json')
        assert len(record['runs']) == 8
        for row in record['runs'].values():
            graphs = [tuple(map(tuple, edges)) for edges in row['graphs']]
            assert len(set(graphs)) == len(graphs) == row['found']
            assert all(formal.valid(edges, obs) for edges in graphs)
            assert row['rr'] == len(graphs)/task['n_ground_truths']
        anchor = formal.program_anchor(formal.NODES, formal.LIMIT, obs)
        before = record['runs']['delete_risk']
        start = perf_counter()
        with patch.object(formal, 'MENU', None):
            graphs, checks, trace, closures = formal.improved(task, anchor)
        assert len(set(graphs)) == len(graphs) <= min(formal.SLOTS, task['n_ground_truths'])
        assert all(formal.valid(edges, obs) for edges in graphs) and checks <= 7
        results.append({'id': item, 'found': len(graphs), 'before': before['found'],
                        'rr': len(graphs)/task['n_ground_truths'], 'checks': checks,
                        'probes': sum(row['probes'] for row in trace),
                        'deletion_traversals': sum(row['deletion_traversals'] for row in trace),
                        'closures': closures, 'seconds': perf_counter()-start, 'graphs': graphs, 'trace': trace})
    summary = {key: sum(row[key] for row in results) for key in
               ('found', 'checks', 'probes', 'deletion_traversals', 'closures', 'seconds')}
    summary.update(rr=mean(row['rr'] for row in results), full=sum(row['found'] == formal.SLOTS for row in results))
    write_json(output/'menu_expansion_followup.json', {'summary': summary, 'records': results,
        'formal_sha': digest(HERE/'run_formal_causal_comparison.py'), 'analysis_sha': digest(__file__)})
    lines = ['## 候选菜单的失分诊断', '',
             'n6_096 和 n6_061 的最后一槽各剩一个可通过单边删除得到的合法图。两题的合法候选都排在第 29 位，超过固定 8 项菜单。仅增加删边预判并未改变正式八组结果。', '',
             f'在同题探索性补测中，取消菜单上限后找到 {summary["found"]} 个合法图，'
             f'{summary["full"]}/24 题找满 8 个，平均 RR 为 {summary["rr"]:.4%}。'
             f'仍执行 {summary["checks"]} 次完整验证，同时执行 {summary["probes"]} 次风险预判与 '
             f'{summary["deletion_traversals"]} 次删边可达性遍历。固定菜单的对应成本是 168、198、55 次。', '',
             '在普通预算下，按需剪枝已经达到覆盖上限，可作为因果任务主结果。每槽只允许一次完整验证时，扩大菜单值得进一步优化，可先筛选保留替代路径的删边，再按需扩大搜索范围。比较效率时同时计入预判、遍历和运行时间。', '',
             '这项补测使用发现失分的同一批题。正式八组成绩沿用冻结的 8 项菜单。核验覆盖 24 题、192 轮方法对照、358 次原始 Qwen 调用和代码哈希。']
    path = output/'REPORT.md'
    body = path.read_text(encoding='utf-8').split('## 候选菜单的失分诊断')[0].rstrip()
    path.write_text(body+'\n\n'+'\n'.join(lines)+'\n', encoding='utf-8')
    print(summary)


if __name__ == '__main__':
    main()
