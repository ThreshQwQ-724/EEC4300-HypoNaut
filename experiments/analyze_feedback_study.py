"""Audit saved experimental artifacts and append interpretation, without inference."""
import argparse
import collections
import csv
import json
import re
import statistics
from pathlib import Path

import networkx as nx
from common import digest
from run_local_causal import write_json

parser = argparse.ArgumentParser()
parser.add_argument('run_dir', type=Path)
args = parser.parse_args()
p = args.run_dir
summary = json.loads((p/'summary.json').read_text(encoding='utf-8'))
config = json.loads((p/'config.json').read_text(encoding='utf-8'))
results = json.loads((p/'feedback_results.json').read_text(encoding='utf-8'))
thinking = json.loads((p/'thinking_results.json').read_text(encoding='utf-8'))
assert len(results) == 48 and len(thinking) == 9
assert len(list((p/'feedback').glob('*/call_*.json'))) == 384
assert len(list((p/'thinking').glob('*/call_*.json'))) == 9
with (p/'all_calls.csv').open(encoding='utf-8-sig') as stream:
    assert len(list(csv.DictReader(stream))) == 384
assert digest(Path(__file__).with_name('run_feedback_study.py')) == config['runner_sha256']
assert digest(Path(__file__).with_name('run_four_groups.py')) == config['scoring_helper_sha256']

first_answers = collections.defaultdict(set)
cycles = collections.Counter()
row_count = 0
for r in results:
    assert r['independent_scoring_matches'] and not r['official']['errors']
    assert len(r['details']) == 8
    first_answers[(r['instance'], r['seed'])].add(r['details'][0]['answer'])
    assert sum(d['prompt_tokens'] for d in r['details']) == r['official']['token_usage']['prompt_tokens']
    assert sum(d['output_tokens'] for d in r['details']) == r['official']['token_usage']['completion_tokens']
    for d in r['details']:
        row_count += 1
        assert d['done_reason'] == 'stop'
        if not d['parsed']:
            graph = nx.DiGraph()
            graph.add_edges_from(re.findall(r'([A-D])\s*->\s*([A-D])', d['answer']))
            assert not nx.is_directed_acyclic_graph(graph)
            cycles[r['group']] += 1
assert all(len(values) == 1 for values in first_answers.values())
verification = {'feedback_calls': row_count, 'thinking_calls': len(thinking),
                'paired_first_answers_identical': True, 'official_and_independent_scoring_match': True,
                'token_accounting_matches': True, 'source_hashes_match': True,
                'feedback_all_normal_stop': True, 'all_parse_failures_contain_cycles': True,
                'cyclic_rejections_by_group': dict(cycles)}
write_json(p/'final_verification.json', verification)

text = ['## 结果解读与失败分析', '',
        '本次原题上的 B 基线复现了上一轮的平均 VR=70.83%、RR=20.83%。相同题目和种子下，四组首个答案完全相同；差异从追加反馈后开始。所有 384 次非思考调用正常结束，没有输出长度截断。', '',
        '原题中，仅观测一致性反馈的 RR 从20.83%升至37.50%（+16.67个百分点），三个种子都找到了3/8个合法图；但VR从70.83%降至62.50%。这意味着它探索到了更多不同的正确图，同时仍生成了错误图。仅重复反馈的VR为83.33%、RR为25.00%；组合反馈的RR反而降至16.67%。', '',
        '三道新题中，重复反馈和组合反馈的平均RR均为19.44%，基线为9.72%（+9.72个百分点）；仅观测一致性反馈为15.28%。正确率分别为：基线25.00%、重复反馈33.33%、一致性反馈37.50%、组合反馈36.11%。新题上的最佳方法与原题不同，没有单一配置在所有指标和题目上占优。', '',
        '下表按每题三个种子平均，仅比较RR：', '',
        '| 题目 | 基线 | 重复反馈 | 一致性反馈 | 组合反馈 |', '|---|---|---|---|---|']
for instance in config['instances']:
    rates = [statistics.mean(r['official']['recovery_rate'] for r in results if r['instance'] == instance and r['group'] == group)
             for group in config['groups']]
    text.append('| '+instance+' | '+' | '.join(f'{v:.2%}' for v in rates)+' |')
text += ['',
         'n4_182的四组都未找到合法图。候选在不同错误结构间变化，存在修复某条约束又破坏其他约束的情况。例如一致性组的一个候选C->A, C->B, C->D满足C的干预，但没有满足A必须到达B和D的要求；下一步又产生了A->C等与观测相反的方向。因而提高唯一率本身不能保证提高合法解覆盖。', '',
         f"共有{sum(cycles.values())}次答案被官方DAG解析器拒绝，全部包含有向环，而不只是文本格式错误。各组次数为："+
         '、'.join(f'{g}={cycles[g]}' for g in config['groups'])+'。循环图不进入官方历史，所以固定种子下同一个坏答案可能反复出现。这里的“重复次数”只统计成功解析的图；它不包含被拒绝的循环图重复。', '',
         '组合反馈在原题上变差，部分与生成循环图并反复失败有关。当前检查器对无法解析的图只给出通用格式/DAG提醒，没有提供具体环路。这是下一轮可改进的实际缺口；本轮没有事后修改反馈再挑选更好的结果。', '',
         '### 思考上限的结果', '',
         '| 上限 | 首答正确次数 | 截断次数 | 平均输出tokens | 平均秒数 |', '|---|---|---|---|---|']
for limit in (2048, 4096, 8192):
    ds = [d for d in thinking if d['limit'] == limit]
    text.append(f"| {limit} | {sum(d['valid'] for d in ds)}/3 | {sum(d['done_reason']=='length' for d in ds)}/3 | "
                f"{statistics.mean(d['output_tokens'] for d in ds):.1f} | {statistics.mean(d['seconds'] for d in ds):.2f} |")
text += ['',
         '同一种子与相同提示下，提高num_predict只是放宽生成上限，并不要求模型生成到上限。正常完成的短答案可能在4096和8192上限下完全相同。该实验只测首答，不能证明思考模式能在8次调用中探索更多不同合法图。', '',
         '### 下一步建议', '',
         '先保留B原版和三种反馈作为可复现消融。更值得尝试的是累计约束反馈与具体环路反馈：保留已满足的可达/不可达条件，指出新增候选破坏了哪些条件；遇到循环图时明确报告环上的边，而不只重复“请输出DAG”。这比继续堆叠笼统的多样性要求更贴近本次失败原因，但尚需实验验证。所有新增模型请求都应计入预算，验证器不得使用标准答案给模型补全图。', '',
         '在更多未参与设计的题目上比较前，暂不宣布某个反馈方法普遍优于其他方法。本文中的“新题”是预先固定的小规模检查集；它们在本轮完成后也应视为开发数据，后续正式结论需要另外保留测试题。', '',
         '最终审计已核对393个原始调用文件、384行反馈指标、9条预算诊断、官方与独立评分、token合计和代码哈希，记录于final_verification.json。']
report = (p/'REPORT.md').read_text(encoding='utf-8').split('## 结果解读与失败分析')[0].rstrip()
(p/'REPORT.md').write_text(report+'\n\n'+'\n'.join(text)+'\n', encoding='utf-8')
print(json.dumps(verification, indent=2))
