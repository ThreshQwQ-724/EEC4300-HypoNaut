"""Add a budget-matched self-review and two-agent debate to the frozen formal evaluation."""
import argparse
from collections import Counter
from pathlib import Path
from statistics import mean
from time import perf_counter

import requests

from common import HERE, digest, load_upstream
from run_formal_causal_comparison import DATA, METHODS as BASE_METHODS, SLOTS, read, valid
from run_four_groups import BASE_URL, MODEL, score_calls
from run_local_causal import write_json


REFERENCE = HERE/'runs/formal_causal_comparison_20261113'
DEBATE_REFERENCE = HERE/'runs/formal_agent_debate_20261113'
REVIEW_LIMIT = 256
SECOND_AGENT_SEED_OFFSET = 1000000
METHODS = {'self_review': '单 Qwen 自审', 'agent_debate': '双 Qwen Debate'}
GUARDED = 'guarded_debate'


def request(session, path, messages, options, seed, limit):
    payload = {'model': MODEL, 'think': False, 'stream': False, 'keep_alive': '10m',
               'options': {**options, 'seed': seed, 'num_predict': limit}, 'messages': messages}
    if path.exists():
        result = read(path)
        if result['request'] != payload:
            raise ValueError(f'Cached request does not match: {path}')
        return result
    start = perf_counter()
    response = session.post(BASE_URL+'/api/chat', json=payload, timeout=(10, 180))
    response.raise_for_status()
    result = {'request': payload, 'response': response.json(), 'seconds': perf_counter()-start}
    temporary = path.with_suffix('.tmp')
    write_json(temporary, result)
    temporary.replace(path)
    return result


def content(call):
    return call['response']['message'].get('content', '')


def critique_prompt(owner):
    draft = 'your draft' if owner else 'the other agent’s draft'
    return (f'Audit {draft} against the task. For every intervention, reason about directed '
            'reachability, including paths of length greater than one. Check for cycles, '
            'edge-limit violations and exact duplicate edge sets in the prior predictions. '
            'State whether you find a concrete issue. If so, name the missing or forbidden path, '
            'cycle, or duplicate and suggest a corrected graph. If you find no issue, advise '
            'keeping the draft. If you propose edits, check the resulting graph against every '
            'observation. Use at most 100 words. Start with Critique: and, if suggesting a graph, '
            'put Alternative: followed by the edge list on a second line.')


def protect_draft(draft, draft_valid, revision, revision_valid, recovered):
    """Preserve legality and new valid coverage when selecting a submission."""
    if not draft_valid:
        return None
    if not revision_valid:
        return 'revision_invalid'
    if draft not in recovered and revision in recovered:
        return 'revision_duplicate'
    return None


def solve(benchmark, session, task, options, folder, seed, method):
    observations = task['observations']
    history, submissions, all_calls, trace = [], [], [], []
    seen, recovered = set(), set()
    limit = options['num_predict']
    debate = method in ('agent_debate', GUARDED)
    for slot in range(1, SLOTS+1):
        proposer_seed = seed if slot % 2 else seed+SECOND_AGENT_SEED_OFFSET
        reviewer_seed = (2*seed+SECOND_AGENT_SEED_OFFSET-proposer_seed) if debate else proposer_seed
        prompt = {'role': 'user', 'content': benchmark.create_prompt(observations, history)}
        prefix = f'{method}_{slot:02d}'
        draft = request(session, folder/f'{prefix}_draft.json', [prompt], options, proposer_seed, limit)
        draft_text = content(draft).strip() or '[empty response]'
        draft_message = {'role': 'assistant', 'content': draft_text}
        critique = {'role': 'user', 'content': critique_prompt(not debate)}
        if debate:
            critique['content'] = f'Other agent’s draft:\n{draft_text}\n\n' + critique['content']
            review_messages = [prompt, critique]
        else:
            review_messages = [prompt, draft_message, critique]
        review = request(session, folder/f'{prefix}_review.json', review_messages, options, reviewer_seed,
                         REVIEW_LIMIT)
        review_text = content(review).strip() or '[empty review]'
        reviewer = 'Peer review' if debate else 'Your review'
        final_prompt = {'role': 'user', 'content': f'{reviewer}:\n{review_text}\n\n'
                        'Verify the review against the original observations. Keep your original '
                        'draft if it already satisfies every observation and differs from all prior '
                        'predictions. Otherwise fix it or use the suggested alternative after checking '
                        'it. Return exactly one Graph: line and no explanation.'}
        revision = request(session, folder/f'{prefix}_final.json', [prompt, draft_message, final_prompt],
                           options, proposer_seed, limit)
        all_calls.extend((draft, review, revision))
        draft_graph = benchmark.parse_llm_response(content(draft))
        graph = benchmark.parse_llm_response(content(revision))
        draft_key = tuple(sorted(draft_graph.edges)) if draft_graph is not None else None
        key = tuple(sorted(graph.edges)) if graph is not None else None
        draft_valid = draft_key is not None and valid(draft_key, observations)
        final_valid = key is not None and valid(key, observations)
        revision_key, revision_valid = key, final_valid
        fallback = (protect_draft(draft_key, draft_valid, key, final_valid, recovered)
                    if method == GUARDED else None)
        if fallback:
            graph, key, final_valid = draft_graph, draft_key, draft_valid
        submissions.append(draft if fallback else revision)
        if method == GUARDED and draft_valid:
            assert final_valid and (draft_key in recovered or key not in recovered)
        trace.append({'slot': slot, 'proposer_seed': proposer_seed, 'reviewer_seed': reviewer_seed,
                      'revision': revision_key, 'revision_valid': revision_valid, 'fallback': fallback,
                      'draft': draft_key, 'final': key, 'draft_valid': draft_valid, 'final_valid': final_valid,
                      'draft_novel': draft_key is not None and draft_key not in seen,
                      'final_novel': key is not None and key not in seen, 'changed': key != draft_key,
                      'draft_new_valid': draft_valid and draft_key not in recovered,
                      'final_new_valid': final_valid and key not in recovered})
        if graph is not None:
            history.append(graph)
            seen.add(key)
        if final_valid:
            recovered.add(key)

    official, counts = score_calls(benchmark, {'observations': observations,
        'ground_truth_graphs': [{'edges': edges} for edges in sorted(recovered)]}, submissions)
    assert len(recovered) == counts['n_recovered_gts'] <= min(SLOTS, task['n_ground_truths'])
    assert all(official_row['valid'] == trace_row['final_valid'] and
               official_row['new_valid'] == trace_row['final_new_valid']
               for official_row, trace_row in zip(official, trace, strict=True))
    return {'rr': len(recovered)/task['n_ground_truths'], 'vr': counts['n_valid']/SLOTS,
            'nr': counts['n_unique_all']/SLOTS, 'found': len(recovered), 'checks': len(history),
            'calls': len(all_calls), 'seconds': sum(call['seconds'] for call in all_calls),
            'tokens': sum(call['response'].get('eval_count', 0) for call in all_calls),
            'input_tokens': sum(call['response'].get('prompt_eval_count', 0) for call in all_calls),
            'truncated_reviews': sum(call['response'].get('done_reason') == 'length' for call in all_calls[1::3]),
            'graphs': sorted(recovered), 'trace': trace, 'official_slots': official}


def report(output, records):
    names = {'qwen_direct': BASE_METHODS['qwen_direct'], **METHODS,
             **{name: label for name, label in BASE_METHODS.items() if name != 'qwen_direct'}}
    summary, diagnosis, comparisons = {}, {}, {}
    lines = ['# HypoSpace 六节点正式对照 · Agent Debate', '',
             '沿用正式评测的 24 道题、每题 8 个提交槽、每种干预次数 4 道。'
             '正式基线从同题原始记录读取。两个新方法均使用 qwen3:8b、关闭思考模式，'
             '每槽依次生成草稿、审查、提交最终图，共 3 次模型调用。只有最终图进入评分和后续历史。', '',
             '双 Qwen 使用同一组权重、独立消息上下文与不同种子。A、B 逐槽交换提案和审查职责。'
             '审查者读题和草稿，指出路径、环路或重复问题；提案者可以采纳或反驳审查建议。'
             '自审组用相同的三阶段流程，但由同一个提案上下文延伸出审查并沿用提案种子。'
             '两个新组共用每槽提案种子、采样参数和输出上限。程序只在评分时核对，不参与选择最终图。', '',
             '| 方法 | RR | VR | NR | 合法图数 | 满 8 图 | 调用 | 输出 token | 秒数 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, label in names.items():
        rows = [record['runs'][name] for record in records]
        stats = {key: mean(row[key] for row in rows) for key in ('rr', 'vr', 'nr')}
        stats.update({key: sum(row[key] for row in rows) for key in ('found', 'calls', 'tokens', 'seconds')})
        stats['full'] = sum(row['found'] == SLOTS for row in rows)
        summary[name] = stats
        lines.append(f"| {label} | {stats['rr']:.4%} | {stats['vr']:.2%} | {stats['nr']:.2%} | "
                     f"{stats['found']} | {stats['full']}/{len(rows)} | {stats['calls']} | "
                     f"{stats['tokens']:,} | {stats['seconds']:.2f} |")
        if name in METHODS:
            slots = [slot for row in rows for slot in row['trace']]
            entry = Counter()
            for slot in slots:
                entry['changed'] += slot['changed']
                for field in ('valid', 'new_valid'):
                    before, after = slot[f'draft_{field}'], slot[f'final_{field}']
                    entry[f'{field}_gain'] += not before and after
                    entry[f'{field}_loss'] += before and not after
            diagnosis[name] = dict(entry) | {'input_tokens': sum(row['input_tokens'] for row in rows),
                                            'truncated_reviews': sum(row['truncated_reviews'] for row in rows)}
    lines += ['', '## 逐题对照', '', '| 题目 | 干预次数 | 合法解数 | 官方提示 | 自审 | Debate | 按需搜索 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for record in records:
        counts = ' | '.join(str(record['runs'][name]['found']) for name in (
            'qwen_direct', 'self_review', 'agent_debate', 'adaptive'))
        lines.append(f"| {record['id']} | {record['observations']} | {record['truth']:,} | {counts} |")
    lines += ['', '## 配对差异和审查诊断', '']
    for other in ('qwen_direct', 'self_review', 'adaptive'):
        deltas = [record['runs']['agent_debate']['found']-record['runs'][other]['found'] for record in records]
        wtl = [sum(d > 0 for d in deltas), sum(d == 0 for d in deltas), sum(d < 0 for d in deltas)]
        effects = {key: summary['agent_debate'][key]-summary[other][key] for key in ('rr', 'vr', 'nr')}
        comparisons[other] = {'win_tie_loss': wtl, **effects}
        lines.append(f"- Debate 对比{names[other]}的逐题合法图数胜／平／负为 {' / '.join(map(str, wtl))}。"
                     f"RR 差 {effects['rr']*100:+.4f} 个百分点，VR 差 {effects['vr']*100:+.2f} 个百分点，"
                     f"NR 差 {effects['nr']*100:+.2f} 个百分点。")
    for name, d in diagnosis.items():
        lines.append(f"- {names[name]}从草稿到最终稿，{d['valid_gain']} 槽由无效变合法，"
                     f"{d['valid_loss']} 槽由合法变无效。新的合法图净变化为 "
                     f"{d['new_valid_gain']-d['new_valid_loss']:+d} 槽。"
                     f"审查输出到达上限 {d['truncated_reviews']} 次。")
    lines += ['', 'RR 按题宏平均，VR 为合法槽比例，NR 为不同可解析候选占槽比例。'
              '本轮每题一个固定种子。原有方法耗时来自上次正式运行；新方法秒数为三阶段请求时间总和。'
              'RR 分母来自已审计的合法解总数。原始调用与逐槽评分保存在同目录。', '']
    write_json(output/'summary.json', {'methods': summary, 'debate_deltas': comparisons, 'review': diagnosis})
    (output/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def report_guarded(output, records):
    names = {'qwen_direct': BASE_METHODS['qwen_direct'], **METHODS,
             GUARDED: '双 Qwen Debate＋校验', 'adaptive': BASE_METHODS['adaptive']}
    lines = ['# HypoSpace 正式对照 · 校验保护的 Agent Debate', '',
             '沿用同一批 24 道六节点题，每题 8 个槽、每题一个固定种子。校验组重新运行完整三阶段对话，'
             '保留原双 Qwen 的模型、参数和调用预算。每次把实际选中的图用于评分和下一槽历史。', '',
             '程序只用题目观测检查草稿与改写图的无环性、边数和可达性。合法草稿遇到无效改写时自动保留。'
             '新合法草稿遇到重复改写时也保留。无效草稿仍允许由合法改写替换。两版都无效时按原流程评分。', '',
             '| 方法 | RR | VR | NR | 不同合法图 | 调用 | 输出 token | 请求秒数 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    summary = {}
    for name, label in names.items():
        rows = [record['runs'][name] for record in records]
        values = {key: mean(row[key] for row in rows) for key in ('rr', 'vr', 'nr')}
        values.update({key: sum(row[key] for row in rows) for key in ('found', 'calls', 'tokens', 'seconds')})
        summary[name] = values
        lines.append(f"| {label} | {values['rr']:.4%} | {values['vr']:.2%} | {values['nr']:.2%} | "
                     f"{values['found']} | {values['calls']} | {values['tokens']:,} | {values['seconds']:.2f} |")
    traces = [step for record in records for step in record['runs'][GUARDED]['trace']]
    diagnosis = {'invalid_rewrites_blocked': sum(step['fallback'] == 'revision_invalid' for step in traces),
                 'duplicate_rewrites_blocked': sum(step['fallback'] == 'revision_duplicate' for step in traces),
                 'invalid_drafts_repaired': sum(not step['draft_valid'] and step['final_valid'] for step in traces),
                 'valid_drafts_lost': sum(step['draft_valid'] and not step['final_valid'] for step in traces)}
    assert diagnosis['valid_drafts_lost'] == 0
    lines += ['', '## 逐题复核', '', '| 题目 | 合法解数 | 原 Debate 合法图 | 校验版合法图 |', '|---|---:|---:|---:|']
    for row in records:
        lines.append(f"| {row['id']} | {row['truth']:,} | {row['runs']['agent_debate']['found']} | "
                     f"{row['runs'][GUARDED]['found']} |")
    comparisons = {}
    for other in ('agent_debate', 'self_review'):
        deltas = [r['runs'][GUARDED]['found']-r['runs'][other]['found'] for r in records]
        comparisons[other] = [sum(d > 0 for d in deltas), sum(d == 0 for d in deltas), sum(d < 0 for d in deltas)]
    lines += ['', f"校验版拦下 {diagnosis['invalid_rewrites_blocked']} 次无效改写和 "
              f"{diagnosis['duplicate_rewrites_blocked']} 次会损失新图的重复改写。"
              f"成功修复 {diagnosis['invalid_drafts_repaired']} 个无效草稿，"
              f"合法草稿降为无效的槽数为 {diagnosis['valid_drafts_lost']}。", '',
              f"校验版对原 Debate 的逐题合法图数胜／平／负为 {' / '.join(map(str, comparisons['agent_debate']))}。", '',
              '旧方法分数和耗时来自已有同题测评记录。校验版时间为三阶段请求时间总和。'
              '由于回退会改变后续提示，整轮分数还受生成历史影响。', '']
    write_json(output/'summary.json', {'methods': summary, 'guard': diagnosis, 'paired': comparisons})
    (output/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--guarded-only', action='store_true', help='Run guarded Debate against frozen comparisons')
    args = parser.parse_args()
    output = args.output
    source = read(REFERENCE/'config.json')
    assert digest(DATA) == source['source_sha'] and source['slots'] == SLOTS
    module = load_upstream('causal', 'run_causal_benchmark.py')
    assert digest(module.__file__) == source['evaluator_sha']
    benchmark = module.CausalBenchmarkEnhanced(str(REFERENCE/'observations.json'))
    tasks = {task['id']: task for task in read(DATA)}
    options = source['options']
    methods = (GUARDED,) if args.guarded_only else tuple(METHODS)
    if args.guarded_only:
        previous = read(DEBATE_REFERENCE/'config.json')
        assert previous['source_config_sha'] == digest(REFERENCE/'config.json')
        assert previous['cases'] == source['cases'] and previous['model'] == source['model']
    output.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        session.trust_env = False
        tags = session.get(BASE_URL+'/api/tags', timeout=10)
        tags.raise_for_status()
        model = next(row for row in tags.json()['models'] if row['name'] == MODEL)
        assert model['digest'] == source['model']
        version = session.get(BASE_URL+'/api/version', timeout=10)
        version.raise_for_status()
        reference_hashes = {case: digest(REFERENCE/case/'result.json') for case in source['cases']}
        config = {'cases': source['cases'], 'methods': list(methods), 'model': model['digest'],
                  'ollama': version.json(), 'options': options, 'think': False, 'review_limit': REVIEW_LIMIT,
                  'slots': SLOTS, 'second_seed_offset': SECOND_AGENT_SEED_OFFSET,
                  'source_config_sha': digest(REFERENCE/'config.json'), 'source_results': reference_hashes,
                  'code_sha': {name: digest(HERE/name) for name in (Path(__file__).name,
                      'run_formal_causal_comparison.py', 'run_four_groups.py', 'stream_causal.py')}}
        if args.guarded_only:
            config['debate_reference'] = {case: {method: digest(DEBATE_REFERENCE/case/f'{method}_result.json')
                                                for method in METHODS} for case in source['cases']}
        config_path = output/'config.json'
        if config_path.exists():
            if read(config_path) != config:
                raise ValueError('Output directory contains a different experiment configuration')
        else:
            write_json(config_path, config)
        records = []
        for index, case in enumerate(source['cases'], 1):
            folder = output/case
            folder.mkdir(exist_ok=True)
            record = read(REFERENCE/case/'result.json')
            if args.guarded_only:
                record['runs'].update({method: read(DEBATE_REFERENCE/case/f'{method}_result.json')
                                       for method in METHODS})
            task = tasks[case]
            assert task['n_ground_truths'] == record['truth']
            # Alternate execution order to reduce systematic cache and warm-up effects.
            order = methods if index % 2 else tuple(reversed(methods))
            for method in order:
                path = folder/f'{method}_result.json'
                if path.exists():
                    result = read(path)
                else:
                    result = solve(benchmark, session, task, options, folder, record['seed'], method)
                    temporary = path.with_suffix('.tmp')
                    write_json(temporary, result)
                    temporary.replace(path)
                record['runs'][method] = result
            records.append(record)
            print(f'{index}/{len(source["cases"])} {case} ' +
                  ' '.join(f'{name}={record["runs"][name]["found"]}' for name in methods), flush=True)
        assert len(records) == len({record['id'] for record in records}) == 24
        (report_guarded if args.guarded_only else report)(output, records)
        print(output.resolve(), flush=True)


if __name__ == '__main__':
    main()
