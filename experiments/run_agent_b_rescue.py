"""Test Agent B against a program using the same feedback-aware backup menu."""
import argparse
from pathlib import Path
from statistics import mean

import requests

from common import HERE, digest
from run_feedback_study import graph_line
from run_formal_agent_debate import REFERENCE, SECOND_AGENT_SEED_OFFSET
from run_four_groups import BASE_URL, MODEL
from run_local_causal import write_json
from run_numbered_agent_search import DATA, LIMIT, MENU, NODES, SLOTS, program_anchor, read, run, valid
from run_qwen_guided_edits import decode, request, schema


OLD = HERE/'runs/numbered_agent_search_tight_b2_20261113'
FORMAL = HERE/'runs/numbered_agent_search_formal_20261113'
METHODS = {'program': '改进菜单＋程序选', 'agent_b': '改进菜单＋Agent B 选'}


def rescue_menu(order, edits, checked, failure):
    failed_edge = edits[failure['graph']]['edge']

    def rank(graph):
        edit = edits[graph]
        a, b = edit['edge']
        parent = set(edit['parent'])
        # A local two-hop witness preserves reachability for either edit direction.
        witness = any((a, mid) in parent and (mid, b) in parent for mid in NODES)
        related = a in failed_edge or b in failed_edge
        return not witness, related

    return sorted((graph for graph in order if graph not in checked), key=rank)[:MENU]


def make_rescue(method, session, folder, seed, observations):
    def select(slot, order, edits, checked, emitted, failure):
        menu = rescue_menu(order, edits, checked, failure)
        if len(menu) < 2 or method == 'program':
            return next(iter(menu), None), None
        options = sorted(menu)
        failed = edits[failure['graph']]
        a, b = failed['edge']
        reason = {'cycle': 'it would create a directed cycle',
                  'forbidden_path': 'it would create a path forbidden by an observation',
                  'missing_required_path': 'it would remove a path required by an observation'}[failure['rejection']]
        lines = [f'{i}. {edits[g]["action"]} {edits[g]["edge"][0]}->{edits[g]["edge"][1]}. '
                 f'Result: {graph_line(g)}' for i, g in enumerate(options, 1)]
        prompt = (
            f'Agent A tried to {failed["action"]} {a}->{b}. The verifier rejected it because {reason}. '
            'You have one check left. Choose an unverified candidate that preserves all observed '
            'reachability and stays acyclic. Positive effects require a directed path, while zero '
            'effects forbid any directed path. The graphs were deduplicated. Return the JSON choice.\n'
            'Observations:\n' + '\n'.join(row['string'] for row in observations)
            + '\nCandidates:\n' + '\n'.join(lines))
        call = request(session, folder/f'B_{slot:02}.json', prompt, seed+SECOND_AGENT_SEED_OFFSET,
                       schema({'choice': {'type': 'integer', 'minimum': 1, 'maximum': len(options)}}))
        choice = decode(call).get('choice')
        parsed = type(choice) is int and 1 <= choice <= len(options)
        picked = options[choice-1] if parsed else menu[0]
        return picked, {'role': 'B', 'choice': picked, 'parsed': parsed, 'menu': menu,
                        'program_choice': menu[0], 'seconds': call['seconds'],
                        'tokens': call['response'].get('eval_count', 0),
                        'input_tokens': call['response'].get('prompt_eval_count', 0)}
    return select


def report(output, records, cohorts):
    lines = ['# Agent B 补位对照', '',
             '每题 8 槽，共享程序首图。每个后续槽至多两次探查。两组的 A 沿用编号版的增边优先菜单。'
             'A 的图被拒绝后，两组从同一规则生成最多 8 张未验证候选，可跨增边、删边及父图。'
             '备选排序优先保留可用两跳替代路径解释的改动，其次避免与失败边共享端点。'
             '程序组取首项；Agent B 组收到具体失败类型，选择编号。无效编号回到程序首项。', '',
             '排序使用父图的局部边信息。每次实际探查才做风险判断和必要的完整验证。'
             '旧编号版结果来自同题历史记录。24 道正式题的旧编号版使用三查预算，因此表中仅列同为两查的新组。', '',
             '| 题集 | 方法 | RR | VR / NR | 合法图 | A 失败 | 补位成功 | B 调用 | B 成功 | 探查 | 请求秒数 |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    summary = {}
    for cohort, cases in cohorts.items():
        rows = [records[case] for case in cases]
        block = {}
        names = (('old', '旧 Agent B'), *METHODS.items()) if cohort == '独立 16 题' else METHODS.items()
        for method, label in names:
            runs = [row[method] for row in rows]
            stats = {key: sum(r[key] for r in runs) for key in ('found', 'probes', 'calls', 'b_calls', 'request_seconds')}
            stats.update({key: mean(r[key] for r in runs) for key in ('rr', 'vr', 'nr')})
            failures = [slot for r in runs for slot in r['trace'] if slot['steps'] and not slot['steps'][0]['accepted']]
            b_slots = [slot for slot in failures if any(c['role'] == 'B' for c in slot['calls'])]
            stats.update(a_failures=len(failures), rescued=sum(any(s['accepted'] for s in slot['steps'][1:]) for slot in failures),
                         b_success=sum(any(s['accepted'] and s['origin'] == 'model' for s in slot['steps'][1:]) for slot in b_slots))
            block[method] = stats
            lines.append(f"| {cohort} | {label} | {stats['rr']:.4%} | {stats['vr']:.2%} | {stats['found']} | "
                         f"{stats['a_failures']} | {stats['rescued']} | {stats['b_calls']} | {stats['b_success']} | "
                         f"{stats['probes']} | {stats['request_seconds']:.2f} |")
        deltas = [row['agent_b']['found']-row['program']['found'] for row in rows]
        block['wins_ties_losses'] = [sum(d > 0 for d in deltas), sum(d == 0 for d in deltas), sum(d < 0 for d in deltas)]
        summary[cohort] = block
    lines += ['', '逐题胜／平／负按合法图数计算。']
    for cohort, block in summary.items():
        lines.append(f"- {cohort}，Agent B 对程序补位为 {' / '.join(map(str, block['wins_ties_losses']))}。")
    lines += ['', '两个题集已经用于先前编号实验，属于同题策略复测。补位后生成历史可能分叉。', '']
    write_json(output/'summary.json', summary)
    (output/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    source = read(REFERENCE/'config.json')
    tight = read(OLD/'config.json')
    assert source['source_sha'] == digest(DATA) and tight['budget'] == 2
    assert tight['source_sha'] == digest(REFERENCE/'config.json') and tight['digest'] == source['model']
    cohorts = {'独立 16 题': tight['cases'], '正式 24 题': source['cases']}
    assert not set(tight['cases']) & set(source['cases'])
    tasks = {task['id']: task for task in read(DATA)}
    config = {'cases': tight['cases']+source['cases'], 'cohorts': cohorts, 'model': MODEL,
              'model_digest': source['model'], 'budget': 2, 'slots': SLOTS,
              'input_sha': digest(DATA), 'old_config_sha': digest(OLD/'config.json'),
              'code_sha': {name: digest(HERE/name) for name in (Path(__file__).name,
                  'run_numbered_agent_search.py', 'run_qwen_tight_budget.py', 'run_qwen_guided_edits.py')}}
    path = output/'config.json'
    if path.exists() and read(path) != config:
        raise ValueError('Output directory contains a different experiment configuration')
    write_json(path, config)
    records = {}
    with requests.Session() as session:
        session.trust_env = False
        tags = session.get(BASE_URL+'/api/tags', timeout=10)
        tags.raise_for_status()
        assert next(row for row in tags.json()['models'] if row['name'] == MODEL)['digest'] == source['model']
        for cohort, cases in cohorts.items():
            reference = OLD if cohort == '独立 16 题' else FORMAL
            for case in cases:
                previous = read(reference/case/'result.json')
                task, seed = tasks[case], previous['seed']
                anchor = program_anchor(NODES, LIMIT, task['observations'])
                assert anchor is not None and valid(anchor, task['observations'])
                results = {'old': previous['runs']['dual']}
                for method in METHODS:
                    folder = output/case/method
                    folder.mkdir(parents=True, exist_ok=True)
                    result_path = folder/'result.json'
                    if result_path.exists():
                        results[method] = read(result_path)
                        continue
                    rescue = make_rescue(method, session, folder, seed, task['observations'])
                    results[method] = run(task, anchor, 2, 'dual', session, folder, seed, rescue=rescue)
                    temporary = result_path.with_suffix('.tmp')
                    write_json(temporary, results[method])
                    temporary.replace(result_path)
                records[case] = results
                print(f"{len(records)}/{len(config['cases'])} {case} "
                      f"{results['program']['found']}/{results['agent_b']['found']}", flush=True)
    report(output, records, cohorts)


if __name__ == '__main__':
    main()
