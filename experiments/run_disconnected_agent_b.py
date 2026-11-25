"""Test independent Qwen path rewiring on exhaustively labeled capped DAGs."""
import argparse
from itertools import combinations, permutations
from pathlib import Path
import random
from time import perf_counter

import requests

from audit_single_edge_components import component_sizes
from common import digest, load_upstream
from run_formal_agent_debate import REFERENCE, SECOND_AGENT_SEED_OFFSET
from run_four_groups import BASE_URL, MODEL
from run_local_causal import write_json
from run_partial_graph_pipeline import forbidden_edges, program_anchor
from run_qwen_guided_edits import decode, read, request, schema
from run_feedback_study import graph_line


def enumerate_answers(nodes, cap):
    edges = list(permutations(nodes, 2))
    bits = {edge: 1 << i for i, edge in enumerate(edges)}
    graphs = []
    for count in range(cap+1):
        for graph in combinations(edges, count):
            reach = {node: set() for node in nodes}
            for a, b in graph:
                reach[a].add(b)
            for mid in nodes:
                for source in nodes:
                    if mid in reach[source]:
                        reach[source].update(reach[mid])
            if all(node not in targets for node, targets in reach.items()):
                graphs.append((graph, reach))
    return bits, graphs


def cases(benchmark):
    for n, cap in ((4, 2), (6, 3)):
        nodes = list('ABCDEF'[:n])
        benchmark.nodes, benchmark.max_edges = nodes, cap
        bits, graphs = enumerate_answers(nodes, cap)
        for source in nodes:
            targets_list = list(combinations([node for node in nodes if node != source], cap))
            if n == 6:
                targets_list = random.Random(921+ord(source)).sample(targets_list, 4)
            for targets in targets_list:
                obs = [{'perturbed_node': source, 'effects': {node: int(node in targets) for node in nodes}}]
                answers = {g for g, reach in graphs if reach[source] == set(targets)}
                anchor = program_anchor(nodes, cap, obs)
                assert anchor in answers
                masks = {sum(bits[edge] for edge in g) for g in answers}
                sizes = component_sizes(masks, bits.values(), sum(bits[edge] for edge in anchor))
                assert len(sizes) > 1 and sizes[0] == 1
                yield nodes, cap, source, targets, obs, anchor, answers, sizes


def proposal(session, path, seed, parent, additions, prompt, role):
    fmt = schema({name: {'type': 'array', 'minItems': 1, 'maxItems': 2, 'uniqueItems': True,
                         'items': {'enum': ['->'.join(edge) for edge in sorted(edges)]}}
                  for name, edges in (('remove', parent), ('add', additions))})
    call = request(session, path, prompt+role, seed, fmt)
    answer = decode(call)
    removed, added = set(), set()
    for name, edges, destination in (('remove', parent, removed), ('add', additions, added)):
        items = answer.get(name)
        if not isinstance(items, list) or not 1 <= len(items) <= 2:
            return None, call['seconds']
        lookup = {'->'.join(edge): edge for edge in edges}
        if any(not isinstance(item, str) or item not in lookup for item in items) or len(set(items)) != len(items):
            return None, call['seconds']
        destination.update(lookup[item] for item in items)
    return tuple(sorted((set(parent)-removed) | added)), call['seconds']


def run(benchmark, session, folder, seed, task):
    nodes, cap, source, targets, obs, anchor, answers, sizes = task
    additions = set(permutations(nodes, 2))-forbidden_edges(obs)-set(anchor)
    prompt = (f'Nodes {", ".join(nodes)}. At most {cap} edges. Under intervention on {source}, '
              f'exactly {", ".join(targets)} must be reachable by directed paths. All other nodes '
              'must be unreachable. The graph must be acyclic. The existing graph has no legal '
              'single-edge edits. Propose a new graph by simultaneously removing 1–2 existing '
              'edges and adding 1–2 new edges. Return JSON arrays remove and add with values '
              f'like "A->B".\nExisting graph: {graph_line(anchor)}\n')
    roles = {'A': 'Find an alternative mechanism.', 'repeat': 'Find an alternative mechanism.',
             'B': 'Use a relay mechanism. Select an affected node to relay the effect to another affected '
                  'node. Reroute a direct edge through that relay while keeping the source connected to both.'}
    drafts = {}
    for label, role in roles.items():
        graph, seconds = proposal(session, folder/f'{label}.json',
                                  seed+(SECOND_AGENT_SEED_OFFSET if label != 'A' else 0),
                                  anchor, additions, prompt, role)
        drafts[label] = {'graph': graph, 'seconds': seconds}
    # A constructive baseline using only the observed route and a known legal parent.
    # Replacing a root-to-target edge by a relay-to-target edge produces a different tree.
    program = []
    for relay, target in permutations(targets, 2):
        graph = tuple(sorted((set(anchor)-{(source, target)}) | {(relay, target)}))
        if graph not in program:
            program.append(graph)
        if len(program) == 2:
            break
    assert len(program) == 2
    scores = {}
    for label, picks in (('A', [drafts['A']['graph']]),
                         ('repeat', [drafts['A']['graph'], drafts['repeat']['graph']]),
                         ('dual', [drafts['A']['graph'], drafts['B']['graph']]), ('program', program)):
        probes = {g for g in picks if g is not None and g != anchor and len(g) <= cap}
        hits = probes & answers
        for graph in probes:
            parsed = benchmark.parse_llm_response(graph_line(graph))
            assert (parsed is not None and benchmark.validate_hypothesis(parsed, obs)) == (graph in hits)
        scores[label] = {'probes': len(probes), 'hits': len(hits), 'new_component_graphs': sorted(hits)}
    return {'id': folder.name, 'nodes': nodes, 'cap': cap, 'source': source, 'targets': targets,
            'anchor': anchor, 'component_sizes': sizes, 'answers': sorted(answers),
            'drafts': drafts, 'program': program, 'scores': scores}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    source = read(REFERENCE/'config.json')
    module = load_upstream('causal', 'run_causal_benchmark.py')
    assert digest(module.__file__) == source['evaluator_sha']
    benchmark = module.CausalBenchmarkEnhanced()
    start = perf_counter()
    rows = []
    with requests.Session() as session:
        session.trust_env = False
        tags = session.get(BASE_URL+'/api/tags', timeout=10)
        tags.raise_for_status()
        assert next(row for row in tags.json()['models'] if row['name'] == MODEL)['digest'] == source['model']
        for i, task in enumerate(cases(benchmark), 1):
            folder = output/f'case_{i:02}'
            folder.mkdir(exist_ok=True)
            row = run(benchmark, session, folder, source['seeds'][(i-1) % len(source['seeds'])]+i, task)
            rows.append(row)
            print(i, [row['scores'][k]['hits'] for k in ('A', 'repeat', 'dual', 'program')], flush=True)
    assert len(rows) == 36
    summary = {}
    for n in (4, 6):
        group = [r for r in rows if len(r['nodes']) == n]
        summary[n] = {name: {**{key: sum(r['scores'][name][key] for r in group) for key in ('probes', 'hits')},
                             'cases_hit': sum(r['scores'][name]['hits'] > 0 for r in group),
                             'rescued_a_failure': sum(r['scores'][name]['hits'] > 0 and r['scores']['A']['hits'] == 0
                                                      for r in group)}
                      for name in ('A', 'repeat', 'dual', 'program')}
    write_json(output/'results.json', {'model': MODEL, 'model_digest': source['model'],
        'runner_sha': digest(__file__), 'component_code_sha': digest(Path(__file__).with_name('audit_single_edge_components.py')),
        'evaluator_sha': digest(module.__file__), 'summary': summary, 'rows': rows, 'seconds': perf_counter()-start})
    print(summary, flush=True)


if __name__ == '__main__':
    main()
