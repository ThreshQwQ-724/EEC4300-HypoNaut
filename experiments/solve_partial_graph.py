"""Generate distinct DAG hypotheses from observations, without answer labels."""
import argparse
from itertools import combinations, islice
import json
from pathlib import Path

import networkx as nx

from graph_completion import complete_observation_count
from run_feedback_study import observations_check
from run_partial_graph_pipeline import Frontier, SLOTS, program_anchor

DEFAULT_CHECKS_PER_SLOT = 12
CHECK_BATCH_SIZE = 4
MAX_SLOTS = 128
CONSTRUCTIVE_ORDER_LIMIT = 32


def validate_anchor(nodes, max_edges, observations, edges):
    if any(len(edge) != 2 or any(node not in nodes for node in edge) for edge in edges):
        raise ValueError('Anchor edges must connect two listed nodes')
    canonical = tuple(sorted({tuple(edge) for edge in edges}))
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(canonical)
    if len(canonical) > max_edges or not nx.is_directed_acyclic_graph(graph):
        raise ValueError('Anchor must be acyclic and within the edge limit')
    if observations_check(nodes, observations, canonical):
        raise ValueError('Anchor does not satisfy the supplied observations')
    return canonical


def constructive_prefix(nodes, max_edges, observations, anchor, target):
    """Generate legal supergraphs by preserving a topological order and observed effects."""
    graphs = {anchor: None}
    if target == 1:
        return graphs, 0, 0
    affected_by = {node: {row['perturbed_node'] for row in observations if row['effects'][node]}
                   for node in nodes}
    sources = {row['perturbed_node'] for row in observations}
    reaches_or_is = {node: affected_by[node] | ({node} & sources) for node in nodes}
    dag = nx.DiGraph()
    dag.add_nodes_from(nodes)
    dag.add_edges_from(anchor)
    if len(anchor) == max_edges:
        return graphs, 0, 0
    generated = 0
    orders = 0
    order_limit = 1 if sources == set(nodes) else CONSTRUCTIVE_ORDER_LIMIT
    for order in islice(nx.all_topological_sorts(dag), order_limit):
        orders += 1
        # Every edge preserves each observed source's allowed descendant set.
        optional = [(a, b) for a, b in combinations(order, 2)
                    if (a, b) not in dag.edges and reaches_or_is[a] <= affected_by[b]]
        for count in range(1, min(len(optional), max_edges-len(anchor))+1):
            for extra in combinations(optional, count):
                generated += 1
                graphs.setdefault(tuple(sorted(anchor+extra)), extra)
                if len(graphs) == target:
                    return graphs, orders, generated
    return graphs, orders, generated


def solve(nodes, max_edges, observations, anchor=None, slots=SLOTS, *,
          checks_per_slot=DEFAULT_CHECKS_PER_SLOT, prune=True, adaptive=True, constructive=False):
    """Return verified unique candidates, checking in batches only when needed."""
    if not nodes or len(set(nodes)) != len(nodes) or not all(isinstance(n, str) and n for n in nodes):
        raise ValueError('Expected a nonempty list of distinct node names')
    if not isinstance(max_edges, int) or not 0 <= max_edges <= len(nodes)*(len(nodes)-1)//2:
        raise ValueError('Invalid DAG edge limit')
    if type(slots) is not int or not 1 <= slots <= MAX_SLOTS:
        raise ValueError(f'Slots must be an integer between 1 and {MAX_SLOTS}')
    if type(checks_per_slot) is not int or checks_per_slot < 1:
        raise ValueError('Neighbor checks per slot must be a positive integer')
    for row in observations:
        source = row['perturbed_node']
        effects = row['effects']
        if (source not in nodes or set(effects) != set(nodes) or effects[source] != 0
                or any(value not in (0, 1) for value in effects.values())):
            raise ValueError('Each observation needs one binary effect per node and zero for its source')
    source = 'program' if anchor is None else 'provided'
    if anchor is None:
        anchor = program_anchor(nodes, max_edges, observations)
    else:
        anchor = validate_anchor(nodes, max_edges, observations, anchor)
    if anchor is None:
        return {'status': 'no_valid_anchor', 'proposals': [], 'checks': 0}
    exact_count = complete_observation_count(nodes, max_edges, observations)
    if constructive:
        target = min(slots, exact_count) if exact_count is not None else slots
        graphs, orders, generated = constructive_prefix(nodes, max_edges, observations, anchor, target)
        proposals = [{'edges': graph, 'origin': source if extra is None else 'observed_effect_preserving',
                      **({'added_edges': extra} if extra is not None else {})} for graph, extra in graphs.items()]
        fallback_checks = []
        fallback_success_slots = []
        fallback_used = len(proposals) < target
        exhausted = False
        if fallback_used:
            frontier = Frontier(nodes, max_edges, observations, anchor, prune=prune)
            for graph in graphs:
                if graph != anchor:
                    frontier.register(graph)
            for slot in range(1, target-len(proposals)+1):
                used = 0
                if adaptive:
                    while not frontier.pool and frontier.queue and used < checks_per_slot:
                        used += frontier.expand(1)
                else:
                    used = frontier.expand(checks_per_slot)
                fallback_checks.append(used)
                options = frontier.shortlist()
                if options:
                    graph = options[0]
                    proposals.append({'edges': graph, 'origin': 'single_edge_edit', 'edit': frontier.emit(graph)})
                    fallback_success_slots.append(slot)
                elif not frontier.queue:
                    exhausted = True
                    break
        completion = 'unproven'
        if len(proposals) == exact_count:
            completion = 'all_legal_graphs_found'
        elif exhausted:
            completion = 'single_edge_component_exhausted'
        constructed_count = len(graphs)
        proposal_slots = list(range(1, constructed_count+1))
        # The fallback probes one candidate at a time in adaptive mode. A failed slot
        # may precede a later success, so retain its actual slot index.
        proposal_slots.extend(constructed_count+i for i in fallback_success_slots)
        return {'status': 'budget_reached' if len(proposals) == slots else 'fewer_candidates_in_current_budget',
                'proposals': proposals, 'checks': 1+sum(fallback_checks), 'proposal_slots': proposal_slots,
                'max_neighbor_checks_per_slot': checks_per_slot, 'observation_pruning': prune,
                'adaptive_budget': adaptive, 'neighbor_checks_by_slot': [0]*(constructed_count-1)+fallback_checks,
                'completion': completion, 'exact_legal_graph_count': exact_count,
                'constructed': constructed_count-1, 'construction_orders': orders,
                'construction_candidates': generated, 'fallback_used': fallback_used}
    frontier = Frontier(nodes, max_edges, observations, anchor, prune=prune)
    proposals = [{'edges': anchor, 'origin': source}]
    proposal_slots = [1]
    checks = 1
    slot_checks = []
    exhausted = False
    for slot in range(2, slots+1):
        if len(proposals) == exact_count:
            break
        if adaptive:
            used = 0
            while not frontier.pool and frontier.queue and used < checks_per_slot:
                used += frontier.expand(min(CHECK_BATCH_SIZE, checks_per_slot-used))
        else:
            used = frontier.expand(checks_per_slot)
        checks += used
        slot_checks.append(used)
        options = frontier.shortlist()
        if not options:
            if not frontier.queue:
                exhausted = True
                break
            continue
        edges = options[0]
        proposals.append({'edges': edges, 'origin': 'single_edge_edit', 'edit': frontier.emit(edges)})
        proposal_slots.append(slot)
    status = 'budget_reached' if len(proposals) == slots else 'fewer_candidates_in_current_budget'
    if len(proposals) == exact_count:
        completion = 'all_legal_graphs_found'
    elif exhausted or (not frontier.queue and not frontier.pool):
        completion = 'single_edge_component_exhausted'
    else:
        completion = 'unproven'
    return {'status': status, 'proposals': proposals, 'checks': checks,
            'max_neighbor_checks_per_slot': checks_per_slot, 'observation_pruning': prune,
            'adaptive_budget': adaptive, 'neighbor_checks_by_slot': slot_checks,
            'proposal_slots': proposal_slots, 'completion': completion, 'exact_legal_graph_count': exact_count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='JSON with nodes, max_edges, observations, and optional anchor')
    parser.add_argument('--case', help='Read a single observation_set_id from a benchmark test JSON')
    parser.add_argument('--slots', type=int, default=SLOTS)
    parser.add_argument('--checks-per-slot', type=int, default=DEFAULT_CHECKS_PER_SLOT)
    parser.add_argument('--no-pruning', action='store_true')
    parser.add_argument('--fixed-budget', action='store_true', help='Expand at every slot up to the check limit')
    parser.add_argument('--constructive', action='store_true', help='Generate observation-preserving supergraphs first')
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding='utf-8'))
    if args.case:
        try:
            case = next(r for r in data['datasets'] if r['observation_set_id'] == args.case)
        except StopIteration:
            parser.error(f'Case not found: {args.case}')
        data = {'nodes': data['metadata']['nodes'], 'max_edges': data['metadata']['max_edges'],
                'observations': case['observations']}
    result = solve(data['nodes'], data['max_edges'], data['observations'], data.get('anchor'), args.slots,
                   checks_per_slot=args.checks_per_slot, prune=not args.no_pruning,
                   adaptive=not args.fixed_budget, constructive=args.constructive)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
