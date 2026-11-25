"""Answer-free completeness certificates for singleton-intervention DAGs."""
from math import comb

import networkx as nx


def complete_observation_count(nodes, max_edges, observations):
    """Return the exact count for full observations, or None for partial ones."""
    effects = {row['perturbed_node']: row['effects'] for row in observations}
    if set(effects) != set(nodes):
        return None
    if any(row['effects'] != effects[row['perturbed_node']] for row in observations):
        raise ValueError('Conflicting observations for a node')
    relation = nx.DiGraph()
    relation.add_nodes_from(nodes)
    relation.add_edges_from((a, b) for a in nodes for b in nodes if effects[a][b])
    if (not nx.is_directed_acyclic_graph(relation)
            or set(nx.transitive_closure_dag(relation).edges) != set(relation.edges)):
        raise ValueError('Complete observations must define a strict transitive reachability relation')
    required = nx.transitive_reduction(relation).number_of_edges()
    optional = relation.number_of_edges() - required
    return sum(comb(optional, k) for k in range(min(optional, max_edges-required)+1))
