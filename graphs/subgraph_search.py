import networkx as nx
from networkx.algorithms import isomorphism


def graph_dict_to_nx(graph_dict):
    G = nx.DiGraph()

    groups = graph_dict["groups"]

    if isinstance(groups, dict):
        group_iter = groups.items()
    else:
        group_iter = enumerate(groups)

    for group_id, group_info in group_iter:
        G.add_node(
            group_id,
            label=str(group_info)
        )

    for edge in graph_dict["edges"]:
        G.add_edge(
            edge["from_group"],
            edge["to_group"],
            action=edge.get("action"),
            probability=edge.get("probability")
        )

    return G


## first approach using direct comparison of graph dicts

def transition_signature(edge):
    source = edge.get("source", edge.get("from_group", edge.get("origin")))
    target = edge.get("target", edge.get("to_group", edge.get("destination")))
    action = edge.get("action", edge.get("label"))

    if source is None or target is None:
        raise KeyError(f"Could not find source/target in edge: {edge}")

    return source, target, action

def get_transition_set(graph_dict):
    return {
        transition_signature(edge)
        for edge in graph_dict["edges"]
    }


def compare_transition_sets(g1, g2):
    e1 = get_transition_set(g1)
    e2 = get_transition_set(g2)

    common = e1 & e2
    only_g1 = e1 - e2
    only_g2 = e2 - e1

    return common, only_g1, only_g2

## seccond approach using networkx isomorphism

def are_graphs_isomorphic(g1, g2):
    nx_g1 = graph_dict_to_nx(g1)
    nx_g2 = graph_dict_to_nx(g2)

    # Define a node matcher that compares the 'label' attribute
    node_matcher = isomorphism.categorical_node_match('label', None)

    # Define an edge matcher that compares the 'action' attribute
    edge_matcher = isomorphism.categorical_edge_match('action', None)

    # Create a DiGraphMatcher object
    gm = isomorphism.DiGraphMatcher(nx_g1, nx_g2, node_match=node_matcher, edge_match=edge_matcher)

    return gm.is_isomorphic()

def get_isomorphism_mapping(g1, g2):
    nx_g1 = graph_dict_to_nx(g1)
    nx_g2 = graph_dict_to_nx(g2)

    # Define a node matcher that compares the 'label' attribute
    node_matcher = isomorphism.categorical_node_match('label', None)

    # Define an edge matcher that compares the 'action' attribute
    edge_matcher = isomorphism.categorical_edge_match('action', None)

    # Create a DiGraphMatcher object
    gm = isomorphism.DiGraphMatcher(nx_g1, nx_g2, node_match=node_matcher, edge_match=edge_matcher)

    if gm.is_isomorphic():
        return gm.mapping  # Returns a dictionary mapping nodes in g1 to nodes in g2
    else:
        return None  # Not isomorphic

def get_maximum_common_subgraph(g1, g2):
    nx_g1 = graph_dict_to_nx(g1)
    nx_g2 = graph_dict_to_nx(g2)

    node_matcher = isomorphism.categorical_node_match("label", None)
    edge_matcher = isomorphism.categorical_edge_match("action", None)

    ismags = isomorphism.ISMAGS(
        nx_g1,
        nx_g2,
        node_match=node_matcher,
        edge_match=edge_matcher
    )

    mappings = list(ismags.largest_common_subgraph())

    if not mappings:
        return None, None

    mapping = mappings[0]
    common_nodes_g1 = list(mapping.keys())
    common_subgraph_g1 = nx_g1.subgraph(common_nodes_g1).copy()

    return common_subgraph_g1, mapping