import matplotlib.pyplot as plt
import networkx as nx

from textToDict import parse_transition_lines, HIGHWAY_GRAPHS, FRUIT_TREE_GRAPHS

def build_nx_graph(edges, edge_labels, node_labels):
    """
    Build a NetworkX DiGraph from:
      edges: dict[(src, dst) -> prob]
      edge_labels: dict[(src, dst) -> action (str)]
      node_labels: dict[node_id -> label (str)]
    """
    G = nx.DiGraph()

    # add nodes with labels
    for node_id, label in node_labels.items():
        G.add_node(node_id, label=label)

    # add edges with probability + action
    for (src, dst), prob in edges.items():
        action = edge_labels.get((src, dst), "")
        G.add_edge(src, dst, prob=prob, action=action)

    return G


def plot_graph(G, node_labels, edges, edge_labels, title="Policy Graph"):
    """
    Plot the graph with node labels and edge labels "aX, p=Y".
    """

    plt.figure(figsize=(8, 6))
    # Layout: you can change to circular_layout, kamada_kawai_layout, etc.
    pos = nx.spring_layout(G, seed=42)

    # Node labels come from node_labels dict
    nx.draw(
        G,
        pos,
        with_labels=True,
        labels=node_labels,
        node_size=800,
        font_size=8,
        arrows=True,
    )

    # Build edge label text
    edge_label_dict = {}
    for (src, dst), prob in edges.items():
        action = edge_labels.get((src, dst), "")
        edge_label_dict[(src, dst)] = f"a{action}, p={prob:.2f}"

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_label_dict, font_size=6)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"./plots/{title.replace(' ', '_')}.png", dpi=300)


if __name__ == "__main__":
    
    # For highway graphs

    graphIndexes = [2, 6, 4, 5,]

    for i in graphIndexes:
        highwayLines = HIGHWAY_GRAPHS[i]

        edges, edge_labels, node_labels = parse_transition_lines(highwayLines)
        G = build_nx_graph(edges, edge_labels, node_labels)

        plot_graph(G, node_labels, edges, edge_labels, title=f"Highway Graph {i}")

    # For fruitTree graphs

    graphIndexes = [1, 3]
    for i in graphIndexes:
        fruitTreeLines = FRUIT_TREE_GRAPHS[i] 

        edges, edge_labels, node_labels = parse_transition_lines(fruitTreeLines)
        G = build_nx_graph(edges, edge_labels, node_labels)

        plot_graph(G, node_labels, edges, edge_labels, title=f"Fruit Tree Graph {i}")
