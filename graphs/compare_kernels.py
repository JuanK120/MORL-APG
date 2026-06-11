from grakel import Graph
from grakel.kernels import WeisfeilerLehman, VertexHistogram, SubgraphMatching
from graphs.utils import parse_graph_dict, print_kernel_table
from sklearn.preprocessing import normalize

def compare_explanation_graphs(
    graph_dicts,
    node_label_mode="translation",
    edge_label_mode="action"
):
    graphs = [
        parse_graph_dict(
            g,
            node_label_mode=node_label_mode,
            edge_label_mode=edge_label_mode
        )
        for g in graph_dicts
    ]

    WLkernel = WeisfeilerLehman(
        base_graph_kernel=VertexHistogram,
        normalize=True
    )

    SMkernel = SubgraphMatching(normalize=True)

    K_wl = WLkernel.fit_transform(graphs)
    K_sm = SMkernel.fit_transform(graphs)

    return K_wl, K_sm