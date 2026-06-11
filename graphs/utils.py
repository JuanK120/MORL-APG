import re
import numpy as np
from grakel import Graph


highwayGraph0 = [
    'Group 1 to Group 1 with p=0.3333333333333333 and action 1',
    'Group 1 to Group 8 with p=0.6666666666666666 and action 1',
    'Group 2 to Group 9 with p=1.0 and action 2',
    'Group 3 to Group 1 with p=0.0016722408026755853 and action 2',
    'Group 3 to Group 2 with p=0.0008361204013377926 and action 2',
    'Group 3 to Group 3 with p=0.9565217391304645 and action 2',
    'Group 3 to Group 8 with p=0.027591973244147146 and action 2',
    'Group 3 to Group 9 with p=0.01337792642140468 and action 2',
    'Group 4 to Group 3 with p=0.5 and action 2',
    'Group 4 to Group 4 with p=0.3333333333333333 and action 2',
    'Group 4 to Group 8 with p=0.16666666666666666 and action 2',
    'Group 5 to Group 3 with p=0.5 and action 2',
    'Group 5 to Group 5 with p=0.5 and action 2',
    'Group 6 to Group 8 with p=1.0 and action 2',
    'Group 7 to Group 6 with p=1.0 and action 2',
    'Group 8 to Group 3 with p=0.49295774647887297 and action 3',
    'Group 8 to Group 8 with p=0.46478873239436597 and action 3',
    'Group 8 to Group 9 with p=0.04225352112676056 and action 4'
]

highwayGraph1 = [
    "Group 1 to Group 3 with p=1.0 and action 2",
    "Group 2 to Group 3 with p=1.0 and action 2",
    "Group 3 to Group 1 with p=0.0009276437847866419 and action 2",
    "Group 3 to Group 2 with p=0.0009276437847866419 and action 2",
    "Group 3 to Group 3 with p=0.9684601113172373 and action 1",
    "Group 3 to Group 4 with p=0.0055658627087198514 and action 2",
    "Group 3 to Group 5 with p=0.0018552875695732839 and action 2",
    "Group 3 to Group 6 with p=0.0018552875695732839 and action 2",
    "Group 3 to Group 7 with p=0.0018552875695732839 and action 2",
    "Group 3 to Group 8 with p=0.01855287569573284 and action 1",
    "Group 4 to Group 3 with p=0.7999999999999999 and action 1",
    "Group 4 to Group 4 with p=0.2 and action 2",
    "Group 5 to Group 3 with p=1.0 and action 2",
    "Group 6 to Group 3 with p=1.0 and action 1",
    "Group 7 to Group 4 with p=1.0 and action 2",
]

highwayGraph2 = [
    "Group 1 to Group 1 with p=0.4347826086956521 and action 2",
    "Group 1 to Group 4 with p=0.043478260869565216 and action 2",
    "Group 1 to Group 5 with p=0.4782608695652173 and action 2",
    "Group 1 to Group 6 with p=0.043478260869565216 and action 2",
    "Group 2 to Group 1 with p=0.5 and action 2",
    "Group 2 to Group 5 with p=0.5 and action 2",
    "Group 3 to Group 4 with p=0.5 and action 2",
    "Group 3 to Group 5 with p=0.5 and action 2",
    "Group 4 to Group 3 with p=0.5 and action 2",
    "Group 4 to Group 5 with p=0.5 and action 2",
    "Group 5 to Group 1 with p=0.004158004158004158 and action 2",
    "Group 5 to Group 2 with p=0.000693000693000693 and action 2",
    "Group 5 to Group 3 with p=0.000693000693000693 and action 4",
    "Group 5 to Group 5 with p=0.9812889812889996 and action 3",
    "Group 5 to Group 6 with p=0.013167013167013165 and action 3",
]

highwayGraph3 = [
    "Group 1 to Group 2 with p=0.5 and action 1",
    "Group 1 to Group 6 with p=0.5 and action 1",
    "Group 2 to Group 1 with p=0.16666666666666666 and action 1",
    "Group 2 to Group 3 with p=0.5 and action 1",
    "Group 2 to Group 5 with p=0.16666666666666666 and action 1",
    "Group 2 to Group 6 with p=0.16666666666666666 and action 1",
    "Group 3 to Group 2 with p=0.4 and action 1",
    "Group 3 to Group 3 with p=0.2 and action 1",
    "Group 3 to Group 5 with p=0.2 and action 1",
    "Group 3 to Group 6 with p=0.2 and action 1",
    "Group 4 to Group 5 with p=1.0 and action 2",
    "Group 5 to Group 2 with p=0.0022123893805309734 and action 2",
    "Group 5 to Group 3 with p=0.0011061946902654867 and action 3",
    "Group 5 to Group 4 with p=0.0011061946902654867 and action 2",
    "Group 5 to Group 5 with p=0.9026548672566423 and action 2",
    "Group 5 to Group 6 with p=0.07743362831858416 and action 2",
    "Group 5 to Group 7 with p=0.015486725663716814 and action 2",
    "Group 6 to Group 1 with p=0.001466275659824047 and action 3",
    "Group 6 to Group 2 with p=0.001466275659824047 and action 3",
    "Group 6 to Group 5 with p=0.10263929618768322 and action 3",
    "Group 6 to Group 6 with p=0.8856304985337237 and action 3",
    "Group 6 to Group 7 with p=0.008797653958944282 and action 4",
]

highwayGraph4 = [
    "Group 1 to Group 1 with p=0.4333333333333333 and action 0",
    "Group 1 to Group 4 with p=0.16666666666666666 and action 0",
    "Group 1 to Group 5 with p=0.03333333333333333 and action 0",
    "Group 1 to Group 6 with p=0.06666666666666667 and action 0",
    "Group 1 to Group 7 with p=0.2333333333333333 and action 0",
    "Group 1 to Group 8 with p=0.06666666666666667 and action 0",
    "Group 2 to Group 2 with p=0.6451612903225803 and action 0",
    "Group 2 to Group 4 with p=0.03225806451612903 and action 0",
    "Group 2 to Group 6 with p=0.06451612903225806 and action 0",
    "Group 2 to Group 7 with p=0.22580645161290322 and action 0",
    "Group 2 to Group 8 with p=0.03225806451612903 and action 0",
    "Group 3 to Group 3 with p=0.3333333333333333 and action 0",
    "Group 3 to Group 7 with p=0.6666666666666666 and action 0",
    "Group 4 to Group 1 with p=0.11538461538461539 and action 0",
    "Group 4 to Group 2 with p=0.07692307692307693 and action 0",
    "Group 4 to Group 4 with p=0.3076923076923077 and action 0",
    "Group 4 to Group 5 with p=0.15384615384615385 and action 0",
    "Group 4 to Group 6 with p=0.038461538461538464 and action 0",
    "Group 4 to Group 7 with p=0.2692307692307693 and action 0",
    "Group 4 to Group 8 with p=0.038461538461538464 and action 0",
    "Group 5 to Group 1 with p=0.13636363636363635 and action 0",
    "Group 5 to Group 2 with p=0.045454545454545456 and action 0",
    "Group 5 to Group 4 with p=0.13636363636363635 and action 0",
    "Group 5 to Group 5 with p=0.40909090909090917 and action 0",
    "Group 5 to Group 6 with p=0.09090909090909091 and action 0",
    "Group 5 to Group 7 with p=0.18181818181818182 and action 0",
    "Group 6 to Group 1 with p=0.07407407407407407 and action 0",
    "Group 6 to Group 2 with p=0.037037037037037035 and action 0",
    "Group 6 to Group 3 with p=0.07407407407407407 and action 0",
    "Group 6 to Group 4 with p=0.037037037037037035 and action 0",
    "Group 6 to Group 5 with p=0.1111111111111111 and action 0",
    "Group 6 to Group 6 with p=0.48148148148148145 and action 0",
    "Group 6 to Group 7 with p=0.18518518518518517 and action 0",
    "Group 7 to Group 1 with p=0.01680672268907563 and action 1",
    "Group 7 to Group 2 with p=0.010504201680672268 and action 0",
    "Group 7 to Group 4 with p=0.018907563025210083 and action 2",
    "Group 7 to Group 5 with p=0.010504201680672268 and action 1",
    "Group 7 to Group 6 with p=0.012605042016806721 and action 1",
    "Group 7 to Group 7 with p=0.9075630252100815 and action 2",
    "Group 7 to Group 8 with p=0.02310924369747899 and action 2",
]

highwayGraph5 = [
    "Group 1 to Group 2 with p=1.0 and action 2",
    "Group 2 to Group 1 with p=0.0011547344110854503 and action 2",
    "Group 2 to Group 2 with p=0.9399538106235691 and action 1",
    "Group 2 to Group 4 with p=0.039260969976905306 and action 1",
    "Group 2 to Group 5 with p=0.0011547344110854503 and action 2",
    "Group 2 to Group 7 with p=0.018475750577367205 and action 1",
    "Group 3 to Group 2 with p=0.3333333333333333 and action 2",
    "Group 3 to Group 3 with p=0.3333333333333333 and action 2",
    "Group 3 to Group 6 with p=0.3333333333333333 and action 2",
    "Group 4 to Group 2 with p=0.27350427350427353 and action 3",
    "Group 4 to Group 3 with p=0.008547008547008548 and action 4",
    "Group 4 to Group 4 with p=0.6837606837606838 and action 4",
    "Group 4 to Group 7 with p=0.03418803418803419 and action 3",
    "Group 5 to Group 2 with p=1.0 and action 4",
    "Group 6 to Group 2 with p=1.0 and action 4",
]

highwayGraph6 = [
    "Group 1 to Group 1 with p=0.9874686716791788 and action 3",
    "Group 1 to Group 5 with p=0.012531328320802004 and action 4",
    "Group 2 to Group 1 with p=1.0 and action 2",
    "Group 3 to Group 2 with p=1.0 and action 2",
    "Group 4 to Group 3 with p=1.0 and action 2",
]

fruitTreeGraph0 = [
    'Group 1 to Group 2 with p=0.9999999999999999 and action 0',
    'Group 2 to Group 9 with p=0.9999999999999999 and action 0',
    'Group 3 to Group 12 with p=1.0 and action 1',
    'Group 4 to Group 1 with p=0.9999999999999999 and action 1',
    'Group 5 to Group 12 with p=1.0 and action 1',
    'Group 6 to Group 12 with p=1.0000000000000002 and action 1',
    'Group 7 to Group 12 with p=1.0 and action 1',
    'Group 8 to Group 13 with p=0.9999999999999999 and action 0',
    'Group 9 to Group 12 with p=1.0 and action 1',
    'Group 10 to Group 13 with p=1.0 and action 0',
    'Group 11 to Group 13 with p=1.0 and action 0',
    'Group 12 to Group 13 with p=1.0 and action 0',
    'Group 13 to Group 14 with p=0.9999999999999999 and action 1'
]

fruitTreeGraph1 = [
    'Group 1 to Group 1 with p=0.49999999999999994 and action 1',
    'Group 1 to Group 3 with p=0.49999999999999994 and action 1',
    'Group 2 to Group 4 with p=0.9999999999999997 and action 1',
    'Group 3 to Group 4 with p=1.0 and action 1',
    'Group 4 to Group 8 with p=1.0 and action 0',
    'Group 5 to Group 8 with p=1.0 and action 1',
    'Group 6 to Group 8 with p=1.0 and action 1',
    'Group 7 to Group 9 with p=1.0 and action 0',
    'Group 8 to Group 8 with p=0.45833333333333337 and action 1',
    'Group 8 to Group 9 with p=0.5416666666666666 and action 0'
]

fruitTreeGraph2 = [
    'Group 1 to Group 3 with p=0.9999999999999999 and action 0',
    'Group 2 to Group 4 with p=0.9999999999999997 and action 1',
    'Group 3 to Group 4 with p=1.0 and action 1',
    'Group 4 to Group 7 with p=0.9999999999999999 and action 0',
    'Group 5 to Group 9 with p=1.0 and action 1',
    'Group 6 to Group 9 with p=1.0000000000000002 and action 1',
    'Group 7 to Group 9 with p=0.4545454545454546 and action 1',
    'Group 7 to Group 10 with p=0.5454545454545455 and action 0',
    'Group 8 to Group 10 with p=0.9999999999999999 and action 0',
    'Group 9 to Group 10 with p=1.0 and action 0',
    'Group 10 to Group 11 with p=0.9999999999999999 and action 0'
]

fruitTreeGraph3 = [
    'Group 1 to Group 1 with p=0.6666666666666666 and action 0',
    'Group 1 to Group 5 with p=0.3333333333333333 and action 1',
    'Group 2 to Group 8 with p=1.0 and action 0',
    'Group 3 to Group 8 with p=1.0 and action 0',
    'Group 4 to Group 8 with p=1.0 and action 0',
    'Group 5 to Group 8 with p=1.0 and action 0',
    'Group 6 to Group 7 with p=1.0 and action 1',
    'Group 7 to Group 9 with p=1.0 and action 0',
    'Group 8 to Group 7 with p=1.0 and action 1'
]

HIGHWAY_GRAPHS = [

    highwayGraph0,
    highwayGraph1,
    highwayGraph2,
    highwayGraph3,
    highwayGraph4,
    highwayGraph5,
    highwayGraph6,

]

FRUIT_TREE_GRAPHS = [

    fruitTreeGraph0,
    fruitTreeGraph1,
    fruitTreeGraph2,
    fruitTreeGraph3,

]


# textToDict.py
import re

def parse_transition_lines(lines):
    """
    lines: list[str] like:
      'Group 1 to Group 2 with p=1.0 and action 2'
    Returns:
      edges: dict[(int,int) -> float]   # transition probability
      edge_labels: dict[(int,int) -> str]  # action as string
      node_labels: dict[int -> str]     # WL vertex labels
    """
    edges = {}
    edge_labels = {}
    node_labels = {}

    pattern = re.compile(
        r"Group\s+(?P<src>\d+)\s+to\s+Group\s+(?P<dst>\d+)\s+with p=(?P<prob>[0-9eE\.\-]+)\s+and action\s+(?P<action>\d+)"
    )

    for line in lines:
        line = line.strip()
        if not line:
            continue
        m = pattern.match(line)
        if not m:
            raise ValueError(f"Cannot parse line: {line}")

        src = int(m.group("src"))
        dst = int(m.group("dst"))
        prob = float(m.group("prob"))
        action = m.group("action")   # keep as string label

        edges[(src, dst)] = prob
        edge_labels[(src, dst)] = action

        # Create simple categorical node labels for WL
        if src not in node_labels:
            node_labels[src] = f"G{src}"
        if dst not in node_labels:
            node_labels[dst] = f"G{dst}"

    return edges, edge_labels, node_labels

def parse_graph_dict(graph_dict, node_label_mode="translation", edge_label_mode="action"): 

    edges = {}
    edge_labels = {}
    node_labels = {}

    for group in graph_dict.get("groups", []):
        group_id = int(group["group"])

        if node_label_mode == "translation":
            label = group.get("translation", f"G{group_id}")

        elif node_label_mode == "important_features":
            features = group.get("important_features", [])
            label = "|".join(features) if features else "none"

        elif node_label_mode == "critical":
            label = f"critical={int(group.get('critical_value', 0))}"

        elif node_label_mode == "group_id":
            label = f"G{group_id}"

        else:
            raise ValueError(f"Unknown node_label_mode: {node_label_mode}")

        node_labels[group_id] = label

    for edge in graph_dict.get("edges", []):
        src = int(edge["from_group"])
        dst = int(edge["to_group"])
        prob = float(edge.get("probability", 1.0))
        action = edge.get("action", "unknown")

        edges[(src, dst)] = prob

        if edge_label_mode == "action":
            edge_labels[(src, dst)] = str(action)

        elif edge_label_mode == "probability_bin":
            edge_labels[(src, dst)] = probability_to_bin(prob)

        elif edge_label_mode == "action_probability_bin":
            edge_labels[(src, dst)] = f"a{action}_p{probability_to_bin(prob)}"

        elif edge_label_mode is None:
            pass

        else:
            raise ValueError(f"Unknown edge_label_mode: {edge_label_mode}")

        # Handle edges to groups that may not appear in graph_dict["groups"]
        if src not in node_labels:
            node_labels[src] = f"G{src}"
        if dst not in node_labels:
            node_labels[dst] = f"G{dst}"

    if edge_label_mode is None:
        return Graph(edges, node_labels=node_labels)

    return Graph(edges, node_labels=node_labels, edge_labels=edge_labels)

def probability_to_bin(prob):
    if prob < 0.25:
        return "low"
    elif prob < 0.75:
        return "medium"
    else:
        return "high"

def print_kernel_table(kernel_matrix):
    n = kernel_matrix.shape[0]

    # Print column header
    header = "     " + "   ".join(f"{j:>4}" for j in range(n))
    print(header)

    # Print each row
    for i in range(n):
        row_vals = "   ".join(f"{kernel_matrix[i,j]:.2f}" for j in range(n))
        print(f"{i:>3}  {row_vals}")

def select_most_similar_pair(kernel_matrix):
    # kernel_matrix is a kernel similarity matrix (n x n) where K[i,j] is the similarity between graph i and graph j.
    # The objective is to find the pair of distinct graphs (i,j) that has the highest similarity.

    k = np.asarray(kernel_matrix, dtype=float).copy()
 
    np.fill_diagonal(k, -np.inf)

    row, col = np.unravel_index(np.argmax(k), k.shape)

    return row, col

def assign_cluster_to_state(clusters, state, attr_names):
    for cluster in clusters:
        boundaries = cluster["boundaries"]
        match = True

        for feature in attr_names:
            if feature not in state:
                continue  # skip derived/non-state features like State Value or Action

            if feature not in boundaries:
                continue

            low, high = boundaries[feature]
            value = state[feature]

            low, high = min(low, high), max(low, high)

            if value is None or value < low or value > high:
                match = False
                break

        if match:
            return cluster["group"]

    return None
def get_next_probable_action(graph_dict, current_group):
    # graph_dict is the graph representation of the policy, with "groups" and "edges"
    # current_group is the group id of the current state
    # This function returns the action with the highest transition probability from the current group

    best_action = None
    best_prob = -1.0
    best_next_group = None

    for edge in graph_dict.get("edges", []):
        if int(edge["from_group"]) == current_group:
            prob = float(edge.get("probability", 0.0))
            action = edge.get("action", None)
            if prob > best_prob:
                best_prob = prob
                best_action = action
                best_next_group = int(edge["to_group"])

    return best_action, best_next_group