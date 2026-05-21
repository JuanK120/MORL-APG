from config import argparser
from CAPS.CAPS_main import CAPS_main
from graphs.compare_kernels import compare_explanation_graphs
from graphs.utils import print_kernel_table, select_most_similar_pair, assign_cluster_to_state

if __name__ == '__main__':

    """"
    This is the main script running the explanation generation algorithm for MO-Policies
    It works by first collecting the initial parameters and then running the policy graph
    computation algorithm to get the graph for each policy, and then collecting all the 
    graphs in a dictionary for later use.
    Then We compute graph comparisson metrics for all generated graphs, and select the best
    for comparisson to generate contrastive explanations. 
    Finally, we generate the contrastive explanations for the best graph and print them out/Log them.
    """

    # Step 1: Collect initial parameters 
    args = argparser()

    paths = [
        r'CAPS/DPMORL/experiments/FruitTree_test/DPMORL.FruitTree.LossNormLamda_0.1/policy-program-0',
        r'CAPS/DPMORL/experiments/FruitTree_test/DPMORL.FruitTree.LossNormLamda_0.1/policy-program-1',
        r'CAPS/DPMORL/experiments/FruitTree_test/DPMORL.FruitTree.LossNormLamda_0.1/policy-program-2',
        r'CAPS/DPMORL/experiments/FruitTree_test/DPMORL.FruitTree.LossNormLamda_0.1/policy-program-3',
        r'CAPS/DPMORL/experiments/FruitTree_test/DPMORL.FruitTree.LossNormLamda_0.1/policy-program-4',
        r'CAPS/DPMORL/experiments/FruitTree_test/DPMORL.FruitTree.LossNormLamda_0.1/policy-program-5',
        r'CAPS/DPMORL/experiments/FruitTree_test/DPMORL.FruitTree.LossNormLamda_0.1/policy-program-6',
    ]

    # Step 2: Run the policy graph computation algorithm for each policy and collect the graphs

    all_graphs = {}

    for pol_idx, model_path in enumerate(paths):
        args.path = model_path 

        print(f"\nRunning policy {pol_idx}: {model_path}")

        explanation = CAPS_main(args)

        all_graphs[f"policy_{pol_idx}"] = explanation


    print(all_graphs)
    print(f"All policies have been tested and graphs collected. {len(all_graphs)} graphs in total.")

    # Step 3: Compute graph comparison metrics and select the best graph for contrastive explanation generation

    pol_names = list(all_graphs.keys())
    graph_dicts = list(all_graphs.values())

    K_wl, K_sm = compare_explanation_graphs(graph_dicts)

    print("Weisfeiler-Lehman Kernel Matrix:")
    print_kernel_table(K_wl)

    print("Subgraph Matching Kernel Matrix:")
    print_kernel_table(K_sm)
    
    # We need to select 2 graphs to compare for generating contrastive explanations, for that, we can use 
    # kernel similarity metrics to select the most similar graphs, as they are more likely to have interesting contrasts.
    # As selection criteria, we implement 3 options, selecting according to:
    # 1. The graph with the highest combined average similarity, for both WL and SM (the ones with the highest 
    #    average similarity between them)
    # 2. The graph with the highest similarity based on WL kernel (most similar in structure)
    # 3. The graph with the highest similarity based on SM kernel (most similar in subgraph patterns)

    if args.compare_criterion == "combined":      
        avg_similarity = (K_wl + K_sm) / 2
        
        id_graph1, id_graph2 = select_most_similar_pair(avg_similarity)


        print(f"Best graphs selected based on combined average similarity: {pol_names[id_graph1]} and {pol_names[id_graph2]}")
    elif args.compare_criterion == "wl":  

        id_graph1, id_graph2 = select_most_similar_pair(K_wl)

        print(f"Best graphs selected based on WL kernel similarity: {pol_names[id_graph1]} and {pol_names[id_graph2]}") 
    elif args.compare_criterion == "sm":  

        id_graph1, id_graph2 = select_most_similar_pair(K_sm)

        print(f"Best graphs selected based on SM kernel similarity: {pol_names[id_graph1]} and {pol_names[id_graph2]}")

    # Step 4: Generate contrastive explanations for the best graph and print/log them out

    test_state = {
        "lvl": 3,
        "pos": 1,
    }
    attr_names=["lvl", "pos"]

    print(
        f"state analysis test: "
        f"{assign_cluster_to_state(graph_dicts[id_graph1]['groups'], test_state, attr_names)}"
    )

    print(f"\n Generating contrastive explanations for the selected graph pair...")