import time
import os
import pickle
from config import argparser
from CAPS.CAPS_main import CAPS_main
from graphs.plot_graph import plot_apg
from graphs.compare_kernels import compare_explanation_graphs
from graphs.utils import print_kernel_table, select_most_similar_pair, assign_cluster_to_state, get_next_probable_action
from graphs.subgraph_search import get_maximum_common_subgraph, compare_transition_sets
from graphs.subgraph_search import build_common_percentage_matrices,percentage_table_g1_in_g2, print_percentage_table
from sample_states import test_states_ft, test_states_hw, test_states_dst
from model_paths import paths_ft, paths_hw, paths_dst, paths_hw2
import time

def run_policies(paths, args):
    for pol_idx, model_path in enumerate(paths):
        args.path = model_path 
        print(f"\nRunning policy {pol_idx}: {model_path}")
        explanation = CAPS_main(args)
        all_graphs[f"policy_{pol_idx}"] = explanation
        pkl_file_path = f"outputs/graphs/{args.env}_{args.num_episodes}_{args.lmbda}_{args.compare_criterion}/policy_{pol_idx}_graph.pkl"
        
        with open(pkl_file_path, 'wb') as f:
            pickle.dump(explanation, f)
    return all_graphs

def read_graphs_from_files(paths):
    for pol_idx, model_path in enumerate(paths):
        pkl_file_path = f"outputs/graphs/{args.env}_{args.num_episodes}_{args.lmbda}_{args.compare_criterion}/policy_{pol_idx}_graph.pkl"
        if os.path.exists(pkl_file_path):
            with open(pkl_file_path, 'rb') as f:
                explanation = pickle.load(f)
                all_graphs[f"policy_{pol_idx}"] = explanation
                print(f"Loaded graph for policy {pol_idx} from {pkl_file_path}.")
        else:
            print(f"Graph file for policy {pol_idx} not found at {pkl_file_path}. Please run the policies to generate the graphs.")
    return all_graphs
    

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
  
    if args.env == "MO_fruitTree":
        paths = paths_ft
    elif args.env == "MO_highway":
        paths = paths_hw2
    elif args.env == "MO_deepSea":
        paths = paths_dst
    else:
        raise ValueError(f"Unknown environment: {args.env}")

    # Step 2: Run the policy graph computation algorithm for each policy and collect the graphs

    all_graphs = {}

    time_graph_phase = time.time()
    test_name= f"outputs/graphs/{args.env}_{args.num_episodes}_{args.lmbda}_{args.compare_criterion}"
    if os.path.exists(test_name):
        print(f"Directory {test_name} already exists.") 
        if args.use_existing:
            print(f"Using existing graphs from {test_name} as --use_existing flag is set to True. Checking for existing graphs...")
            if len(os.listdir(test_name)) < len(paths):
                raise ValueError(f"The directory {test_name} is empty or contains "+
                      f"fewer files than expected but the --use_existing flag is  set to True. "+
                      f"Please check if the directory contains the expected graph files.")
            else:
                print(f"""length of files in {test_name}: {len(os.listdir(test_name))}, 
                expected: {len(paths)}.
                Re-running the policies to collect new graphs""")
                all_graphs = read_graphs_from_files(paths)
                print(f"Loaded existing graphs from {f'{test_name}'} . {len(all_graphs)} graphs in total.")
        else:
            print(f"Re-running the policies to collect new graphs as --use_existing flag is set to False.")
            all_graphs = run_policies(paths, args)
            print(f"All policies have been tested and graphs collected. {len(all_graphs)} graphs in total.")
    else:
        print(f"Directory {test_name} does not exist. Creating it and running the policies to collect new graphs.")
        os.makedirs(test_name)
        all_graphs = run_policies(paths, args)
        print(f"All policies have been tested and graphs collected. {len(all_graphs)} graphs in total.")

    print(f"All graphs collected:")
    for policy_name, graph in all_graphs.items():
        plot_apg(
            graph,
            title=f"{args.env}_{policy_name}",
            save_path=test_name
        )

    time_graph_phase = time.time() - time_graph_phase

    # Step 3: Compute graph comparison metrics and select the best graph for contrastive explanation generation

    time_comparison_phase = time.time() 

    pol_names = list(all_graphs.keys())
    graph_dicts = list(all_graphs.values())

    K_wl, K_sm = compare_explanation_graphs(graph_dicts)
      
    avg_similarity = (K_wl + K_sm) / 2

    print("Weisfeiler-Lehman Kernel Matrix:")
    print_kernel_table(K_wl)

    print("Subgraph Matching Kernel Matrix:")
    print_kernel_table(K_sm)

    print("Avg. simillarity Matrix:")
    print_kernel_table(avg_similarity)
    
    # We need to select 2 graphs to compare for generating contrastive explanations, for that, we can use 
    # kernel similarity metrics to select the most similar graphs, as they are more likely to have interesting contrasts.
    # As selection criteria, we implement 3 options, selecting according to:
    # 1. The graph with the highest combined average similarity, for both WL and SM (the ones with the highest 
    #    average similarity between them)
    # 2. The graph with the highest similarity based on WL kernel (most similar in structure)
    # 3. The graph with the highest similarity based on SM kernel (most similar in subgraph patterns)

    if args.compare_criterion == "combined":
        
        id_graph1, id_graph2 = select_most_similar_pair(avg_similarity)


        print(f"Best graphs selected based on combined average similarity: {pol_names[id_graph1]} and {pol_names[id_graph2]}")
    elif args.compare_criterion == "wl":  

        id_graph1, id_graph2 = select_most_similar_pair(K_wl)

        print(f"Best graphs selected based on WL kernel similarity: {pol_names[id_graph1]} and {pol_names[id_graph2]}") 
    elif args.compare_criterion == "sm":  

        id_graph1, id_graph2 = select_most_similar_pair(K_sm)

        print(f"Best graphs selected based on SM kernel similarity: {pol_names[id_graph1]} and {pol_names[id_graph2]}")


    print(f"policy 1 selected for explanation: \n {graph_dicts[id_graph1]}")
    print(f"policy 2 selected for explanation: \n{graph_dicts[id_graph2]}")

    time_comparison_phase = time.time() - time_comparison_phase

    # Step 4: Generate contrastive explanations for the best graph and print/log them out

    ## action explanation generation phase

    time_explanation_phase = time.time()

    if args.env == "MO_fruitTree":
        test_states = test_states_ft
    elif args.env == "MO_highway":
        test_states = test_states_hw
    elif args.env == "MO_deepSea":
        test_states = test_states_dst
    else:
        raise ValueError(f"Unknown environment: {args.env}")
    
    print(f"\n Generating contrastive explanations for the selected graph pair...")

    for i, test_state in enumerate(test_states):

        print(f"\n\n--- Contrastive explanation for test state {i}: {test_state} ---")

        group_for_state_graph1 = assign_cluster_to_state(graph_dicts[id_graph1]['groups'], test_state, graph_dicts[id_graph1]['state_features'])
        group_for_state_graph2 = assign_cluster_to_state(graph_dicts[id_graph2]['groups'], test_state, graph_dicts[id_graph2]['state_features'])


        print(
            f"state analysis test for {pol_names[id_graph1]}: "
            f"{group_for_state_graph1}"
        )

        print(
            f"state analysis test for {pol_names[id_graph2]}: "
            f"{group_for_state_graph2}"
        )


        
        next_action_policy_1, next_abstract_state_policy_1 = get_next_probable_action(graph_dicts[id_graph1], group_for_state_graph1)
        next_action_policy_2, next_abstract_state_policy_2 = get_next_probable_action(graph_dicts[id_graph2], group_for_state_graph2)

        print(f"Next probable action for {pol_names[id_graph1]}: {next_action_policy_1}")
        print(f"Next probable action for {pol_names[id_graph2]}: {next_action_policy_2}")


    
    time_action_explanation_phase = time.time() - time_explanation_phase

    ## edge explanation generation phase
    
    time_explanation_phase = time.time()

    common, only_g1, only_g2 = compare_transition_sets(
        graph_dicts[id_graph1],
        graph_dicts[id_graph2]
    )

    print("Common transitions:")
    for edge in common:
        print(edge)

    print("Unique to policy 1:")
    for edge in only_g1:
        print(edge)

    print("Unique to policy 2:")
    for edge in only_g2:
        print(edge)

    time_edge_explanation_phase = time.time() - time_explanation_phase

    ## subgraph explanation generation phase

    time_explanation_phase = time.time()

    common_subgraph, mapping = get_maximum_common_subgraph(
        graph_dicts[id_graph1],
        graph_dicts[id_graph2]
    )

    print("MCS mapping:", mapping)

    if common_subgraph is not None:
        print("Common nodes:", common_subgraph.nodes(data=True))
        print("Common edges:", common_subgraph.edges(data=True))
    else:
        print("No common subgraph found.")

    

    time_subgraph_explanation_phase = time.time() - time_explanation_phase

    # shared subgraph percentage for all nodes and edges between all graphs, to give a sense of how similar the policies are in terms of their structure and transitions

    time_subgraph_percentage_explanation_phase = time.time() 

    print("\n\n--- Shared subgraph percentage for all nodes and edges between all graphs ---")

    print(f"Best graphs selected based on {args.compare_criterion} kernel similarity: {pol_names[id_graph1]} and {pol_names[id_graph2]}")

    print("Weisfeiler-Lehman Kernel Matrix:")
    print_kernel_table(K_wl)

    print("Subgraph Matching Kernel Matrix:")
    print_kernel_table(K_sm)

    print("Avg. simillarity Matrix:")
    print_kernel_table(avg_similarity)

    print("Shared subgraph percentage for all nodes and edges between all graphs:")

    common_nodes_percentage, common_edges_percentage = build_common_percentage_matrices(graph_dicts)

    print("nodes:")
    print_percentage_table(common_nodes_percentage)
    
    print("edges:")
    print_percentage_table(common_edges_percentage)

    print("Shared percentage from g1 to g2 in nodes and edges between all graphs:")
    g1_in_g2_nodes_percentage, g1_in_g2_edges_percentage = percentage_table_g1_in_g2(graph_dicts) 

    print("nodes:")
    print_percentage_table(g1_in_g2_nodes_percentage)
    
    print("edges:")
    print_percentage_table(g1_in_g2_edges_percentage)

    time_subgraph_percentage_explanation_phase = time.time() - time_subgraph_percentage_explanation_phase

    #

    # Calculate total time for explanation generation phase

    total_explanation_phase_time = time_action_explanation_phase + time_edge_explanation_phase + time_subgraph_explanation_phase + time_subgraph_percentage_explanation_phase

    print(f"""\n\n--- Summary of execution --- \n\n
        -- args used --  \n
        Environment: {args.env} \n
        Number of episodes: {args.num_episodes} \n
        Lambda value: {args.lmbda} \n
        Graph comparison criterion: {args.compare_criterion} \n
        used stored graphs: {args.use_existing} \n
        -- times for each phase --  \n
        Graph generation phase: {time_graph_phase:.2f} seconds \n
        Graph comparison phase: {time_comparison_phase:.2f} seconds \n
        Action explanation generation phase: {time_action_explanation_phase:.2f} seconds \n
        Edge explanation generation phase: {time_edge_explanation_phase:.2f} seconds \n
        Subgraph explanation generation phase: {time_subgraph_explanation_phase:.2f} seconds \n
        Subgraph percentage explanation generation phase: {time_subgraph_percentage_explanation_phase:.2f} seconds \n
        Total explanation generation phase: {total_explanation_phase_time:.2f} seconds
        \n\n--- End of execution ---
    """)  

