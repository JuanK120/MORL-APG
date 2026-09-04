import argparse

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='grid') #Environment (grid, cart, mountain)
    parser.add_argument('--path', default='') #Path to RLlib pre-trained model
    parser.add_argument('--num_episodes', type=int, default=3) #Number of episodes to collect data from
    parser.add_argument('--calc_fidelity', type=str2bool, nargs='?', const=True, default=False) #calculate fidelity of generated graphs
    parser.add_argument('--alpha', type=float, default=0.015) #Alpha parameter
    parser.add_argument('--k', type=int, default=3) #Number of graphs to produce
    parser.add_argument('--max_height', type=int, default=10) #Maximum height of CLTree
    parser.add_argument('--lmbda', type=float, default=1) #Lambda value from RL training
    parser.add_argument('--hayes_baseline', type=str2bool, nargs='?', const=True, default=False) #Whether to use Hayes and Shah 2017 baseline for explanations
    parser.add_argument('--topin_baseline', type=str2bool, nargs='?', const=True, default=False) #Whether to use Topin and Veloso 2019 baseline for apg gen
    parser.add_argument('--zahavy_baseline', type=str2bool, nargs='?', const=True, default=False) #Whether to cluster states according to Zahavy methodology
    parser.add_argument('--alg', default='DQN') #Training algorithm. DQN and PPO supported currently
    ############## Policy Return Plot args ##############
    parser.add_argument('--plot_returns', type=str2bool, nargs='?', const=True, default=True) #Whether to plot the return distributions of the trained policies
    parser.add_argument('--batch_size', type=int, default=100) #Number of episodes to average into one plotted point
    parser.add_argument('--final_episodes', type=int, default=100) #Number of final episodes from each policy to plot
    ############## AutoPred args ##############
    parser.add_argument('--autoPred', type=str2bool, nargs='?', const=True, default=False) #Whether to use the automatic predicate generation algorithm instead of hand-crafted predicates
    parser.add_argument('--shap_selection', type=str2bool, nargs='?', const=True, default=False) #Whether to use SHAP for feature selection in AutoPred (if False, uses feature selection based on decision tree path instead)
    parser.add_argument('--use_all_features', type=str2bool, nargs='?', const=True, default=False) #Whether to use all features for predicate generation
    ############## graph selection args ##############
    parser.add_argument('--compare_criterion', default='combined') #Criterion for comparing graphs ('combined', 'wl', 'sm')
    ############## re-run policies arg ##############
    parser.add_argument('--use_existing', type=str2bool, nargs='?', const=True, default=False) #Whether to use existing graphs or re-run the policies to collect new data (True or False)

    args = parser.parse_args()
    return args

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', '1', 'yes', 'y'):
        return True
    elif v.lower() in ('false', '0', 'no', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')