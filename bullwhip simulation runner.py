from env3run import MultiAgentInvManagementDiv
from env2 import MultiAgentInvManagementDiv1
import gymnasium as gym
from ray.rllib.algorithms.ppo import PPOConfig
from gymnasium.spaces import Dict, Box
import numpy as np 
from ray.rllib.models import ModelCatalog
import ray 
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.policy import Policy
from ray import tune 
from ray.tune.logger import pretty_print
from ray import tune 
import torch 
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import json
import os



ray.init()

# Register environment - OR
def env_creator1(config):
    return MultiAgentInvManagementDiv1(config = config)
config = {"bullwhip": True}
tune.register_env("MultiAgentInvManagementDiv1", env_creator1)   

# Register environment - sS
def env_creator2(config):
    return MultiAgentInvManagementDiv(config = config)
config = {"bullwhip": True}
tune.register_env("MultiAgentInvManagementDiv", env_creator2)

#load the trained policy marl
#checkpoint_path = "/Users/nikikotecha/ray_results/PPO/PPO_MultiAgentInvManagementDiv_8ed7e_00000_0_2023-11-14_12-08-44/checkpoint_000070/policies/default_policy"

#single agent policy loading 
#checkpoint_path1 = "/Users/nikikotecha/ray_results/PPO_MultiAgentInvManagementDiv_2023-11-16_15-53-14a7azmbdo/checkpoint_000070/"
#checkpoint_path = "/Users/nikikotecha/ray_results/PPO/PPO_MultiAgentInvManagementDiv_1466d_00000_0_2023-11-18_13-47-57/checkpoint_000070"

#this one has a series of policies 
#checkpoint_path = "/Users/nikikotecha/ray_results/PPO_MultiAgentInvManagementDiv_2023-11-14_14-24-23z24j6i1z/checkpoint_000070"

#this one has a default policy 
checkpoint_path = "/Users/nikikotecha/ray_results/PPO/PPO_MultiAgentInvManagementDiv_4b099_00000_0_2023-11-13_15-06-59/checkpoint_000070"
trial = "/Users/nikikotecha/ray_results/PPO/PPO_MultiAgentInvManagementDiv_60221_00000_0_2023-11-20_11-53-10/checkpoint_000070"
newenv = "/Users/nikikotecha/ray_results/PPO/PPO_MultiAgentInvManagementDiv_711b2_00000_0_2023-11-21_17-43-12/checkpoint_000060"
newenv_optp = "/Users/nikikotecha/ray_results/PPO/PPO_MultiAgentInvManagementDiv_82dea_00000_0_2023-11-21_19-23-55/checkpoint_000060" #marl_opt num 2 
single_agent = "/Users/nikikotecha/ray_results/PPO/PPO_MultiAgentInvManagementDiv_03040_00000_0_2023-11-22_17-24-37/checkpoint_000060"
marl_opt = "/Users/nikikotecha/ray_results/PPO/PPO_MultiAgentInvManagementDiv_e11c4_00000_0_2023-11-21_22-25-30/checkpoint_000060"
trained_policy_single = Algorithm.from_checkpoint(single_agent)
trained_policy_multi = Algorithm.from_checkpoint(marl_opt)
or_policy = "/Users/nikikotecha/ray_results/PPO/PPO_MultiAgentInvManagementDiv_e41a5_00000_0_2023-11-29_14-30-49/checkpoint_000060"
#OR = "/Users/nikikotecha/ray_results/PPO/PPO_MultiAgentInvManagementDiv1_658b6_00000_0_2023-11-30_13-28-50/checkpoint_000002"
OR = "/Users/nikikotecha/ray_results/PPO/PPO_MultiAgentInvManagementDiv1_fa9c5_00000_0_2023-12-06_11-09-52/checkpoint_000060"
#trained_policy_or = Algorithm.from_checkpoint(OR)
#gor = "/Users/nikikotecha/ray_results/PPO/PPO_MultiAgentInvManagementDiv1_692cb_00000_0_2023-12-06_12-46-01/checkpoint_000060" with gcn conv 
#gor = '/Users/nikikotecha/ray_results/PPO/PPO_MultiAgentInvManagementDiv1_07f5a_00000_0_2023-12-07_12-42-07/checkpoint_000060'
#trained_policy_gor = Algorithm.from_checkpoint(gor)
gnn = "/Users/nikikotecha/ray_results/PPO/PPO_MultiAgentInvManagementDiv_a0efc_00000_0_2023-12-01_11-34-46/checkpoint_000060"
gnn_policy = Algorithm.from_checkpoint(gnn)
gnn2 = "/Users/nikikotecha/ray_results/PPO/PPO_MultiAgentInvManagementDiv_d8e44_00000_0_2023-12-01_15-46-52/checkpoint_000060"
#gnn2_policy = Algorithm.from_checkpoint(gnn2)
gss3= "/Users/nikikotecha/ray_results/PPO/PPO_MultiAgentInvManagementDiv_a1f97_00000_0_2023-12-11_13-53-16/checkpoint_000060"
#rained_policy_gss3 = Algorithm.from_checkpoint(gss3)
gss4 = "/Users/nikikotecha/ray_results/PPO/PPO_MultiAgentInvManagementDiv_77f3d_00000_0_2023-12-14_11-54-00/checkpoint_000060"
#trained_policy_gss4 = Algorithm.from_checkpoint(gss4)
gss5 = "/Users/nikikotecha/ray_results/PPO/PPO_MultiAgentInvManagementDiv_bf9cf_00000_0_2023-12-14_13-36-13/checkpoint_000060"
#trained_policy_gss5 = Algorithm.from_checkpoint(gss5)
"""algo_config= (
    PPOConfig()
    .environment(
        env= "MultiAgentInvManagementDiv",
        env_config={
            "num_agents": 6,
        },
    )
    .multi_agent(
        policies={'0_00_0', '0_00_1', '1_01_0', '1_01_1', '2_02_0', '2_02_1'},
        # Map "agent0" -> "pol0", etc...
        policy_mapping_fn=(
            lambda agent_id, episode, worker, **kwargs: (
        print(f"Agent ID: {agent_id}"),
        str(agent_id)
    )[1]
    )
    )
    .build()
)

algo_config.restore(checkpoint_path)
"""


config1 = {"bullwhip": True}
env_OR = MultiAgentInvManagementDiv1(config1)
action_spaceOR = env_OR.action_space
env_SS = MultiAgentInvManagementDiv(config1)
act_spaceSS = env_SS

print("as 0r", action_spaceOR)
print("act space ss", act_spaceSS)

num_runs = 20
def run_simulation(num_periods_, trained_policy, env):
    obs, infos = env.reset()
    all_infos = []
    all_profits = []
    all_backlog = []
    all_inv = []
    all_or =[]
    all_demand = []
    all_s1 = []
    all_s2 = []
    for _ in range(num_periods_):
        print("num period {}".format(_))
        actions = {}
        for agent_id in obs.keys():
            print("obs",obs[agent_id])
            action = trained_policy.compute_single_action(obs[agent_id])
            actions[agent_id] = action
        print("action dictionary", actions)
        obs, rewards, done, truncated, infos = env.step(actions)
        all_infos.append(infos)
        all_profits.append(infos['overall_profit'])
        for agent_id in obs.keys():
            all_backlog.append((agent_id, infos[agent_id]['reward']))
            all_demand.append((agent_id, infos[agent_id]['demand']))
            all_inv.append((agent_id, infos[agent_id]['inv']))
            all_or.append((agent_id, infos[agent_id]['actual order']))
            #all_s1.append((agent_id, infos[agent_id]['rescales1']))
            #all_s2.append((agent_id, infos[agent_id]['rescales2']))

        _ +=1
    return all_infos, all_profits, all_backlog, all_inv, all_or, all_demand, all_s1, all_s2


def average_simulation(num_runs, 
                       trained_policy, 
                       num_periods_,
                       env):
    #initialise to store variables
    av_infos = []
    av_profits = []
    av_backlog =[]
    av_inv = []
    av_or =[]
    av_demand = []
    av_s1 = []
    av_s2 = []
    for run in range(num_runs):
        all_infos, all_profits, all_backlog, all_inv, all_or, all_demand, all_s1, all_s2 = run_simulation(num_periods_, trained_policy, env)
        av_infos.append(all_infos)
        av_profits.append(all_profits)
        av_backlog.append(all_backlog)
        av_inv.append(all_inv)
        av_or.append(all_or)
        av_demand.append(all_demand)
        av_s1.append(all_s1)
        av_s2.append(all_s2)
    
    return av_infos, av_profits, av_backlog, av_inv, av_or, av_demand, av_s1, av_s2



av_infos, av_profits, av_backlog, av_inv, av_or, av_demand, av_s1, av_s2 = average_simulation(num_runs, trained_policy=trained_policy_single, num_periods_=50, env=env_SS)
mav_infos, mav_profits, mav_backlog, mav_inv, mav_or, mav_demand, mav_s1, mav_s2 = average_simulation(num_runs, trained_policy=trained_policy_multi, num_periods_=50, env=env_SS)
#gav_infos, gav_profits, gav_backlog, gav_inv, gav_or, gav_demand, gav_s1, gav_s2 = average_simulation(num_runs, trained_policy=gnn_policy, num_periods_=50, env=env_SS)
#goav_infos, goav_profits, goav_backlog, goav_inv, goav_or, goav_demand, goav_s1, goav_s2 = average_simulation(num_runs, trained_policy=trained_policy_or, num_periods_=50, env=env_OR)

profit_data = {"av_profits": av_profits, 
               "mav_profits": mav_profits,
               #"gav_profits": gav_profits,
               #"goav_profits": goav_profits
                }

output_file = 'bullwhip_profit_data.json'
with open(output_file, 'w') as file:
    json.dump(profit_data, file)

absolute_path = os.path.abspath(output_file)
print(f"The JSON file is saved at: {absolute_path}")

#all nodes e.g. profit
def process_all_nodes_data(av_profits1, av_profits2, config):
    average_profit_list1  = np.mean(av_profits1, axis =0)
    std_profit_list1 = np.std(av_profits1, axis=0)
    average_profit_list2 = np.mean(av_profits2, axis=0)
    std_profit_list2 = np.std(av_profits2, axis =0)
    """    average_profit_list3 = np.mean(av_profits3, axis=0)
        std_profit_list3 = np.std(av_profits2, axis =0)
        average_profit_list4 = np.mean(av_profits4, axis=0)
        std_profit_list4 = np.std(av_profits4, axis =0)"""

    cumulative_profit_list1 = np.cumsum(average_profit_list1, axis =0)  # Calculate cumulative profit
    cumulative_profit_list2 = np.cumsum(average_profit_list2, axis = 0)
    """    cumulative_profit_list3 = np.cumsum(average_profit_list3, axis = 0)
        cumulative_profit_list4 = np.cumsum(average_profit_list4, axis = 0)"""

    period = range(1, len(average_profit_list1)+1)
    plt.figure()
    plt.plot(period, cumulative_profit_list1, label = "Single Agent", color = 'red', linestyle = 'solid')
    plt.fill_between(period, cumulative_profit_list1 + std_profit_list1, cumulative_profit_list1 - std_profit_list1, color = 'red', alpha = 0.5)
    plt.plot(period, cumulative_profit_list2, label = "Multi Agent", color = 'blue', linestyle = 'dashed')
    plt.fill_between(period, cumulative_profit_list2 + std_profit_list2, cumulative_profit_list2 - std_profit_list2, color = 'blue', alpha = 0.5)
    """plt.plot(period, cumulative_profit_list3, label = "Graph Multi Agent", color = 'green', linestyle = 'solid')
    plt.fill_between(period, cumulative_profit_list3 + std_profit_list3, cumulative_profit_list3 - std_profit_list3, color = 'green', alpha = 0.5)
    plt.plot(period, cumulative_profit_list4, label = "Multi Agent OR", color = 'orange', linestyle = 'solid')
    plt.fill_between(period, cumulative_profit_list4 + std_profit_list4, cumulative_profit_list4 - std_profit_list4, color = 'orange', alpha = 0.5)"""

    #plt.fill_between(period, cumulative_profit_list - std_profit_list, cumulative_profit_list + std_profit_list, color='gray', alpha=0.5)
    plt.xlabel('Period')
    plt.ylabel('Average Overall Profit')
    if config["bullwhip"] == True:
        # Add vertical lines at time periods 20 and 30
        plt.axvline(x=20, color='green', linestyle='--', label = 'Disruption Start')
        plt.axvline(x=30, color='green', linestyle='--', label = 'Disruption End')
    plt.legend()
    plt.show()

def process_all_nodes_data_from_file(file_path, config):
    with open(file_path, 'r') as file:
        data = json.load(file)

    av_profits = data["av_profits"]
    mav_profits = data["mav_profits"]
    #gav_profits = data["gav_profits"]
    #goav_profits = data["goav_profits"]

    process_all_nodes_data(av_profits, mav_profits, config)

process_all_nodes_data_from_file(output_file, config)

#individual nodes e.g. backlog
def process_ind_nodes_data(av_backlog, num_runs, config):
    agent_data = {}
    for period_data in av_backlog:
        for agent_id, value in period_data:
            if agent_id in agent_data:
                agent_data[agent_id].append(value)
            else:
                agent_data[agent_id] = [value]
    print(agent_data)
    # Calculate the average backlog value for each period for each agent

    split_data = {agent_id: np.array_split(data, num_runs) for agent_id, data in agent_data.items()}

    # Calculate the average for each period within each run
    average_period_data = {agent_id: [np.mean(np.array(run_data), axis=0) for run_data in zip(*split_data[agent_id])] for agent_id in agent_data}

    return average_period_data

#average_backlog_data = process_ind_nodes_data(mav_backlog, num_runs, config)
#average_demand_data = process_ind_nodes_data(mav_demand, num_runs, config)
#average_inv_data = process_ind_nodes_data(mav_inv, num_runs, config)
#average_or_data = process_ind_nodes_data(mav_or, num_runs, config)
#average_s1_data = process_ind_nodes_data(av_s1, num_runs, config)
#average_s2_data = process_ind_nodes_data(mav_s2, num_runs, config)

def plot_data(average_backlog_data, average_demand_data, config):

    num_rows = 3 
    num_cols = 2

    fig, axes = plt.subplots(num_rows, num_cols, figsize = (12,8))
    for i, (agent_id, data) in enumerate(average_backlog_data.items()):
        row = i // num_cols
        col = i % num_cols
        ax = axes[row, col]
        period = range(1, len(data)+1)
        ax.plot(period, data, label= f'Agent ID: {agent_id}')
        if config["bullwhip"] == True:
            ax.axvline(x=15, color='red', linestyle='--', label='Bullwhip Effect Start')
            ax.axvline(x=20, color='green', linestyle='--', label='Bullwhip Effect End')
            demand_data = average_demand_data[agent_id]
            ax.plot(period, demand_data, label= 'Demand')
        ax.set_xlabel('Period')
        ax.set_ylabel('Average Demand')
        ax.legend()
    plt.tight_layout()
    plt.show()

#plot_data(average_backlog_data, average_demand_data, config)
#plot_data(average_s1_data, average_s2_data, config)

def plot_single_data(average_backlog_data,config):

    num_rows = 3 
    num_cols = 2

    fig, axes = plt.subplots(num_rows, num_cols, figsize = (12,8))
    for i, (agent_id, data) in enumerate(average_backlog_data.items()):
        row = i // num_cols
        col = i % num_cols
        ax = axes[row, col]
        period = range(1, len(data)+1)
        ax.plot(period, data, label= f'Agent ID: {agent_id}')
        if config["bullwhip"] == True:
            ax.axvline(x=20, color='red', linestyle='--', label='Bullwhip Effect Start')
            ax.axvline(x=30, color='green', linestyle='--', label='Bullwhip Effect End')
    plt.tight_layout()
    plt.show()

#plot_single_data(average_backlog_data, config)
#plot_single_data(average_inv_data, config)
#plot_single_data(average_or_data, config)
#plot_single_data(average_demand_data, config)


#heat maps e.g. order replenishment and inventory 
def heat_maps(average_period_data1, average_period_data2, average_period_data3):
    num_rows = 3
    num_cols = 2
    fig, axes = plt.subplots(num_rows, num_cols, figsize = (12,8))

    for i, (agent_id, data) in enumerate(average_period_data1.items()):
        row = i // num_cols
        col = i % num_cols
        ax = fig.add_subplot(num_rows, num_cols)
        #period = range(1, len(data)+1)
        df_1 = pd.DataFrame(average_period_data1[agent_id])
        df_2 = pd.DataFrame(average_period_data2[agent_id])
        df_3 = pd.DataFrame(average_period_data3[agent_id])

        combined_df = pd.concat([df_1, df_2, df_3], axis = 1)
        correlation_matrix = combined_df.corr()

        plt.figure(figsize = (8,6))
        sns.heatmap(correlation_matrix, annot=True, cmap = 'coolwarm', vmin = -1, vmax = 1, ax= ax)
        ax.set_xticklabels(["Inv", "S1", "S2"], rotation = 0)
        ax.set_yticklabels(["Inv", "S1", "S2"], rotation = 0)
        ax.set_title(f'Agent ID: {agent_id}', pad = 10)

    fig.tight_layout()
    plt.subplots_adjust(top = 0.9)
    plt.show()

    plt.tight_layout()
    plt.show()

#heat_maps(average_inv_data, average_s1_data,average_demand_data)

       
"""plt.figure(figsize = (8,6))
        sns.heatmap(correlation_matrix, annot=True, cmap = 'coolwarm', vmin = -1, vmax = 1, ax= ax)
        ax.set_xticklabels(["Inv", "S1", "S2", "Demand"], rotation = 0)
        ax.set_yticklabels(["Inv", "S1", "S2", "Demand"], rotation = 0)
        ax.set_title(f'Agent ID: {agent_id}', pad = 10)

    fig.tight_layout()
    plt.subplots_adjust(top = 0.9)
    plt.show()"""

#heat_maps(average_inv_data, average_s1_data,average_demand_data)


"""def process_data(b):
    data_dict = {
        'period':{},
        'reward': {},
        'demand': {},
        'ship': {},
        'acquisition': {},
        'actual order': {},
        'profit': {},
    }

    for run_data in b:
        # Extract data from all_infos for all nodes
        for node_info in run_data:
            for node_id, node_data in node_info.items():
                # Iterate through each metric and append the data to the respective node's list
                for metric in data_dict.keys():
                    if node_id not in data_dict[metric]:
                        data_dict[metric][node_id] = []
                    data_dict[metric][node_id].append(node_data[metric])
    
    return data_dict

data = process_data(b)
print(data)"""


"""import matplotlib.pyplot as plt
import numpy as np

# Initialize a dictionary to store data for each metric for each node
data_dict = {
    'period': {},
    'demand': {},
    'ship': {},
    'acquisition': {},
    'actual order': {},
    'profit': {},
}

# Extract data from all_infos for all nodes
for node_info in all_infos:
    for node_id, node_data in node_info.items():
        # Iterate through each metric and append the data to the respective node's list
        for metric in data_dict.keys():
            if node_id not in data_dict[metric]:
                data_dict[metric][node_id] = []
            data_dict[metric][node_id].append(node_data[metric])

# Now you have a dictionary containing data for each metric for each node
print("Data Dictionary:", data_dict)

import matplotlib.pyplot as plt

# Define the list of node IDs 
node_ids = ['node_0', 'node_1', 'node_2', 'node_3', 'node_4', 'node_5']

# Create subplots for each metric
metrics = data_dict.keys()
num_metrics = len(metrics)

for node_id in node_ids:
    ship_data = data_dict['ship'][node_id]
    demand_data = data_dict['demand'][node_id]
    ratio = [demand/ship for ship, demand in zip(ship_data, demand_data)]
    plt.plot(data_dict['period'][node_id], ratio, label = node_id)

plt.tight_layout()
plt.show()
"""



"""ship_demand_ratio = [ship / demand for ship, demand in zip(all_ships, all_demands)]

# Create a plot for the ship-to-demand ratio
plt.figure(figsize=(12, 8))
plt.plot(all_periods, ship_demand_ratio, label='Ship/Demand Ratio')
plt.xlabel('Period')
plt.ylabel('Ship/Demand Ratio')
plt.legend()
plt.title('Ship-to-Demand Ratio vs. Period for All Nodes')

# Show the plot
plt.show()"""

"""# Create a separate graph for each metric with 6 lines (one for each node)
metrics = ['Demand', 'Ship', 'Acquisition', 'Actual Order', 'Profit[0]']

for metric_data, metric_name in zip([all_demands, all_ships, all_acquisitions, all_actual_orders, all_profits], metrics):
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle(f'{metric_name} vs. Period for All Nodes')

    for node_idx in range(6):
        ax.plot(all_periods[node_idx::6], metric_data[node_idx::6], label=f'Node_{node_idx}')

    # Set labels and legend
    ax.set_xlabel('Period')
    ax.set_ylabel(metric_name)
    ax.legend()

# Show the plots
plt.show()"""


print("doneee")
ray.shutdown()


