from env3run import MultiAgentInvManagementDiv
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

ray.init()

# Register environment
def env_creator(config):
    return MultiAgentInvManagementDiv(config = config)
config = {"bullwhip": True}
tune.register_env("MultiAgentInvManagementDiv", env_creator)   # noqa: E501

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
newenv_optp = "/Users/nikikotecha/ray_results/PPO/PPO_MultiAgentInvManagementDiv_82dea_00000_0_2023-11-21_19-23-55/checkpoint_000060"
single_agent = "/Users/nikikotecha/ray_results/PPO/PPO_MultiAgentInvManagementDiv_e11c4_00000_0_2023-11-21_22-25-30/checkpoint_000060"
trained_policy_single = Algorithm.from_checkpoint(single_agent)
trained_policy_multi = Algorithm.from_checkpoint(newenv_optp)
#trained_policy = Policy.from_checkpoint(checkpoint_path)

algo_config= (
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





config = {"bullwhip": True}
env = MultiAgentInvManagementDiv(config)
num_runs = 20
def run_simulation(num_periods_, trained_policy):
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
            all_s1.append((agent_id, infos[agent_id]['rescales1']))
            all_s2.append((agent_id, infos[agent_id]['rescales2']))

        _ +=1
    return all_infos, all_profits, all_backlog, all_inv, all_or, all_demand, all_s1, all_s2


def average_simulation(num_runs, 
                       trained_policy, 
                       num_periods_):
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
        all_infos, all_profits, all_backlog, all_inv, all_or, all_demand, all_s1, all_s2 = run_simulation(num_periods_, trained_policy)
        av_infos.append(all_infos)
        av_profits.append(all_profits)
        av_backlog.append(all_backlog)
        av_inv.append(all_inv)
        av_or.append(all_or)
        av_demand.append(all_demand)
        av_s1.append(all_s1)
        av_s2.append(all_s2)
    return av_infos, av_profits, av_backlog, av_inv, av_or, av_demand, av_s1, av_s2

av_infos, av_profits, av_backlog, av_inv , av_or , av_demand, av_s1, av_s2 = average_simulation(num_runs, trained_policy = trained_policy_single, num_periods_=50)
mav_infos, mav_profits, mav_backlog, mav_inv , mav_or , mav_demand, mav_s1, mav_s2 = average_simulation(num_runs, trained_policy = trained_policy_multi, num_periods_=50)

#all nodes e.g. profit
def process_all_nodes_data(av_profits1, av_profits2, config):
    average_profit_list1  = np.mean(av_profits1, axis =0)
    std_profit_list1 = np.std(av_profits1, axis=0)
    average_profit_list2 = np.mean(av_profits2, axis=0)
    std_profit_list2 = np.std(av_profits2, axis =0)

    cumulative_profit_list1 = np.cumsum(average_profit_list1, axis=0)  # Calculate cumulative profit
    cumulative_profit_list2 = np.cumsum(average_profit_list2, axis = 0)

    period = range(1, len(average_profit_list1)+1)
    plt.figure()
    plt.plot(period, cumulative_profit_list1, label = "Single Agent")
    plt.plot(period, cumulative_profit_list2, label = "Multi Agent")
    #plt.fill_between(period, cumulative_profit_list - std_profit_list, cumulative_profit_list + std_profit_list, color='gray', alpha=0.5)
    plt.xlabel('Period')
    plt.ylabel('Average Overall Profit')
    if config["bullwhip"] == True:
        # Add vertical lines at time periods 20 and 30
        plt.axvline(x=20, color='blue', linestyle='--')
        plt.axvline(x=30, color='blue', linestyle='--')
    plt.legend()
    plt.show()

    return average_profit_list

average_profit_list = process_all_nodes_data(av_profits, mav_profits, config)

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

#average_backlog_data = process_ind_nodes_data(av_backlog, num_runs, config)
#average_demand_data = process_ind_nodes_data(av_demand, num_runs, config)
#average_inv_data = process_ind_nodes_data(av_inv, num_runs, config)
#average_or_data = process_ind_nodes_data(av_or, num_runs, config)
#average_s1_data = process_ind_nodes_data(av_s1, num_runs, config)
#average_s2_data = process_ind_nodes_data(av_s2, num_runs, config)

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
plot_data(average_s1_data, average_s2_data, config)

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

plot_single_data(average_backlog_data, config)
plot_single_data(average_inv_data, config)
plot_single_data(average_or_data, config)
plot_single_data(average_demand_data, config)


#heat maps e.g. order replenishment and inventory 
def heat_maps(average_period_data1, average_period_data2, average_period_data3, average_period_data4):
    num_rows = 3
    num_cols = 2
    fig, axes = plt.subplots(num_rows, num_cols, figsize = (12,8))

    for i, (agent_id, data) in enumerate(average_period_data1.items()):
        row = i // num_cols
        col = i % num_cols
        ax = axes[row, col]
        #period = range(1, len(data)+1)
        df_inv = pd.DataFrame(average_period_data1[agent_id])
        df_or = pd.DataFrame(average_period_data2[agent_id])
        df_backlog = pd.DataFrame(average_period_data3[agent_id])
        df_demand = pd.DataFrame(average_period_data4[agent_id])

        combined_df = pd.concat([df_inv, df_or, df_backlog, df_demand], axis = 1)
        correlation_matrix = combined_df.corr()

        #correlation_matrix = df_inv.corrwith(df_or)
        plt.figure(figsize = (8,6))
        sns.heatmap(correlation_matrix, annot=True, cmap = 'coolwarm', vmin = -1, vmax = 1, ax= ax)
        ax.set_xticklabels(["Inv", "S1", "S2", "Demand"], rotation = 0)
        ax.set_yticklabels(["Inv", "S1", "S2", "Demand"], rotation = 0)
        ax.set_title(f'Agent ID: {agent_id}', pad = 10)

    fig.tight_layout()
    plt.subplots_adjust(top = 0.9)
    plt.show()

#heat_maps(average_inv_data, average_s1_data, average_s2_data, average_demand_data)

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


