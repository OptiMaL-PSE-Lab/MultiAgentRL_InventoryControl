from email import policy
from env3rundiv import MultiAgentInvManagementDiv
#from env2 import MultiAgentInvManagementDiv1
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
#import seaborn as sns
import pandas as pd
#from mpl_toolkits.mplot3d import Axes3D
import json
import os
from ccmodel import CentralisedCriticModel, FillInActions
from modelpool import GNNActorCriticModelPool
from model import GNNActorCriticModel


ray.init()
# Register environment - OR
#def env_creator1(config):
#    return MultiAgentInvManagementDiv1(config = config)
#config = {"bullwhip": True}
#tune.register_env("MultiAgentInvManagementDiv1", env_creator1)   
 
model_config = {"model": "central"}
output_file = 'mappo_12_1.json'
# Register environment - sS
def env_creator2(config):
    return MultiAgentInvManagementDiv(config = config)
config = {"bullwhip": False}
tune.register_env("MultiAgentInvManagementDiv", env_creator2)
config1 = {"bullwhip": False}
env_SS = MultiAgentInvManagementDiv(config1)
ModelCatalog.register_custom_model("cc_model", CentralisedCriticModel)
ModelCatalog.register_custom_model("gnn_model", GNNActorCriticModel)
ModelCatalog.register_custom_model("gnnpool_model", GNNActorCriticModelPool)

def create_network(connections):
    num_nodes = max(connections.keys())
    network = np.zeros((num_nodes + 1, num_nodes + 1))
    for parent, children in connections.items():
        if children:
            for child in children:
                network[parent][child] = 1

    return network

def get_stage(node, network):
    reached_root = False
    stage = 0
    counter = 0
    if node == 0:
        return 0
    while not reached_root:
        for i in range(len(network)):
            if network[i][node] == 1:
                stage += 1
                node = i
                if node == 0:
                    return stage
        counter += 1
        if counter > len(network):
            raise Exception("Infinite Loop")
        
connections = {0: [1,2], 1:[3,4], 2:[4, 5], 3:[], 4:[], 5:[]}



agent_ids = []
network = create_network(connections)
echelons = {node: get_stage(node, network) for node in range(len(network))}
agent_ids = [f"{echelons[node]}_{node:02d}" for node in range(len(network))]

ng1 =  r"c:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-05-19_23-11-22vjup3rv8\checkpoint_000060"
ng1p = Algorithm.from_checkpoint(ng1,
                                 policy_ids= agent_ids,
                                policy_mapping_fn= lambda agent_id, episode, worker, **kwargs: (
        print(f"Agent ID: {agent_id}"),
        str(agent_id)
    )[1])


def obs_reformat(agent_obs, **kw):
    """Rewrites the agent obs to include opponent data for training."""
    agents = [*agent_obs]
    num_agents = len(agents)
    obs_space = len(agent_obs[agents[0]])

    new_obs = dict()
    for agent in agents:
        new_obs[agent] = dict()
        new_obs[agent]["own_obs"] = agent_obs[agent]
        new_obs[agent]["opponent_obs"] = np.zeros((num_agents - 1)*obs_space)
        new_obs[agent]["opponent_action"] = np.zeros(2*(num_agents - 1))
        i = 0
        for other_agent in agents:
            if agent != other_agent:
                new_obs[agent]["opponent_obs"][i*obs_space:i*obs_space + obs_space] = agent_obs[other_agent]
                i += 1

    return new_obs

num_runs = 20
def run_simulation(num_periods_, trained_policy, env, model_config):
    all_infos = []
    all_profits = []
    all_backlog = []
    all_inv = []


    obs, infos = env.reset()
    for _ in range(num_periods_):
        actions = {}
        for agent_id in obs.keys():
            if model_config["model"]== "central":
                obschange = obs_reformat(obs)
                action = trained_policy.compute_single_action(obschange[agent_id], policy_id = agent_id)
            else:
                action = trained_policy.compute_single_action(obs[agent_id], policy_id = agent_id) 
            actions[agent_id] = action

        obs, rewards, done, truncated, infos = env.step(actions)
        all_infos.append(infos)
        all_profits.append(infos['overall_profit'])
        all_backlog.append(infos['total_backlog'])
        all_inv.append(infos['total_inventory'])

        _ +=1
    
    return all_infos, all_profits, all_backlog, all_inv


def average_simulation(num_runs, 
                       trained_policy, 
                       num_periods_,
                       env, 
                       model_config):
    #initialise to store variables
    av_infos = []
    av_profits = []
    av_backlog =[]
    av_inv = []

    for run in range(num_runs):
        all_infos, all_profits, all_backlog, all_inv = run_simulation(num_periods_, trained_policy, env, model_config)
        av_infos.append(all_infos)
        av_profits.append(all_profits)
        av_backlog.append(all_backlog)
        av_inv.append(all_inv)

    
    return av_infos, av_profits, av_backlog, av_inv



av_infos, av_profits, av_backlog, av_inv  = average_simulation(num_runs, trained_policy=ng1p, num_periods_=50, env=env_SS, model_config = model_config)


average_profit_list  = np.mean(av_profits, axis =0)
cumulative_profit_list = np.cumsum(average_profit_list, axis = 0)
std_profit_list = np.std(av_profits, axis =0)
print(f"Average Profit: {cumulative_profit_list[-1]}")
print(f"Standard Deviation Profit: {std_profit_list[-1]}")

#last_values_backlog = [backlog[-1] for backlog in av_backlog]
last_values_backlog = np.mean(av_backlog, axis = 0)
average_backlog = np.mean(last_values_backlog)
std_deviation_backlog = np.std(last_values_backlog)
median_backlog = np.median(last_values_backlog)
print(f"Average Backlog : {average_backlog}")
print(f"Standard Deviation Backlog : {std_deviation_backlog}")
print(f"Median Backlog : {median_backlog}")

#last_values_inv = [inv[-1] for inv in av_inv]
last_values_inv = np.median(av_inv, axis = 0)
average_inv = np.median(last_values_inv)
std_deviation_inv = np.std(last_values_inv)
print(f"Average Inventory : {average_inv}")
print(f"Standard Deviation Inventory : {std_deviation_inv}")


profit_data = {"av_profits": av_profits, 
                }

"""
with open(output_file, 'w') as file:
    json.dump(profit_data, file)

absolute_path = os.path.abspath(output_file)
print(f"The JSON file is saved at: {absolute_path}")
"""