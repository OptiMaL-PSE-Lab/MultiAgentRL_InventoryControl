from email import policy
from env3SA import InvManagementDiv
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
from gymnasium.wrappers import EnvCompatibility


ray.init()

output_file = 'single_6.json'

config = {}
# Register environment
def env_creator(config):
    return EnvCompatibility(InvManagementDiv(config = config))
config = {}
tune.register_env("InvManagementDiv", env_creator)   # noqa: E501



ng1 =  r"C:\Users\nk3118\ray_results\PPO_InvManagementDiv_2024-06-06_17-02-51jv28gwfr\checkpoint_000001"
ng1p = Algorithm.from_checkpoint(ng1) #load the trained policy

env_SS = InvManagementDiv(config = config)
model_config = {}
num_runs = 20
def run_simulation(num_periods_, trained_policy, env, model_config):
    all_infos = []
    all_profits = []
    all_backlog = []
    all_inv = []


    obs = env.reset()
    for _ in range(num_periods_):

        actions = trained_policy.compute_single_action(obs) 

        state, rewards, done, infos = env.step(actions)
        all_infos.append(infos)
        all_profits.append(infos['profit'])
        all_backlog.append(infos['backlog'])
        all_inv.append(infos['inv'])

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


with open(output_file, 'w') as file:
    json.dump(profit_data, file)

absolute_path = os.path.abspath(output_file)
print(f"The JSON file is saved at: {absolute_path}")
