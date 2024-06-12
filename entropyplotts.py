from collections import ChainMap
import json
from math import e
from turtle import color 
import matplotlib.pyplot as plt
import numpy as np
import os 
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from torch import layout


def training_figures(file_paths_list, iteration_to_check, number_agents):

    fig, ax = plt.subplots(1,1,figsize=(10,6), layout = 'constrained')
    all_entropies = []
    for file_paths , label, no_agent in zip(file_paths_list, ['0.0', '0.1', '0.2', '0.5', '1.0', '2.0'], number_agents):
        for path in file_paths:
            with open(path, 'r') as f:
                json_str = f.read()
                json_list = json_str.split('\n')

            results_list = []
            for json_obj in json_list:
                if json_obj.strip():
                    results_list.append(json.loads(json_obj))

            entropy_vales = [] 

            for result in results_list:
                iteration = result['training_iteration']
                entropy_vales.append(result['info']['learner']['0_00']['learner_stats']['entropy'])
        all_entropies.append(entropy_vales)


    fig, ax = plt.subplots(1,1,figsize=(10,6), layout = 'constrained')
    cols = ['tab:blue','tab:orange','tab:red']
    for sublist, col in zip(all_entropies, cols): 
        ax.plot(sublist, color = col, linestyle = 'solid')
    ax.legend(['0.0', '0.1', '0.5'], frameon=False, fontsize=14)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('Training Iteration')
    ax.set_ylabel('Entropy')
    ax.set_xlim(0, 60)
    plt.show()
    fig.savefig('entropy_noises.png', dpi= 1200)

    return all_entropies


mappo_24 = [r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-05-06_15-41-461marte8k\result.json"]
ippo_24 =  [r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-05-06_15-42-44nu1jjpuk\result.json"]
gmappo_24 = [r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-05-06_15-45-588zbz5wvr\result.json"]
g2_noise0 = [r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-05-07_08-10-33cjpows2l\result.json"]
g2_noise1 = [r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-05-07_08-35-34vpwck8mj\result.json"]

file_paths_24 = [mappo_24, ippo_24, gmappo_24, g2_noise0, g2_noise1]


mappo_12 = [r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-05-02_13-36-43l0j31ce0\result.json"]
ippo_12 =  [r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-05-01_16-12-06owdt78yd\result.json"]
gmappo_12 = [r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-05-01_16-13-281w63y_j_\result.json"]
g2_12_noise0 = [r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-05-01_12-00-15zbm_m4ax\result.json"]
g2_12_noise1 = [r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-05-01_16-11-11wakdzoi7\result.json"]

file_paths_12 = [mappo_12, ippo_12, gmappo_12, g2_12_noise0, g2_12_noise1]

noise_18_0 = [r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-05-12_18-09-12pea61thx\result.json"]
noise_18_01 = [r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-05-12_18-10-223euvrde2\result.json"]
noise_18_02 = [r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-05-15_11-14-05l1k6u06k\result.json"]
noise_18_05 = [r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-05-14_13-47-33xyca2ajl\result.json"]
noise_18_75 = [r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-05-15_11-16-012_kinfxk\result.json"]
noise_18_10 = [r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-05-14_13-48-08fskh2eg9\result.json"]
noise_18_20 = [r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-05-16_10-29-44k80w0cni\result.json"]
file_paths_noise_all  = [noise_18_0, noise_18_01, noise_18_02, noise_18_05, noise_18_10, noise_18_20]
file_paths_noise = [noise_18_0, noise_18_01, noise_18_05]
iteration_to_check = 60 
number_agents = [18,18, 18]

training_figures(file_paths_noise, iteration_to_check, number_agents)
