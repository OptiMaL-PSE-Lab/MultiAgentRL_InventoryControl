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


    fig, ax = plt.subplots(1,1,figsize=(12,8), layout = 'constrained')
    cols = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
    #cols = ['tab:blue', 'tab:red', 'tab:brown']
    for sublist, col in zip(all_entropies, cols): 
        ax.plot(sublist, color = col, linestyle = 'solid')
    ax.legend(['0.0', '0.5', '2.0', '0.5', '1.0', '2.0'], frameon=False, fontsize=14)
    #ax.spines['right'].set_visible(False)
    #ax.spines['top'].set_visible(False)
    ax.set_xlabel('Training Iteration')
    ax.set_ylabel('Entropy')
    ax.set_xlim(0, 60)
    plt.show()
    #fig.savefig('entropy_noise.png', dpi= 1200)

    return all_entropies


ippo18 = [r"ray_results/PPO_MultiAgentInvManagementDiv_2024-04-11_09-18-09gpdriji_/result.json"]
mappo18 = [r"ray_results/PPO_MultiAgentInvManagementDiv_2024-04-10_20-41-03g7yt7xc3/result.json"]
gmappo18 = [r"ay_results/PPO_MultiAgentInvManagementDiv_2024-04-10_20-40-58vb6xruob/result.json"]
g2_18 = [r"ray_results/PPO_MultiAgentInvManagementDiv_2024-04-10_22-16-53zax54im2/result.json"]
g2_18noise = [r"ray_results/PPO_MultiAgentInvManagementDiv_2024-05-15_11-14-05l1k6u06k/result.json"]
file_paths18 = [ippo18, mappo18, gmappo18, g2_18, g2_18noise]

noise_18_0 = [r"C:ray_results/PPO_MultiAgentInvManagementDiv_2024-05-12_18-09-12pea61thx/result.json"]
noise_18_01 = [r"C:ray_results/PPO_MultiAgentInvManagementDiv_2024-05-12_18-10-223euvrde2/result.json"]
noise_18_02 = [r"C:ray_results/PPO_MultiAgentInvManagementDiv_2024-05-15_11-14-05l1k6u06k/result.json"]
noise_18_05 = [r"ray_results/PPO_MultiAgentInvManagementDiv_2024-05-14_13-47-33xyca2ajl/result.json"]
noise_18_75 = [r"ray_results/PPO_MultiAgentInvManagementDiv_2024-05-15_11-16-012_kinfxk/result.json"]
noise_18_10 = [r"ray_results/PPO_MultiAgentInvManagementDiv_2024-05-14_13-48-08fskh2eg9/result.json"]
noise_18_20 = [r"ray_results/PPO_MultiAgentInvManagementDiv_2024-05-16_10-29-44k80w0cni/result.json"]
file_paths_noise_all  = [noise_18_0, noise_18_01, noise_18_02, noise_18_05, noise_18_10, noise_18_20]
file_paths_noise = [noise_18_0, noise_18_05, noise_18_20]

iteration_to_check = 60 
number_agents = [18,18,18,18,18]

training_figures(file_paths_noise, iteration_to_check, number_agents)
