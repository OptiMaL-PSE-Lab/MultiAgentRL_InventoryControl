from collections import ChainMap
import json 
import matplotlib.pyplot as plt
import numpy as np
import os 
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from torch import layout

# --------------------------------------------  ng1 --------------------------------------------


ng1_3n2p = [
    r"C:\Users\nk3118/ray_results\PPO_MultiAgentInvManagementDiv_2024-02-07_11-56-05zpasv7px\result.json",
    r"C:\Users\nk3118/ray_results\PPO_MultiAgentInvManagementDiv_2024-02-07_11-58-32ffzoj9sm\result.json",
    r"C:\Users\nk3118/ray_results\PPO_MultiAgentInvManagementDiv_2024-02-07_12-01-076mbux2kk\result.json"
]

ng1_6n2p = [
    r"C:\Users\nk3118/ray_results\PPO_MultiAgentInvManagementDiv_2024-02-06_14-44-053nyf2_uz\result.json",
r"C:\Users\nk3118/ray_results\PPO_MultiAgentInvManagementDiv_2024-02-06_14-51-29zpqpq7tt\result.json",
r"C:\Users\nk3118/ray_results\PPO_MultiAgentInvManagementDiv_2024-02-06_15-05-47hm60m7n8\result.json",
]

ng1_12n2p = [
    r"C:\Users\nk3118/ray_results\PPO_MultiAgentInvManagementDiv_2024-02-07_13-35-30e87zubfu\result.json",
r"C:\Users\nk3118/ray_results\PPO_MultiAgentInvManagementDiv_2024-02-07_13-39-20d38mqks0\result.json",
r"C:\Users\nk3118/ray_results\PPO_MultiAgentInvManagementDiv_2024-02-07_13-44-47hj9dvneu\result.json",
]

ng1_16n2p = [r"C:\Users\nk3118\Documents\sS\result_ng1_32.json" #combined the two files below
    #r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-02-28_10-52-50py6yw7c9\result.json",
            # r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-02-28_13-44-25ek8bd981\result.json"
]

ng1_24n2p = [r"c:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-03-04_11-11-347xe53e4m/result.json"]

# --------------------------------------------  ng2 --------------------------------------------

ng2_3n2p = [
    r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-02-14_17-31-39a0c4zn9f\result.json",
r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-02-14_17-39-58jcz3bfq6\result.json",
r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-02-14_20-07-33jjinsoss\result.json",
]

ng2_6n2p = [
r"c:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-02-15_23-38-04f9uwxe_r\result.json",
]

ng2_12n2p = [
#r"c:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-02-19_10-45-192_ypvuv1\result.json",
r"c:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-03-11_14-06-217y1zrgx2\result.json" #this is new vs
]

ng2_16n2p = [r"C:\Users\nk3118\Documents\sS\result_ng2_32.json", #combined the two files below
             r"c:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-03-12_15-42-223423r374\result.json"
    #r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-02-28_10-43-55wdgfzx11\result.json",
#r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-02-28_14-04-01auuos13n\result.json",
]

ng2_24n2p = [r"C:\Users\nk3118/ray_results\PPO_MultiAgentInvManagementDiv_2024-03-04_19-37-31qum_qv81\result.json"]

# -------------------------------------------- g1 --------------------------------------------

g1_3n2p = [
    #r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-02-14_14-01-01fcmfw_2i\result.json",
#r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-02-14_14-00-29pje09paj\result.json",
#r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-02-14_13-58-25n4fwicdl\result.json",
#r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-03-11_09-21-52eoxu8djo\result.json" #this is new vs
r"c:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-03-20_11-38-088hhtc7tg\result.json" #this is new vs

]

g1_6n2p = [
#r"c:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-02-15_11-15-57r79y_26u\result.json",
#r"c:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-02-15_11-17-17ckbvo_zj\result.json",
#r"c:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-02-15_23-41-23yiqe3u6s\result.json", 
r"c:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-03-09_19-10-07vq1_rwzf\result.json", #this is new vs
]

g1_12n2p = [
#r"c:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-02-16_10-23-49deadjktl\result.json",
#r"c:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-02-16_10-31-08bfqq3dv0\result.json",
#r"c:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-02-19_10-43-271w6m6xd5\result.json",
#r"c:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-02-20_18-02-24i9zbiioa\result.json",
#r"c:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-02-26_11-39-5432h72q5x\result.json",
r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-03-10_23-32-256mmmvasp\result.json" #this is new vs
]

g1_16n2p = [#r"c:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-02-27_13-58-27v_j7y4wt\result.json", 
            #r"C:\Users\nk3118\Documents\sS\result_g1_32.json",
            r"c:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-03-10_00-33-224winipu1\result.json" #this is new vs 
    ]

g1_24n2p = [#r"C:\Users\nk3118\Documents\sS\result_g1_48.json",
            r"c:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-03-06_20-59-44ar37rczw\result.json"] #this is new vs 

# -------------------------------------------- g1 --------------------------------------------

g2_3n2p = [#r"C:\Users\nk3118\Documents\sS\result_g2_6.json" #the below only goes to 55 so it combines the other file
    #r"c:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-02-21_21-03-10xa39f2kh\result.json",
    r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-03-11_11-19-354ekr_jrt\result.json" #this is new vs

]

g2_6n2p = [#r"c:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-02-22_14-29-24vla7q9im\result.json",
           r"c:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-03-09_14-55-28kuvg86t4\result.json" #this is new vs
           ]

g2_12n2p = [
            #r"c:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-02-21_11-56-29ipv5qji4\result.json",
             #r"c:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-02-22_20-14-23tu6r2u9b\result.json",
             r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-03-10_13-48-310gk7ntjx\result.json" #this is new vs
]

g2_16n2p = [
    #r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-02-26_23-03-100_1btam3\result.json",
    r"c:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-03-09_01-54-194ml6r6d_\result.json" #new vs
]

g2_24n2p = [#r"C:\Users\nk3118\Documents\sS\result_g2_48.json",
            r"c:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-03-07_22-58-1331e7d1y_\result.json" #new vs
            ]

iteration_to_check = 60

ng2_new6 = [r"c:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-03-20_14-15-0053qk6p3z\result.json"]
#file paths grouped per method 
file_paths6 = [ng1_3n2p, ng2_new6, g1_3n2p, g2_3n2p]
#file_paths12 =[ng1_6n2p, ng2_6n2p, g1_6n2p, g2_6n2p]
#file_paths24 = [ng2_12n2p, ng1_12n2p, g1_12n2p, g2_12n2p]
file_paths32 = [ng1_16n2p, ng2_16n2p, g2_16n2p, g1_16n2p]
file_paths48 = [ng1_24n2p, ng2_24n2p, g2_24n2p, g1_24n2p]

g1= [r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-04-04_12-04-12fzniw9d1\result.json"]
ng2 = [r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-04-04_11-56-101jbkgqou\result.json"]
g2 = [r"c:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-04-04_15-33-39sojltnl4\result.json"]
n4 = [r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-04-05_10-53-524go0l40j\result.json"]

g124= [r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-04-08_14-54-09e34hq7ae\result.json"]
ng224 = [r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-04-08_14-50-43wxg9s41x\result.json"]
g224 = [r"c:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-04-10_10-14-32uhxhdvnp\result.json"]
ng124 = [r"c:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-04-08_15-16-48c488e0lg\result.json"]

file_paths12 = [n4, ng2, g1, g2]
file_paths24 = [ng124, ng224, g124, g224]
number_agents = [1,1,1,1,1]

mappo_24 = [r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-05-06_15-41-461marte8k\result.json"]
ippo_24 =  [r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-05-06_15-42-44nu1jjpuk\result.json"]
gmappo_24 = [r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-05-06_15-45-588zbz5wvr\result.json"]
g2_noise0 = [r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-05-07_08-10-33cjpows2l\result.json"]
g2_noise1 = [r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-05-07_08-35-34vpwck8mj\result.json"]

file_paths = [mappo_24, ippo_24, gmappo_24, g2_noise0, g2_noise1]

def normalize_rewards(rewards, num_agents):
    return [reward / num_agents for reward in rewards]

def normalize_rewards2(rewards):
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    normalized_rewards = [(reward - mean_reward) / std_reward for reward in rewards]
    return normalized_rewards


def training_figures(file_paths_list, iteration_to_check, number_agents):

    mean_training_times = []
    stds_training_times = []
    fig, ax = plt.subplots(1,1,figsize=(10,6), layout = 'constrained')

    for file_paths , label, no_agent in zip(file_paths_list, ['IPPO', 'MAPPO', 'GMAPPO', 'GMAPPO with Pooling', 'GMAPPO with Pooling & Noise'], number_agents):
        all_rewards = []
        mean_training_times_path = []
        stds_training_times_path = []
        for path in file_paths:
            with open(path, 'r') as f:
                json_str = f.read()
                json_list = json_str.split('\n')

            results_list = []
            for json_obj in json_list:
                if json_obj.strip():
                    results_list.append(json.loads(json_obj))

            episode_reward_mean = []
            time_step = []
            for result in results_list:
                iteration = result['training_iteration']
                episode_reward_mean.append(result['episode_reward_mean'])
                time_step.append(result['time_this_iter_s'])

            time_step = np.array(time_step)
            z_scores = stats.zscore(time_step)
            time_steps = time_step[np.abs(z_scores) < 3]

            mean_training_times_path.append(np.median(time_step))
            stds_training_times_path.append(np.std(time_steps))  

        normalized_rewards = normalize_rewards(episode_reward_mean, no_agent)
        #normalized_rewards = normalize_rewards2(episode_reward_mean)
        all_rewards.append(normalized_rewards)

        mean_training_times.append(np.mean(mean_training_times_path))
        stds_training_times.append(np.mean(stds_training_times_path))

        max_iterations = max(len(rewards) for rewards in all_rewards)
        padded_rewards = [r + [np.nan] * (max_iterations - len(r)) for r in all_rewards]

        avg_reward = np.nanmean(padded_rewards, axis=0)
        std_reward = np.nanstd(padded_rewards, axis=0)

        iteration_index = min(iteration_to_check, max_iterations - 1)

        highest_avg_reward_path = file_paths[np.argmax(avg_reward[iteration_index])]
        ax.plot(range(max_iterations), avg_reward, label=label)
        #ax.fill_between(range(max_iterations), avg_reward - std_reward, avg_reward + std_reward, alpha=0.3, label='Standard Deviation')
        ax.set_xlabel('Iteration', fontsize=18)
        ax.set_ylabel('Normalized Reward',fontsize=18)
        #ax.set_yscale('log')
    ax.legend(frameon = False, fontsize=14)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    labels = ['6', '12', '24', '32', '48']
    #for label in labels:
    #    fig.savefig(f'training_{label}.png', dpi = 1200)
    plt.show()

    return highest_avg_reward_path, mean_training_times, stds_training_times, avg_reward, std_reward


highest_avg_reward_path6, mean_training_times6, stds_training_times6, avg_reward6, std_reward6 = training_figures(file_paths, iteration_to_check, number_agents)
'''
highest_avg_reward_path12, mean_training_times12, stds_training_times12, avg_reward12, std_reward12 = training_figures(file_paths12, iteration_to_check, number_agents)
highest_avg_reward_path24, mean_training_times24, stds_training_times24, avg_reward24, std_reward24 = training_figures(file_paths24, iteration_to_check, number_agents)
highest_avg_reward_path32, mean_training_times32, stds_training_times32, avg_reward32, std_reward32 = training_figures(file_paths32, iteration_to_check, number_agents)
highest_avg_reward_path48, mean_training_times48, stds_training_times48, avg_reward48, std_reward48 = training_figures(file_paths48, iteration_to_check, number_agents)

mean_training_times_dict = {
    '3n2p': mean_training_times6,
    '6n2p': mean_training_times12,
    '12n2p': mean_training_times24,
    '16n2p': mean_training_times32,
    '24n2p': mean_training_times48,
}

stds_training_times_dict = {    
    '3n2p': stds_training_times6,
    '6n2p': stds_training_times12,
    '12n2p': stds_training_times24,
    '16n2p': stds_training_times32,
    '24n2p': stds_training_times48,
}
'''

def error_bars_method(mean_training_times_dict, stds_training_times_dict):
    agents = ['6', '12', '24', '32', '48']  # Example agent names
    methods = ['ng1', 'ng2', 'g1', 'g2']  # Example method names

    # Convert dictionaries to arrays
    mean_training_times = np.array([mean_training_times_dict[key] for key in mean_training_times_dict])
    stds_training_times = np.array([stds_training_times_dict[key] for key in stds_training_times_dict])

    # Plot error bar graphs for each method
    fig, axs = plt.subplots(1, len(methods), figsize=(12, 6), sharey=True)
    for i, method in enumerate(methods):
        ax = axs[i]
        x = np.arange(len(agents))
        y = mean_training_times[:, i]
        yerr = stds_training_times[:, i]
        ax.errorbar(x, y, yerr=yerr, fmt='o', capsize=5)
        ax.set_title(method)
        ax.set_xticks(x)  # Set the ticks based on the positions
        ax.set_xticklabels(agents)  # Set the tick labels to be the agent names
    ax.legend(frameon = False, fontsize=14)
    axs[0].set_ylabel('Mean Training Time')
    axs[-1].set_xlabel('Agents')
    plt.tight_layout()
    plt.show()

def error_bars_method_all(mean_training_times_dict, stds_training_times_dict):
    agents = ['6', '12', '24', '32', '48']  # Example agent names
    methods = ['IPPO', 'MAPPO', 'GMAPPO', 'GMAPPO with Pooling']  # Example method names

    # Convert dictionaries to arrays
    mean_training_times = np.array([mean_training_times_dict[key] for key in mean_training_times_dict])
    stds_training_times = np.array([stds_training_times_dict[key] for key in stds_training_times_dict])

    # Plot error bar graphs for each method
    fig, ax = plt.subplots(1, 1, figsize=(12, 6), sharey=True)
    for i, method in enumerate(methods):
        x = np.arange(len(agents))
        y = mean_training_times[:, i]
        yerr = stds_training_times[:, i]
        ax.errorbar(x, y, yerr=yerr, fmt='o', capsize=5, label = method)
        ax.set_xticks(x)  # Set the ticks based on the positions
        ax.set_xticklabels(agents)  # Set the tick labels to be the agent names
    ax.legend(frameon = False, fontsize=14)
    ax.set_ylabel('Medium Training Time per iteration, s')
    ax.set_xlabel('Number of Agents')
    ax.legend(frameon = False, fontsize=14)
    plt.tight_layout()
    fig.savefig('training_times.png', dpi = 1200)
    plt.show()

def error_bars_agents(mean_training_times_dict, stds_training_times_dict):
    agents = ['6', '12', '24', '32', '48']  # Example agent names
    methods = ['ng1', 'ng2', 'g1', 'g2']  # Example method names

    mean_training_times = np.array([mean_training_times_dict[key] for key in mean_training_times_dict])
    stds_training_times = np.array([stds_training_times_dict[key] for key in stds_training_times_dict])

    plt.figure(figsize=(12, 6))
    for i, agent in enumerate(agents):
        plt.subplot(1, len(agents), i + 1)
        x = np.arange(len(methods))
        y = mean_training_times[i, :]
        yerr = stds_training_times[i, :]
        plt.errorbar(x, y, yerr=yerr, fmt='o', capsize=5)
        plt.title(agent)
        plt.xlabel('Methods')
        plt.ylabel('Mean Training Time')

    plt.tight_layout()
    plt.show()
   
def heatmap(mean_training_times_dict, stds_training_times_dict):
    mean_training_times = np.array([mean_training_times_dict[key] for key in mean_training_times_dict])
    stds_training_times = np.array([stds_training_times_dict[key] for key in stds_training_times_dict])

    agents = ['6', '12', '24', '32', '48']  # Example agent names
    methods = ['ng1', 'ng2', 'g1', 'g2']  # Example method names

    plt.figure(figsize=(10, 6))
    colors = [(0, 'green'), (0.5, 'yellow'), (1, 'red')]  # Green to Yellow to Red
    cmap = LinearSegmentedColormap.from_list('custom', colors)

    plt.imshow(mean_training_times, cmap=cmap, interpolation='nearest')

    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label('Mean Training Time')

    # Add labels
    plt.title('Mean Training Time Heatmap')
    plt.xlabel('Methods')
    plt.ylabel('Agents')
    plt.xticks(np.arange(len(methods)), methods)
    plt.yticks(np.arange(len(agents)), agents)
    plt.show()

#error_bars_method_all(mean_training_times_dict, stds_training_times_dict)
# error_bars_agents(mean_training_times_dict, stds_training_times_dict)
#heatmap(mean_training_times_dict, stds_training_times_dict)

def plot_mean_rewards(file_paths, label, color, linestyle):
    all_rewards = []

    for path in file_paths:
        with open(path, 'r') as f:
            json_str = f.read()
            json_list = json_str.split('\n')

        results_list = []
        for json_obj in json_list:
            if json_obj.strip():
                results_list.append(json.loads(json_obj))

        episode_reward_mean = [result['episode_reward_mean'] for result in results_list]
        all_rewards.append(episode_reward_mean)

    max_iterations = max(len(rewards) for rewards in all_rewards)
    padded_rewards = [r + [np.nan] * (max_iterations - len(r)) for r in all_rewards]

    avg_reward = np.nanmean(padded_rewards, axis=0)
    std_reward = np.nanstd(padded_rewards, axis=0)

    iteration_to_check = min(70, max_iterations - 1)

    plt.plot(range(max_iterations), avg_reward, label=label, color=color, linestyle = linestyle)
    #plt.fill_between(range(max_iterations), avg_reward - std_reward, avg_reward + std_reward, alpha=0.3, label=f'Standard Deviation ({label})', color='grey')

iteration_to_check = 70