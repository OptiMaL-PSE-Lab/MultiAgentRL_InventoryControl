import json 
import matplotlib.pyplot as plt
import numpy as np
import os 
import statistics

'''
todo: add checkpoint to the runn.py file so you can access the compute_action mean & sd
will need to restore the checkpoint on here 
'''
ng1_3n2p = [
    r"/Users/nikikotecha/Documents/PhD/trainingss/results/PPO_MultiAgentInvManagementDiv_2024-02-07_11-56-05zpasv7px/result.json",
    r"/Users/nikikotecha/Documents/PhD/trainingss/results/PPO_MultiAgentInvManagementDiv_2024-02-07_11-58-32ffzoj9sm/result.json",
    r"/Users/nikikotecha/Documents/PhD/trainingss/results/PPO_MultiAgentInvManagementDiv_2024-02-07_12-01-076mbux2kk/result.json"
]

ng1_6n2p = [
    r"/Users/nikikotecha/Documents/PhD/trainingss/results/PPO_MultiAgentInvManagementDiv_2024-02-06_14-44-053nyf2_uz/result.json",
r"/Users/nikikotecha/Documents/PhD/trainingss/results/PPO_MultiAgentInvManagementDiv_2024-02-06_14-51-29zpqpq7tt/result.json",
r"/Users/nikikotecha/Documents/PhD/trainingss/results/PPO_MultiAgentInvManagementDiv_2024-02-06_15-05-47hm60m7n8/result.json",
]

ng1_12n2p = [
    r"/Users/nikikotecha/Documents/PhD/trainingss/results/PPO_MultiAgentInvManagementDiv_2024-02-07_13-35-30e87zubfu/result.json",
r"/Users/nikikotecha/Documents/PhD/trainingss/results/PPO_MultiAgentInvManagementDiv_2024-02-07_13-39-20d38mqks0/result.json",
r"/Users/nikikotecha/Documents/PhD/trainingss/results/PPO_MultiAgentInvManagementDiv_2024-02-07_13-44-47hj9dvneu/result.json",
]

ng1_3n4p = [
    r"/Users/nikikotecha/Documents/PhD/trainingss/results/PPO_MultiAgentInvManagementDiv_2024-02-07_18-59-03ef41kyhv/result.json",
r"/Users/nikikotecha/Documents/PhD/trainingss/results/PPO_MultiAgentInvManagementDiv_2024-02-07_18-59-36d1eopod3/result.json",
r"/Users/nikikotecha/Documents/PhD/trainingss/results/PPO_MultiAgentInvManagementDiv_2024-02-07_18-59-457tnpdc9p/result.json",
]

ng2_3n2p = [
    r"/Users/nikikotecha/Documents/PhD/trainingss/results/PPO_MultiAgentInvManagementDiv_2024-02-14_17-31-39a0c4zn9f/result.json",
r"/Users/nikikotecha/Documents/PhD/trainingss/results/PPO_MultiAgentInvManagementDiv_2024-02-14_17-39-58jcz3bfq6/result.json",
r"/Users/nikikotecha/Documents/PhD/trainingss/results/PPO_MultiAgentInvManagementDiv_2024-02-14_20-07-33jjinsoss/result.json",

]

ng2_6n2p = [
r"/Users/nikikotecha/Documents/PhD/trainingss/results/PPO_MultiAgentInvManagementDiv_2024-02-15_23-38-04f9uwxe_r/result.json",
]

ng2_12n2p = [
r"/Users/nikikotecha/Documents/PhD/trainingss/results/PPO_MultiAgentInvManagementDiv_2024-02-19_10-45-192_ypvuv1/result.json",
]

g1_3n2p = [
    r"/Users/nikikotecha/Documents/PhD/trainingss/results/PPO_MultiAgentInvManagementDiv_2024-02-14_14-01-01fcmfw_2i/result.json",
r"/Users/nikikotecha/Documents/PhD/trainingss/results/PPO_MultiAgentInvManagementDiv_2024-02-14_14-00-29pje09paj/result.json",
r"/Users/nikikotecha/Documents/PhD/trainingss/results/PPO_MultiAgentInvManagementDiv_2024-02-14_13-58-25n4fwicdl/result.json",
]

g1_6n2p = [
r"/Users/nikikotecha/Documents/PhD/trainingss/results/PPO_MultiAgentInvManagementDiv_2024-02-15_11-15-57r79y_26u/result.json",
r"/Users/nikikotecha/Documents/PhD/trainingss/results/PPO_MultiAgentInvManagementDiv_2024-02-15_11-17-17ckbvo_zj/result.json",
r"/Users/nikikotecha/Documents/PhD/trainingss/results/PPO_MultiAgentInvManagementDiv_2024-02-15_23-41-23yiqe3u6s/result.json",
]

g1_12n2p = [
r"/Users/nikikotecha/Documents/PhD/trainingss/results/PPO_MultiAgentInvManagementDiv_2024-02-16_10-23-49deadjktl/result.json",
r"/Users/nikikotecha/Documents/PhD/trainingss/results/PPO_MultiAgentInvManagementDiv_2024-02-16_10-31-08bfqq3dv0/result.json",
r"/Users/nikikotecha/Documents/PhD/trainingss/results/PPO_MultiAgentInvManagementDiv_2024-02-19_10-43-271w6m6xd5/result.json",
]

g2_3n2p = [

]

g2_6n2p =[

]

g2_12n2p = [

]

iteration_to_check = 60
file_paths1 = [ng1_3n2p, ng2_3n2p, g1_3n2p]
file_paths2 = [ng1_6n2p, ng2_6n2p, g1_6n2p]
file_paths3 = [ng1_12n2p, ng2_12n2p, g1_12n2p]
number_agents = [1,1,1]

scale1 = [ng1_3n2p, ng1_6n2p, ng1_12n2p]
scale2 = [ng2_3n2p, ng2_6n2p, ng2_12n2p]
scale3 = [g1_3n2p, g1_6n2p, g1_12n2p]
number_agents1 = [6,12,24]
def normalize_rewards(rewards, num_agents):
    return [reward / num_agents for reward in rewards]

def normalize_rewards2(rewards):
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    normalized_rewards = [(reward - mean_reward) / std_reward for reward in rewards]
    return normalized_rewards


def training_figures(file_paths_list, iteration_to_check, number_agents):
    plt.figure(figsize=(8, 6))

    mean_time_iters = []
    std_time_iters = []
    for file_paths , label, no_agent in zip(file_paths_list, ['ng1_3n2p', 'ng2_3n2p', 'g1_3n2p'], number_agents):
        all_rewards = []
        for path in file_paths:
            with open(path, 'r') as f:
                json_str = f.read()
                json_list = json_str.split('\n')

            results_list = []
            for json_obj in json_list:
                if json_obj.strip():
                    results_list.append(json.loads(json_obj))
                    
            time_step = []
            episode_reward_mean = []
            for result in results_list:
                episode_reward_mean.append(result['episode_reward_mean'])
                time_step.append(result['time_this_iter_s'])

            mean_time = np.mean(time_step)
            mean_time_iters.append(mean_time)
            std_time = np.std(time_step)
            std_time_iters.append(std_time)

            #normalized_rewards = normalize_rewards(episode_reward_mean, no_agent)
            normalized_rewards = normalize_rewards2(episode_reward_mean)
            all_rewards.append(normalized_rewards)

        max_iterations = max(len(rewards) for rewards in all_rewards)
        padded_rewards = [r + [np.nan] * (max_iterations - len(r)) for r in all_rewards]

        avg_reward = np.nanmean(padded_rewards, axis=0)
        std_reward = np.nanstd(padded_rewards, axis=0)

        iteration_index = min(iteration_to_check, max_iterations - 1)

        highest_avg_reward_path = file_paths[np.argmax(avg_reward[iteration_index])]

        plt.plot(range(max_iterations), avg_reward, label=label)
        plt.fill_between(range(max_iterations), avg_reward - std_reward, avg_reward + std_reward, alpha=0.3, label='Standard Deviation', color='g')

    print("mean, std", mean_time_iters, std_time_iters) 
    #print("mean, std", m, s)    
    #plt.errorbar(file_paths_list, mean_time_iters, std_time_iters)

    #plt.xlabel('Iteration')
    #plt.ylabel('Reward')
    #plt.title('Training curves')
    #plt.legend()
    #plt.show()
    current_directory = os.getcwd()
    filename = 'training.png'
    #plt.savefig(filename, format='png', bbox_inches='tight', dpi=500)
    full_path = os.path.join(current_directory, filename)

    # Check if the file exists
    #if os.path.exists(full_path):
    #    print(f'The file {filename} has been saved in the following directory:\n{current_directory}')
    #else:
    #    print(f'The file {filename} was not found in the current working directory.')


    return highest_avg_reward_path


#highest_avg_reward_path = training_figures(file_paths1, iteration_to_check, number_agents)
#highest_avg_reward_path = training_figures(file_paths2, iteration_to_check, number_agents)
#highest_avg_reward_path = training_figures(file_paths3, iteration_to_check, number_agents)

highest_avg_reward_path = training_figures(scale1, iteration_to_check, number_agents1)
highest_avg_reward_path = training_figures(scale2, iteration_to_check, number_agents1)
highest_avg_reward_path = training_figures(scale3, iteration_to_check, number_agents1)



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