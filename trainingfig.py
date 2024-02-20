import json 
import matplotlib.pyplot as plt
import numpy as np
import os 
'''
todo: add checkpoint to the runn.py file so you can access the compute_action mean & sd
will need to restore the checkpoint on here 
'''

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

ng1_3n4p = [
    r"C:\Users\nk3118/ray_results\PPO_MultiAgentInvManagementDiv_2024-02-07_18-59-03ef41kyhv\result.json",
r"C:\Users\nk3118/ray_results\PPO_MultiAgentInvManagementDiv_2024-02-07_18-59-36d1eopod3\result.json",
r"C:\Users\nk3118/ray_results\PPO_MultiAgentInvManagementDiv_2024-02-07_18-59-457tnpdc9p\result.json",
]

ng2_3n2p = [
    r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-02-14_17-31-39a0c4zn9f\result.json",
r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-02-14_17-39-58jcz3bfq6\result.json",
r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-02-14_20-07-33jjinsoss\result.json",

]
g1_3n2p = [
    r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-02-14_14-01-01fcmfw_2i\result.json",
r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-02-14_14-00-29pje09paj\result.json",
r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-02-14_13-58-25n4fwicdl\result.json",
]

iteration_to_check = 60
file_paths = [ng1_3n2p, ng2_3n2p, g1_3n2p]
number_agents = [1,1,1]


def normalize_rewards(rewards, num_agents):
    return [reward / num_agents for reward in rewards]

def normalize_rewards2(rewards):
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    normalized_rewards = [(reward - mean_reward) / std_reward for reward in rewards]
    return normalized_rewards


def training_figures(file_paths_list, iteration_to_check, number_agents):
    plt.figure(figsize=(8, 6))

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

            episode_reward_mean = []
            for result in results_list:
                episode_reward_mean.append(result['episode_reward_mean'])

            normalized_rewards = normalize_rewards(episode_reward_mean, no_agent)
            #normalized_rewards = normalize_rewards2(episode_reward_mean)
            all_rewards.append(normalized_rewards)


        max_iterations = max(len(rewards) for rewards in all_rewards)
        padded_rewards = [r + [np.nan] * (max_iterations - len(r)) for r in all_rewards]

        avg_reward = np.nanmean(padded_rewards, axis=0)
        std_reward = np.nanstd(padded_rewards, axis=0)

        iteration_index = min(iteration_to_check, max_iterations - 1)

        highest_avg_reward_path = file_paths[np.argmax(avg_reward[iteration_index])]

        plt.plot(range(max_iterations), avg_reward, label=label)
        plt.fill_between(range(max_iterations), avg_reward - std_reward, avg_reward + std_reward, alpha=0.3, label='Standard Deviation', color='g')


    plt.xlabel('Iteration')
    plt.ylabel('Reward')
    plt.title('Training curves')
    plt.legend()
    plt.show()
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


highest_avg_reward_path = training_figures(file_paths, iteration_to_check, number_agents)

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