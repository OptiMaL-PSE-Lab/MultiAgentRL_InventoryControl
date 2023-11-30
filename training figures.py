import json 
import matplotlib.pyplot as plt
import numpy as np
'''
todo: add checkpoint to the runn.py file so you can access the compute_action mean & sd
will need to restore the checkpoint on here 
'''

file_paths_env3 = [
    "/Users/nikikotecha/ray_results/PPO_MultiAgentInvManagementDiv_2023-11-14_14-24-23z24j6i1z/result.json",
"/Users/nikikotecha/ray_results/PPO_MultiAgentInvManagementDiv_2023-11-13_11-42-28z2eiouux/result.json", 
"/Users/nikikotecha/ray_results/PPO/PPO_MultiAgentInvManagementDiv_4b099_00000_0_2023-11-13_15-06-59/result.json",
"/Users/nikikotecha/ray_results/PPO/PPO_MultiAgentInvManagementDiv_8ed7e_00000_0_2023-11-14_12-08-44/result.json",
]

single_agentreal = ["/Users/nikikotecha/ray_results/PPO_MultiAgentInvManagementDiv_2023-11-16_15-53-14a7azmbdo/result.json",
                    "/Users/nikikotecha/ray_results/PPO/PPO_MultiAgentInvManagementDiv_60221_00000_0_2023-11-20_11-53-10/result.json"] #second one?
single = ["/Users/nikikotecha/ray_results/PPO/PPO_MultiAgentInvManagementDiv_e5e06_00000_0_2023-11-21_20-52-35/result.json"] #this was single on new env, env3
single_agent_env3 = [#"/Users/nikikotecha/ray_results/PPO/PPO_MultiAgentInvManagementDiv_1466d_00000_0_2023-11-18_13-47-57/result.json", 
                #"/Users/nikikotecha/ray_results/PPO/PPO_MultiAgentInvManagementDiv_4c57c_00000_0_2023-11-19_14-45-35/result.json",
                #"/Users/nikikotecha/ray_results/PPO_MultiAgentInvManagementDiv_2023-11-20_09-57-19wjot7gsb/result.json",
                #"/Users/nikikotecha/ray_results/PPO_MultiAgentInvManagementDiv_2023-11-20_10-15-30oinxwunj/result.json" this was when we did max(o, svalues)
                #"/Users/nikikotecha/ray_results/PPO_MultiAgentInvManagementDiv_2023-11-20_10-37-17pwgqlbgr/result.json" this was when we did max(o, rescaled s values)
                 #"/Users/nikikotecha/ray_results/PPO/PPO_MultiAgentInvManagementDiv_711b2_00000_0_2023-11-21_17-43-12/result.json" , #this was env3 file, unoptimised parameters,
                 "/Users/nikikotecha/ray_results/PPO/PPO_MultiAgentInvManagementDiv_e11c4_00000_0_2023-11-21_22-25-30/result.json", #this was one policy (single agent) with env3run file with optimised parameters
                 "/Users/nikikotecha/ray_results/PPO/PPO_MultiAgentInvManagementDiv_3e283_00000_0_2023-11-22_15-31-45/result.json"]

multi_agent_env3 = ["/Users/nikikotecha/ray_results/PPO/PPO_MultiAgentInvManagementDiv_82dea_00000_0_2023-11-21_19-23-55/result.json", #this was env3run file with optimised parameters 
                    "/Users/nikikotecha/ray_results/PPO/PPO_MultiAgentInvManagementDiv_03040_00000_0_2023-11-22_17-24-37/result.json"]
file_paths_env2 = []
trial = ["/Users/nikikotecha/ray_results/PPO/PPO_MultiAgentInvManagementDiv1_9ca91_00000_0_2023-11-30_15-24-54/result.json"]

def training_figures(file_paths_list, iteration_to_check):
    plt.figure(figsize=(8, 6))

    for file_paths , label in zip(file_paths_list, ['Single Agent', 'Multi Agent']):
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
            all_rewards.append(episode_reward_mean)

        max_iterations = max(len(rewards) for rewards in all_rewards)
        padded_rewards = [r + [np.nan] * (max_iterations - len(r)) for r in all_rewards]

        avg_reward = np.nanmean(padded_rewards, axis=0)
        std_reward = np.nanstd(padded_rewards, axis=0)

        iteration_index = min(iteration_to_check, max_iterations - 1)

        highest_avg_reward_path = file_paths[np.argmax(avg_reward[iteration_index])]

        plt.plot(range(max_iterations), avg_reward, label=path)
        plt.fill_between(range(max_iterations), avg_reward - std_reward, avg_reward + std_reward, alpha=0.3, label='Standard Deviation', color='g')


    plt.xlabel('Iteration')
    plt.ylabel('Reward')
    plt.title('Training curves')
    plt.legend()
    plt.show()

    return highest_avg_reward_path

iteration_to_check = 70

#file_paths_list = [single_agent_env3, multi_agent_env3]
list_trial = [trial]
highest_avg_reward_path = training_figures(list_trial, iteration_to_check)
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
    plt.fill_between(range(max_iterations), avg_reward - std_reward, avg_reward + std_reward, alpha=0.3, label=f'Standard Deviation ({label})', color='grey')

iteration_to_check = 70

plt.figure(figsize=(8, 6))
plot_mean_rewards(list_trial, label='Single Agent', color='blue', linestyle ='dashed')
#plot_mean_rewards(multi_agent_env3, label='Multi Agent', color='orange', linestyle = 'solid')

plt.xlabel('Iteration')
plt.ylabel('Reward')
plt.legend()
plt.show()

"""def training_figures(file_paths, iteration_to_check):
    all_rewards = []
    for path in file_paths:
        with open(path, 'r') as f:  # noqa: E501
            json_str = f.read()
            json_list = json_str.split('\n')  


        results_list = []
        for json_obj in json_list:
            if json_obj.strip():  # Skip empty lines
                results_list.append(json.loads(json_obj))
        
        episode_reward_mean = []
        for result in results_list:
            episode_reward_mean.append(result['episode_reward_mean'])
        all_rewards.append(episode_reward_mean)


    max_iterations = max(len(rewards) for rewards in all_rewards)
    padded_rewards = [r + [np.nan] * (max_iterations - len(r)) for r in all_rewards]

    avg_reward = np.nanmean(padded_rewards, axis = 0)
    std_reward = np.nanstd(padded_rewards, axis = 0)
    
    iteration_index = min(iteration_to_check, max_iterations - 1)
    
    highest_avg_reward_path = print(file_paths[np.argmax(avg_reward[iteration_index])])


    plt.figure(figsize=(8, 6))
    plt.plot(range(max_iterations), avg_reward, label='Average Reward', color='b')
    plt.fill_between(range(max_iterations), avg_reward - std_reward, avg_reward + std_reward, alpha=0.3, label='Standard Deviation', color='g')
    plt.xlabel('Iteration')
    plt.ylabel('Reward')
    plt.title('Training curves env3')
    plt.legend()
    plt.show()

    
    return highest_avg_reward_path

iteration_to_check = 70

highest_avg_reward_path = training_figures(file_paths_env2, iteration_to_check)
highest_avg_reward_path = training_figures(single_agent, iteration_to_check)
"""

# list of JSON objects in 'results_list'

"""
episode_reward_min = []
episode_reward_max = []
episode_reward_mean = []
time = []
current_kl_coeff = []
kl = []
entropy =[]
lr = [] 
policy_loss = []

for result in results_list:
    episode_reward_min.append(result['episode_reward_min'])
    episode_reward_max.append(result['episode_reward_max'])
    episode_reward_mean.append(result['episode_reward_mean'])
    time.append(result['training_iteration'])  # Assuming iteration is available

    learner_stats = result['info']['learner']['default_policy']['learner_stats']
    entropy.append(learner_stats.get('entropy', 0.0))  # Use a default value if 'entropy' is missing
    current_kl_coeff.append(learner_stats.get('cur_kl_coeff', 0.0))  # Use a default value if 'cur_kl_coeff' is missing
    kl.append(learner_stats.get('kl', 0.0))
    policy_loss.append(learner_stats.get('policy_loss', 0.0))  # Use a default value if 'policy_loss' is missing

    


plt.plot(time, episode_reward_min, label='Min Reward')
plt.plot(time, episode_reward_max, label='Max Reward')
plt.plot(time, episode_reward_mean, label='Mean Reward')
plt.fill_between(time, episode_reward_min, episode_reward_max, alpha = 0.3)  # noqa: E501
plt.xlabel('Iteration')
plt.ylabel('Episode Reward')
plt.title('Episode Rewards over Iterations')
plt.legend()
plt.show()

# Create a figure for the first plot (current kl coefficient)
plt.figure(figsize=(10, 5))
plt.plot(time, kl, label='Current KL Coefficient', color='red')
plt.xlabel('Iteration')
plt.ylabel('Current KL Coefficient')
plt.title('Current KL Coefficient over Iterations')
plt.legend()
plt.show()

# Create a figure for the second plot (entropy)
plt.figure(figsize=(10, 5))
plt.plot(time, entropy, label='Entropy', color='green')
plt.xlabel('Iteration')
plt.ylabel('Entropy')
plt.title('Entropy over Iterations')
plt.legend()
plt.show()


# Create a figure for the second plot (policy loss)
plt.figure(figsize=(10, 5))
plt.plot(time, policy_loss, label='Entropy', color='green')
plt.xlabel('Iteration')
plt.ylabel('policyc loss')
plt.title('policy loss over Iterations')
plt.legend()
plt.show()"""