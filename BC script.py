from env2 import MultiAgentInvManagementDiv
import gymnasium as gym
from gymnasium.spaces import Dict, Box
import numpy as np 
from ray.rllib.models import ModelCatalog
import ray 
import time
from ray import tune 
from ray import air
from ray.rllib.algorithms.qmix import QMixConfig
from ray.tune.logger import pretty_print
import os 
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.policy.policy import Policy
from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig


"""
Behavioural Cloning Script. Steps: 
1. Pre-train & checkpoint [run.py file]
2. Create new environment with a larger number of agents
3. Reload weighted policies 
4. Clone weighted policies 
5. Re-train - look at convergence rate [compare against training just from scratch]
"""
ray.shutdown()
ray.init(resources={"CUSTOM_RESOURCE":100})


config = {"num_nodes": 4, 
          "num_products": 2, 
          "connections": {0: [1], 1:[2,3], 2:[], 3:[]}}

num_nodes = config["num_nodes"]
num_products = config["num_products"]

# Set script seed
SEED = 52
np.random.seed(seed=SEED)

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

# Agent/Policy ids
agent_ids = []
network = create_network(config["connections"])
echelons = {node: get_stage(node, network) for node in range(len(network))}

agent_ids = []
agent_ids = [f"{echelons[node]}_{node:02d}_{product}" for node in range(len(network)) for product in range(num_products)]

"""for i in range(num_nodes * num_products):
    agent_id = "node_" + str(i)
    agent_ids.append(agent_id)"""

print("agent_ids", agent_ids)

test_env = MultiAgentInvManagementDiv(config)
obs_space = test_env.observation_space
print(obs_space)
act_space = test_env.action_space
print(act_space)
num_agents = test_env.num_agents
print(num_agents)
size = obs_space.shape[0]
size_action = act_space.shape[0]


opponent_obs_space = Box(low=np.tile(obs_space.low, num_agents-1), 
                        high=np.tile(obs_space.high, num_agents-1),
                         dtype=np.float64, shape=(size*(num_agents-1),))

opponent_act_space = Box(
low = np.tile(act_space.low, (num_agents -1)), 
high = np.tile(act_space.high, (num_agents -1))
)
"""

#opponent_act_space = Box(low=np.tile(act_space.low, num_agents-1), 
#                        high=np.tile(act_space.high, num_agents-1),
#                         dtype=np.float64, shape=(act_space.shape[0]*(num_agents-1),))

"""
print(type(obs_space))
print(type(opponent_act_space))
print(type(opponent_act_space))

cc_obs_space = Dict({
        "own_obs": obs_space,
        "opponent_obs": opponent_obs_space,
        "opponent_action": opponent_act_space,
    })

print(cc_obs_space)


# Define policies to train
policy_graphs = {}
for i in range(num_agents):
    policy_graphs[agent_ids[i]] = None, cc_obs_space, act_space, {}

# Policy Mapping function to map each agent to appropriate stage policy
def policy_mapping_fn(agent_id, episode, **kwargs):
    for i in range(num_nodes * num_products):
        print("agent id in policy map fn", agent_id)
        if agent_id.startswith(agent_ids[i][0]) and agent_id.endswith(agent_ids[i][2]):
            return agent_ids[i]
        else: 
            raise Exception("policy mapping function not working")

def central_critic_observer(agent_obs, **kw):
    #agent observation includes all agents observation data for 
        #training which enable centrailised training 
    agents = [*agent_obs]
    num_agents = len(agents)
    obs_space = len(agent_obs[agents[0]])

    new_obs = dict()
    for agent in agents:
        new_obs[agent] = dict()
        new_obs[agent]["own_obs"] = agent_obs[agent]
        new_obs[agent]["opponent_obs"] = np.zeros((num_agents - 1)*obs_space)
        new_obs[agent]["opponent_action"] = np.zeros((num_agents - 1))
        i = 0
        for other_agent in agents:
            if agent != other_agent:
                new_obs[agent]["opponent_obs"][i*obs_space:i*obs_space + obs_space] \
                    = agent_obs[other_agent]
                i += 1

    return new_obs



# Register environment

def env_creator(config):
    return MultiAgentInvManagementDiv(config = config)
tune.register_env("MultiAgentInvManagementDiv", env_creator)   # noqa: E501


# Policy Mapping function to map each agent to appropriate stage policy
def new_policy_mapping_fn(agent_id, episode, **kwargs):
    agent_ids = ["0_00_0", "0_00_1", "1_01_0", "1_01_1", "2_02_0", "2_02_1"]

    for i in range(num_nodes * num_products):
        print("agent id policy map fn", agent_id)
        if agent_id.startswith(agent_ids[i][0]) and agent_id.endswith(agent_ids[i][5]):
            return agent_ids[i]

    raise Exception("policy mapping function not working")

path_to_checkpoint = "/Users/nikikotecha/ray_results/PPO_MultiAgentInvManagementDiv_2023-11-09_15-37-08s_ldp4fp/checkpoint_000001"

algo_w_2_policies = Algorithm.from_checkpoint(
            checkpoint = path_to_checkpoint,
            policy_ids={"0_00_0", "0_00_1", "1_01_0", "1_01_1", "2_02_0", "2_02_1"},  # <- restore only those policy IDs here.
            policy_mapping_fn=new_policy_mapping_fn,  # <- use this new mapping fn.
        )

# Test, whether we can train with this new setup.
algo_w_2_policies.train()
# Terminate the new algo.
algo_w_2_policies.stop()




"""
rl_config = {             
    "multiagent": {
    # Policy mapping function to map agents to policies
        "policy_mapping_fn": policy_mapping_fn,
        "policies": policy_graphs,
        "observation_fn": central_critic_observer
    },

   #"model": {
        #"custom_model": "gnn_cc_model",
        #"custom_model_config": {},
        #"use_lstm": True,
        #"max_seq_len": 10},

    "max_seq_len": 10,
    "env": "MultiAgentInvManagementDiv", 
    "env_config": {"seed": SEED},
    #"train_batch_size": 1000, 
    #"num_workers": 1,
    #"batch_mode": "truncate_episodes",
    }


algo_config = PPOConfig()
algo_config.__dict__.update(**rl_config)

algo = algo_config.build()

#timestamp = str(int(time.time()))
#timestamp_dir = os.path.join("/Users/nikikotecha/Documents/PhD/sS/Checkpoint", timestamp)
base_dir = os.path.join(timestamp_dir, "checkpoint")
#os.makedirs(timestamp_dir, exist_ok = True)
train_agent = True

if train_agent:  
    # Training
    iters = 2
    min_iter_save = 3
    checkpoint_interval = 3
    results = []
    epc =[]
    reward =[]
    time1 = []
    timestep = 0 
    for i in range(iters):
        timestep += i
        time1.append(timestep)
        res = algo.train()
        #checkpoint_dir = os.path.join(base_dir + str(i))
        #os.makedirs(checkpoint_dir, exist_ok=True)  # Create the directory if it doesn't exist
        #save_result = algo.save_checkpoint(checkpoint_dir)
        res["config"] = 0
        results.append(res)
        epc.append(i+1)
        reward.append(res['episode_reward_mean'])
        if (i + 1) % 1 == 0:
            print('\rIter: {}\tReward: {:.2f}'.format(
                i + 1, res['episode_reward_mean']), end='')
            
print("done with training")
"""
    
"""A different variation on how to save and restore the checkpoints in ray 
results = tune.Tuner(
    "PPO", 
    param_space= rl_config.to_dict(),
    run_config= air.RunConfig(
        stop= {"training_iteration": 2},
    verbose=1,
    checkpoint_config= air.CheckpointConfig(
        checkpoint_frequency=1, checkpoint_at_end=True
    )
    )
).fit()
print("pre-traininf done")
best_checkpoint = results.get_best_result().checkpoint
print("best checkpoint was {}".format(best_checkpoint))
#restored_policies = Policy.from_checkpoint()
"""

