from env3run import MultiAgentInvManagementDiv
import gymnasium as gym
from gymnasium.spaces import Dict, Box, Discrete
import numpy as np 
from ray.rllib.models import ModelCatalog
import ray 
from ray import tune 
from ray import air
import os 
from ray.rllib.policy.policy import Policy
import time 
from ray.rllib.algorithms.ppo import PPOConfig
import json 
from ray.rllib.policy.policy import PolicySpec #For policy mapping
from model import GNNActorCriticModel
from ccmodel import CentralizedCriticModel
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.policy.sample_batch import SampleBatch

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

ModelCatalog.register_custom_model("gnn_model", GNNActorCriticModel)
ModelCatalog.register_custom_model("cc_model", CentralizedCriticModel)
#import ray.rllib.algorithms
#from ray.rllib.algorithms.maddpg.maddpg import MADDPGConfig
ray.shutdown()
ray.init(log_to_driver= False)


def central_critic_observer(agent_obs, **kw):
    """Rewrites the agent obs to include opponent data for training."""

    new_obs = {
        0: {
            "own_obs": agent_obs[0],
            "opponent_obs": agent_obs[1],
            "opponent_action": 0,  # filled in by FillInActions
        },
        1: {
            "own_obs": agent_obs[1],
            "opponent_obs": agent_obs[0],
            "opponent_action": 0,  # filled in by FillInActions
        },
    }
    return new_obs


config = {"connections":{0: [1], 1:[2], 2:[]},
          "num_products":2, 
          "num_nodes":3}

# Test environment
test_env = MultiAgentInvManagementDiv(config)
obs_space = test_env.observation_space
act_space = test_env.action_space
num_agents = test_env.num_agents
size = obs_space.shape[0]
opponent_obs_space = Box(low=np.tile(obs_space.low, num_agents-1), high=np.tile(obs_space.high, num_agents-1),
                         dtype=np.float64, shape=(obs_space.shape[0]*(num_agents-1),))
opponent_act_space = Box(low=np.tile(act_space.low, num_agents-1), high=np.tile(act_space.high, num_agents-1),
                         dtype=np.float64, shape=(act_space.shape[0]*(num_agents-1),))
cc_obs_space = Dict({
    "own_obs": obs_space,
    "opponent_obs": opponent_obs_space,
    "opponent_action": opponent_act_space,
})

# Set script seed
SEED = 52
np.random.seed(seed=SEED)

num_products = config["num_products"]
num_nodes = config["num_nodes"]
num_agents = num_products * num_nodes

test_env = MultiAgentInvManagementDiv(config)
obs_space = test_env.observation_space
print("obs shape env",obs_space.shape)


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


def policy_dict():
    return {f"{agent_id}": PolicySpec() for agent_id in agent_ids}

policy_graphs = {}
for i in range(num_agents):
    policy_graphs[agent_ids[i]] = None, cc_obs_space, act_space, {}

print("cc obs space", cc_obs_space, cc_obs_space.shape)

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    '''Maps each Agent's ID with a different policy. So each agent is trained with a diff policy.'''
    get_policy_key = lambda agent_id: f"{agent_id}"  # noqa: E731
    return get_policy_key(agent_id)

# Policy Mapping function to map each agent to appropriate stage policy
def policy_mapping_fn1(agent_id, episode, **kwargs):
    for i in range(num_nodes * num_products):
        if agent_id.startswith(agent_ids[i]):
            return agent_ids[i]

        
def central_critic_observer(agent_obs, **kw):
    """agent observation includes all agents observation data for 
        training which enable centrailised training ."""
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


algo_w_5_policies = (
    PPOConfig()
    .environment(
        env= "MultiAgentInvManagementDiv",
        env_config={
            "num_agents": num_agents,
        },
    )
    .multi_agent(
        policies= policy_graphs,
        #observation_fn = central_critic_observer, 
        # Map "agent0" -> "pol0", etc...
        policy_mapping_fn=(
            lambda agent_id, episode, worker, **kwargs: (
        print(f"Agent ID: {agent_id}"),
        str(agent_id)
    )[1]
    )
    )
    .training(
        model = {"fcnet_hiddens": [128,128],
                 "custom_model": "gnn_model",
                 }
    )
    .build()
)

iterations = 1
for i in range(iterations):
    algo_w_5_policies.train()
    path_to_checkpoint = algo_w_5_policies.save()
    print(
                "An Algorithm checkpoint has been created inside directory: "
                f"'{path_to_checkpoint}'. It should contain 5 policies in the 'policies/' sub dir."
            )

# Let's terminate the algo for demonstration purposes.

algo_w_5_policies.stop()
print("donee")

"""
marl_config = {             
    "multiagent": {
    # Policy mapping function to map agents to policies
        "policy_mapping_fn": policy_mapping_fn,
        "policies": policy_dict(),
        "observation_fn": central_critic_observer,
        #"policies_to_train":["0_00_0", "0_00_1", "1_01_0", "1_01_1", "2_02_0", "2_02_1"]
    },
    "max_seq_len": 10,
    "env": "MultiAgentInvManagementDiv", 
    "env_config": {"seed": SEED},
    "custom": {"custom model": "gnn_model"}
    }


algo_config = PPOConfig()
algo_config.__dict__.update(**marl_config)

#sets the config's checkpointing settings
algo_config.checkpointing(export_native_model_files=True, 
                          checkpoint_trainable_policies_only = True)

algo = algo_config.build()

results = tune.Tuner(
        "PPO",
        param_space=algo_config.to_dict(),
        run_config=air.RunConfig(
            stop={"training_iteration": 60},
            verbose=1,
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=1, checkpoint_at_end=True
            ),
        ),
    ).fit()
print("Pre-training done.")

best_checkpoint = results.get_best_result().checkpoint
print(f".. best checkpoint was: {best_checkpoint}")
"""