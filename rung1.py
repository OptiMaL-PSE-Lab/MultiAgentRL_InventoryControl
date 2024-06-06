
from env3run import MultiAgentInvManagementDiv
import gymnasium as gym
from gymnasium.spaces import Dict, Box
import numpy as np 
from ray.rllib.models import ModelCatalog
import ray 
from ray import tune 
from ray import air
from ray.rllib.algorithms.qmix import QMixConfig
from ray.tune.logger import pretty_print
import os 
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.policy.policy import Policy
import time 
from ray.rllib.algorithms.ppo import PPOConfig
import json 
from ray.rllib.policy.policy import PolicySpec #For policy mapping
from model import GNNActorCriticModel

config = {"connections":{0: [1], 1:[2], 2:[3], 3:[4], 4:[5], 5:[]},
          "num_products":2, 
          "num_nodes":6}

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

ModelCatalog.register_custom_model("gnn_model", GNNActorCriticModel)

ray.shutdown()
ray.init(log_to_driver= False)


num_nodes = config["num_nodes"]
num_periods = 2
num_products = config["num_products"]

# Set script seed
SEED = 52
np.random.seed(seed=SEED)

"""# Agent/Policy ids
agent_ids = []
for i in range(num_nodes * num_products):
    agent_id = "node_" + str(i)
    agent_ids.append(agent_id)"""

# Test environment
test_env = MultiAgentInvManagementDiv(config)
obs_space = test_env.observation_space
print("obs space",obs_space)
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
"""opponent_act_space = Box(low=np.tile(act_space.low, num_agents-1), 
                        high=np.tile(act_space.high, num_agents-1),
                         dtype=np.float64, shape=(act_space.shape[0]*(num_agents-1),))
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

print(agent_ids)
# Define policies to train
policy_graphs = {}
for i in range(num_agents):
    policy_graphs[agent_ids[i]] = None, cc_obs_space, act_space, {}


def policy_dict():
    return {f"{agent_id}": PolicySpec() for agent_id in agent_ids}

print("PG Keys", policy_graphs.keys())

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    '''Maps each Agent's ID with a different policy. So each agent is trained with a diff policy.'''
    print("agent id in policy mapping function", agent_id)
    get_policy_key = lambda agent_id: f"{agent_id}"  # noqa: E731
    return get_policy_key(agent_id)

# Policy Mapping function to map each agent to appropriate stage policy
def policy_mapping_fn1(agent_id, episode, **kwargs):
    for i in range(num_nodes * num_products):
        if agent_id.startswith(agent_ids[i]):
            return agent_ids[i]

# Policy Mapping function to map each agent to appropriate stage policy
def single_policy_mapping_fn(agent_id, episode, **kwargs):
    for i in range(num_nodes * num_products):
        if agent_id.startswith(agent_ids[i]):
            return '0_00_0'
        
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

#def model_creator():
#    model = GNNCentralizedCriticModel(obs_space = obs_space, action_space = act_space, \
#                num_outputs = 2, model_config = model_config, name=None, state = state)
#    return model



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
        policies= policy_dict(),
        # Map "agent0" -> "pol0", etc...
        policy_mapping_fn=(
            lambda agent_id, episode, worker, **kwargs: (
        print(f"Agent ID: {agent_id}"),
        str(agent_id)
    )[1]
    )
    )
    .build()
)
iterations = 60
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