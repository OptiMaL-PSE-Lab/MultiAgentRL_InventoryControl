
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
#from model import GNNActorCriticModel

config = {"connections":{0: [1], 1:[2], 2:[]},
          "num_products":2, 
          "num_nodes":3}

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

#ModelCatalog.register_custom_model("gnn_model", GNNActorCriticModel)
#import ray.rllib.algorithms
#from ray.rllib.algorithms.maddpg.maddpg import MADDPGConfig
ray.shutdown()
ray.init(resources={"CUSTOM_RESOURCE":100}, log_to_driver= False)

#todo - create centralised critic model class in models.py file 


num_nodes = 3
num_periods = 2
num_products = 2 

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
agent_ids = [f"{echelons[node]}_{node:02d}_{product}" for node in range(len(network)) for product in range(num_products)]
print(agent_ids)

# Define policies to train
policy_graphs = {}
for i in range(num_agents):
    policy_graphs[agent_ids[i]] = None, cc_obs_space, act_space, {}

print(policy_graphs)

def policy_mapping(agent_id, episode, worker, **kwargs):
    return agent_id

def policy_dict():
    return {f"{agent_id}": PolicySpec(config=config.overrides(agent_id=agent_id)) for agent_id in agent_ids}

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    '''Maps each Agent's ID with a different policy. So each agent is trained with a diff policy.'''
    print("agent id in policy mapping function", agent_id)
    get_policy_key = lambda agent_id: f"{agent_id}" 
    return str(get_policy_key(agent_id))

        
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


marl_config = {             
    "multiagent": {
    # Policy mapping function to map agents to policies
        "policy_mapping_fn": policy_mapping,
        "policies": {'0_00_0': PolicySpec(), 
                     '0_00_1': PolicySpec(), 
                     '1_01_0': PolicySpec(),
                     '1_01_1': PolicySpec(),
                     '2_02_0': PolicySpec(),
                     '2_02_1': PolicySpec(),
                     },
        #"observation_fn": central_critic_observer,
        #will automatically train all policies if left empty
        #"policies_to_train":["0_00_0", "0_00_1", "1_01_0", "1_01_1", "2_02_0", "2_02_1"]
    },
    #"max_seq_len": 10,
    "env": "MultiAgentInvManagementDiv", 
    "env_config": {"seed": SEED},
    #"config": {"model": {"custom_model": "gnn_model"}}
    }


algo_config = PPOConfig()
algo_config.__dict__.update(**marl_config)

#sets the config's checkpointing settings
algo_config.checkpointing(export_native_model_files=True, 
                          checkpoint_trainable_policies_only = True)

algo = algo_config.build()
print("built")
results = tune.Tuner(
        "PPO",
        param_space=algo_config.to_dict(),
        run_config=air.RunConfig(
            stop={"training_iteration": 1},
            verbose=1,
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=1, checkpoint_at_end=True
            ),
        ),
    ).fit()
print("Pre-training done.")

best_checkpoint = results.get_best_result().checkpoint
print(f".. best checkpoint was: {best_checkpoint}")
# Let's terminate the algo for demonstration purposes.
#algo_config.stop()



"""timestamp = str(int(time.time()))
timestamp_dir = os.path.join("/Users/nikikotecha/Documents/PhD/sS/Checkpoint", timestamp)
os.makedirs(timestamp_dir, exist_ok = True)
save_path = os.path.join(timestamp_dir, "checkpoint")
ensure_dir(save_path)
load_path = "/Users/nikikotecha/Documents/PhD/sS/Checkpoint"
LP_load_path = "LP_results/sS"
load_iteration = str(100)
load_agent_path = load_path + '/checkpoint_000' + load_iteration + '/checkpoint-' + load_iteration

train_agent = True
if train_agent:  
    # Training
    iters = 80
    results = []
    epc =[]
    reward =[]
    time = []
    timestep = 0 
    for i in range(iters):
        print("iters: {}".format(i))
        res = algo.train()
        #weights = algo.get_weights()
        algo.save(save_path)
"""
#restored_policy = Policy.from_checkpoint("/Users/nikikotecha/Documents/PhD/sS/Checkpoint/1697121757/checkpoint/checkpoint_000004/policies/default_policy")
#print("Restored policy")

"""for i in range(iters):
        timestep += i
        time.append(timestep)
        res = algo.train()
        ensure_dir(save_path + str(i))
        algo.save_checkpoint(save_path +str(i))
else:
    restored_policy = Policy.from_checkpoint("Checkpoints/policies/default_policy/policy_state.pkl")
    print("restored policy")"""

"""else:
    algo.restore(load_agent_path)
    results_load_path = load_path + '/results.npy'
    results = np.load(results_load_path, allow_pickle= True)"""



"""if save_agent:
    json_env_config = save_path + '/env_config.json'
    ensure_dir(json_env_config)
    with open(json_env_config, 'w') as fp:
        for key, value in config.items():
            if isinstance(value, np.ndarray):
                config[key] = config[key].tolist()
        json.dump(config, fp)
    results_save_path = save_path + '/results.npy'
    np.save(results_save_path, results)
"""
"""
A different variation on how to save and restore the checkpoints in ray 
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



results = tune.Tuner(
    "PPO", 
    param_space= rl_config,
    run_config= air.RunConfig(
        stop= {"training_iteration": 2},
    verbose=1,
    checkpoint_config= air.CheckpointConfig(
        checkpoint_frequency=1, checkpoint_at_end=True
    )
    )
).fit()
print("pre-training done")


"""


"""
configg = PPOConfig().\
    framework("torch").\
        rollouts(num_rollout_workers=1).\
        resources(num_cpus_per_worker=1).\
            environment(env = "MultiAgentInvManagementDiv", 
                        env_config={"disable_env_checking": True}).\
        multi_agent(policies=policy_graphs, 
                    policy_mapping_fn = policy_mapping_fn)

if ray.is_initialized(): 
    ray.shutdown()

ray.init()
train_steps = 2
experiment_name = "my_exp"
tuner = tune.Tuner("PPO", param_space=configg, 
                   run_config = air.RunConfig(
                       name = experiment_name, 
                       stop={"timesteps_total": train_steps},
                       checkpoint_config= air.CheckpointConfig(checkpoint_frequency=50, checkpoint_at_end=True),
                   ))
result_grid = tuner.fit()

ray.shutdown()

"""