from env3 import MultiAgentInvManagementDiv
import gymnasium as gym
from gymnasium.spaces import Dict, Box
import numpy as np 
from ray.rllib.models import ModelCatalog
import ray 
from ray.rllib.algorithms.algorithm import Algorithm
from ray import tune 
from ray.tune.logger import pretty_print
from ray import tune , air 
import torch 
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.examples.env.multi_agent import MultiAgentCartPole
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.policy.policy import PolicySpec #For policy mapping


ray.init()

# Register environment
def env_creator(config):
    return MultiAgentInvManagementDiv(config = config)
config = {"bullwhip": False}
tune.register_env("MultiAgentInvManagementDiv", env_creator)   # noqa: E501

"""#load the trained policy 
checkpoint_path = '/Users/nikikotecha/ray_results/PPO/PPO_MultiAgentInvManagementDiv_18a00_00000_0_2023-10-31_12-00-34/checkpoint_000070'
trained_policy = Algorithm.from_checkpoint(checkpoint_path)
trained_policy.train()
path_to_checkpoint = trained_policy.save()
print(
    "An Algorithm checkpoint has been created inside directory: "
    f"'{path_to_checkpoint}'. It should contain 5 policies in the 'policies/' sub dir."
)
# Let's terminate the algo for demonstration purposes.
trained_policy.stop()
"""
def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    '''Maps each Agent's ID with a different policy. So each agent is trained with a diff policy.'''
    print("agent id in policy mapping function", agent_id)
    get_policy_key = lambda agent_id: f"{agent_id}" 
    return str(get_policy_key(agent_id))

marl_config = {             
    "multiagent": {
    # Policy mapping function to map agents to policies
        "policy_mapping_fn": policy_mapping_fn,
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
    #"config": {"model": {"custom_model": "gnn_model"}}
    }


algo_config = PPOConfig()
algo_config.__dict__.update(**marl_config)

#sets the config's checkpointing settings
algo_config.checkpointing(export_native_model_files=True, 
                          checkpoint_trainable_policies_only = True)

algo = algo_config.build()
print("built")

train_steps = 60 
experiment_name = "trial"
tuner = tune.Tuner("PPO", param_space=algo_config.to_dict(),
                              run_config=air.RunConfig(
                                        name =  experiment_name,
                                        stop={"timesteps_total": train_steps}, #maybe change for "stop" above
                                        checkpoint_config=air.CheckpointConfig(checkpoint_frequency=50, checkpoint_at_end=True),
                                        #verbose= 0
                                )
                     )
result_grid = tuner.fit() #brings back a list!
    # ANALIZE RESULTS
    # https://docs.ray.io/en/latest/rllib/rllib-env.html
    # RLlib reports separate training statistics for each policy in the return from train(), along with the combined reward.
    # Get rewards per policy




"""algo_w_2_policies = Algorithm.from_checkpoint(
            checkpoint=path_to_checkpoint,
            policy_ids={"0_00_0", "0_00_1"},  # <- restore only those policy IDs here.
            policy_mapping_fn=new_policy_mapping_fn,  # <- use this new mapping fn.
        )

# Test, whether we can train with this new setup.
algo_w_2_policies.train()
# Terminate the new algo.
algo_w_2_policies.stop()"""

"""rl_config = {             
            "multiagent": {
            # Policy mapping function to map agents to policies
                "policy_mapping_fn": policy_mapping_fn,
                "policies": {"0_00_0", "0_001", "1_01_0", "1_01_1", "2_02_0", "2_02_1"},
                "observation_fn": central_critic_observer,
                "policies_to_train":["0_00_0", "0_001", "1_01_0", "1_01_1", "2_02_0", "2_02_1"]
            },
            "max_seq_len": 10,
            "env": "MultiAgentInvManagementDiv", 
            "env_config": {"seed": SEED},
            }

        #algo_w_5_policies.checkpointing(export_native_model_files=True, 
        #                          checkpoint_trainable_policies_only = True)

        # .. train one iteration ..
        algo_w_5_policies.train()
        # .. and call `save()` to create a checkpoint.
        path_to_checkpoint = algo_w_5_policies.save()
        print(
            "An Algorithm checkpoint has been created inside directory: "
            f"'{path_to_checkpoint}'. It should contain 5 policies in the 'policies/' sub dir."
        )
        # Let's terminate the algo for demonstration purposes.
        algo_w_5_policies.stop()

        # We will now recreate a new algo from this checkpoint, but only with 2 of the
        # original policies ("pol0" and "pol1"). Note that this will require us to change the
        # `policy_mapping_fn` (instead of mapping 5 agents to 5 policies, we now have
        # to map 5 agents to only 2 policies).


        def new_policy_mapping_fn(agent_id, episode, worker, **kwargs):
            return "0_00_0" if agent_id in ["agent0", "agent1"] else "0_00_1"


        algo_w_2_policies = Algorithm.from_checkpoint(
            checkpoint=path_to_checkpoint,
            policy_ids={"pol0", "pol1"},  # <- restore only those policy IDs here.
            policy_mapping_fn=new_policy_mapping_fn,  # <- use this new mapping fn.
        )

        # Test, whether we can train with this new setup.
        algo_w_2_policies.train()
        # Terminate the new algo.
        algo_w_2_policies.stop()

        """  # noqa: E999