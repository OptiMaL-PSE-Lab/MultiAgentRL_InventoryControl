
from env3SA import InvManagementDiv
import gymnasium as gym
from gymnasium.spaces import Dict, Box
import numpy as np 
import ray 
from ray import tune 
from ray import air
from ray.tune.logger import pretty_print
import os 
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.algorithms.ppo import PPOConfig
import json 
from gymnasium.wrappers import EnvCompatibility
#from model import GNNActorCriticModel


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


# Set script seed
SEED = 52
np.random.seed(seed=SEED)




# Register environment
def env_creator(config):
    return EnvCompatibility(InvManagementDiv(config = config))
config = {}
tune.register_env("InvManagementDiv", env_creator)   # noqa: E501

"""
rl_config = {             
    "max_seq_len": 10,
    "env": "InvManagementDiv", 
    "env_config": {"seed": SEED},
    }

algo_config = PPOConfig()
algo_config.__dict__.update(**rl_config)

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
"""

algo_w_5_policies = (
    PPOConfig()
    .environment(
        env= "InvManagementDiv",
    )
    .rollouts(
        batch_mode="complete_episodes",
            num_rollout_workers=0,
            # TODO(avnishn) make a new example compatible w connectors.
            enable_connectors=False,)
    .training()
    .build()
)
iterations = 120
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