"""An example of implementing a centralized critic with ObservationFunction.
The advantage of this approach is that it's very simple and you don't have to
change the algorithm at all -- just use callbacks and a custom model.
However, it is a bit less principled in that you have to change the agent
observation spaces to include data that is only used at train time.
See also: centralized_critic.py for an alternative approach that instead
modifies the policy to add a centralized value function.
"""

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Dict, Box, Discrete
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork 
from ray.rllib.utils.annotations import override
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor

torch, nn = try_import_torch()

class CentralisedCriticModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        state_dim = 10
        # Actor: Neural network for policy
        self.actor = FullyConnectedNetwork(
            Box(
                low = -np.ones(state_dim),
                high = np.ones(state_dim),
                dtype = np.float64,
                shape = (state_dim,)), 
                action_space, num_outputs, model_config, name + '_action')
        # Critic: Neural network for state-value estimation
        self.critic = FullyConnectedNetwork(
            obs_space, action_space, 1, model_config, name+ '_vf')

        self._model_in = None 
    def forward(self, input_dict, state, seq_lens):
        
        self._model_in = [input_dict["obs_flat"], state, seq_lens]

        return self.actor({
            "obs": input_dict["obs"]["own_obs"]
        }, state, seq_lens)

    def value_function(self):

        value_out, _ = self.critic({
            "obs": self._model_in[0]
        }, self._model_in[1], self._model_in[2])

        noise = torch.randn_like(value_out) * 0.2  # Gaussian noise with mean 0 and std 0.1
        value_out += noise

        return torch.reshape(value_out, [-1])


class FillInActions(DefaultCallbacks):
    """Fills in the opponent actions info in the training batches."""

    def on_postprocess_trajectory(self, worker, episode, agent_id, policy_id,
                                  policies, postprocessed_batch,
                                  original_batches, **kwargs):
        to_update = postprocessed_batch[SampleBatch.CUR_OBS]
        action_encoder = ModelCatalog.get_preprocessor_for_space(
            Box(
                low=-1,
                high=1,
                dtype=np.float64,
                shape=(2,)
            )
        )
        agents = [*original_batches]
        agents.remove(agent_id)
        num_agents = len(agents)

        for i in range(num_agents):
            other_id = agents[i]

            # set the opponent actions into the observation
            _, opponent_batch = original_batches[other_id]
            opponent_actions = np.array([
                action_encoder.transform(np.clip(a, -1, 1))
                for a in opponent_batch[SampleBatch.ACTIONS]
            ])
        # Update only the corresponding columns for the actions of each opponent agent
            start_col = i * 2 # Start column index for opponent actions
            end_col = start_col + 2 # End column index for opponent actions
            to_update[:, start_col:end_col] = np.squeeze(opponent_actions)  # <--------------------------









