import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()

class GNNLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNLayer, self).__init__()
        self.conv1 = nn.Linear(input_dim, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, adjacency_matrix, node_features):
        x = F.relu(self.conv1(torch.matmul(adjacency_matrix, node_features)))
        x = self.conv2(torch.matmul(adjacency_matrix, x))
        return x

class GNNActorCriticModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        state_dim = obs_space.shape[0]
        action_dim = num_outputs
        message_dim = 64  # You can adjust this based on your requirements
        gnn_hidden_dim = 32  # You can adjust this based on your requirements

        # GNN for message passing
        self.gnn = GNNLayer(state_dim, gnn_hidden_dim, message_dim)
        # Actor: Neural network for policy
        self.actor = FullyConnectedNetwork(obs_space, action_space, num_outputs, model_config, name)
        # Critic: Neural network for state-value estimation
        self.critic = FullyConnectedNetwork(obs_space, action_space, 1, model_config, name)

    def forward(self, input_dict, state, seq_lens):
        state_tensor = input_dict["obs"].float()

        #adjacency_matrix = input_dict["adjacency_matrix"].float()
        adjacency_matrix = [    [0, 1, 0],
                                [0, 0, 1],
                                [0, 0, 0]
                            ].float()
        # GNN-based message generation
        message = self.gnn(adjacency_matrix, state_tensor)
        # Concatenate message with state for actor input
        actor_input = torch.cat([state_tensor, message], dim=1)
        # Actor: Select action
        action_logits, _ = self.actor({"obs": actor_input}, state, seq_lens)
        # Critic: Estimate state value
        value = self.critic({"obs": actor_input}, state, seq_lens)
        return action_logits, [], value

    def value_function(self):
        return self._value_out
