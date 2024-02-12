import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.utils.framework import try_import_torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn.pool import global_mean_pool
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing


torch, nn = try_import_torch()

class GNNLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNLayer, self).__init__()
        self.conv1 = nn.Linear(input_dim, hidden_dim)
        self.conv = GCNConv(hidden_dim, hidden_dim) 
        self.conv2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x))
        x = self.conv(x, edge_index)
        x = F.relu(x) 
        x = self.conv2(x)
        return x

class GNNActorCriticModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        state_dim = 10
        action_dim = num_outputs
        message_dim = 10  
        gnn_hidden_dim = 64  
        
        # GNN for message passing [input, hidden, output]
        self.gnn = GNNLayer(state_dim, gnn_hidden_dim, message_dim)
        # Actor: Neural network for policy
        self.actor = FullyConnectedNetwork(obs_space, action_space, num_outputs, model_config, name)
        # Critic: Neural network for state-value estimation
        self.critic = FullyConnectedNetwork(obs_space, action_space, 1, model_config, name)

    def forward(self, input_dict, state, seq_lens):
        statetensor = input_dict["obs"]["own_obs"]
        state_tensor = statetensor.unsqueeze(1)
        opponent_obs = input_dict["obs"]["opponent_obs"].reshape(32,5,10)
        node_features_concat = torch.cat((state_tensor, opponent_obs), dim = 1)

        adjacency_matrix = [
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [1, 1, 0, 0, 1, 1],
            [1, 1, 0, 0, 1, 1],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0]
        ]

        # Convert it to a PyTorch tensor
        adj_t = torch.tensor(adjacency_matrix)

        edge_index_single = adj_t.nonzero(as_tuple=False).t().contiguous()

        # Repeat the edge index tensor along the batch dimension (32 times)
        batch_edge_index = torch.cat([edge_index_single] * 32, dim=1)

        data = Data (x = node_features_concat, edge_index=batch_edge_index)

        # GNN-based message generation
        message = self.gnn(data)

        # Concatenate message with state for actor input
        actor_input = torch.cat([state_tensor, message], dim=1)

        # Actor: Select action
        action_logits, _ = self.actor({"obs": actor_input}, state, seq_lens)
        # Critic: Estimate state value
        value = self.critic({"obs": state_tensor}, state, seq_lens)
        return action_logits, [], value

    def value_function(self):
        return self._value_out
