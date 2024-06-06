import random
import numpy as np
from env3run import MultiAgentInvManagementDiv
import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from scipy.optimize import minimize

# Create the environment
config = {}
env = MultiAgentInvManagementDiv(config)

def _update_inventory_state(env):
        '''
        Get current state of the system: Inventory position at each echelon
        Inventory at hand + Pipeline inventory - backlog up to the current stage 
        (excludes last stage since no inventory there, nor replenishment orders placed there).
        '''
        n = env.period
        m = env.num_stages
        if n>=1:
            IP = np.cumsum(env.I[n,:] + env.T[n,:] - env.B[n-1,:-1])
            #print(IP)
        else:
            IP = np.cumsum(env.I[n,:] + env.T[n,:])
        env.state = IP
        return env.state

def min_max_action(env, z_min, z_max): # (s, S) policy
    '''
    Sample action (number of units to request) based on a min-max policy (order up to z_max if inv is below z_min)
    z = [integer list; dimension |Stages| - 1] base stock level (no inventory at the last stage)
    '''
    assert len(z_min) == len(z_max)
    
    n = env.period
    c = env.supply_capacity
    m = env.num_stages
    IP = _update_inventory_state(env) # extract inventory position (current state)
    
    try:
        dimz = len(z_min)
    except:
        dimz = 1
    assert dimz == m-1, "Wrong dimension on base stock level vector. Should be # Stages - 1."
    
    R = np.zeros(3)
    #print(type(IP))
    #print(IP)
    # calculate total inventory position at the beginning of period n
    for i in range(len(IP)):
        if IP[i] < z_min[i]:
            R[i] = z_max[i] - IP[i] # replenishmet order to reach z_max
        else:
            R[i] = 0

    # check if R can actually be fulfilled (capacity and inventory constraints)
    Im1 = np.append(env.I[n,1:], np.Inf) # available inventory at the m+1 stage
                                        # NOTE: last stage has unlimited raw materials
    Rpos = np.column_stack((np.zeros(len(R)),R)) # augmented materix to get replenishment only if positive
    A = np.column_stack((c, np.max(Rpos,axis=1), Im1)) # augmented matrix with c, R, and I_m+1 as columns
    
    R = np.min(A, axis = 1) # replenishmet order to reach zopt (capacity constrained)
    
    return R


def min_max_black_box(bounds):
    z_min = bounds[:int(len(bounds)/2)]
    z_max = bounds[int(len(bounds)/2):]
    
    env = MultiAgentInvManagementDiv(config)
    N_episodes = 10 # NOTE: we chose this
    total_reward_dummy = np.array([])

    for i in range(N_episodes):
        obs = env.reset()
        done = False
        
        #obs, reward, done = env.sample_action()
        
        while not done:
            #print(env.I)
            action = min_max_action(env, z_min, z_max)

            # Take the defined action (place an order), and advance to the next time period by taking a "step"
            obs, reward, done, _ = env.step(action)
            total_reward_dummy = np.append(total_reward_dummy, [reward])
            
    avg_total_rewards = np.average(total_reward_dummy)
    
    return avg_total_rewards


n_restarts = 100
best_min_max = -99999

for i in range(n_restarts):    
    z_up = np.array([random.randint(0, 10) for i in range(3)])
    z_low = np.array([random.randint(0, z_up[int(i)]) for i in range(len(z_up))])
    x0 = np.concatenate((z_low, z_up))
    
    opt_min_max = minimize(min_max_black_box, x0)
    #print(opt_base_stock.fun)
    
    if opt_min_max.fun > best_min_max:
        best_min_max = opt_min_max.fun




states = []
actions = []
num_episodes = 1000
for _ in range(num_episodes):
    state = env.reset()
    done = False 
    while not done:
        action = {}
        for agent_id in env.agents:
            action[agent_id] = min_max_action(env, z_min, z_max)
            states.append(state[agent_id])
            actions.append(action[agent_id])
        state, reward, done, _ = env.step(action)

states = np.array(states)
actions = np.array(actions)


class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(10, 64)
        self.conv2 = GCNConv(64, 10)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

# Initialize the GNN
gnn = GNN()

# Define a loss function and an optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(gnn.parameters(), lr=0.01)

# Convert the states and actions to PyTorch tensors
states = torch.tensor(states, dtype=torch.float)
actions = torch.tensor(actions, dtype=torch.float)

# Pretrain the GNN
for epoch in range(100):
    # Forward pass
    outputs = gnn(states)
    loss = criterion(outputs, actions)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()