import numpy as np 
import matplotlib.pyplot as plt
from env3rundiv import MultiAgentInvManagementDiv

config = {}
env = MultiAgentInvManagementDiv(config= config)

import numpy as np

# Parameters
T = 50  # number of periods
max_inventory = env.inv_max  # maximum inventory level at each echelon
max_order = env.order_max  # maximum order quantity at each echelon

# Define the structure of the multi-echelon system
connecions = env.connections

# Define costs for each echelon
holding_costs = env.node_cost

ordering_costs = env.stock_cost

shortage_costs = env.backlog_cost

retailer_demand = env.retailer_demand

# State space and action space
max_inventory = max_inventory[0]
max_order = max_order[0]
inventory_levels = np.arange(max_inventory + 1)
order_quantities = np.arange(max_order + 1)

# Initialize cost-to-go function for each echelon
cost_to_go = {echelon: np.zeros((T + 1, max_inventory + 1)) for echelon in connecions}

# Dynamic programming for the bottom echelons upwards
def backward_induction(echelon, demand=None):
    for t in range(T - 1, -1, -1):
        for i in inventory_levels:
            min_cost = float('inf')
            best_order = 0
            for q in order_quantities:
                if demand is not None:
                    next_inventory = i + q - demand[t]
                else:
                    next_inventory = i + q
                next_inventory = int(max(0, min(next_inventory, max_inventory)))  # bound inventory
                holding_costs_total = holding_costs[echelon] * max(0, next_inventory)
                shortage_costs_total = shortage_costs[echelon] * max(0, -next_inventory)
                order_costs_total = ordering_costs[echelon] if q > 0 else 0
                if echelon in connecions and connecions[echelon]:
                    print(t+1 , next_inventory)
                    children_costs = sum(cost_to_go[child][t+1, int(next_inventory)] for child in connecions[echelon])
                else:
                    children_costs = 0
                total_cost = holding_costs_total + shortage_costs_total + order_costs_total + children_costs
                if total_cost < min_cost:
                    min_cost = total_cost
                    best_order = q
            cost_to_go[echelon][t, i] = min_cost

# Perform backward induction for the leaf nodes first
for echelon in [3, 4, 5]:
    backward_induction(echelon, retailer_demand[echelon])

# Perform backward induction for intermediate nodes
for echelon in [1, 2]:
    backward_induction(echelon)

# Perform backward induction for the top node (central warehouse)
backward_induction(0)

# Extract optimal policy for each echelon
optimal_policy = {echelon: np.zeros((T, max_inventory + 1), dtype=int) for echelon in connecions}
for echelon in connecions:
    for t in range(T):
        for i in inventory_levels:
            best_order = 0
            min_cost = float('inf')
            for q in order_quantities:
                if echelon in retailer_demand:
                    next_inventory = i + q - retailer_demand[echelon][t]
                else:
                    next_inventory = i + q
                next_inventory = max(0, min(next_inventory, max_inventory))
                holding_costs_total = holding_costs[echelon] * max(0, next_inventory)
                shortage_costs_total = shortage_costs[echelon] * max(0, -next_inventory)
                order_costs_total = ordering_costs[echelon] if q > 0 else 0
                if echelon in connecions and connecions[echelon]:
                    children_costs = sum(cost_to_go[child][t + 1, int(next_inventory)] for child in connecions[echelon])
                else:
                    children_costs = 0
                total_cost = holding_costs_total + shortage_costs_total + order_costs_total + children_costs
                if total_cost < min_cost:
                    min_cost = total_cost
                    best_order = q
            optimal_policy[echelon][t, i] = best_order

# Find the (s, S) levels for each echelon
s_levels = {echelon: np.zeros(T) for echelon in connecions}
S_levels = {echelon: np.zeros(T) for echelon in connecions}
for echelon in connecions:
    for t in range(T):
        for i in inventory_levels:
            if optimal_policy[echelon][t, i] > 0:
                s_levels[echelon][t] = i
                S_levels[echelon][t] = i + optimal_policy[echelon][t, i]
                break

# Print the optimal (s, S) levels for each echelon
for echelon in connecions:
    print(f"Optimal (s, S) levels per period for echelon {echelon}:")
    for t in range(T):
        print(f"Period {t + 1}: s = {s_levels[echelon][t]}, S = {S_levels[echelon][t]}")
    print()
