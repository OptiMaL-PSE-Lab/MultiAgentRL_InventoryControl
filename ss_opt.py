import numpy as np
from scipy.optimize import minimize
from env3run import MultiAgentInvManagementDiv

env = MultiAgentInvManagementDiv(config = {})

num_products = env.num_products
num_periods = env.num_periods
num_nodes = env.num_nodes 
num_stages = env.num_stages
node_price = env.node_price
node_cost = env.node_cost
stock_cost = env.stock_cost
backlog_cost = env.backlog_cost
init_inv = env.inv_init
# Generating demand for each product and period
demand = np.random.poisson(5, (num_products, num_nodes, num_periods))

# Define the cost function to minimize
def cost_function(params, init_inv, demand, num_products, num_nodes, node_price, node_cost, stock_cost, backlog_cost):
    s_values = params[:num_products * num_nodes]
    S_values = params[num_products * num_nodes:]

    s_values = s_values.reshape((num_products, num_nodes))
    S_values = S_values.reshape((num_products, num_nodes))

    total_cost = 0

    for i in range(num_products):
        for j in range(num_nodes):
            s = s_values[i, j]
            S = S_values[i, j]

            inventory = init_inv[j,i]  # Start with an initial order

            for t in range(num_periods):
                d = demand[i, j, t]

                inventory -= d
                if inventory < 0:
                    total_cost += -abs(inventory) * stock_cost[j,i] - abs(inventory) * backlog_cost[j,i]
                    inventory = 0
                else:
                    total_cost = inventory * node_price[j,i] 

                if inventory <= s:
                    total_cost += S * node_price[j, i]
                    inventory += S

    return total_cost

# Example parameters
initial_guess = [20] * num_products * num_nodes + [60] * num_products * num_nodes

# Use scipy's minimize function to find optimal s, S
result = minimize(cost_function, initial_guess, args=(init_inv, demand, num_products, num_nodes, node_price, node_cost, stock_cost, backlog_cost), method='L-BFGS-B')

# Extract the optimal values
optimal_params = result.x

# Ensure the cost function returns a scalar
optimal_cost = cost_function(optimal_params, init_inv, demand, num_products, num_nodes, node_price, node_cost, stock_cost, backlog_cost)

optimal_s_values = optimal_params[:num_products * num_nodes].reshape((num_products, num_nodes))
optimal_S_values = optimal_params[num_products * num_nodes:].reshape((num_products, num_nodes))

print("Optimal s values:")
print(optimal_s_values)
print("\nOptimal S values:")
print(optimal_S_values)
print("\nOptimal cost:")
print(optimal_cost)
