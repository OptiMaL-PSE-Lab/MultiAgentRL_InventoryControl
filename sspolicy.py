import numpy as np 
from scipy.optimize import minimize
from scipy.optimize import differential_evolution


def ss_policy(reorder_point, order_up_to, env):
    '''
    Implements an (s, S) policy. This means that for each node
    in the network, if the inventory at that node falls to or
    below the reorder point s, we will re-order inventory to
    bring it to the order-up-to level S.
    '''
    inv_ech = env.inv[env.period, :] 

    orders_needed = np.where(inv_ech <= reorder_point, order_up_to - inv_ech, 0)
    # Ensure that actions can be fulfilled by checking constraints
    actions = np.minimum(env.order_max, orders_needed)
    return actions


def dfo_func(policy, env, demand=None, *args):
    '''
    Runs an episode based on current (s, S) model settings. This allows us to use our environment for the DFO optimizer.
    '''
    reorder_point, order_up_to = policy[:env.num_nodes], policy[env.num_nodes:]
    
    if demand is None:
        env.reset()  # Ensure env is fresh
    else:
        env.reset(customer_demand=demand)  # Ensure env is fresh
    
    rewards = []
    done = False
    while not done:
        action = ss_policy(reorder_point, order_up_to, env)
        state, reward, done, _ = env.step(action)
        rewards.append(reward)
    
    rewards = np.array(rewards)
    prob = env.dist.pmf(env.customer_demand, **env.dist_param)
    
    return -np.sum(rewards)

def optimize_inventory_policy(env, fun, init_policy, method, demand):
    if init_policy is None:
        init_policy = np.concatenate([np.ones(env.num_nodes) * (env.mu - env.sigma), 
                                      np.ones(env.num_nodes) * (env.mu + env.sigma)])
    
    bounds_1 = [(0, 50)] * (len(init_policy)//2)
    bounds_2 = [(40, 100)] * (len(init_policy)//2)
    bounds = bounds_1 + bounds_2
    # Optimize policy
    if demand is None:
        out = minimize(fun=fun, x0=init_policy, args=env, method=method)
        #out = differential_evolution(fun, bounds=bounds, args=(env,), maxiter=1000, popsize=100, disp=True)
    else:
        out = minimize(fun=fun, x0=init_policy, args=(env, demand), method=method)

        #out = differential_evolution(fun, bounds=bounds, args=(env,demand), maxiter=1000, popsize=100, disp=True)
    
    policy = out.x.copy()
    
    # Policy must be positive integer
    policy = np.round(np.maximum(policy, 0), 0).astype(int)
    
    return policy, out
