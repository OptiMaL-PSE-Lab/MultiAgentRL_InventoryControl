from cgi import test
from env2SA import InvManagementDiv#
import numpy as np 
from sspolicy import optimize_inventory_policy, dfo_func, ss_policy

seed = 52
np.random.seed(seed=seed)
num_tests = 200 

# Create environment
config = {}
env = InvManagementDiv(config=config)

test_demand = env.dist.rvs(size=(num_tests, (len(env.retailers)), env.num_periods), **env.dist_param)

#DFO 
init_policy = np.ones(env.num_nodes * 2) 
init_policy[:env.num_nodes] *=10
init_policy[env.num_nodes:] *=70

print("Initial policy: {}".format(init_policy))
print(test_demand[0,:])
policy, out = optimize_inventory_policy(env, dfo_func, init_policy = init_policy ,method='Powell', demand=test_demand[0,:])
print("Re-order levels, s: {}".format(policy[:env.num_nodes]))
print("Order up to levels, S: {}".format(policy[env.num_nodes:]))
print("DFO Info:\n{}".format(out))


dfo_rewards = []
backlog_values = [] 
inv_values = []
for i in range(20):
    demand = test_demand[i, :]
    env.reset(customer_demand=demand , noisy_delay=False, noisy_delay_threshold = 0)
    dfo_reward = 0
    done = False
    n = 0 
    while not done:
        dfo_action = ss_policy(policy[:env.num_nodes], policy[env.num_nodes:], env)
        #print('dfo action',dfo_action)
        s, r, done, info = env.step(dfo_action)
        b = info['backlog'] 
        i = info['inv']
        dfo_reward += r
        if done: # if the episode is done 
            backlog_values.append(b)
            inv_values.append(i)
    dfo_rewards.append(dfo_reward)

print(dfo_rewards)
print(len(backlog_values))
print(len(inv_values))


print(f'Mean DFO rewards is {np.mean(dfo_rewards)}')
print(f'median DFO rewards is {np.median(dfo_rewards)}')
print(f'Median DFO rewards is {np.min(dfo_rewards)}')
print(f'Std DFO rewards is {np.std(dfo_rewards)}')
print(f'Backlog values is {np.median(backlog_values)}')
print(f'Inventory values is {np.median(inv_values)}')
print(f'Inventory values STD is {np.std(inv_values)}')