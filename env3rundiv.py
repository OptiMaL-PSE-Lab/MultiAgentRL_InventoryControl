from os import truncate
from pickle import FALSE
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import copy
import gymnasium as gym
from gymnasium.spaces import Dict, Box
import numpy as np
import matplotlib.pyplot as plt
import torch
from ray.rllib.env.wrappers.multi_agent_env_compatibility import MultiAgentEnvCompatibility
from scipy.stats import poisson, randint
"""
This environment is for a multi product, multi echelon supply chain 
action space is (2,) defined as s and S parameters 
"""

def create_network(connections):
    num_nodes = max(connections.keys())
    network = np.zeros((num_nodes + 1, num_nodes + 1))
    for parent, children in connections.items():
        if children:
            for child in children:
                network[parent][child] = 1

    return network


def get_stage(node, network):
    reached_root = False
    stage = 0
    counter = 0
    if node == 0:
        return 0
    while not reached_root:
        for i in range(len(network)):
            if network[i][node] == 1:
                stage += 1
                node = i
                if node == 0:
                    return stage
        counter += 1
        if counter > len(network):
            raise Exception("Infinite Loop")

def get_retailers(network):
    retailers = []
    for i in range(len(network)):
        if not any(network[i]):
            retailers.append(i)

    return retailers

class MultiAgentInvManagementDiv(MultiAgentEnv):
    def __init__(self, config, **kwargs):

        self.config = config.copy()
        self.bullwhip = config.get("bullwhip", False)
        # Number of Periods in Episode
        self.num_periods = config.get("num_periods", 50)

        # Structure
        self.num_nodes = config.get("num_nodes", 6)
        self.connections = config.get("connections", {0: [1,2], 1:[3,4], 2:[4, 5], 3:[], 4:[], 5:[]})
        self.network = create_network(self.connections)
        self.order_network = np.transpose(self.network)
        self.retailers = get_retailers(self.network)

        #determine the echelon number 
        self.echelons = {node: get_stage(node, self.network) for node in range(len(self.network))}

        self.node_names = []
        self.node_names = [f"{self.echelons[node]}_{node:02d}" for node in range(len(self.network))]

        #node names is defined by [echelon number_node number_product number]. e.g. 0_00_0 [echelon 0, node 0, product 0]

        self.non_retailers = list()
        for i in range(self.num_nodes):
            if i not in self.retailers:
                self.non_retailers.append(i)
        self.upstream_node = dict()
        for i in range(1, self.num_nodes):
            self.upstream_node[i] = np.where(self.order_network[i] == 1)[0][0]


        self.num_stages = get_stage(node=int(self.num_nodes - 1), network=self.network) + 1
    
        self.a = config.get("a", -1)
        self.b = config.get("b", 1)

        self.num_agents = config.get("num_agents", self.num_nodes)
        self.inv_init = config.get("init_inv", np.ones(self.num_nodes)*100)
        self.inv_target = config.get("inv_target", np.ones(self.num_nodes) * 10) #0?
        self.prev_actions = config.get("prev_actions", True)
        self.prev_demand = config.get("prev_demand", True)
        self.prev_length = config.get("prev_length", 1)
        delay_init = np.array([1,2,3,1,1,2,1,1,2,3,4,5,1,2,1,1,1,1,1,1,1,1,2,1,2,1,1,2,1,2,1,2,1,2,1,1,3,1,1,2,3,3,2,1,1,2,3,1,1,1,1,1])
        self.delay = delay_init
        self.max_delay = np.max(self.delay)

        #if there is no maximum delay, then no time dependency == False 
        if self.max_delay == 0:
            self.time_dependency = False
        else:
            self.time_dependency = True 

        # Price of goods
        stage_price = np.arange(self.num_stages+1) + 2
        stage_cost = np.arange(self.num_stages+1) + 1

        self.node_price = np.zeros(self.num_nodes)
        self.node_cost = np.zeros(self.num_nodes)

        for i in range(self.num_nodes):
            stage = get_stage(i, self.network)
            self.node_price[i] = 2 * stage_price[stage] #this was x2 during training 
            self.node_cost[i] = 0.5 * stage_cost[stage] #this was x0.5 during training
            
        #self.price = config.get("price", np.flip(np.arange((self.num_stages + 1, self.num_products)) + 1))

        # Stock Holding and Backlog cost
        self.stock_cost = config.get("stock_cost", np.ones(self.num_nodes)*0.5)
        self.backlog_cost = config.get("backlog_cost", np.ones(self.num_nodes)*2.5) #this was x2 during training but changed to 2.5 for bullwhip runner
        self.initn = self.node_price

        # customer demand 
        self.demand_dist = config.get("demand_dist", "poisson")
        self.SEED = config.get("seed", 52)
        np.random.seed(seed=int(self.SEED))
        
        # Customer demand noise
        self.noisy_demand = config.get("noisy_demand", False)
        self.noisy_demand_threshold = config.get("noisy_demand_threshold", 0.5)

        # Lead time noise
        self.noisy_delay = config.get("noisy_delay", False)
        self.noisy_delay_threshold = config.get("noisy_delay_threshold", 0.5)


        # Capacity
        self.inv_max = config.get("inv_max", np.ones(self.num_nodes, dtype=np.int16)* 100)
        order_max = np.zeros(self.num_nodes)
        for i in range(1, self.num_nodes):
            indices = np.where(self.order_network[i] == 1)
            order_max[i] = self.inv_max[indices].max()

        order_max[0] = self.inv_max[0]
        self.order_max = config.get("order_max", order_max)

        # Number of downstream nodes of a given node and max demand epr producct at a given node
        self.num_downstream = dict()
        self.demand_max = copy.deepcopy(self.inv_max)


        for i in range(self.num_nodes):
            self.num_downstream[i] = np.sum(self.network[i])
            downstream_max_demand = 0
            for j in range(len(self.network[i])):
                if self.network[i][j] == 1:
                    downstream_max_demand += self.order_max[j]
            if downstream_max_demand > self.demand_max[i]:
                self.demand_max[i] = downstream_max_demand

        self.done = set()

        #action space is continuos. the decisions are s (reorder point) and S (order up to level)
        self.action_space = Box(
            low=-1,
            high=1,
            dtype=np.float64,
            shape=(2,)
        )


        # observation space (Inventory position at each echelon, 
        # which is any integer value)
        # elif not self.time_dependency and not self.prev_actions and self.prev_demand:
        #self.observation_space = Box(
        #            low=np.ones(3 + self.prev_length + 1, dtype=np.float64) * self.a,
        #            high=np.ones(3 + self.prev_length + 1, dtype=np.float64) * self.b,
        #            dtype=np.float64,
        #            shape=(3 + self.prev_length + 1,))

        #time dependency, prevv_actions and prev_demand
        self.observation_space = Box(
                    low=np.ones(3 + self.prev_length*2 + self.max_delay, dtype=np.float64) * self.a,
                    high=np.ones(3 + self.prev_length*2 + self.max_delay, dtype=np.float64) * self.b,
                    dtype=np.float64,
                    shape=(3 + self.max_delay + self.prev_length*2,))
                    


        self.state = {}

        # Error catching
        assert isinstance(self.num_periods, int)
        # Check maximum possible order is less than inventory capacity for each node
        for i in range(len(self.order_max) - 1):
            if self.order_max[i] > self.inv_max[i + 1]:
                break
                raise Exception('Maximum order cannot exceed maximum inventory of upstream node')



        # Maximum order of first node cannot exceed its own inventory
        assert self.order_max[0] <= self.inv_max[0]

        self.reset()



    def reset(self, customer_demand=None, noisy_delay=False, noisy_delay_threshold=0, seed = None, options = None):
        """
        Create and initialize all variables.
        Nomenclature:
            inv = On hand inventory at the start of each period at each node (except last one).
            order_u = Pipeline inventory at the start of each period at each node (except last one).
            order_r = Replenishment order placed at each period at each node (except last one).
            demand = demand at each node
            ship = Sales performed at each period at each node.
            backlog = Backlog at each period at each node.
            profit = Total profit at each node.
        """

        periods = self.num_periods
        num_nodes = self.num_nodes

        if noisy_delay:
            self.noisy_delay = noisy_delay
            self.noisy_delay_threshold = noisy_delay_threshold

        if customer_demand is not None:
            self.customer_demand = customer_demand
        else:
            # Custom customer demand
            if self.demand_dist == "custom":
                self.customer_demand = self.config.get("customer_demand",
                                                       np.ones((len(self.retailers), self.num_periods),
                                                               dtype=np.int16) * 5)
            # Poisson distribution
            elif self.demand_dist == "poisson":
                self.mu = self.config.get("mu", 5)
                self.dist = poisson
                self.dist_param = {'mu': self.mu}
                self.customer_demand = self.dist.rvs(size=(len(self.retailers), self.num_periods), **self.dist_param)
            # Uniform distribution
            elif self.demand_dist == "uniform":
                lower_upper = self.config.get("lower_upper", (1, 5))
                lower = lower_upper[0]
                upper = lower_upper[1]
                self.dist = randint
                self.dist_param = {'low': lower, 'high': upper}
                if lower >= upper:
                    raise Exception('Lower bound cannot be larger than upper bound')
                self.customer_demand = self.dist.rvs(size=(len(self.retailers), self.num_periods), **self.dist_param)
            else:
                raise Exception('Unrecognised, Distribution Not Implemented')

            if self.noisy_demand:
                for k in range(len(self.retailers)):
                    for j in range(self.num_periods):
                        double_demand = np.random.uniform(0, 1)
                        zero_demand = np.random.uniform(0, 1)
                        if double_demand <= self.noisy_demand_threshold:
                            self.customer_demand[k, j] = 2 * self.customer_demand[k, j]
                        if zero_demand <= self.noisy_demand_threshold:
                            self.customer_demand[k, j] = 0

        # Assign customer demand to each retailer
        self.retailer_demand = dict()
        for i in range(self.customer_demand.shape[0]):
            self.retailer_demand[self.retailers[i]] = self.customer_demand[i]

        # simulation result lists
        self.inv = np.zeros([periods + 1, num_nodes])  # inventory at the beginning of each period
        self.order_r = np.zeros([periods, num_nodes])  # replenishment order (last node places no replenishment orders)
        self.order_u = np.zeros([periods + 1, num_nodes])  # Unfulfilled order
        self.ship = np.zeros([periods, num_nodes])  # units sold
        self.acquisition = np.zeros([periods, num_nodes])
        self.backlog = np.zeros([periods + 1, num_nodes])  # backlog
        self.demand = np.zeros([periods + 1, num_nodes])
        if self.time_dependency:
            self.time_dependent_state = np.zeros([periods, num_nodes, self.max_delay])

        # Initialise list of dicts tracking goods shipped from one node to another
        self.ship_to_list = []
        for i in range(self.num_periods):
            # Shipping dict
            ship_to = dict()
            for node in self.non_retailers:
                ship_to[node] = dict()
                for d_node in self.connections[node]:
                    ship_to[node][d_node] = 0

            self.ship_to_list.append(ship_to)

        self.backlog_to = dict()
        for i in range(self.num_nodes):
            if len(self.connections[i]) > 1:
                self.backlog_to[i] = dict()
                for node in self.connections[i]:
                    self.backlog_to[i][node] = 0

        # initialization
        self.period = 0  # initialize time
        for node in self.retailers:
            self.demand[self.period, node] = self.retailer_demand[node][self.period]
        self.inv[self.period, :] = self.inv_init  # initial inventory

        # set state
        self._update_state()
        infos = {}
        return self.state, infos

    def _update_state(self):
        # Dictionary containing observation of each agent
        #dictionary containing observation of each agent 
        self.obs = {}
        

        t = self.period
        m = self.num_nodes
        #p = self.num_products

        for i in range(m):
            # Each agent observes five things at every time-step
            # Their inventory, backlog, demand received, 
            # acquired inventory from upstream node
            # and inventory sent to downstream node which forms observation/state vector
            agent = self.node_names[i] # Get agent name
            node = i 
            #product = i % p 
            #node = i //p 
            #time dependent, prev actions, prev demand not share_network
            self.obs_vector = np.zeros(3 + self.prev_length*2 + self.max_delay)

            # Initialise state vector

            self.prev_demand = True 
            self.prev_actions = True

            if self.prev_demand:
                demand_history = np.zeros(self.prev_length)
                for j in range(self.prev_length):
                    if j < t:
                        demand_history[j] = self.demand[t - 1 - j, node]

                demand_history = self.rescale(demand_history, 
                                              np.zeros(self.prev_length),
                                              np.ones(self.prev_length)*self.demand_max[node],
                                              self.a, self.b)
                
            if self.time_dependency:
                delay_states = np.zeros(self.max_delay)
                if t >=1 :
                    
                    delay_states = np.minimum(self.time_dependent_state[t-1, node, :], 
                                              np.ones(self.max_delay)* self.inv_max[node]*2)
                    
                delay_states = self.rescale(delay_states, np.zeros(self.max_delay),
                                            np.ones(self.max_delay)*self.inv_max[node]*2, 
                                            self.a, self.b)
            if self.prev_actions:
                order_history = np.zeros(self.prev_length)
                for j in range(self.prev_length):
                    if j < t:
                        order_history[j] = self.order_r[t - 1 - j, node]
                order_history = self.rescale(order_history, np.zeros(self.prev_length),
                                              np.ones(self.prev_length)*self.order_max[node],
                                              self.a, self.b)
                
                '''
            we haven't considered prev actions as this may be inherently included in the previous demand 
            if self.prev_actions:
                order_history = np.zeros(self.prev_length)
                for j in range(self.prev_length):
                    if j < t:
                        order_history[j] = self.order_r[t - 1 - j, i]
                order_history = self.rescale(order_history, np.zeros(self.prev_length),
                                              np.ones(self.prev_length)*self.order_max[i],
                                              self.a, self.b)
                '''  # noqa: E501

            self.obs_vector[0] = self.rescale(
                self.inv[t, node], 0, self.inv_max[node], self.a, self.b)

            self.obs_vector[1] = self.rescale(
                self.backlog[t, node], 0, self.demand_max[node], self.a, self.b)

            self.obs_vector[2] = self.rescale(
                self.order_u[t, node], 0, self.order_max[node], self.a, self.b)

            demand_history = self.obs_vector[3: 3+self.prev_length] 

                        #if share network 
                        #self.obs_vector[len(self.obs_vector) - 1] = self.rescale(i, 0, self.num_nodes, self.a, self.b)  # noqa: E501
            
            
            if self.time_dependency and self.prev_actions and self.prev_demand:


                self.obs_vector[3:3+self.prev_length] = demand_history

                self.obs_vector[3+self.prev_length:3+self.prev_length*2] = order_history

                self.obs_vector[3+self.prev_length*2:3+self.prev_length*2+self.max_delay] = delay_states[0]
            self.obs[agent] = self.obs_vector

        self.state = self.obs.copy()

    def step(self, action_dict):
        """
        Update state, transition to next state/period/time-step
        :param action_dict:
        :return:
        """
        t = self.period
        m = self.num_nodes

        # Get replenishment order at each node
        for i in range(self.num_nodes):
            node_name = self.node_names[i]
            node = i 
            s_value1, S_value2 = action_dict[node_name]
            self.s_value1 = s_value1
            self.S_value2 = S_value2
            self.rescales1 = self.rev_scale(self.s_value1, 0, self.order_max[node], self.a, self.b)
            self.rescales2= self.rev_scale(self.S_value2, 0, self.order_max[node], self.a, self.b)

            if self.inv[t, node] < self.rescales1:
                order_quant = max(0, self.rescales2 - self.inv[t, node])
            else:
                order_quant = 0
                    
            if self.rescales2 < self.rescales1:
                self.rescales2 = self.rescales1
                    
            self.order_r[t, node] = order_quant

            self.order_r[t, node] = np.round(self.order_r[t, node], 0).astype(int)

        self.order_r[t, :] = np.minimum(np.maximum(self.order_r[t, :], np.zeros(self.num_nodes)), self.order_max)

        # Demand of goods at each stage
        # Demand at last (retailer stages) is customer demand
        for node in self.retailers:
            self.demand[t, node] = np.minimum(self.retailer_demand[node][t], self.inv_max[node])  # min for re-scaling
        # Demand at other stages is the replenishment order of the downstream stage
        for i in range(self.num_nodes):
            if i not in self.retailers:
                for j in range(i, len(self.network[i])):
                    if self.network[i][j] == 1:
                        self.demand[t, i] += self.order_r[t, j]

        # Update acquisition, i.e. goods received from previous node
        self.update_acquisition()

        # Amount shipped by each node to downstream node at each time-step. This is backlog from previous time-steps
        # And demand from current time-step, This cannot be more than the current inventory at each node
        self.ship[t, :] = np.minimum(self.backlog[t, :] + self.demand[t, :], self.inv[t, :] + self.acquisition[t, :])

        # Get amount shipped to downstream nodes
        for i in self.non_retailers:
            # If shipping to only one downstream node, the total amount shipped is equivalent to amount shipped to
            # downstream node
            if self.num_downstream[i] == 1:
                self.ship_to_list[t][i][self.connections[i][0]] = self.ship[t, i]
            # If node has more than one downstream nodes, then the amount shipped needs to be split appropriately
            elif self.num_downstream[i] > 1:
                # Extract the total amount shipped in this period
                ship_amount = self.ship[t, i]
                # If shipment equal to or more than demand, send ordered amount to each downstream node
                if self.ship[t, i] >= self.demand[t, i]:
                    # If there is backlog, fulfill it first then fulfill demand
                    if self.backlog[t, i] > 0:
                        # Fulfill backlog first
                        while_counter = 0  # to exit infinite loops if error
                        # Keep distributing shipment across downstream nodes until there is no backlog or no goods left
                        while sum(list(self.backlog_to[i].values())) > 0 and ship_amount > 0:
                            # Keep distributing shipped goods to downstream nodes
                            for node in self.connections[i]:
                                # If there is a backlog towards a downstream node ship a unit of product to that node
                                if self.backlog_to[i][node] > 0:
                                    self.ship_to_list[t][i][node] += 1  # increase amount shipped to node
                                    self.backlog_to[i][node] -= 1  # decrease its corresponding backlog
                                    ship_amount -= 1  # reduce amount of shipped goods left

                            # Counter to escape while loop with error if infinite
                            while_counter += 1
                            if while_counter > self.demand_max[i] * 2:
                                raise Exception("Infinite Loop 1")

                        # If there is still left-over shipped goods fulfill current demand if any
                        if ship_amount > 0 and self.demand[t, i] > 0:
                            # Create a dict of downstream nodes' demand/orders
                            outstanding_order = dict()
                            for node in self.connections[i]:
                                outstanding_order[node] = self.order_r[t, node]

                            while_counter = 0
                            # Keep distributing shipment across downstream nodes until there is no backlog or no
                            # outstanding orders left
                            while ship_amount > 0 and sum(list(outstanding_order.values())) > 0:
                                for node in self.connections[i]:
                                    if outstanding_order[node] > 0:
                                        self.ship_to_list[t][i][node] += 1  # increase amount shipped to node
                                        outstanding_order[node] -= 1  # decrease its corresponding outstanding order
                                        ship_amount -= 1  # reduce amount of shipped goods left

                                # Counter to escape while loop with error if infinite
                                while_counter += 1
                                if while_counter > self.demand_max[i]:
                                    raise Exception("Infinite Loop 2")

                            # Update backlog if some outstanding order unfulfilled
                            for node in self.connections[i]:
                                self.backlog_to[i][node] += outstanding_order[node]

                    # If there is no backlog
                    else:
                        for node in self.connections[i]:
                            self.ship_to_list[t][i][node] += self.order_r[t, node]
                            ship_amount = ship_amount - self.order_r[t, node]
                        if ship_amount > 0:
                            print("WTF")

                # If shipment is insufficient to meet downstream demand
                elif self.ship[t, i] < self.demand[t, i]:
                    while_counter = 0
                    # Distribute amount shipped to downstream nodes
                    if self.backlog[t, i] > 0:
                        # Fulfill backlog first
                        while_counter = 0  # to exit infinite loops if error
                        # Keep distributing shipment across downstream nodes until there is no backlog or no goods left
                        while sum(list(self.backlog_to[i].values())) > 0 and ship_amount > 0:
                            # Keep distributing shipped goods to downstream nodes
                            for node in self.connections[i]:
                                # If there is a backlog towards a downstream node ship a unit of product to that node
                                if self.backlog_to[i][node] > 0:
                                    self.ship_to_list[t][i][node] += 1  # increase amount shipped to node
                                    self.backlog_to[i][node] -= 1  # decrease its corresponding backlog
                                    ship_amount -= 1  # reduce amount of shipped goods left

                            # Counter to escape while loop with error if infinite
                            while_counter += 1
                            if while_counter > self.demand_max[i]:
                                raise Exception("Infinite Loop 3")

                    else:
                        # Keep distributing shipped goods to downstream nodes until no goods left
                        while ship_amount > 0:
                            for node in self.connections[i]:
                                # If amount being shipped less than amount ordered
                                if self.ship_to_list[t][i][node] < self.order_r[t, node] + self.backlog_to[i][node]:
                                    self.ship_to_list[t][i][node] += 1  # increase amount shipped to node
                                    ship_amount -= 1  # reduce amount of shipped goods left

                            # Counter to escape while loop with error if infinite
                            while_counter += 1
                            if while_counter > self.demand_max[i]:
                                raise Exception("Infinite Loop 4")

                    # Log unfulfilled order amount as backlog
                    for node in self.connections[i]:
                        self.backlog_to[i][node] += self.order_r[t, node] - self.ship_to_list[t][i][node]

        # Update backlog demand increases backlog while fulfilling demand reduces it
        self.backlog[t + 1, :] = self.backlog[t, :] + self.demand[t, :] - self.ship[t, :]
        # Capping backlog to allow re-scaling
        self.backlog[t + 1, :] = np.minimum(self.backlog[t + 1, :], self.demand_max)

        # Update time-dependent states
        if self.time_dependency:
            self.time_dependent_acquisition()

        # Update unfulfilled orders/ pipeline inventory
        self.order_u[t + 1, :] = np.minimum(
            np.maximum(
                self.order_u[t, :] + self.order_r[t, :] - self.acquisition[t, :],
                np.zeros(self.num_nodes)),
            self.inv_max)


        # Update inventory
        self.inv[t + 1, :] = np.minimum(
            np.maximum(
                self.inv[t, :] + self.acquisition[t, :] - self.ship[t, :],
                np.zeros(self.num_nodes)),
            self.inv_max)

        # Calculate rewards
        rewards, profit, total_profit = self.get_rewards()

        # Update period
        self.period += 1
        # Update state
        self._update_state()

        # determine if simulation should terminate
        done = {
            "__all__": self.period >= self.num_periods,
        }

        infos = {}
        for i in range(m):
            meta_info = dict()
            meta_info['period'] = self.period
            meta_info['demand'] = self.demand[t, i]
            meta_info['ship'] = self.ship[t, i]
            meta_info['acquisition'] = self.acquisition[t, i]
            meta_info['actual order'] = self.order_r[t, i]
            meta_info['profit'] = profit[i]
            meta_info['backlog'] = self.backlog[t, i]
            meta_info['inventory'] = self.inv[t, i]
            node = self.node_names[i]
            infos[node] = meta_info
        total_backlog = sum(node_info['backlog'] for node_info in infos.values())
        total_inventory = sum(node_info['inventory'] for node_info in infos.values())
        
        infos['total_backlog'] = total_backlog
        infos['overall_profit'] = total_profit
        infos['total_inventory'] = total_inventory

        truncated = {}
        for node_id in self.node_names:
            if self.period >= self.num_periods:
                truncated[node_id] = True
            else:
                truncated[node_id] = False
        truncated['__all__'] = all(truncated.values())

        return self.state, rewards, done, truncated, infos

    def get_rewards(self):
        rewards = {}
        profit = np.zeros(self.num_nodes)
        m = self.num_nodes
        t = self.period
        reward_sum = 0
        for i in range(m):
            agent = self.node_names[i]
            reward = self.node_price[i] * self.ship[t, i] \
                - self.node_cost[i] * self.order_r[t, i] \
                - self.stock_cost[i] * np.abs(self.inv[t + 1, i] - self.inv_target[i]) \
                - self.backlog_cost[i] * self.backlog[t + 1, i]

            reward_sum += reward
            profit[i] = reward
        
            #if self.independent:
            #    rewards[agent] = reward

        #if not self.independent:
            #for i in range(m):
            #agent = self.node_names[i]
            rewards[agent] = reward_sum/self.num_nodes

        total_profit = reward_sum
        return rewards, profit, total_profit 

    def update_acquisition(self):
        """
        Get acquisition at each node
        :return: None
        """
        m = self.num_nodes
        t = self.period

        # Acquisition at node 0 is unique since delay is manufacturing delay instead of shipment delay
        if t - self.delay[0] >= 0:
            extra_delay = False
            if self.noisy_delay:
                delay_percent = np.random.uniform(0, 1)
                if delay_percent <= self.noisy_delay_threshold:
                    extra_delay = True

            self.acquisition[t, 0] += self.order_r[t - self.delay[0], 0]
            if extra_delay and t < self.num_periods - 1:
                self.acquisition[t + 1, 0] += self.acquisition[t, 0]
                self.acquisition[t, 0] = 0
        else:
            self.acquisition[t, 0] = self.acquisition[t, 0]

        # Acquisition at subsequent stage is the delayed shipment of the upstream stage
        for i in range(1, m):
            if t - self.delay[i] >= 0:
                extra_delay = False
                if self.noisy_delay:
                    delay_percent = np.random.uniform(0, 1)
                    if delay_percent <= self.noisy_delay_threshold:
                        extra_delay = True
                self.acquisition[t, i] += \
                self.ship_to_list[t - self.delay[i]][self.upstream_node[i]][i]
                if extra_delay and t < self.num_periods - 1:
                    self.acquisition[t + 1, i] += self.acquisition[t, i]
                    self.acquisition[t, i] = 0

            else:
                self.acquisition[t, i] = self.acquisition[t, i]

    def time_dependent_acquisition(self):
        """
        Get time-dependent states
        :return: None
        """
        m = self.num_nodes
        t = self.period

        # Shift delay down with every time-step
        if self.max_delay > 1 and t >= 1:
            self.time_dependent_state[t, :, 0:self.max_delay - 1] = self.time_dependent_state[t - 1, :,
                                                                    1:self.max_delay]

        # Delayed states of first node
        self.time_dependent_state[t, 0, self.delay[0] - 1] = self.order_r[t, 0]
        # Delayed states of rest of n:
        for i in range(1, m):
            self.time_dependent_state[t, i, self.delay[i] - 1] = \
                self.ship_to_list[t][self.upstream_node[i]][i]

    def rescale(self, val, min_val, max_val, A=-1, B=1):
        if isinstance(val, np.ndarray):
            a = np.ones(np.size(val)) * A
            b = np.ones(np.size(val)) * B
        else:
            a = A
            b = B

        val_scaled = a + (((val - min_val) * (b - a)) / (max_val - min_val))

        return val_scaled

    def rev_scale(self, val_scaled, min_val, max_val, A=-1, B=1):
        if isinstance(val_scaled, np.ndarray):
            a = np.ones(np.size(val_scaled)) * A
            b = np.ones(np.size(val_scaled)) * B
        else:
            a = A
            b = B

        val = (((val_scaled - a) * (max_val - min_val)) / (b - a)) + min_val

        return val