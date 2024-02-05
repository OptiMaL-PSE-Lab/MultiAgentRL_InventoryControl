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
        #coeff 
        self.np = config.get("np", 2)
        self.nc = config.get("nc", 0.5)
        self.bc = config.get("bs", 2.5)

        # Structure
        self.num_products = config.get("num_products",2)
        self.num_nodes = config.get("num_nodes", 6)

        self.connections = config.get("connections", {0: [1,2], 1:[3,4,5], 2:[3,4,5], 3:[], 4:[], 5:[]})
        self.network = create_network(self.connections)
        self.order_network = np.transpose(self.network)
        self.retailers = get_retailers(self.network)

        #determine the echelon number 
        self.echelons = {node: get_stage(node, self.network) for node in range(len(self.network))}

        self.node_names = []
        self.node_names = [f"{self.echelons[node]}_{node:02d}_{product}" for node in range(len(self.network)) for product in range(self.num_products)]
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

        self.num_agents = config.get("num_agents", self.num_nodes * self.num_products)
        self.inv_init = config.get("init_inv", np.ones((self.num_nodes, self.num_products))*30)
        self.inv_target = config.get("inv_target", np.ones((self.num_nodes, self.num_products)) * 1)
        self.prev_actions = config.get("prev_actions", True)
        self.prev_demand = config.get("prev_demand", True)
        self.prev_length = config.get("prev_length", 1)
        delay_init = np.array([1,2,3,4,5,6,7])
        self.delay = delay_init
        self.max_delay = np.max(self.delay)

        #if there is no maximum delay, then no time dependency == False 
        if self.max_delay == 0:
            self.time_dependency = False
        else:
            self.time_dependency = True 

        # Price of goods
        stage_price = np.arange(self.num_stages) + 2
        stage_cost = np.arange(self.num_stages) + 1

        self.node_price = np.zeros((self.num_nodes, self.num_products))
        self.node_cost = np.zeros((self.num_nodes, self.num_products))

        for i in range(self.num_nodes):
            for product in range(self.num_products):
                stage = get_stage(i, self.network)
                self.node_price[i][product] = 2 * stage_price[stage] #this was x2 during training 
                self.node_cost[i][product] = 0.5 * stage_cost[stage] #this was x0.5 during training
            
        #self.price = config.get("price", np.flip(np.arange((self.num_stages + 1, self.num_products)) + 1))

        # Stock Holding and Backlog cost
        self.stock_cost = config.get("stock_cost", np.ones((self.num_nodes, self.num_products)))
        self.backlog_cost = config.get("backlog_cost", np.ones((self.num_nodes, self.num_products))*2.5) #this was x2 during training but changed to 2.5 for bullwhip runner


        # customer demand 
        self.demand_dist = config.get("demand_dist", "poisson")
        self.SEED = config.get("seed", 52)
        np.random.seed(seed=int(self.SEED))
        
        #Lead time noise 
        self.noise_delay = config.get("noise_delay", False)
        self.noise_delay_threshold = config.get("noise_delay_threshold", 0)

        # Capacity
        self.inv_max = config.get("inv_max", np.ones((self.num_nodes, self.num_products), dtype=np.int16)\
             * 100)
        order_max = np.zeros((self.num_nodes, self.num_products))
        for i in range(1, self.num_nodes):
            for product in range(self.num_products):
                indices = np.where(self.order_network[i] == 1)
                order_max[i][product] = self.inv_max[indices, product].max()

        order_max[0] = self.inv_max[0]
        self.order_max = config.get("order_max", order_max)

        # Number of downstream nodes of a given node and max demand epr producct at a given node
        self.num_downstream = dict()
        self.demand_max = copy.deepcopy(self.inv_max)

        # Initialize a dictionary to store maximum downstream demand per product at each node
        downstream_max_demand_per_product = dict()

        for i in range(self.num_nodes):
            self.num_downstream[i] = np.sum(self.network[i])

            #max demand per product for this node 
            downstream_max_demand = np.zeros(self.num_products)
            
            for j in range(len(self.network[i])):
                if self.network[i][j] == 1:
                    downstream_max_demand += self.order_max[j]
            #updates the overall demand max based on the max demand per products at this node 
            for product in range(self.num_products):
                if downstream_max_demand[product] > self.demand_max[i][product]:
                    self.demand_max[i][product] = downstream_max_demand[product]
                    downstream_max_demand_per_product[(i, product)] = downstream_max_demand[product]

        self.downstream_max_demand_per_product = downstream_max_demand_per_product

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
            for product in range(self.num_products):
                if self.order_max[i][product] > self.inv_max[i + 1][product]:
                    break
                    raise Exception('Maximum order for product cannot exceed \
                        maximum inventory of upstream node')



        # Maximum order of first node per product cannot exceed its own inventory
        for product in range(self.num_products):
            assert self.order_max[0][product] <= self.inv_max[0][product]



        
        self.state = {}


        self.reset()



    def reset(self, customer_demand=None, noisy_delay=True, noisy_delay_threshold=0.1, \
              seed = None, options = None):
        """
        Create and initialize all variables.
        Nomenclature:
            inv = On hand inventory at the start of each period at each node 
            (except last one).
            order_u = Pipeline inventory at the start of each period at each node 
            (except last one).
            order_r = Replenishment order placed at each period at each node 
            (except last one).
            demand = demand at each node
            ship = Sales performed at each period at each node.
            backlog = Backlog at each period at each node.
            profit = Total profit at each node.
        """

        periods = self.num_periods
        num_nodes = self.num_nodes
        num_products = self.num_products


        '''
        customer demand shape has to be length of retailers so that we can then assign 
        the demand for each retailer
        retailers are defined as the end nodes. this will then be used to look at the 
        bullwhip effect 
        todo: need to define the demand as seasonal !!!!
        '''

        if customer_demand is not None:
            self.customer_demand = customer_demand
        else:
            if self.demand_dist == "custom":
                self.customer_demand = self.config.get("customer_demand", \
                            np.ones((len(self.retailers), self.num_periods, self.num_products), \
                            dtype=np.int16) * 5)
            # Poisson distribution
            elif self.demand_dist == "poisson":
                self.mu = self.config.get("mu", 5)
                self.dist = poisson
                self.dist_param = {'mu': self.mu}
                self.customer_demand = np.ones((len(self.retailers), self.num_periods, self.num_products), \
                            dtype=np.int16)
                for product in range(self.num_products):
                    demand_pattern = np.random.poisson(self.mu, size=(len(self.retailers), self.num_periods))
                    self.customer_demand[:,:, product] = demand_pattern * 6
            # Uniform distribution
            elif self.demand_dist == "uniform":
                lower_upper = self.config.get("lower_upper", (1, 5))
                lower = lower_upper[0]
                upper = lower_upper[1]
                self.dist = randint
                self.dist_param = {'low': lower, 'high': upper}
                self.customer_demand = np.ones((len(self.retailers), self.num_periods, self.num_products), \
                            dtype=np.int16)
                if lower >= upper:
                    raise Exception('Lower bound cannot be larger than upper bound')
                for product in range(self.num_products):
                    self.customer_demand[:,:, product] = self.dist.rvs(size=(len(self.retailers), \
                                                    self.num_periods, self.num_products), **self.dist_param)
            elif self.demand_dist == "seasonal":
                num_cycles = 2  # You can adjust this value to control the number of cycles per year
                seasonal_pattern = 100 + 50 * np.sin(2 * np.pi * np.arange(periods) / (periods / num_cycles))
                #seasonal_pattern = 5

                self.customer_demand = np.ones((len(self.retailers), self.num_periods, num_products), dtype = np.int16)  # noqa: E501
                for product in range(num_products):
                    demand_pattern = np.random.poisson(seasonal_pattern, size=(len(self.retailers), self.num_periods))
                    self.customer_demand[:,:, product] = demand_pattern
            else:
                raise Exception('Unrecognised, Distribution Not Implemented')
            
        bullwhip_time_introduction = 20
        bullwhip_duration = 10
        demand_factor = 5

        if self.bullwhip == True:
            if bullwhip_time_introduction < self.num_periods:
                end_bullwhip_time = min(bullwhip_time_introduction + bullwhip_duration, self.num_periods)
                self.customer_demand[:, bullwhip_time_introduction:end_bullwhip_time, :] *= demand_factor



        # Assign customer demand to each retailer and product 
        self.retailer_demand = dict()
        for i in range(self.customer_demand.shape[0]):
            retailer = self.retailers[i]
            print(self.retailers)
            self.retailer_demand[retailer] = dict()

            for product in range(self.num_products):
                self.retailer_demand[retailer][product] = self.customer_demand[i, :, product]

        # simulation result lists
        self.inv = np.zeros([periods + 1, num_nodes, num_products])  
        # inventory at the beginning of each period
        self.order_r = np.zeros([periods, num_nodes, num_products])  
        # replenishment order (last node places no replenishment orders)
        self.order_u = np.zeros([periods + 1, num_nodes, num_products])  # Unfulfilled order
        self.ship = np.zeros([periods, num_nodes, num_products])  # units sold
        self.acquisition = np.zeros([periods, num_nodes, num_products])
        self.backlog = np.zeros([periods + 1,num_nodes, num_products])  # backlog
        self.demand = np.zeros([periods + 1, num_nodes, num_products])
        if self.time_dependency:
            self.time_dependent_state = np.zeros([periods, num_nodes, num_nodes, self.num_products])


        # Initialise list of dicts tracking goods shipped from one node to another
        self.ship_to_list = []
        for i in range(self.num_periods):
            # Shipping dict
            ship_to = {}
            for node in self.non_retailers:
                ship_to[node] = {}
                for d_node in self.connections[node]:
                    ship_to[node][d_node] = {}
                    for product in range(self.num_products):
                        ship_to[node][d_node][product] = 0

            self.ship_to_list.append(ship_to)

        self.backlog_to = dict()
        for i in range(self.num_nodes):
            if len(self.connections[i]) > 1:
                self.backlog_to[i] = dict()
                for node in self.connections[i]:
                    if node not in self.backlog_to[i]:
                        self.backlog_to[i][node] = dict()
                    for product in range(self.num_products):
                        self.backlog_to[i][node][product] = 0 

        # initialization
        self.period = 0  # initialize time
        for node in self.retailers:
            for product in range(self.num_products):
                self.demand[self.period, node, product] = self.retailer_demand[node][product][self.period]
        self.inv[self.period, :, :] = self.inv_init  # initial inventory

        # set state
        self._update_state()

        infos = {}
        
        return self.state , infos

    def _update_state(self):
        # Dictionary containing observation of each agent
        #dictionary containing observation of each agent 
        self.obs = {}
        

        t = self.period
        m = self.num_nodes
        p = self.num_products

        for i in range(m * p):
            # Each agent observes five things at every time-step
            # Their inventory, backlog, demand received, 
            # acquired inventory from upstream node
            # and inventory sent to downstream node which forms observation/state vector
            agent = self.node_names[i] # Get agent name
            product = i % p 
            node = i //p 
            #time dependent, prev actions, prev demand not share_network
            self.obs_vector = np.zeros(3 + self.prev_length*2 + self.max_delay)

            # Initialise state vector

            self.prev_demand = True 
            self.prev_actions = True

            if self.prev_demand:
                demand_history = np.zeros(self.prev_length)
                for j in range(self.prev_length):
                    if j < t:
                        demand_history[j] = self.demand[t - 1 - j, node, product]

                demand_history = self.rescale(demand_history, 
                                              np.zeros(self.prev_length),
                                              np.ones(self.prev_length)*self.demand_max[node, product],
                                              self.a, self.b)
                
            if self.time_dependency:
                delay_states = np.zeros(self.max_delay)
                if t >=1 :
                    
                    delay_states = np.minimum(self.time_dependent_state[t-1, node, product, :], 
                                              np.ones(self.max_delay)* self.inv_max[node, product]*2)
                    
                delay_states = self.rescale(delay_states, np.zeros(self.max_delay),
                                            np.ones(self.max_delay)*self.inv_max[node, product]*2, 
                                            self.a, self.b)
            if self.prev_actions:
                order_history = np.zeros(self.prev_length)
                for j in range(self.prev_length):
                    if j < t:
                        order_history[j] = self.order_r[t - 1 - j, node, product]
                order_history = self.rescale(order_history, np.zeros(self.prev_length),
                                              np.ones(self.prev_length)*self.order_max[node, product],
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
                self.inv[t, node, product], 0, self.inv_max[node, product], self.a, self.b)

            self.obs_vector[1] = self.rescale(
                self.backlog[t, node, product], 0, self.demand_max[node, product], self.a, self.b)

            self.obs_vector[2] = self.rescale(
                self.order_u[t, node, product], 0, self.order_max[node, product], self.a, self.b)

            demand_history = self.obs_vector[3: 3+self.prev_length] 

                        #if share network 
                        #self.obs_vector[len(self.obs_vector) - 1] = self.rescale(i, 0, self.num_nodes, self.a, self.b)  # noqa: E501
            
            
            if self.time_dependency and self.prev_actions and self.prev_demand:


                self.obs_vector[3:3+self.prev_length] = demand_history

                self.obs_vector[3+self.prev_length:3+self.prev_length*2] = order_history

                self.obs_vector[3+self.prev_length*2:3+self.prev_length*2+self.max_delay] = delay_states[0]
                #TODO: NEED TO DOUBLE CHECK THIS AND WHY ITS NOT RIGHT! 
            self.obs[agent] = self.obs_vector
        #print("x in update state before obs.copy", self.x)

        self.state = self.obs.copy()

    def step(self, action_dict):
        """
        Update state, transition to next state/period/time-step
        :param action_dict:
        :return:
        """
        t = self.period
        m = self.num_nodes
        p = self.num_products
        
        # Get replenishment order at each node

        node_ids = []

        for echelon in range(len(self.echelons)):
            for node in range(len(self.network)):
                for product in range(self.num_products):
                    node_name = f"{self.echelons[node]}_{node:02d}_{product}"
                    node_ids.append(node_name)

                    #adjusting the code so its now based on making decisions based on s and S 
                    #print("action dictionary in step",action_dict)
                    s_value1, S_value2 = action_dict[node_name]
                    self.s_value1 = s_value1
                    self.S_value2 = S_value2
                    print("s_values", (self.s_value1, self.S_value2))
                    self.rescales1 = self.rev_scale(self.s_value1, 0, self.order_max[node, product], self.a, self.b)
                    self.rescales2= self.rev_scale(self.S_value2, 0, self.order_max[node, product], self.a, self.b)

                    #order_quant = max(0, self.S_value2 - self.s_value1)
                    if self.inv[t, node, product] < self.rescales1:
                        order_quant = max(0, self.rescales2 - self.rescales1)
                    else:
                        order_quant = 0
                    
                    if self.rescales2 < self.rescales1:
                        self.rescales2 = self.rescales1
                    
                    #order_quant = max(0, self.rescales2 - self.rescales1)
                    #print("order_quant",order_quant)
                    self.order_r[t, node, product] = order_quant

                    #self.order_r[t, node, product] = self.rev_scale(order_quant, 0, self.order_max[node, product], self.a, self.b)
                    #print("order rescale", self.order_r[t, node, product])
                    self.order_r[t, node, product] = np.round(self.order_r[t, node, product], 0).astype(int)

                    #print("rescaled values", (rescales1, rescales2))          
        self.order_r[t, :, :] = np.minimum(
            np.maximum(self.order_r[t, :, :], np.zeros((self.num_nodes, self.num_products))), self.order_max)
        
        print(node_ids)

        # Demand of goods at each stage
        # Demand at last (retailer stages) is customer demand
        for node in self.retailers:
            for product in range(self.num_products):
                self.demand[t, node, product] = np.minimum(
                self.retailer_demand[node][product][t], self.inv_max[node, product])  # min for re-scaling
        
        # Demand at other stages is the replenishment order of the downstream stage
        for i in range(self.num_nodes):
            if i not in self.retailers:
                for j in range(i, len(self.network[i])):
                    if self.network[i][j] == 1:
                        for product in range(self.num_products):
                            self.demand[t, i, product] += self.order_r[t, j, product]
        # Update acquisition, i.e. goods received from previous node
        self.update_acquisition()

        # Amount shipped by each node to downstream node at each time-step. 
        # This is backlog from previous time-steps
        # And demand from current time-step, 
        # This cannot be more than the current inventory at each node
        self.ship[t, :, :] = np.minimum(
            self.backlog[t, :, :] + self.demand[t, :, :], 
            self.inv[t, :, :] + self.acquisition[t, :, :])

        # Get amount shipped to downstream nodes
        for product in range(self.num_products):
            for i in self.non_retailers:
                # If shipping to only one downstream node, 
                # the total amount shipped is equivalent to amount shipped to
                # downstream node
                if self.num_downstream[i] == 1:
                    self.ship_to_list[t][i][self.connections[i][0]][product] = self.ship[t, i, product]
                # If node has more than one downstream nodes, 
                # then the amount shipped needs to be split appropriately
                elif self.num_downstream[i] > 1:
                    # Extract the total amount shipped in this period
                    ship_amount = self.ship[t, i, product]
                    # If shipment equal to or more than demand, 
                    # send ordered amount to each downstream node
                    if self.ship[t, i, product] >= self.demand[t, i, product]:
                        # If there is backlog, fulfill it first then fulfill demand
                        if self.backlog[t, i, product] > 0:
                            # Fulfill backlog first
                            while_counter = 0  # to exit infinite loops if error
                            # Keep distributing shipment across downstream nodes 
                            # until there is no backlog or no goods left
                            while sum(list(self.backlog_to[i][product].values())) > 0 \
                            and ship_amount > 0:
                                # Keep distributing shipped goods to downstream nodes
                                for node in self.connections[i]:
                                    # If there is a backlog towards a downstream node ship 
                                    # a unit of product to that node
                                    if self.backlog_to[i][node][product] > 0:
                                        self.ship_to_list[t][i][node][product] += 1  
                                        # increase amount shipped to node
                                        self.backlog_to[i][node][product] -= 1  
                                        # decrease its corresponding backlog
                                        ship_amount -= 1  
                                        # reduce amount of shipped goods left

                                # Counter to escape while loop with error if infinite
                                while_counter += 1
                                if while_counter > self.demand_max[i][product] * 2:
                                    raise Exception("Infinite Loop 1")

                            # If there is still left-over shipped goods fulfill 
                            # current demand if any
                            if ship_amount > 0 and self.demand[t, i, product] > 0:
                                # Create a dict of downstream nodes' demand/orders
                                outstanding_order = dict()
                                for node in self.connections[i]:
                                    for product in range(self.num_products):
                                        outstanding_order[node][product] = self.order_r[t, node, product]

                                while_counter = 0
                                # Keep distributing shipment across downstream nodes until 
                                # there is no backlog or no outstanding orders left
                                while ship_amount > 0 and \
                                            sum(list(outstanding_order.values())) > 0:
                                    for node in self.connections[i]:
                                        for product in range(self.num_products):
                                            if outstanding_order[node][product] > 0:
                                                self.ship_to_list[t][i][node][product] += 1  
                                                # increase amount shipped to node
                                                outstanding_order[node][product] -= 1  
                                                # decrease its corresponding outstanding order
                                                ship_amount -= 1  
                                                # reduce amount of shipped goods left

                                    # Counter to escape while loop with error if infinite
                                    while_counter += 1
                                    if while_counter > self.demand_max[i][product]:
                                        raise Exception("Infinite Loop 2")

                                # Update backlog if some outstanding order unfulfilled
                                for node in self.connections[i]:
                                    for product in range(self.num_products):
                                        self.backlog_to[i][node][product] += outstanding_order[node][product]

                        # If there is no backlog
                        else:
                            for node in self.connections[i]:
                                for product in range(self.num_products):
                                    self.ship_to_list[t][i][node][product] += self.order_r[t, node, product]
                                    ship_amount = ship_amount - self.order_r[t, node, product]
                            if ship_amount > 0:
                                print("WTF")

                    # If shipment is insufficient to meet downstream demand
                    elif self.ship[t, i, product] < self.demand[t, i, product]:
                        while_counter = 0
                        # Distribute amount shipped to downstream nodes
                        if self.backlog[t, i, product] > 0:
                            # Fulfill backlog first
                            while_counter = 0  # to exit infinite loops if error
                            # Keep distributing shipment across downstream nodes 
                            # until there is no backlog or no goods left
                            while sum(list(self.backlog_to[i].values())) > 0 \
                                                        and ship_amount > 0:
                                # Keep distributing shipped goods to downstream nodes
                                for node in self.connections[i]:
                                    # If there is a backlog towards a downstream node 
                                    # ship a unit of product to that node
                                    for product in range(self.num_products):
                                        if self.backlog_to[i][node][product] > 0:
                                            self.ship_to_list[t][i][node][product] += 1 
                                            # increase amount shipped to node
                                            self.backlog_to[i][node][product] -= 1  
                                            # decrease its corresponding backlog
                                            ship_amount -= 1  
                                            # reduce amount of shipped goods left

                                # Counter to escape while loop with error if infinite
                                while_counter += 1
                                if while_counter > self.demand_max[i][product]:
                                    raise Exception("Infinite Loop 3")

                        else:
                            # Keep distributing shipped goods to downstream nodes until 
                            # no goods left
                            while ship_amount > 0:
                                for node in self.connections[i]:
                                    for product in range(self.num_products):
                                        # If amount being shipped less than amount ordered
                                        if self.ship_to_list[t][i][node][product] < \
                                        self.order_r[t, node, product] + self.backlog_to[i][node][product]:
                                            if not isinstance(self.ship_to_list[t][i], dict):
                                                raise KeyError("Expected a dictionary, but found {}, {}".format(type(self.ship_to_list[t][i]), self.ship_to_list))
                                            else:
                                                self.ship_to_list[t][i][node][product] += 1  
                                                # increase amount shipped to node
                                                ship_amount -= 1  
                                            # reduce amount of shipped goods left

                                # Counter to escape while loop with error if infinite
                                while_counter += 1
                                if while_counter > self.demand_max[i][product]:
                                    raise Exception("Infinite Loop 4")

                        # Log unfulfilled order amount as backlog
                        for node in self.connections[i]:
                            for product in range(self.num_products):
                                self.backlog_to[i][node][product] += \
                                self.order_r[t, node, product] - self.ship_to_list[t][i][node][product]
        # Update backlog demand increases backlog while fulfilling demand reduces it

        self.backlog[t + 1, :, :] = \
            self.backlog[t, :, :] + self.demand[t, :, :] - self.ship[t, :, :]
        # Capping backlog to allow re-scaling
        self.backlog[t + 1, :, :] = np.minimum(self.backlog[t + 1, :, :], self.demand_max)

        # Update time-dependent states
        if self.time_dependency:
            self.time_dependent_acquisition()

        # Update unfulfilled orders/ pipeline inventory
        self.order_u[t + 1, :, :] = np.minimum(
            np.maximum(
                self.order_u[t, :, :] + self.order_r[t, :, :] - self.acquisition[t, :, :],
                np.zeros((self.num_nodes, self.num_products))),
            self.inv_max)

        # Update inventory
        self.inv[t + 1, :, :] = np.minimum(
            np.maximum(
                self.inv[t, :, :] + self.acquisition[t, :, :] - self.ship[t, :, :],
                np.zeros((self.num_nodes, self.num_products))),
                self.inv_max)
        
        print("inv", self.inv)
        print("demand", self.demand)
        print("acq", self.acquisition)
        print("ship", self.ship)
        # Calculate rewards
        rewards, profit, total_profit = self.get_rewards()
        self.rewards = rewards
        # Update period
        self.period += 1
        # Update state
        self._update_state()
        # determine if simulation should terminate
        done = {
            "__all__": self.period >= self.num_periods,
        }
        #done is the same as terminated 

        truncated = {}

        for node_id in node_ids:
            if self.period >= self.num_periods:
                truncated[node_id] = True
            else:
                truncated[node_id] = False
        truncated['__all__'] = all(truncated.values())
        
        infos = {}
        
        print("period {}, rewards {}".format(self.period, self.rewards))
        #infos['__all__'] = {}  
        for i in range(m * p):
            node = i // p
            product = i % p 
            meta_info = dict()
            node_name = self.node_names[i]
            meta_info['period'] = self.period
            meta_info['reward'] = self.rewards[node_name]
            meta_info['demand'] = self.demand[t, node, product] 
            meta_info['ship'] = self.ship[t, node, product]
            meta_info['acquisition'] = self.acquisition[t, node, product]
            meta_info['actual order'] = self.order_r[t, node, product]
            meta_info['profit'] = profit[node][product]
            meta_info['backlog'] = self.backlog[t, node, product]
            meta_info['inv'] = self.inv[t, node, product]
            meta_info['rescales1'] = self.rescales1
            meta_info['rescales2'] = self.rescales2

            
            infos[node_name] = meta_info

#need to comment when bullwhip simulation 
        if self.bullwhip == True:
            infos['overall_profit'] = total_profit

        """        for i in range(m):
                    for n in range(p):
                        meta_info = dict()
                        meta_info['period'] = self.period
                        meta_info['demand'] = self.demand[t, i, n] 
                        meta_info['ship'] = self.ship[t, i, n]
                        meta_info['acquisition'] = self.acquisition[t, i, n]
                        meta_info['actual order'] = self.order_r[t, i, n]
                        meta_info['profit'] = profit[i]
                        node = self.node_names[i]
                        infos[node] = meta_info"""

        print("rewards in step", rewards)
        for key,value in rewards.items():
            if isinstance(value, np.ndarray):
                print(f"key step:{key}, shape: {value.shape}")
            else:
                print("not an array")
        return self.state, rewards, done, truncated, infos
    
    def get_rewards(self):
        rewards = {}
        m = self.num_nodes
        p = self.num_products
        t = self.period
        reward_sum = np.zeros((m,p))
        profit = np.zeros((m,p))
        total_profit = 0 
        for node in range(m):
            for product in range(p):
                index = node * p + product
                agent = self.node_names[index]
                #if 20 <= t <= 30:
                #    self.node_price[node, product] = 1.03 * self.node_price[node, product]

                reward = self.node_price[node, product] * self.ship[t, node, product] \
                    - self.node_cost[node, product] * self.order_r[t, node, product] \
                    - self.stock_cost[node, product] * np.abs(self.inv[t + 1, node, product] - self.inv_target[node, product]) \
                    - self.backlog_cost[node, product] * self.backlog[t + 1, node, product]
                print("reward sclar value",reward)
                reward_sum[node][product] += reward
                profit[node][product] = reward_sum[node][product]
                rewards[agent] = reward

                total_profit += profit[node][product]
                print(total_profit)
                print(rewards)
                print(profit) 
        ''' 
            this will depend as to whether reward is indiviual or for all 
            if self.independent:
                    rewards[agent] = reward

            if not self.independent:
                for i in range(m):
                    agent = self.node_names[i]
                    rewards[agent] = reward_sum/self.num_nodes
        '''
        

        for key,value in rewards.items():
            if isinstance(value, np.ndarray):
                print(f"key:{key}, shape: {value.shape}")
            else:
                print("not an array")

        return rewards , profit, total_profit

    def update_acquisition(self):
        """
        Get acquisition at each node
        :return: None
        """

        m = self.num_nodes
        t = self.period
        self.noise_delay = False
        self.noise_delay_threshold = 50/100
        # Acquisition at node 0 is unique since 
        # delay is manufacturing delay instead of shipment delay
        for product in range(self.num_products):
            if t - self.delay[0] >= 0:
                extra_delay = False
                if self.noise_delay:
                    delay_percent = np.random.uniform(0, 1)
                    if delay_percent <= self.noise_delay_threshold:
                        extra_delay = True

                self.acquisition[t, 0, product] += self.order_r[t - self.delay[0], 0, product]
                if extra_delay and t < self.num_periods - 1:
                    self.acquisition[t + 1, 0, product] += self.acquisition[t, 0, product]
                    self.acquisition[t, 0, product] = 0
            else:
                self.acquisition[t, 0, product] = self.acquisition[t, 0, product]

        # Acquisition at subsequent stage is the delayed shipment of the upstream stage
        for product in range(self.num_products):
            for i in range(1, m):
                if t - self.delay[i] >= 0:
                    extra_delay = False
                    if self.noise_delay:
                        delay_percent = np.random.uniform(0, 1)
                        if delay_percent <= self.noise_delay_threshold:
                            extra_delay = True
                    self.acquisition[t, i, product] += \
                    self.ship_to_list[t - self.delay[i]][self.upstream_node[i]][i][product]
                    if extra_delay and t < self.num_periods - 1:
                        self.acquisition[t + 1, i, product] += self.acquisition[t, i, product]
                        self.acquisition[t, i, product] = 0

                else:
                    self.acquisition[t, i, product] = self.acquisition[t, i, product]



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
        for product in range(self.num_products):
            # Delayed states of first node
            if self.delay[0] >0:
                self.time_dependent_state[t, 0, self.delay[0] - 1, product] = self.order_r[t, 0, product]
            # Delayed states of rest of n:
            for i in range(1, m):
                if self.delay[i]>0:
                    self.time_dependent_state[t, i, self.delay[i] - 1, product] = \
                    self.ship_to_list[t][self.upstream_node[i]][i][product]
            
    def rescale(self, val, min_val, max_val, A=-1, B=1):
        if isinstance(val, np.ndarray):
            a = np.ones(val.shape) * A
            b = np.ones(val.shape) * B
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
    
config = {}
test_env = MultiAgentInvManagementDiv(config=config)
print(test_env.obs)
for i in range(10):
    test_env.step
    i+=1
print(test_env.ship_to_list)

print("environment has been tested individually")
