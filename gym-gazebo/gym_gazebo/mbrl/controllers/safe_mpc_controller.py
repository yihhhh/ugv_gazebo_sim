'''
@Author: Zuxin Liu
@Email: zuxinl@andrew.cmu.edu
@Date:   2020-03-24 10:42:50
@LastEditTime: 2020-07-29 21:34:37
@Description:
'''

import numpy as np
from ..optimizers import RandomOptimizer, CEMOptimizer, RCEOptimizer

class SafeMPC(object):
    optimizers = {"CEM": CEMOptimizer, "RANDOM": RandomOptimizer, "RCE": RCEOptimizer}

    def __init__(self, env, mpc_config, layout, cost_fn = None, n_ensembles=0):
        # mpc_config = config["mpc_config"]
        self.type = mpc_config["optimizer"].upper()
        self.horizon = mpc_config["horizon"]
        self.gamma = mpc_config["gamma"]
        self.beta = 0.4
        self.n_ensembles = n_ensembles
        self.action_low = np.array(env.action_space.low) # array (dim,)
        self.action_high = np.array(env.action_space.high) # array (dim,)
        self.action_dim = env.action_space.shape[0]

        # self.popsize = conf["popsize"]
        # self.particle = conf["particle"]
        # self.init_mean = np.array([conf["init_mean"]] * self.horizon)
        # self.init_var = np.array([conf["init_var"]] * self.horizon)

        if len(self.action_low) == 1: # auto fill in other dims
            self.action_low = np.tile(self.action_low, [self.action_dim]) # (act dim)
            self.action_high = np.tile(self.action_high, [self.action_dim ])
       
        lb = np.tile(self.action_low, [self.horizon])
        ub = np.tile(self.action_high, [self.horizon])

        self.sol_dim = self.horizon*self.action_dim

        optimizer_config = mpc_config[self.type]
        self.popsize = optimizer_config["popsize"]
        
        self.optimizer = SafeMPC.optimizers[self.type](sol_dim=self.sol_dim,
            upper_bound=ub, lower_bound=lb, **optimizer_config)

        assert cost_fn is not None, " cost function is not defined! "
        self.cost_fn = cost_fn
        self.layout = layout

        self.optimizer.setup(self.rce_cost_function)

        self.reset()

    def reset(self):
        """Resets this controller (clears previous solution, calls all update functions).

        Returns: None
        """
        #print('set init mean to 0')
        # self.prev_sol = np.tile((self.action_low + self.action_high) / 2, [self.horizon])
        # self.init_var = np.tile(np.square(self.action_low - self.action_high) / 16, [self.horizon])
        pass

    def act(self, model, state):
        '''
        :param state: model, (numpy array) current state
        :return: (float) optimal action
        '''
        self.model = model
        self.state = state

        soln, var = self.optimizer.obtain_solution()

        action = soln[-self.action_dim:]
        return action

    def rce_cost_function(self, actions):
        """
        Calculate the cost given a sequence of actions
        Parameters:
        ----------
            @param numpy array - actions : size should be (popsize x sol_dim)

        Return:
        ----------
            @param numpy array - cost : length should be of batch_size
        """
        actions = actions.reshape((-1, self.horizon, self.action_dim)) # [pop size, horizon, action_dim]
        #actions = np.tile(actions, (self.particle, 1, 1)) # 
        cost_rewards = np.zeros(self.popsize)#*self.particle)
        cost_constraints = np.zeros(self.popsize)#*self.particle)
        state = np.repeat(self.state.reshape(1, -1), self.popsize, axis=0) # [pop size, state dim]
        for t in range(self.horizon):
            action = actions[:, t, :]  # numpy array (pop size x action dim)
            x = np.concatenate((state, action), axis=1)
            state_next = self.model.predict(x) #+ state

            cost_reward = state_next[:, -1]  # compute cost
            cost_reward = cost_reward.reshape(cost_rewards.shape)
            cost_rewards += cost_reward * self.gamma**t

            cost_const = self.cost_fn(self.layout, state_next[:, 0], state_next[:, 1])  # compute cost
            cost_const = cost_const.reshape(cost_constraints.shape)
            cost_constraints += cost_const * self.beta**t
            state = state_next

        return cost_rewards, cost_constraints