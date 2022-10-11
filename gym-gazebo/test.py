import argparse
import numpy as np
import os
import torch
import time
from tqdm import tqdm

import gym
import gym_gazebo

from gym_gazebo import SafeMPC, RegressionModelEnsemble, CostModel
import utils

import wandb

def run(config, args):
    # setup environment
    env_name = "env_{}".format(args.env_id)
    env_config = config['env_config']
    if args.render:
        env_config['render'] = True
    if args.record:
        env_config['record'] = True
    env = gym.make("GazeboCarNav-v0", config=config['env_config'], layout=env_config[env_name])
    env.reset()

    # MPC and dynamic model config
    mpc_config = config['mpc_config']
    mpc_config["optimizer"] = args.optimizer.upper()
    dynamic_config = config['dynamic_config']
    assert args.load is not None

    dynamic_config["load"] = True
    dynamic_config["load_folder"] = args.load

    config["arguments"] = vars(args)

    state_dim, action_dim = env.observation_size, env.action_size
    if args.ensemble>0:
        dynamic_config["n_ensembles"] = args.ensemble
    dynamic_model = RegressionModelEnsemble(state_dim+action_dim, state_dim, config=dynamic_config)
    mpc_controller = SafeMPC(env, mpc_config, cost_fn=env.cost_fn, reward_fn=env.reward_fn, n_ensembles=dynamic_config["n_ensembles"])

    # Prepare random collected dataset
    print("start running ...")
    start_time = time.time()
    ep_len = 0
    obs, ep_ret, ep_cost, done = env.reset(), 0, 0, False
    mpc_controller.reset()
    if args.render:
            env.render(0.0)
    while not done:    
        action = np.squeeze(np.array([mpc_controller.act(model=dynamic_model, state=obs)]))
        obs_next, reward, done, info = env.step(action)
        if args.render:
            env.render(reward)
        ep_ret += reward
        ep_cost += info["cost"]
        obs = obs_next
        ep_len += 1
        print("ep {0} : cost={1}, ret={2}".format(ep_len, info["cost"], reward))
    if args.record:
        np.save(os.path.join(args.load, 'action_list.npy'), env.action_list)
    env.close()

    print("ep_cost: {0}, ep_ret: {1}, ep_len: {2}".format(ep_cost, ep_ret, ep_len))

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', type=int, default=0, help="environment index")
    parser.add_argument('--render', action='store_true', help="render the environment")
    parser.add_argument('--record', action='store_true', help="record the vel_cmd")
    parser.add_argument('--load',type=str, default=None, help="load the trained dynamic model, data buffer, and cost model from a specified directory")
    parser.add_argument('--ensemble',type=int, default=0, help="number of model ensembles, if this argument is greater than 0, then it will replace the default ensembles number in config.yml") # number of ensembles
    parser.add_argument('--optimizer','-o',type=str, default="rce", help=" determine the optimizer, selected from `rce`, `cem`, or `random` ") # random, cem or CCE
    parser.add_argument('--config', '-c', type=str, default='./config.yml', help="specify the path to the configuation file of the models")

    args = parser.parse_args()
    print("Reading configurations ...")
    config = utils.load_config(args.config)

    run(config, args)
    
