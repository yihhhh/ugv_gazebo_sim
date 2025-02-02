import argparse
import numpy as np
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
    env = gym.make("GazeboCarNav-v0", config=config['env_config'], layout=env_config[env_name])
    env.reset()

    # MPC and dynamic model config
    mpc_config = config['mpc_config']
    mpc_config["optimizer"] = args.optimizer.upper()
    dynamic_config = config['dynamic_config']
    if args.load is not None:
        dynamic_config["load"] = True
        dynamic_config["load_folder"] = args.load
    if args.save:
        save_dir = './checkpoints/{0}/{1}'.format(args.group_id, args.exp_id)
        dynamic_config["save"] = True
        dynamic_config["save_folder"] = save_dir

    config["arguments"] = vars(args)
    if not args.debug:
        utils.wandb_update(config)

    state_dim, action_dim = env.observation_size, env.action_size
    if args.ensemble>0:
        dynamic_config["n_ensembles"] = args.ensemble
    dynamic_model = RegressionModelEnsemble(state_dim+action_dim, state_dim, config=dynamic_config)
    mpc_controller = SafeMPC(env, mpc_config, cost_fn=env.cost_fn, reward_fn=env.reward_fn, n_ensembles=dynamic_config["n_ensembles"])

    # Prepare random collected dataset
    start_time = time.time()
    if not args.test:
        pretrain_episodes = 50 if args.load is None and not args.debug else 10
        pretrain_max_step = 50
        print("collecting random episodes...")
        data_num = 0
        for epi in tqdm(range(pretrain_episodes)):
            obs = env.reset(random=True)
            done = False
            i = 0
            while not done and i<pretrain_max_step:
                action = env.action_space.sample()
                obs_next, reward, done, info = env.step(action)
                if args.render:
                    env.render(reward)
                x, y = np.concatenate((obs, action)), obs_next
                dynamic_model.add_data_point(x, y)
                data_num += 1
                i += 1
                obs = obs_next
        print("Finish to collect %i data "%data_num)

    # training the model
    if args.load is None:
        dynamic_model.reset_model()
        print("resetting model")
        dynamic_model.fit(use_data_buf=True, normalize=True)

    # Main loop: collect experience in env and update/log each epoch
    total_len = 0 # total interactions
    total_epi = 0
    total_ep_ret = 0
    total_ep_cost = 0
    for epoch in tqdm(range(args.epoch)): # update models per epoch
        for test_episode in range(args.episode): # collect data for episodes length
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
                total_ep_ret += reward
                ep_cost += info["cost"]
                total_ep_cost += info["cost"]
                total_len += 1
                if not args.test:
                    x = np.concatenate((obs, action))
                    y = obs_next #- obs
                    dynamic_model.add_data_point(x, y)
                obs = obs_next 
            if not args.debug:
                wandb.log({"EpRet": ep_ret, "EpCost": ep_cost})
                wandb.log({"Epoch": epoch, 
                        "Episode": total_epi, 
                        "AvgEpRet": total_ep_ret/total_len, 
                        "AvgEpCost": total_ep_cost/total_len,
                        "TotalEnvInteracts": total_len,
                        "Time": time.time()-start_time})
            total_epi += 1
        # training the model
        if not args.test:
            dynamic_model.fit(use_data_buf=True, normalize=True)
    env.close()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--group_id','-g', default='trial', help="group id of wandb")
    parser.add_argument('--exp_id','-e', default='trial', help="experiment id of wandb")
    parser.add_argument('--debug', action='store_true', help="debug mode, turn off wandb")
    parser.add_argument('--env_id', type=int, default=0, help="environment index")
    parser.add_argument('--epoch', type=int, default=60, help="maximum epochs to train")
    parser.add_argument('--episode', type=int, default=10, help="determines how many episodes data to collect for each epoch")
    parser.add_argument('--render','-r', action='store_true', help="render the environment")
    parser.add_argument('--test', '-t', action='store_true', help="test the performance of pretrained models without training")

    parser.add_argument('--save', action='store_true', help="save the trained dynamic model, data buffer, and cost model")
    parser.add_argument('--load',type=str, default=None, help="load the trained dynamic model, data buffer, and cost model from a specified directory")
    parser.add_argument('--ensemble',type=int, default=0, help="number of model ensembles, if this argument is greater than 0, then it will replace the default ensembles number in config.yml") # number of ensembles
    parser.add_argument('--optimizer','-o',type=str, default="rce", help=" determine the optimizer, selected from `rce`, `cem`, or `random` ") # random, cem or CCE
    parser.add_argument('--config', '-c', type=str, default='./config.yml', help="specify the path to the configuation file of the models")

    args = parser.parse_args()
    print("Reading configurations ...")
    if not args.debug:
        utils.wandb_init("limo_project", args.group_id, args.exp_id)
    config = utils.load_config(args.config)

    run(config, args)
    
