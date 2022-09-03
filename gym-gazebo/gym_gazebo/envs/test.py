# import os
# import sys
# import subprocess

# launchfile = "limo_ackerman.launch"
# command = "roslaunch limo_gazebo_sim " + launchfile
# _roslaunch = subprocess.Popen(command.split())


import time
import gym
import gym_gazebo
import numpy as np

env = gym.make("GazeboCarNav-v0")
env.seed(0)
env.reset()

for i in range(100):
    # x = np.random.uniform(-0.3, 0.3)
    # z = np.random.uniform(-0.3, 0.3)
    # x = 0.1
    # # z = -0.3
    # z = -0.1
    # action = [x, z]
    # print('action = ', action)
    # env.step(action)
    obs = env.reset()
    print(obs.shape)
    time.sleep(100)
# env.step([0.0, 0.0])
# env.close()
