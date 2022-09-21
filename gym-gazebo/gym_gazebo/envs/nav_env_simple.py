import math
from pyexpat import model
import time

import rospy
import numpy as np
import matplotlib.pyplot as plt

# import gym
from gym import spaces
from gym.utils import seeding

from gym_gazebo.envs.gazebo_env import GazeboEnv
from gazebo_msgs.msg import ModelStates, ModelState
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Quaternion, PoseStamped
import tf
from scipy.spatial.transform import Rotation as R

from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from sensor_msgs.msg import LaserScan

# import CMap2D
# from CMap2D import CMap2D, gridshow

DEFAULT_LAYOUT = dict(
    region_bound=2.5, 
    collision_region=0.45,
    cost_region=0.6,
    goal_region=0.3,
    robot_pos=[-0.5, -1.5, np.pi], 
    goal_pos=[-1.5, 1.5], 
    cyls_pos=[[-1.5, -0.5], [-0.5, -0.5], [0.5, -0.5]]
)

class Dict2Obj(object):
    # Turns a dictionary into a class
    def __init__(self, dictionary):
        for key in dictionary:
            setattr(self, key, dictionary[key])

    def __repr__(self):
        return "%s" % self.__dict__

class GazeboCarNavEnvSimple(GazeboEnv):

    def __init__(self, config, layout = Dict2Obj(DEFAULT_LAYOUT)):
        super().__init__()

        self.layout = layout
        self.region_bound = self.layout.region_bound
        self.config = config
        self.reward_distance = self.config.reward_distance

        self.robot_pos = np.zeros((3, ))
        self.robot_vel = np.zeros((3, ))
        self.goal_pos = np.zeros((2, ))
        self.last_dist_goal = 0
        self.goal_region = self.layout.goal_region
        self.collision_region = self.layout.collision_region
        self.cost_region = self.layout.cost_region

        self.lvel_lim = self.config.lvel_lim  # linear velocity limit
        self.rvel_lim = self.config.rvel_lim  # rotational velocity limit
        self.ego_obs_dim = 6  # 7
        self.obs_dim = self.ego_obs_dim + 1
        self.act_dim = 2
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1, 1, (self.act_dim,), dtype=np.float32)
        self.key_to_slice = {}
        self.key_to_slice["goal"] = slice(0, 2)

        # ROS
        self.cmd_topic = "/cmd_vel"
        self.laser_topic = "/limo/scan"
        self.goal_topic = "/goal"
        self.robot_name = "limo/"
        self.frame = "/base_link"
        self.timeout = 100

        rospy.init_node("GazeboCarNavEnv", anonymous=True)

        self.vel_pub = rospy.Publisher(self.cmd_topic, Twist, queue_size=1)
        self.goal_pub = rospy.Publisher(self.goal_topic, PoseStamped, queue_size=1)

        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
    
    def dist_xy(self):
        dx, dy = self.robot_pos[:2] - self.goal_pos[:2]
        return np.hypot(dx, dy)
    
    def reward(self):
        dist_goal = self.dist_xy()
        reward = (self.last_dist_goal - dist_goal) * self.reward_distance
        self.last_dist_goal = dist_goal
        return reward
    
    def cost(self):
        for cyl in self.cyls:
            x, y = self.placements[cyl]
            dist = np.hypot(x-self.robot_pos[0], y-self.robot_pos[1])
            if dist <= self.cost_region:
                return 1.0
        return 0.0
    
    def _gazebo_pause(self):
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except rospy.ServiceException as e:
            print("/gazebo/pause_physics service call failed: {}".format(e))

    def _gazebo_unpause(self):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException as e:
            print("/gazebo/unpause_physics service call failed: {}".format(e))
        
    def get_model_states(self):
        model_states = None
        success = False
        while model_states is None or not success:
            try:
                model_states = rospy.wait_for_message("/gazebo/model_states", ModelStates, timeout=self.timeout)
            finally:
                success = True
        return model_states
    
    def goal_met(self):
        dist_goal = self.dist_xy()
        if dist_goal <= self.goal_region:
            return True
        return False
    
    def collision(self):
        for cyl in self.cyls:
            x, y = self.placements[cyl]
            dist = np.hypot(x-self.robot_pos[0], y-self.robot_pos[1])
            if dist <= self.collision_region:
                return True
        return False
    
    def process_ego_obs(self):
        # x, y, yaw, xdot, ydot, yawdot 
        # in world frame
        return np.concatenate([self.robot_pos, self.robot_vel])
    
    def step(self, action):

        # Unpause simulation to make observation
        self._gazebo_unpause()

        reward = 0
        cost = 0
        for k in range(self.config.action_repeat):
            vel_cmd = Twist()
            vel_cmd.linear.x = self.lvel_lim*action[0]
            vel_cmd.angular.z = self.rvel_lim*action[1]
            self.vel_pub.publish(vel_cmd)

            # get robot and obstacles states
            model_states = self.get_model_states()

            # pause simulation
            self._gazebo_pause()

            # set robot pos
            self.robot_pos[0] = model_states.pose[-1].position.x
            self.robot_pos[1] = model_states.pose[-1].position.y
            q = model_states.pose[-1].orientation
            _, _, yaw = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
            self.robot_pos[2] = yaw
            # set robot vel
            self.robot_vel[0] = model_states.twist[-1].linear.x
            self.robot_vel[1] = model_states.twist[-1].linear.y
            self.robot_vel[2] = model_states.twist[-1].angular.z

            # calculate reward
            reward += self.reward()

            # calaulate cost
            cost += self.cost()

            # check collision
            collision = self.collision()

            # check out of bound
            out_bound = np.hypot(self.robot_pos[0], self.robot_pos[1]) > self.region_bound

            # Increment internal timer
            self.t += 1

            # goal reach check
            goal_met = self.goal_met()
            if goal_met:
                reward += 2.0

            # get ego obs
            ego_obs = self.process_ego_obs()

            # unpause
            self._gazebo_unpause()

            self.publish_goal_pos()

            if goal_met or \
               self.t == self.config.max_episode_length or \
               collision or \
               out_bound:
                break

        cost = 1 if cost > 0 else 0

        # pause simulation
        self._gazebo_pause()

        done = collision or out_bound or self.t == self.config.max_episode_length
        info = {"cost":cost, "goal_met":goal_met}
        obs = np.concatenate([ego_obs, np.array([reward])])
        # print("ego obs: {0}\nreward: {1}\ncost: {2}\ngoal_met: {3}\n\n".format(ego_obs, reward, cost, goal_met))

        return obs, reward, done, info
    
    def reset_robot_pos(self):
        self.robot_pos = np.array(self.layout.robot_pos)
        self.robot_vel = np.zeros((3, ))
        state = ModelState()
        state.model_name = self.robot_name
        state.pose.position.x = self.robot_pos[0]
        state.pose.position.y = self.robot_pos[1]
        quaternion = tf.transformations.quaternion_from_euler(0, 0, self.robot_pos[2])
        state.pose.orientation.x = quaternion[0]
        state.pose.orientation.y = quaternion[1]
        state.pose.orientation.z = quaternion[2]
        state.pose.orientation.w = quaternion[3]
        state.twist.linear.x = 0.0
        state.twist.linear.y = 0.0
        state.twist.angular.z = 0.0

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            set_state(state)
        except rospy.ServiceException as e:
            print("/gazebo/set_model_state call failed: {}".format(e))
        return
    
    def reset_layout(self):
        self.cyls_num = len(self.layout.cyls_pos)
        self.cyls = []
        for i in range(self.cyls_num):
            self.cyls.append("cyl" + str(i))

        self.placements = {}
        for i, cyl in enumerate(self.cyls):
            self.placements[cyl] = self.layout.cyls_pos[i]

        self.placements["goal"] = self.layout.goal_pos
        self.goal_pos = np.array(self.layout.goal_pos)
        self.last_dist_goal = self.dist_xy()
        
        # set cyls states by SetModelStates
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            for model_name, pos in self.placements.items():
                state = ModelState()
                state.model_name = model_name
                state.pose.position.x = pos[0]
                state.pose.position.y = pos[1]
                set_state(state)

        except rospy.ServiceException as e:
            print("/gazebo/set_model_state call failed: {}".format(e))
    
    def reset(self):
        # Reset internal timer
        self.t = 0

        # Unpause simulation to make observation
        self._gazebo_unpause()

        # sample goal_pos and obstacles' pos
        self.reset_robot_pos()
        self.reset_layout()

        ego_obs = self.process_ego_obs()
        obs = np.concatenate([ego_obs, np.array([0.0])])

        # pause simulation
        self._gazebo_pause()

        return obs
    
    def publish_goal_pos(self):
        goal = PoseStamped()
        goal.header.frame_id = "odom"

        goal.pose.position.x = self.goal_pos[0]
        goal.pose.position.y = self.goal_pos[1]

        self.goal_pub.publish(goal)
    
    def render(self):
        self.fig = plt.figure()
        # plot obs
    
    @staticmethod
    def cost_fn(layout, robot_x, robot_y):
        batch_size = robot_x.shape[0]
        dist = np.repeat(np.inf, batch_size)
        for cyl in layout.cyls_pos:
            dist = np.minimum(dist, np.hypot(np.repeat(cyl[0], batch_size)-robot_x, np.repeat(cyl[1], batch_size)-robot_y))
        return np.where(dist <= np.repeat(layout.cost_region, batch_size), 1.0, 0.0)
    
    @property
    def observation_size(self):
        return self.obs_dim

    @property
    def action_size(self):
        return self.act_dim

    @property
    def action_range(self):
        return float(self.action_space.low[0]), float(self.action_space.high[0])