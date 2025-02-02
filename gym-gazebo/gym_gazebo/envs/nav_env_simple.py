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
        self.reward_angle = self.config.reward_angle
        self.reward_goal_met = self.config.reward_goal_met
        self.stack_obs = self.config.stack_obs
        self.action_repeat = self.config.action_repeat

        self.robot_pos = np.zeros((3, ))
        self.robot_vel = np.zeros((3, ))
        self.goal_pos = np.zeros((2, ))
        self.last_dist_goal = 0
        self.last_angle_goal = 0
        self.goal_region = self.layout.goal_region
        self.collision_region = self.layout.collision_region
        self.cost_region = self.layout.cost_region

        self.lvel_lim = self.config.lvel_lim  # linear velocity limit
        self.rvel_lim = self.config.rvel_lim  # rotational velocity limit
        self.ego_obs_dim = 8
        self.obs_dim = self.ego_obs_dim if not self.stack_obs else self.ego_obs_dim * self.action_repeat
        self.act_dim = 2
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1, 1, (self.act_dim,), dtype=np.float32)
        self.key_to_slice = {}
        self.key_to_slice["goal"] = slice(0, 2)
        self.seed()

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

        if self.config.render:
            self.fig, self.ax = plt.subplots()
            self.ax.set_aspect('equal', 'box')
        
        if self.config.record:
            self.action_list = []
    
    def seed(self):
        ''' Set internal random state seeds '''
        self._seed = np.random.randint(2**32)

    def dist_xy(self):
        dx, dy = self.robot_pos[:2] - self.goal_pos[:2]
        return np.hypot(dx, dy)

    def dist_yaw(self):
        dx, dy = self.goal_pos[:2] - self.robot_pos[:2]
        angle_to_goal = np.arctan2(dy, dx)
        dist_yaw = abs(abs(angle_to_goal - self.robot_pos[2] + np.pi)%(2*np.pi)-np.pi)
        return dist_yaw
    
    def reward(self):
        dist_goal = self.dist_xy()
        angle_goal = self.dist_yaw()
        reward = (self.last_dist_goal - dist_goal) * self.reward_distance + (self.last_angle_goal - angle_goal) * self.reward_angle
        self.last_dist_goal = dist_goal
        self.last_angle_goal = angle_goal
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
    
    def process_obs(self):
        # x, y, yaw, xdot, ydot, yawdot, cos(yaw), sin(yaw), goal_x, goal_y, (obstacle state)
        # in world frame
        yaw = self.robot_pos[2]
        obs = np.concatenate([self.robot_pos, self.robot_vel, np.array([np.cos(yaw), np.sin(yaw)]).reshape(-1)])
        return obs
    
    def step(self, action):

        # Unpause simulation to make observation
        self._gazebo_unpause()

        if self.stack_obs:
            cat_obs = np.zeros(self.obs_dim)

        reward = 0
        cost = 0
        for k in range(self.action_repeat):
            vel_cmd = Twist()
            vel_cmd.linear.x = self.lvel_lim*action[0]
            vel_cmd.angular.z = self.rvel_lim*action[1]
            self.vel_pub.publish(vel_cmd)
            if self.config.record:
                self.action_list.append(vel_cmd)

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
                reward += self.reward_goal_met

            # get ego obs
            obs = self.process_obs()

            if self.stack_obs:
                cat_obs[k*self.ego_obs_dim :(k+1)*self.ego_obs_dim] = obs

            # unpause
            self._gazebo_unpause()

            self.publish_goal_pos()

            if goal_met or \
               self.t == self.config.max_episode_length or \
               collision or \
               out_bound:
                if k != self.action_repeat-1 and self.stack_obs:
                    for j in range(k+1, self.action_repeat):
                        cat_obs[j*self.ego_obs_dim :(j+1)*self.ego_obs_dim] = obs 
                break

        cost = 1 if cost > 0 else 0

        # pause simulation
        self._gazebo_pause()

        done = goal_met or collision or out_bound or self.t == self.config.max_episode_length
        info = {"cost":cost, "goal_met":goal_met}

        if self.stack_obs:
            return cat_obs, reward, done, info
        else:
            return obs, reward, done, info
    
    def sample_robot_pos(self):
        success = False
        while not success:
            x_sample = self.rs.uniform(-self.region_bound, self.region_bound)
            y_sample = self.rs.uniform(-self.region_bound, self.region_bound)
            for i in range(self.cyls_num):
                x, y = self.placements[self.cyls[i]]
                dist = np.hypot(x-x_sample, y-y_sample)
                if dist < self.collision_region:
                    if i == self.cyls_num - 1:
                        i = 0
                    break
            if i == self.cyls_num - 1:
                success = True

        yaw_sample = self.rs.uniform(-np.pi, np.pi, 1)
        return np.array([x_sample, y_sample, yaw_sample])

    def reset_robot_pos(self, random=False):
        if random:
            self.robot_pos = self.sample_robot_pos()
        else:
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
        self.cyls_pos = np.array(self.layout.cyls_pos)  # (clys num * 2)
        self.cyls = []
        for i in range(self.cyls_num):
            self.cyls.append("cyl" + str(i))

        self.placements = {}
        for i, cyl in enumerate(self.cyls):
            self.placements[cyl] = self.layout.cyls_pos[i]

        self.placements["goal"] = self.layout.goal_pos
        self.goal_pos = np.array(self.layout.goal_pos)
        self.last_dist_goal = self.dist_xy()
        self.last_angle_goal = self.dist_yaw()
        
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
    
    def reset(self, random=False):
        # Increment seed
        self._seed += 1
        self.rs = np.random.RandomState(self._seed)

        # Reset internal timer
        self.t = 0

        # Unpause simulation to make observation
        self._gazebo_unpause()

        # sample goal_pos and obstacles' pos
        self.reset_robot_pos(random)
        self.reset_layout()

        obs = self.process_obs()

        if self.stack_obs:
            for k in range(self.action_repeat):
                cat_obs = obs if k == 0 else np.concatenate((cat_obs, obs))

        # pause simulation
        self._gazebo_pause()

        if self.stack_obs:
            return cat_obs
        else:
            return obs
    
    def publish_goal_pos(self):
        goal = PoseStamped()
        goal.header.frame_id = "odom"

        goal.pose.position.x = self.goal_pos[0]
        goal.pose.position.y = self.goal_pos[1]

        self.goal_pub.publish(goal)
    
    def render(self, reward):
        if not self.config.render:
            pass

        self.ax.clear()

        # plot obstacle
        for i in range(self.cyls_num):
            self.ax.add_patch(plt.Circle(self.cyls_pos[i, :], self.layout.collision_region, color='b', alpha=0.8))
        
        # plot goal
        self.ax.scatter(self.goal_pos[0], self.goal_pos[1], color='r', marker='x')
        
        # plot robot
        self.ax.scatter(self.robot_pos[0], self.robot_pos[1], color='g', marker='o')
        yaw = self.robot_pos[2]
        if yaw >=0 and yaw < 0.5*np.pi:
            x_sgn = 1
            y_sgn = 1
        elif yaw >= 0.5*np.pi and yaw < np.pi:
            x_sgn = -1
            y_sgn = 1
        elif yaw < 0 and yaw >= -0.5*np.pi:
            x_sgn = 1
            y_sgn = -1
        else:
            x_sgn = -1
            y_sgn = -1
        x_sgn *= 0.5/np.sqrt(1+np.tan(yaw)**2)
        y_sgn *= 0.5*abs(np.tan(yaw))/np.sqrt(1+np.tan(yaw)**2)
        self.ax.arrow(self.robot_pos[0], self.robot_pos[1], x_sgn, y_sgn, head_width=0.05, head_length=0.08, color='g')
        # self.ax.arrow(self.robot_pos[0], self.robot_pos[1], self.goal_pos[0] - self.robot_pos[0], self.goal_pos[1] - self.robot_pos[1], head_width=0.05, head_length=0.08, color='r')

        self.ax.set_xlim(-self.layout.region_bound, self.layout.region_bound)
        self.ax.set_ylim(-self.layout.region_bound, self.layout.region_bound)
        self.ax.set_xticks(np.linspace(-self.layout.region_bound, self.layout.region_bound, 6))
        self.ax.set_yticks(np.linspace(-self.layout.region_bound, self.layout.region_bound, 6))
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_title('reward: {0}, dist to goal: {1} m'.format(np.round(reward, 4), np.round(self.dist_xy(), 4)))
        # self.ax.set_title('reward: {0}, dist: {1}, yaw: {2}, angle: {3}'.format(np.round(reward, 4), 
        # np.round(self.dist_xy(), 4), 
        # np.round(self.robot_pos[2], 4),
        # np.round(self.dist_yaw(), 4)))
        
        plt.grid()
        plt.get_current_fig_manager().window.setGeometry(100, 100, 640, 640)
        plt.show(block=False)
        plt.pause(0.0001)


    def cost_fn(self, robot_state):
        batch_size = robot_state.shape[0]
        dist = np.repeat(np.inf, batch_size)
        if not self.stack_obs:
            robot_x, robot_y = robot_state[:, 0], robot_state[:,1]
            for cyl in self.layout.cyls_pos:
                dist = np.minimum(dist, np.hypot(np.repeat(cyl[0], batch_size)-robot_x, np.repeat(cyl[1], batch_size)-robot_y))
        else:
            for k in range(self.action_repeat):
                robot_x, robot_y = robot_state[:, k*self.ego_obs_dim], robot_state[:, k*self.ego_obs_dim+1]
                for cyl in self.layout.cyls_pos:
                    dist = np.minimum(dist, np.hypot(np.repeat(cyl[0], batch_size)-robot_x, np.repeat(cyl[1], batch_size)-robot_y))
        return np.where(dist <= np.repeat(self.layout.cost_region, batch_size), 1.0, 0.0)

    def reward_fn(self, robot_state, robot_state_next):
        batch_size = robot_state.shape[0]
        goal_x = np.repeat(self.goal_pos[0], batch_size)
        goal_y = np.repeat(self.goal_pos[1], batch_size)

        if not self.stack_obs:
            robot_x, robot_y, robot_yaw = robot_state[:, 0], robot_state[:, 1], robot_state[:, 2]
            robot_x_next, robot_y_next, robot_yaw_next = robot_state_next[:, 0], robot_state_next[:, 1], robot_state_next[:, 2]

            dx, dy = goal_x-robot_x, goal_y-robot_y
            dist = np.hypot(dx, dy)
            angle = abs(np.arctan2(dy, dx) - robot_yaw)

            next_dx, next_dy = goal_x-robot_x_next, goal_y-robot_y_next
            next_dist = np.hypot(next_dx, next_dy)
            next_angle = abs(np.arctan2(next_dy, next_dx) - robot_yaw_next)

            reward = (dist - next_dist) * self.reward_distance + (angle - next_angle) * self.reward_angle
            reward += np.where(next_dist<=self.goal_region, self.reward_goal_met, 0)
            return reward
            
        else:
            reward = 0.0
            for k in range(self.action_repeat):
                robot_x, robot_y, robot_yaw = robot_state[:, k*self.ego_obs_dim], robot_state[:, k*self.ego_obs_dim+1], robot_state[:, k*self.ego_obs_dim+2]
                robot_x_next, robot_y_next, robot_yaw_next = robot_state_next[:, k*self.ego_obs_dim], robot_state_next[:, k*self.ego_obs_dim+1], robot_state_next[:, k*self.ego_obs_dim+2]

                dx, dy = goal_x-robot_x, goal_y-robot_y
                dist = np.hypot(dx, dy)
                angle = abs(np.arctan2(dy, dx) - robot_yaw)

                next_dx, next_dy = goal_x-robot_x_next, goal_y-robot_y_next
                next_dist = np.hypot(next_dx, next_dy)
                next_angle = abs(np.arctan2(next_dy, next_dx) - robot_yaw_next)

                reward += (dist - next_dist) * self.reward_distance + (angle - next_angle) * self.reward_angle
                reward += np.where(next_dist<=self.goal_region, self.reward_goal_met, 0)
            return reward
    
    @property
    def observation_size(self):
        return self.obs_dim

    @property
    def action_size(self):
        return self.act_dim

    @property
    def action_range(self):
        return float(self.action_space.low[0]), float(self.action_space.high[0])