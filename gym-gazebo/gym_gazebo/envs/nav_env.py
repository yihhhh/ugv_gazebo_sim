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
# import tf
# from tf.transformations import euler_from_quaternion
from scipy.spatial.transform import Rotation as R

from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from sensor_msgs.msg import LaserScan

# import CMap2D
from CMap2D import CMap2D, gridshow


DEFAULT_CONFIG = dict(
    action_repeat=2,
    max_episode_length=100,
    lidar_dim=16,  # total: 450
    use_dist_reward=False,
    stack_obs=False,
    reward_distance=1.0,  # reward scale
    placements_margin=1.2,  # min distance of obstacles between each other
    goal_region=0.3,
    collision_region=0.45,
    cost_region=0.6,
    lidar_max_range=5.0,
    lvel_lim=0.3,  # linear velocity limit
    rvel_lim=1.0,  # rotational velocity limit

    # grid map
    use_grid_map=True,
    xmax=4,
    ymax=4,
    resolution=0.1,
    render=True
)


class Dict2Obj(object):
    # Turns a dictionary into a class
    def __init__(self, dictionary):
        for key in dictionary:
            setattr(self, key, dictionary[key])

    def __repr__(self):
        return "%s" % self.__dict__


class GazeboCarNavEnv(GazeboEnv):
    def __init__(self, level=1, seed=0, config=DEFAULT_CONFIG):
        super().__init__()

        if level == 1:
            self.x_range = [-2.5, 2.5]
            self.y_range = [-2.5, 2.5]
            self.region_bound = 4.0
            self.cyls_num = 7
            self.cyls = []
            for i in range(self.cyls_num):
                self.cyls.append("cyl" + str(i))
        elif level == 2:
            self.x_range = [-3.5, 3.5]
            self.y_range = [-3.5, 3.5]
            self.cyls_num = 14
            self.region_bound = 6.0
            self.cyls = []
            for i in range(self.cyls_num):
                self.cyls.append("cyl" + str(i))

        self.config = Dict2Obj(config)
        self.reward_distance = self.config.reward_distance
        self.placements_margin = self.config.placements_margin
        self.placements = {}
        self.seed(seed)

        self.robot_pos = np.zeros((3, ))
        self.robot_vel = np.zeros((3, ))
        self.goal_pos = np.zeros((3, ))
        self.last_dist_goal = 0
        self.goal_region = self.config.goal_region
        self.collision_region = self.config.collision_region
        self.cost_region = self.config.cost_region
        self.lidar_max_range = self.config.lidar_max_range

        self.lvel_lim = self.config.lvel_lim  # linear velocity limit
        self.rvel_lim = self.config.rvel_lim  # rotational velocity limit
        self.ego_obs_dim = 5  # 7
        self.obs_dim = self.ego_obs_dim + self.config.lidar_dim
        self.act_dim = 2
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1, 1, (self.act_dim,), dtype=np.float32)
        self.key_to_slice = {}
        self.key_to_slice["goal"] = slice(0, 2)

        self.first_time = True

        if self.config.use_grid_map:
            self.gridmap = CMap2D()
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
            self.xmax = self.config.xmax
            self.ymax = self.config.ymax
            self.limits = np.array([[-self.xmax, self.xmax],
                                    [-self.ymax, self.ymax]],
                                    dtype=np.float32)

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

    def dist_yaw(self):
        dyaw = self.robot_pos[2] - self.goal_pos[2]
        return np.min([np.abs(dyaw), 2*np.pi-np.abs(dyaw)])

    def seed(self, seed=None):
        ''' Set internal random state seeds '''
        self._seed = np.random.randint(2**32) if seed is None else seed

    def reward(self):
        dist_goal = self.dist_xy()  # + self.dist_yaw()
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

    def _gazebo_reset(self):
        # # Resets the state of the environment and returns an initial observation.
        # rospy.wait_for_service('/gazebo/reset_simulation')
        # try:
        #     # reset_proxy.call()
        #     self.reset_proxy()
        #     self.unpause()
        # except rospy.ServiceException as e:
        #     print("/gazebo/reset_simulation service call failed: {}".format(e))
        pass
        # rospy.exceptions.

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
        dist_goal = self.dist_xy()  # + self.dist_yaw()
        # print(dist_goal)
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

    def step(self, action):

        # Unpause simulation to make observation
        self._gazebo_unpause()

        reward = 0
        cost = 0
        # print(self.t, self.config.max_episode_length)
        if self.config.use_grid_map and self.config.render:
            plt.ion()
        for k in range(self.config.action_repeat):
            vel_cmd = Twist()
            vel_cmd.linear.x = self.lvel_lim*action[0]
            vel_cmd.angular.z = self.rvel_lim*action[1]
            self.vel_pub.publish(vel_cmd)

            # get lidar obs
            lidar_obs = self.process_lidar_obs()

            if self.config.use_grid_map and self.config.render:
                # print(self.gridmap.occupancy().shape, self.gridmap.occupancy())
                self.ax.imshow(1-self.gridmap.occupancy(), cmap="gray")
                plt.pause(0.1)
                # plt.show()

            # gridshow(self.gridmap.occupancy(), extent=self.gridmap.get_extent_xy())
            # plt.show(0.1)

            # get robot and obstacles states
            model_states = self.get_model_states()

            # pause simulation
            self._gazebo_pause()

            # set robot pos
            self.robot_pos[0] = model_states.pose[-1].position.x
            self.robot_pos[1] = model_states.pose[-1].position.y
            q = model_states.pose[-1].orientation
            yaw = R.from_quat([q.x, q.y, q.z, q.w]).as_euler('z')
            self.robot_pos[2] = yaw
            # set robot vel
            self.robot_vel[0] = model_states.twist[-1].linear.x
            self.robot_vel[1] = model_states.twist[-1].linear.y
            self.robot_vel[2] = model_states.twist[-1].angular.z

            # calculate reward
            reward += self.reward()
            # print(f"reward = {reward}")

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
                self.sample_goal_pos()

            # get ego obs
            ego_obs = self.process_ego_obs()
            cat_obs = np.concatenate([ego_obs, lidar_obs])
            if self.config.use_grid_map:
                cat_obs = [ego_obs, self.gridmap.occupancy()]

            # unpause
            self._gazebo_unpause()

            self.publish_goal_pos()

            if goal_met or \
               self.t == self.config.max_episode_length or \
               collision or \
               out_bound:
                break

        cost = 1 if cost > 0 else 0
        if self.config.use_grid_map and self.config.render:
            plt.ioff()
        # pause simulation
        self._gazebo_pause()

        done = collision or out_bound or self.t == self.config.max_episode_length
        info = {"cost":cost, "goal_met":goal_met}

        # if goal_met:
        #     print("goal_met")
        # if collision:
        #     print("collision")
        # if out_bound:
        #     print("out_bound")
        # if self.t == self.config.max_episode_length:
        #     print("max_episode_length")
        # # print('=='*20)

        return cat_obs, reward, done, info

    def process_lidar_obs(self):

        lidar_data = None
        success = False
        while lidar_data is None or not success:
            try:
                lidar_data = rospy.wait_for_message(self.laser_topic, LaserScan, timeout=self.timeout)
            finally:
                success = True

        if self.config.use_grid_map:
            # lidar_data_tmp = lidar_data.copy()
            self.lidar_min_range = lidar_data.range_min
            # print(type(lidar_data.ranges))
            lidar_ranges = np.array(lidar_data.ranges, dtype=np.float32)
            lidar_ranges[lidar_ranges >= lidar_data.range_max] = lidar_data.range_max
            lidar_ranges[lidar_ranges <= lidar_data.range_min] = lidar_data.range_min
            lidar_data.ranges = lidar_ranges
            self.gridmap.from_scan(lidar_data, limits=self.limits, resolution=self.config.resolution, legacy=False)

        self.lidar_min_range = lidar_data.range_min
        # self.lidar_max_range = lidar_data.range_max
        lidar_ranges = np.array(lidar_data.ranges)
        idx = np.linspace(0, lidar_ranges.shape[0], self.config.lidar_dim, 
                          endpoint=False, dtype=int)
        lidar_ranges = lidar_ranges[idx]
        lidar_ranges[lidar_ranges >= self.lidar_max_range] = self.lidar_max_range
        lidar_ranges[lidar_ranges <= self.lidar_min_range] = self.lidar_min_range
        lidar_ranges /= self.lidar_max_range
        return 1.0 - lidar_ranges

    def sample_layout(self):
        # if self.first_time:
        self.placements = {}
        # sample goal_pos and obstacles' pos
        sample_size = self.cyls_num + 1
        success = False
        while not success:
            valid = True
            x_samples = self.rs.uniform(self.x_range[0], self.x_range[1], sample_size)
            y_samples = self.rs.uniform(self.y_range[0], self.y_range[1], sample_size)
            dists = np.hypot(x_samples, y_samples)
            if np.sum(dists < self.placements_margin) > 0:
                continue
            for i in range(sample_size):
                dists = np.hypot(x_samples-x_samples[i], y_samples-y_samples[i])
                if np.sum(dists < self.placements_margin) > 1:
                    valid = False
                    break
            success = valid

        for i, cyl in enumerate(self.cyls):
            self.placements[cyl] = [x_samples[i], y_samples[i]]
        yaw_sample = self.rs.uniform(-np.pi, np.pi, 1)
        self.placements["goal"] = [x_samples[-1], y_samples[-1], yaw_sample[0]]
        # self.first_time = False
        
        self.goal_pos = np.array(self.placements["goal"])
        self.last_dist_goal = self.dist_xy()  # + self.dist_yaw()
        
        # set cyls states by SetModelStates
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            # for model_name in self.cyls:
            for model_name, pos in self.placements.items():
                # pos = self.placements[model_name]
                # print(model_name, pos)
                state = ModelState()
                state.model_name = model_name
                state.pose.position.x = pos[0]
                state.pose.position.y = pos[1]
                # state.reference_frame = self.frame
                if len(pos) == 3:
                    quaternion = tf.transformations.quaternion_from_euler(0, 0, pos[2])
                    state.pose.orientation.x = quaternion[0]
                    state.pose.orientation.y = quaternion[1]
                    state.pose.orientation.z = quaternion[2]
                    state.pose.orientation.w = quaternion[3]
                set_state(state)

        except rospy.ServiceException as e:
            print("/gazebo/set_model_state call failed: {}".format(e))

    def process_ego_obs(self):
        # dx, dy, cos(dyaw), sin(dyaw), xdot, ydot, yawdot 
        # in world frame
        dxy = self.robot_pos[:2] - self.goal_pos[:2]
        dist_yaw = self.dist_yaw()
        return np.concatenate([dxy, self.robot_vel])
        # return np.concatenate([dxy, [np.cos(dist_yaw), np.sin(dist_yaw)], self.robot_vel])

    def reset_robot_pos(self):
        self.robot_pos = np.zeros((3, ))
        self.robot_vel = np.zeros((3, ))
        state = ModelState()
        state.model_name = self.robot_name
        state.pose.position.x = self.robot_pos[0]
        state.pose.position.y = self.robot_pos[1]
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

    def sample_goal_pos(self):

        success = False
        while not success:
            x_sample = self.rs.uniform(self.x_range[0], self.x_range[1])
            y_sample = self.rs.uniform(self.y_range[0], self.y_range[1])
            dist = np.hypot(x_sample, y_sample)
            if dist < self.placements_margin:
                continue
            dist = np.hypot(x_sample-self.goal_pos[0], y_sample-self.goal_pos[1])
            if dist < self.placements_margin:
                continue
            for i in range(self.cyls_num):
                x, y = self.placements[self.cyls[i]]
                dist = np.hypot(x-x_sample, y-y_sample)
                if dist < self.placements_margin:
                    if i == self.cyls_num - 1:
                        i = 0
                    break
            if i == self.cyls_num - 1:
                success = True

        yaw_sample = self.rs.uniform(-np.pi, np.pi, 1)
        self.placements["goal"] = [x_sample, y_sample, yaw_sample[0]]
        self.goal_pos = np.array(self.placements["goal"])
        self.last_dist_goal = self.dist_xy()  # + self.dist_yaw()
        # print(f"goal pos = {self.goal_pos}")

    def publish_goal_pos(self):
        goal = PoseStamped()
        # goal.header.seq = 1
        # goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = "odom"

        goal.pose.position.x = self.goal_pos[0]
        goal.pose.position.y = self.goal_pos[1]
        quaternion = tf.transformations.quaternion_from_euler(0, 0, self.goal_pos[2])
        goal.pose.orientation.x = quaternion[0]
        goal.pose.orientation.y = quaternion[1]
        goal.pose.orientation.z = quaternion[2]
        goal.pose.orientation.w = quaternion[3]

        self.goal_pub.publish(goal)
        # state = ModelState()
        # state.model_name = "goal"
        # state.pose.position.x = self.goal_pos[0]
        # state.pose.position.y = self.goal_pos[1]
        # quaternion = tf.transformations.quaternion_from_euler(0, 0, self.goal_pos[2])
        # state.pose.orientation.x = quaternion[0]
        # state.pose.orientation.y = quaternion[1]
        # state.pose.orientation.z = quaternion[2]
        # state.pose.orientation.w = quaternion[3]
        # rospy.wait_for_service('/gazebo/set_model_state')
        # try:
        #     set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        #     set_state(state)
        # except rospy.ServiceException as e:
        #     print("/gazebo/set_model_state call failed: {}".format(e))

    def reset(self):
        self._seed += 1  # Increment seed
        self.rs = np.random.RandomState(self._seed)
        self.t = 0    # Reset internal timer

        # Unpause simulation to make observation
        self._gazebo_unpause()

        # sample goal_pos and obstacles' pos
        self.reset_robot_pos()
        self.sample_layout()

        # read and process lidar data
        lidar_obs = self.process_lidar_obs()
        ego_obs = self.process_ego_obs()

        obs = np.concatenate([ego_obs, lidar_obs])

        # pause simulation
        self._gazebo_pause()

        return obs

    @property
    def observation_size(self):
        return self.obs_dim

    @property
    def action_size(self):
        return self.act_dim

    @property
    def action_range(self):
        return float(self.action_space.low[0]), float(self.action_space.high[0])