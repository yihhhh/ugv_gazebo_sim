import math
import rospy
import numpy as np

from gym import spaces
from gym.utils import seeding

from gym_gazebo.envs.gazebo_env import GazeboEnv
from gazebo_msgs.msg import ModelStates, ModelState
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Quaternion, PoseStamped, PoseWithCovarianceStamped
import tf

from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from sensor_msgs.msg import LaserScan


env_config = {"pkg_name": "limo_gazebo_sim",
              "launch": "limo_ackerman.launch",
              "cmd_topic": "/cmd_vel",
              "laser_topic": "/limo/scan"}


class GazeboCarParkingEnv(GazeboEnv):
    def __init__(self):

        super().__init__()
        self.cmd_topic = "/cmd_vel"
        self.laser_topic = "/limo/scan"
        self.goal_pos_topic = "/limo/goal"
        self.map_frame_id = "/map"
        self.robot_pos_topic = "/initialpose"

        rospy.init_node("gym", anonymous=True)

        self.vel_pub = rospy.Publisher(self.cmd_topic, Twist, queue_size=1)
        self.goal_pos_pub = rospy.Publisher(self.goal_pos_topic, PoseStamped, queue_size=1)
        self.robot_pos_pub = rospy.Publisher(self.robot_pos_topic, PoseWithCovarianceStamped, queue_size=1)

        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _gazebo_pause(self):
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except rospy.ServiceException as e:
            print("/gazebo/pause_physics service call failed: {}".format(e))

    def _gazebo_unpause(self):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException as e:
            print(e)
            print("/gazebo/unpause_physics service call failed")

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

    def set_robot_pose(self):

        state = ModelState()
        state.model_name = "limo/"
        x = self.np_random.uniform(-0.5, 0.5)
        y = self.np_random.uniform(-0.5, 0.5)
        z = 0
        state.pose.position.x = x
        state.pose.position.y = y
        state.pose.position.z = z
        theta = self.np_random.uniform(-math.pi, math.pi)
        print(x, y, z, theta)
        quaternion = tf.transformations.quaternion_from_euler(0, 0, theta)
        state.pose.orientation.x = quaternion[0]
        state.pose.orientation.y = quaternion[1]
        state.pose.orientation.z = quaternion[2]
        state.pose.orientation.w = quaternion[3]

        self.robot_pos = PoseWithCovarianceStamped()
        self.robot_pos.header.frame_id = self.map_frame_id
        self.robot_pos.header.stamp = rospy.Time.now()
        self.robot_pos.pose.pose.position.x = x
        self.robot_pos.pose.pose.position.y = y
        self.robot_pos.pose.pose.position.z = z
        self.robot_pos.pose.pose.orientation.x = quaternion[0]
        self.robot_pos.pose.pose.orientation.y = quaternion[1]
        self.robot_pos.pose.pose.orientation.z = quaternion[2]
        self.robot_pos.pose.pose.orientation.w = quaternion[3]
        covariance = [0 for i in range(36)]
        # covariance[0, 0] = 0.1
        # covariance[1, 1] = 0.1
        # covariance
        self.robot_pos.pose.covariance = covariance
        self.robot_pos_pub.publish(self.robot_pos)

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            set_state(state)
        except rospy.ServiceException as e:
            print("Service call failed: {}".format(e))
        return

    def set_goal_pose(self):
        self.goal_pos = PoseStamped()
        self.goal_pos.header.frame_id = self.map_frame_id
        self.goal_pos.header.stamp = rospy.Time.now()
        x = self.np_random.uniform(-0.5, 0.5)
        y = self.np_random.uniform(-0.5, 0.5)
        z = 0
        self.goal_pos.pose.position.x = x
        self.goal_pos.pose.position.y = y
        self.goal_pos.pose.position.z = z
        theta = self.np_random.uniform(-math.pi, math.pi)
        print(x, y, z, theta)
        quaternion = tf.transformations.quaternion_from_euler(0, 0, theta)
        self.goal_pos.pose.orientation.x = quaternion[0]
        self.goal_pos.pose.orientation.y = quaternion[1]
        self.goal_pos.pose.orientation.z = quaternion[2]
        self.goal_pos.pose.orientation.w = quaternion[3]
        self.goal_pos_pub.publish(self.goal_pos)

    def step(self, action):

        # vel_cmd = Twist()
        # vel_cmd.linear.x = action[0]
        # vel_cmd.angular.z = action[1]
        # vel_cmd.linear.x = 0.0
        # vel_cmd.angular.z = 0.0
        # self.vel_pub.publish(vel_cmd)

        # modelstates = rospy.wait_for_message("/gazebo/model_states", ModelStates, timeout=5)
        # print(modelstates.name)
        self.set_robot_pose()
        self.set_goal_pose()

        # laser_data = None
        # success = False
        # while laser_data is None or not success:
        #     try:
        #         laser_data = rospy.wait_for_message(self.laser_topic, LaserScan, timeout=5)
        #     finally:
        #         success = True

        # # state, _ = self.discrete_observation(laser_data, 5)
        # laser_len = len(laser_data.ranges)
        # print(laser_len)
        # # state = [ego_pos, goal_pos, laser_data]
        # reward = 0
        # done = 0
        # # return laser_data, reward, done, {}

    def get_observation(self):
        pass

    def reset(self):
        self._gazebo_reset()
        # time.sleep(0.1)

        # self._gazebo_unpause()

        # Read laser data
        laser_data = None
        success = False
        while laser_data is None or not success:
            try:
                laser_data = rospy.wait_for_message(self.laser_topic, LaserScan, timeout=5)
            finally:
                success = True

        # self._gazebo_pause()
        print(len(laser_data.ranges))
        # state = self.discrete_observation(laser_data, 5)
        state = 0
        return state
