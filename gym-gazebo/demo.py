import numpy as np
import os
import argparse

import rospy
from geometry_msgs.msg import Twist


def run_demo(args):

    action_list = np.load(os.path.join(args.load, 'action_list.npy'), allow_pickle=True)
    print(len(action_list))

    rospy.init_node("cmd", anonymous=True)
    r = rospy.Rate(args.rate)
    vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
    for vel_cmd in action_list:
        vel_pub.publish(vel_cmd)
        rospy.loginfo(vel_cmd)
        r.sleep()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--load',type=str, default=None, help="load the action list")
    parser.add_argument('--rate',type=float, default=5.2, help="sleep rate (Hz)")
    args = parser.parse_args()

    run_demo(args)