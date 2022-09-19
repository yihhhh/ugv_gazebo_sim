from .gazebo_env import GazeboEnv
from .parking_env import GazeboCarParkingEnv
from .nav_env import GazeboCarNavEnv
from gym.envs.registration import register

register(
    id='GazeboCarNav-v0',
    entry_point='gym_gazebo.envs:GazeboCarNavEnv',
    # More arguments here
)
