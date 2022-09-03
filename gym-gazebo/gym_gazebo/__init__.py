import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

# Gazebo
# ----------------------------------------

# F1 manual
register(
    id='GazeboCarParking-v0',
    entry_point='gym_gazebo.envs:GazeboCarParkingEnv',
    # More arguments here
)

register(
    id='GazeboCarNav-v0',
    entry_point='gym_gazebo.envs:GazeboCarNavEnv',
    # More arguments here
)
