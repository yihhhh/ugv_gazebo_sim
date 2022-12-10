# Gym Gazebo Environment for LIMO
## Docker Environment Setting
### 1. Create image
To run the simulation in docker, an image to support both ros-melodic and pytorch can be created by run the following in the terminal:
```shell
cd dockerfile
docker build -f dockerfile_nvidia_ros_melodic -t nvidia_ros_melodic .
docker build -f dockerfile_pytorch -t pytorch_ros .
docker build -f dockerfile_limo_base -t limo_base .
```
### 2. Create a container over the image
Once the `limo_base` image is built, a container over the image can be created by run the following in the terminal:
```shell
sh run_limo_base.sh
```
In this container, a conda environment called `mytorch` is already installed to support pytorch and can be activated using alias `start_mytoch` in the terminal.

This repo is also cloned in this container, under the directory `/limo_ws/src/`.
### 3. Build ros-tf for python 3 environment
```shell
start_mytorch && cd /tf && export ROS_PYTHON_VERSION=3 && export CMAKE_PREFIX_PATH=/opt/ros/melodic && catkin_make
```
### 4. Pull the repository
```shell
cd /limo_ws/src/ugv_gazebo_sim && source /tf/devel/setup.bash && git pull
```
### 5. Run a new command in the running container to run Gazebo
Open a new terminal and run the following:
```shell
docker exec -it <container_name> /bin/bash
source /opt/ros/melodic/setup.bash && cd /limo_ws && catkin_make
source devel/setup.bash && roslaunch limo_gazebo_sim GazeboCarNavEnv.launch
```
## Run Training
```shell
cd /limo_ws/src/ugv_gazebo_sim/gym-gazebo
python run.py --group_id <group_id> --exp_id <exp_id> --env_id <env_id> --epoch <epoch> --episode <episode> --ensemble <number of ensemble model> --optimizer <optimizer type> --config <config file>
```
**Note**:       
Parse the parameter `--debug` to use the debug mode, then no log will be recorded.      
Parse the parameter `--save` to save the model that being trained.      
Parse the parameter `--render` to visualize the training step in python.        
Parse the model file path via parameter `--load` to continue train on the pre-trained model.

## Run evaluation
```shell
cd /limo_ws/src/ugv_gazebo_sim/gym-gazebo
python evaluate.py --env_id <env_id> --load <model file path> --config <config file>
```
**Note**:       
Parse the parameter `--record` to save the trajectory into a npy file.      
Parse the parameter `--render` to visualize the training step in python.     