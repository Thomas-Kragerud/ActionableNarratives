# ROS 1 Setup

This project relies heavily on ROS 1. One method of setting this up is using Docker. See the DockerReadme.md for more information about how to do this with our provided Docker and docker-compose files.

# Redis

Although we would like to stick to only using ROS 1, because of some compatability issues regarding python versions when doing experiments with ChatGPT4, we have also included Redis in our setup. Redis is however only needed for the ChatGPT4 experiments.

# MiniGPT-4

Download the MiniGPT-4 checkpoint from the following link: https://drive.google.com/file/d/1aVbfW7nkCSYx99_vCRyP1sOlQiWVSnAl/view, and put it in the ```MiniGPT-4``` folder.
This model is based on Llama2.

## Running the Unified Llama2 Model for Decision Making

Make sure you have Python installed on your system and the necessary dependencies are set up following the [MiniGPT-4 GitHub Repository](https://github.com/Vision-CAIR/MiniGPT-4). Additionally, ensure you have a GPU (with memory >= 8G) that can be utilized (in this case, GPU ID 0 is specified).

Open your terminal or command prompt and execute the following command:

```bash
cd MiniGPT-4
python analysis_node.py --cfg-path eval_configs/minigptv2_eval.yaml --gpu-id 0
```

## Running the Hierarchical Llama2 + Llama3 Model for Decision Making

Make sure you have Python installed on your system and the necessary dependencies are set up following the [MiniGPT-4 GitHub Repository](https://github.com/Vision-CAIR/MiniGPT-4). Additionally, ensure you have two GPUs (with memory >= 12G) that can be utilized for each of the models.

Open a terminal or command prompt and execute the following command:

```bash
cd MiniGPT-4
python vision_node_llama2.py --cfg-path eval_configs/minigptv2_eval.yaml --gpu-id 1
```

Open another terminal or command prompt and execute the following command:

```bash
cd MiniGPT-4
python analysis_llama3.py 
```

# GPT4-Vision 
To run the gpt4 vision code is located in the no_ros_workspace folder and is run from 310_env virtual environment. To be able to run this code you need to have redis downloaded on your ubuntu machine and start the local server. The gpt4_vision.py file can then be run without any arguments with "python3 gpt4_vision.py". 

## Full pipline 
To run the full pipline you need 5 processes running. On the robot you need to do the following 
```bash
sudo stop start_system_service
roslaunch hiwonder_controller hiwonder_controller.launch 
roslaunch hiwonder_peripreals depth_cam.launch
```
then to activate the microfon you need to cd into the audio folder. Change to bash by typing bash. source /dev/opt/ros/melodic/setup.bash, then source devil/setup.bash then

```bash roslaunch audio_capture capture.launch ```



# Folder descriptions

- ```ros_robot_default_ws```: The default ROS workspace that is on our Robot by default. Some code for capturing images are used in there??
- ```ros_robot_custom_ws```: A custom ROS workspace that we also have on the robot. Mainly for streaming of Audio / Speech.
    - By sourcing this workspace (on the robot), and launching ```roslaunch audio_capture capture.launch```, one can make a stream of audio data from the robot.
- ```ros_computer_ws```: A workspace for running the ROS nodes that we run on our computer.
    - ```src/hw1_pkg/scripts```: Here you find the ```livewhisper4.py``` script, that we made for streaming audio from the robot, and detect voice, and feed it into the whisper model. Also the ```controller.py``` script is here, which is our simple script for controlling the robot. The controller requires the ```roslaunch hiwonder_controller hiwonder_controller.launch```-command being run on the robot, for making it listen to inputs.
- ```no_ros_computer_ws```: This is the folder where we use Redis for communicating with ChatGPT4?? And do some other stuff.
- ```MiniGPT-4```: The folder where the MiniGPT-4 model is stored.
