# Use an official ROS base image
FROM ros:noetic-robot
#FROM ros:melodic-ros-base

SHELL ["/bin/bash", "-c"]
RUN source ros_entrypoint.sh
RUN echo "source /ros_entrypoint.sh" >> /root/.bashrc

# Revert the shell back to sh for subsequent commands
SHELL ["/bin/sh", "-c"]

RUN sudo apt-get update
RUN sudo apt-get install apt-utils -y
RUN sudo apt-get install iputils-ping -y
RUN sudo apt-get install python3-pip -y
RUN pip3 install PyYAML
RUN sudo apt-get install python3-catkin-pkg -y
RUN sudo apt-get install ros-noetic-joy -y
RUN sudo apt-get install ros-noetic-rviz-visual-tools -y
RUN sudo apt-get install git -y
RUN sudo apt-get install wget -y
#RUN sudo apt-get install libopencv-dev -y
#RUN sudo apt install ros-noetic-camera-info-manager* -y
#RUN sudo apt install ros-noetic-rgbd-launch libuvc-dev -y
#RUN sudo apt install libgflags-dev  ros-noetic-image-geometry ros-noetic-camera-info-manager \ 
#    ros-noetic-image-transport ros-noetic-image-publisher libgoogle-glog-dev libusb-1.0-0-dev libeigen3-dev -y
#RUN sudo apt-get install ros-noetic-navigation -y
#RUN sudo apt-get install -y ros-noetic-joy ros-noetic-costmap-2d ros-noetic-nav-core ros-noetic-sound-play ros-noetic-amcl \
#    ros-noetic-slam-gmapping ros-noetic-move-base ros-noetic-controller-interface ros-noetic-gazebo-ros-control ros-noetic-joint-state-controller \
#    ros-noetic-effort-controll ros-noetic-moveit-msgs ros-noetic-teleop-twist-keyboard ros-noetic-slam-gmapping ros-noetic-map-server ros-noetic-qt-gui \
#    ros-noetic-kdl-parser ros-noetic-combined-robot-hw ros-noetic-combined-robot-hw-tests ros-noetic-controller-manager ros-noetic-diff-drive-controller \
#    ros-noetic-force-torque-sensor-controller ros-noetic-gripper-action-controller ros-noetic-imu-sensor-controller ros-noetic-position-controll \
#    ros-noetic-ros-control ros-noetic-ros-controll ros-noetic-rqt-joint-trajectory-controller ros-noetic-velocity-controll ros-noetic-cv-bridge \
#    ros-noetic-polled-camera ros-noetic-camera-info-manager ros-noetic-tf-conversions ros-noetic-opencv-apps libopencv-dev ros-noetic-rqt \
#    ros-noetic-rqt-common-plugins ros-noetic-ur-client-library -y
#RUN sudo apt install ros-noetic-pcl-ros -y
#RUN sudo apt install pcl-tools -y
#RUN sudo apt install cmake pkg-config -y
#RUN sudo apt install ros-noetic-desktop -y
RUN apt-get update && apt-get install -y \
    libx11-xcb1 \
    libgl1-mesa-glx \
    libfontconfig1 \
    libxrender1 \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN sudo apt-get update
RUN sudo apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base gstreamer1.0-plugins-good \ 
    gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-tools gstreamer1.0-x \
    gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio -y

RUN sudo apt install net-tools -y
RUN sudo apt-get install -y alsa-base alsa-utils

RUN apt-get update && apt-get install -y \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    && rm -rf /var/lib/apt/lists/*