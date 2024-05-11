#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import redis
import json
import threading
from PIL import Image
from io import BytesIO

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, quaternion_from_euler


class RobotController:
    def __init__(self):
        rospy.init_node('robot_controller_redis', anonymous=False)
        self.cmd_vel_pub = rospy.Publisher('/hiwonder_controller/cmd_vel', Twist, queue_size=10)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.rate = rospy.Rate(10)  # 10 Hz
        self.initial_pose = None
        self.current_pose = None
        self.target_distance = 2.0
        self.angular_speed = 0.5
        self.linear_speed = 0.4
        self.command = None
        self.robot_cmd_output_channel = "robot_cmd_output"
        self.redis_client = redis.Redis(host="localhost", port=6379, db=0)
        self.redis_thread = threading.Thread(target=self.sub_to_redis)
        self.redis_thread.start()

    def sub_to_redis(self):
        pubsub = self.redis_client.pubsub()
        pubsub.subscribe(self.robot_cmd_output_channel)
        for message in pubsub.listen():
            if message["type"] == "message":
                data = json.loads(message["data"])
                self.command = data["direction"]
                self.initial_pose = self.current_pose
                rospy.loginfo("Received command to move: {}".format(self.command))
                self.execute_command()

    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose

    def execute_command(self):
        while self.initial_pose is None or self.current_pose is None:
            rospy.loginfo("Waiting for initial and current poses...")
            rospy.sleep(0.1)

        if self.command == "forward":
            self.move_forward()
        elif self.command in ["right", "left"]:
            self.turn()

    def move_forward(self):
        while not rospy.is_shutdown() and self.euclidean_distance(self.initial_pose,
                                                                  self.current_pose) < self.target_distance:
            self.publish_velocity(self.linear_speed, 0)
        self.stop_robot()

    def turn(self):
        while not rospy.is_shutdown() and self.calculate_angle(self.initial_pose, self.current_pose) < 0.52:
            angular_z = self.angular_speed if self.command == "right" else -self.angular_speed
            self.publish_velocity(0, angular_z)
        self.stop_robot()

    def publish_velocity(self, linear_x, angular_z):
        cmd_vel = Twist()
        cmd_vel.linear.x = linear_x
        cmd_vel.angular.z = angular_z
        self.cmd_vel_pub.publish(cmd_vel)

    def stop_robot(self):
        self.publish_velocity(0, 0)
        rospy.loginfo("Movement complete. Ready for next command.")
        self.command = None
        self.initial_pose = None

    def euclidean_distance(self, pose1, pose2):
        return ((pose1.position.x - pose2.position.x) ** 2 + (pose1.position.y - pose2.position.y) ** 2) ** 0.5

    def calculate_angle(self, initial, current):
        _, _, yaw1 = euler_from_quaternion(
            [initial.orientation.x, initial.orientation.y, initial.orientation.z, initial.orientation.w])
        _, _, yaw2 = euler_from_quaternion(
            [current.orientation.x, current.orientation.y, current.orientation.z, current.orientation.w])
        return abs(yaw2 - yaw1)


if __name__ == "__main__":
    controller = RobotController()
    rospy.spin()  # This keeps the node alive and responsive to ROS callbacks and thread operations
