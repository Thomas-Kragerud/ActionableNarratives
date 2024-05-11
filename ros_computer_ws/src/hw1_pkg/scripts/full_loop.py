#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, quaternion_from_euler

from std_msgs.msg import String

class RobotController:
    def __init__(self):
        rospy.init_node('robot_controller', anonymous=False)
        rospy.Subscriber("/full_response", String, self.llm_callback)
        self.cmd_vel_pub = rospy.Publisher('/hiwonder_controller/cmd_vel', Twist, queue_size=10)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.rate = rospy.Rate(10)  # 10 Hz
        self.initial_pose = None
        self.current_pose = None
        self.target_distance = 2.0  # Default forward distance in meters
        self.angular_speed = 0.5  # rad/s
        self.linear_speed = 0.4  # m/s
        self.command = None

    def llm_callback(self, msg):
        response = msg.data.lower()
        commands = {"move forward": "forward", "straight": "forward", "turn right": "right", "right": "right",
                    "turn left": "left", "left": "left"}
        found_command = False

        for key_phrase, command in commands.items():
            if key_phrase in response:
                print(f"Command: {command.capitalize()}")
                self.command = command
                self.initial_pose = self.current_pose  # Reset initial pose on new command
                self.execute_command()
                found_command = True
                break

        if not found_command:
            print("No valid command found.")

    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose

    def execute_command(self):
        while not rospy.is_shutdown() and self.current_pose and self.command:
            if self.command == "forward":
                self.move_forward()
            elif self.command in ["right", "left"]:
                self.turn()
            self.rate.sleep()

    def move_forward(self):
        distance = self.euclidean_distance(self.initial_pose, self.current_pose)
        if distance < self.target_distance:
            self.publish_velocity(self.linear_speed, 0)
        else:
            self.stop_robot()

    def turn(self):
        angle_turned = self.calculate_angle(self.initial_pose, self.current_pose)
        if angle_turned < 0.52:  # Approximately 30 degrees in radians
            angular_z = self.angular_speed if self.command == "right" else -self.angular_speed
            self.publish_velocity(0, angular_z)
        else:
            self.stop_robot()

    def publish_velocity(self, linear_x, angular_z):
        cmd_vel = Twist()
        cmd_vel.linear.x = linear_x
        cmd_vel.angular.z = angular_z
        self.cmd_vel_pub.publish(cmd_vel)

    def stop_robot(self):
        self.publish_velocity(0, 0)
        print("Movement complete. Ready for next command.")
        self.command = None  # Clear command to wait for new input

    def euclidean_distance(self, pose1, pose2):
        return ((pose1.position.x - pose2.position.x) ** 2 + (pose1.position.y - pose2.position.y) ** 2) ** 0.5

    def calculate_angle(self, initial, current):
        _, _, yaw1 = euler_from_quaternion([initial.orientation.x, initial.orientation.y, initial.orientation.z, initial.orientation.w])
        _, _, yaw2 = euler_from_quaternion([current.orientation.x, current.orientation.y, current.orientation.z, current.orientation.w])
        return abs(yaw2 - yaw1)

if __name__ == '__main__':
    try:
        controller = RobotController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
