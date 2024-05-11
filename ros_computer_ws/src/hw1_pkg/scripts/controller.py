#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, quaternion_from_euler

from std_msgs.msg import String

from scipy.spatial.transform import Rotation as R
import numpy as np
from typing import NamedTuple
import os
import math


os.environ['ROS_IP'] = '10.43.208.163' #'10.43.70.164'
#os.environ['ROS_HOSTNAME'] = '10.43.208.163'
os.environ['ROS_MASTER_URI'] = 'http://10.43.70.164:11311'

def clip(val, min_val, max_val):
    return max(min(val, max_val), min_val)


class LowpassFilter:

    def __init__(self, filet_coeff, val=0):
        self.val = val
        self.filet_coeff = filet_coeff

    def update(self, new_val):
        self.val = self.filet_coeff * new_val + (1 - self.filet_coeff) * self.val

    def get_val(self):
        return self.val

class PIDController:

    def __init__(self, kp, ki, kd, delta_t):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        #self.v_ref_integral = 0
        self.last_pos = 0
        self.last_err_p = 0
        self.delta_t = delta_t

    def get_output(self, ref_pos, current_pos, curr_vel):
        
        # Proportional error
        ref_vel = (ref_pos - self.last_pos) / self.delta_t
        err_p = ref_vel - curr_vel
        self.last_pos = ref_pos

        # Integral
        err_i = ref_pos - current_pos

        # Derivative
        err_d = (err_p - self.last_err_p) / self.delta_t
        self.last_err_p = err_p

        #print("Err_i:", err_i, "Err_p:", err_p, "Err_d:", err_d)

        return self.kp * err_p + self.ki * err_i + self.kd * err_d
    
class AngleController:

    def __init__(self, delta_t, kp=1, ki=0.3, kd=0.05) -> None:

        self.angle = LowpassFilter(filet_coeff=0.3, val=0)
        self.angle_vel = LowpassFilter(filet_coeff=0.3, val=0)
        self.angle_controller = PIDController(kp=kp, ki=ki, kd=kd, delta_t=delta_t)
    
    def get_output(self, data: Odometry, target_angle):

        orientation = R.from_quat([data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w])
        angle_z = orientation.as_euler('zyx')[0]
        angular_velocity = data.twist.twist.angular.z

        self.angle.update(angle_z)
        self.angle_vel.update(angular_velocity)

        ang_vel_out = self.angle_controller.get_output(target_angle, self.angle.get_val(), self.angle_vel.get_val())

        #print("Target angle:", target_angle, "Angle:", self.angle.get_val())

        return ang_vel_out
    
class PosController:

    def __init__(self, vel: int, delta_t: float) -> None:
        self.vel = vel
        self.delta_t = delta_t
        self.last_angle = 0

        
    def get_output(self, data: Odometry, target_pos):

        vel_output = np.array([0, 0, 0])

        position = np.array([data.pose.pose.position.x, data.pose.pose.position.y, data.pose.pose.position.z])
        orientation = R.from_quat([data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w])

        direction = target_pos - position
        #print("Direction:", direction)
        dir_size = np.linalg.norm(direction)

        if dir_size < 0.2:
            return vel_output, 0
        
        direction = direction * self.vel / dir_size

        self.last_angle = np.arctan2(direction[1], direction[0])

        return orientation.apply(direction, inverse=True), self.last_angle


class RobotController:
    def __init__(self):
        rate = 10
        rospy.init_node('robot_controller', anonymous=False)

        self.cmd_vel_pub = rospy.Publisher('/hiwonder_controller/cmd_vel', Twist, queue_size=10)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.instruction_sub = rospy.Subscriber('/robot_instructions', String, self.instruction_callback)
        self.rate = rospy.Rate(rate)  # 10 Hz

        self.target_pos = np.array([0, 0, 0])
        #self.target_pos = np.array([0, 0, 0])
        self.target_angle = 0
        self.target_angle = 0

        self.current_odom = None

        self.angle_output = LowpassFilter(filet_coeff=0.6, val=0)
        self.vel_output = LowpassFilter(filet_coeff=0.6, val=0)

        self.pos_controller = PosController(vel=0.15, delta_t=1/rate)
        self.angle_controller = AngleController(delta_t=1/rate, kp=1.7, ki=0.5, kd=0.2)

    def instruction_callback(self, msg):
        
        distance = 0.2
        angle = 0.5

        # Get into world frame position
        if self.current_odom is not None:
            orientation = R.from_quat([self.current_odom.pose.pose.orientation.x, self.current_odom.pose.pose.orientation.y, self.current_odom.pose.pose.orientation.z, self.current_odom.pose.pose.orientation.w])
            angle_z = orientation.as_euler('zyx')[0]

            position = np.array([self.current_odom.pose.pose.position.x, self.current_odom.pose.pose.position.y, self.current_odom.pose.pose.position.z])

            self.target_angle = angle_z + angle
            #self.target_pos = position + orientation.apply([distance, 0, 0])
            self.target_pos = position + np.array([distance * np.cos(self.target_angle), distance * np.sin(self.target_angle), 0])


    def odom_callback(self, msg):
        self.current_odom = msg
        vel_output, angle = self.pos_controller.get_output(msg, self.target_pos)
        ang_output = self.angle_controller.get_output(msg, self.target_angle)
        self.angle_output.update(ang_output)
        self.vel_output.update(vel_output)

        angle_out = self.angle_output.get_val()
        
        if abs(angle_out) < 0.07:
            angle_out = 0

        cmd_vel = Twist()
        cmd_vel.linear.x = self.vel_output.get_val()[0]
        cmd_vel.linear.y = self.vel_output.get_val()[1]
        cmd_vel.angular.z = clip(angle_out, -0.5, 0.5)
        self.cmd_vel_pub.publish(cmd_vel)

    def publish_velocity(self, linear_x, angular_z):
        cmd_vel = Twist()
        cmd_vel.linear.x = linear_x
        cmd_vel.angular.z = angular_z
        self.cmd_vel_pub.publish(cmd_vel)

if __name__ == '__main__':
    try:
        controller = RobotController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
