#!/usr/bin/env python3
# encoding: utf-8
import rospy
from geometry_msgs.msg import Twist
from hiwonder_servo_msgs.msg import MultiRawIdPosDur
from hiwonder_kinematics.kinematics_control import set_pose_target
from hiwonder_servo_controllers.bus_servo_control import set_servos
from your_package_name.srv import ResetArm, ResetArmResponse


class ArmResetService:

    def __init__(self):
        rospy.init_node("arm_reset_service")
        self.joints_pub = rospy.Publisher('/servo_controllers/port_id_1/multi_id_pos_dur', MultiRawIdPosDur, queue_size=1)
        self.service = rospy.Service('reset_arm_position', ResetArm, self.reset_arm_position)
