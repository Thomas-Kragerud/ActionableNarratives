#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image as Im
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger, TriggerResponse
import ros_numpy
import tf
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

import base64
from PIL import Image
import redis
import json
from io import BytesIO
import math


class ImagePoseService:
    def __init__(self):
        print("Starter image cap service")
        rospy.init_node('image_pose_service', anonymous=False)
        print("i am here")
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        print("after client")
        self.tf_listener = tf.TransformListener()
        print("after tf_listener")
        self.move_base_client = actionlib.SimpleActionClient('/robot_1/move_base', MoveBaseAction)
        print("after move base")
        self.move_base_client.wait_for_server()
        print("after move base")

        self.service = rospy.Service('/robot_1/capture_images_and_poses', Trigger, self.handle_capture_request)
        print("after service")
        # self.image_sub = rospy.Subscriber('/usb_cam_simple/image_raw', Im, self.image_callback)
        self.image_sub = rospy.Subscriber('/robot_1/depth_cam/rgb/image_raw', Im, self.image_callback)

        self.images_and_poses = []

    def image_callback(self, img_msg):
        try:
            (trans, rot) = self.tf_listener.lookupTransform('/robot_1/map', '/base_link', rospy.Time(0))
            pose = PoseStamped()
            pose.header.frame_id = "robot_1/map"
            pose.pose.position.x = trans[0]
            pose.pose.position.y = trans[1]
            pose.pose.position.z = trans[2]
            pose.pose.orientation.x = rot[0]
            pose.pose.orientation.y = rot[1]
            pose.pose.orientation.z = rot[2]
            pose.pose.orientation.w = rot[3]

            pil_image = Image.fromarray(ros_numpy.numpify(img_msg))
            buffer = BytesIO()
            pil_image.save(buffer, format="JPEG")
            encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

            self.images_and_poses.append({"image": encoded_image, "pose": pose})
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            pass

    def handle_capture_request(self, req):
        self.images_and_poses = []
        for i in range(6):
            goal = MoveBaseGoal()
            goal.target_pose.header.frame_id = "robot_1/map"
            goal.target_pose.header.stamp = rospy.Time.now()
            goal.target_pose.pose.orientation.z = math.sin(math.pi / 6 * (i + 1))
            goal.target_pose.pose.orientation.w = math.cos(math.pi / 6 * (i + 1))
            self.move_base_client.send_goal(goal)
            self.move_base_client.wait_for_result()
            rospy.sleep(1)  # Wait for image capture and to stabilize after rotation

        self.send_to_redis()
        return TriggerResponse(success=True, message="Images and poses captured and sent.")

    def send_to_redis(self):
        data = {
            "images_and_poses": self.images_and_poses,
        }
        self.redis_client.publish("llm_cmd_input", json.dumps(data))
        rospy.loginfo("Data sent to Redis")


if __name__ == "__main__":
    server = ImagePoseService()
    rospy.spin()