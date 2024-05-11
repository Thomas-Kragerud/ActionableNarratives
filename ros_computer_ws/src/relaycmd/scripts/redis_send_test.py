#!/usr/bin/env python3
# -*_ coding: utf-8 -*-

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image as Im
import ros_numpy

import base64
from PIL import Image

import redis
import json
from io import BytesIO
import time

class RelayNode:
    """ Relay information from robot and whisper to llm"""
    def __init__(self):
        self.newest_encoded_image: str = ""
        self.newest_text: str = ""
        self.has_new_image = False
        self.has_new_text = False

        self.redis_client = redis.Redis(host="localhost", port=6379, db=0)
        self.last_sent_time = time.time()
        self.interval = 10

        print("Started redis to LLM")
        rospy.init_node("mac_relay_node", anonymous=True)
        print("Started Node")
        # Subscribe to the image topic
        #self.pic_sub = rospy.Subscriber('/usb_cam_simple/image_raw', Im, self.image_callback)
        self.pic_sub = rospy.Subscriber('/depth_cam/rgb/image_raw', Im, self.image_callback)
        self.text_sub = rospy.Subscriber("/transcription", String, self.text_callback)
        print("finished subbing")
    def image_callback(self, img_msg):
        #print("Recived image")
        pil_image = Image.fromarray(ros_numpy.numpify(img_msg))
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG")
        self.newest_encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        self.has_new_image = True
        self.try_send()

    def text_callback(self, txt_msg):
        print("Recived Text")
        response = txt_msg.data.lower()
        self.newest_text = response
        self.has_new_text = True
        self.try_send()

    def try_send(self):
        current_time = time.time()

        if (
                self.has_new_image and
                self.has_new_text and
                (current_time - self.last_sent_time >= self.interval)
        ):
            self.send_to_redis()
            self.last_sent_time = current_time
            self.has_new_image = False
            self.has_new_text = False

    def send_to_redis(self):
        data = {
            "image": self.newest_encoded_image,
            "user_input": self.newest_text
        }
        # lpush ads combined data to the Redis queue
        #self.redis_client.lpush('queue:tasks', json.dumps(data))
        self.redis_client.publish("llm_cmd_input", json.dumps(data))
        print("Data sent to Redis")


if __name__ == "__main__":
    node = RelayNode()
    rospy.spin()




