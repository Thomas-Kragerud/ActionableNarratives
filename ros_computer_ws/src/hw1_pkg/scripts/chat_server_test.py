#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import String

response_received = True  # Initialize as True to allow first input immediately

def callback(msg):
    global response_received
    rospy.loginfo("Received message: %s", msg.data)
    response_received = True  # Set the flag to True when a response is received

def pub_it():
    global response_received
    rospy.init_node("user_input_publisher", anonymous=True)
    pub = rospy.Publisher("/user_input", String, queue_size=10)
    rospy.Subscriber("/full_response", String, callback)

    while not rospy.is_shutdown():
        if response_received:  # Allow input if a response has been received or if it's the first input
            user_input = input("Enter a message: ")
            if user_input == "":
                break

            pub.publish(user_input)
            rospy.loginfo("Published message: %s", user_input)
            response_received = False  # Reset the flag after sending a message

if __name__ == '__main__':
    try:
        pub_it()
    except rospy.ROSInterruptException:
        pass
