import rospy
from audio_common_msgs.msg import AudioData
import os

"""
PARAMETERS
 * /audio/audio_capture/bitrate: 128
 * /audio/audio_capture/channels: 1
 * /audio/audio_capture/depth: 16
 * /audio/audio_capture/device: 
 * /audio/audio_capture/dst: appsink
 * /audio/audio_capture/format: mp3
 * /audio/audio_capture/sample_format: S16LE
 * /audio/audio_capture/sample_rate: 16000
 * /rosdistro: melodic
 * /rosversion: 1.14.13
"""

class AudioSaver:
    def __init__(self, topic_name, mp3_path):
        self.topic_name = topic_name
        self.mp3_path = mp3_path
        self.mp3_file = open(self.mp3_path, 'wb')  # Open file in binary write mode
        
        rospy.init_node('audio_saver', anonymous=True)
        rospy.Subscriber(self.topic_name, AudioData, self.audio_callback)
        rospy.on_shutdown(self.shutdown_hook)  # Ensure file is closed on shutdown

    def audio_callback(self, msg):
        """
        Callback function that saves received audio data to an MP3 file.
        """
        self.mp3_file.write(msg.data)

    def shutdown_hook(self):
        """
        Clean up resources when ROS node is shutting down.
        """
        self.mp3_file.close()
        print(f"Audio data saved to {self.mp3_path}")

if __name__ == '__main__':
    topic = '/audio/audio'  # Adjust this to your audio topic
    output_file = 'output_audio.mp3'  # Path to save the MP3 file
    audio_saver = AudioSaver(topic, output_file)
    rospy.spin()  # Keep the node running
