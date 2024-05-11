#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
import whisper
from audio_common_msgs.msg import AudioData
from pydub import AudioSegment
from queue import Queue
import io
import numpy as np
import threading
import torch
import os
from pydub.effects import normalize

os.environ['ROS_IP'] = '10.43.208.163' #'10.43.70.164'
#os.environ['ROS_HOSTNAME'] = '10.43.208.163'
os.environ['ROS_MASTER_URI'] = 'http://10.43.70.164:11311'

class ROSWhisper:
    def __init__(self):
        rospy.init_node('whisper_transcriber', anonymous=True)
        self.pub = rospy.Publisher('transcription', String, queue_size=10)
        self.model = whisper.load_model("base")
        self.bytes_buffer = bytes()
        self.audio_segment_queue = Queue()
        self.audio_for_processing = Queue()
        self.active_speech_buffer = np.zeros(0)
        self.active_speech_buffer2 = AudioSegment.empty()
        self.sample_rate = 16000
        self.bytes_buffer_secs = 0.4
        self.no_speech_treshold_secs = 1
        self.minimum_speech_duration = 1

        rospy.Subscriber('/audio/audio', AudioData, self.audio_callback) #, queue_size=20, buff_size=15000)

    def audio_segment_preprocess(self, audio_segment: AudioSegment) -> np.ndarray:

        # Convert to mono if not already
        if audio_segment.channels > 1:
            audio_segment = audio_segment.set_channels(1)

        # Resample to 16 kHz if necessary
        if audio_segment.frame_rate != 16000:
            audio_segment = audio_segment.set_frame_rate(16000)

        audio_raw = np.array(audio_segment.get_array_of_samples(), dtype=np.int16)

        return audio_segment, audio_raw

    def audio_callback(self, msg):
        """
        Callback function that receives MP3 audio data and buffers it.
        Converts it into an AudioSegment and puts it into the queue for processing.
        """
        #print("Msg data length:", len(msg.data))
        self.bytes_buffer += msg.data
        #print("Bytes length:", len(self.bytes_buffer))
        if len(self.bytes_buffer) > self.bytes_buffer_secs*self.sample_rate:  # Buffer size can be adjusted based on your data
            audio_data = io.BytesIO(self.bytes_buffer)
            audio_segment = AudioSegment.from_file(audio_data, format="mp3")

            self.bytes_buffer = bytes()  # Clear the buffer after successful decoding

            self.audio_segment_queue.put(audio_segment)

    def detect_speech(self, raw_data: bytes):
        # Assume 16-bit mono audio, change this if your audio format is different
        bytes_per_sample = 2
        num_channels = 1  # Set to 2 if stereo

        # Calculate the number of bytes in a 30 ms segment
        frame_duration_ms = 30  # frame duration in milliseconds
        samples_per_frame = int(self.sample_rate * frame_duration_ms / 1000)
        bytes_per_frame = samples_per_frame * bytes_per_sample * num_channels

        # Loop over the raw data in chunks of 30 milliseconds
        for i in range(0, len(raw_data) - bytes_per_frame + 1, bytes_per_frame):
            segment = raw_data[i:i+bytes_per_frame]
            if len(segment) == bytes_per_frame:
                is_speech = self.vad.is_speech(segment, self.sample_rate)
                if is_speech:
                    #print("Detected speech")
                    return True
                
            else:
                print("Skipped check")

        return False
    
    def detect_speech2(self, raw_data: np.ndarray):
        raw_data = raw_data.astype(np.float32)
        freq = np.argmax(np.abs(np.fft.rfft(raw_data))) * self.sample_rate / len(raw_data)
        return np.sqrt(np.mean(raw_data**2)) > self.threshold and self.vocals[0] <= freq <= self.vocals[1]
    
    def detect_speech3(self, raw_data: np.ndarray):
        fft_data = np.fft.rfft(raw_data)
        frequencies = np.fft.rfftfreq(len(raw_data), 1/self.sample_rate)
        avg_amp = np.mean(np.abs(fft_data[frequencies > 1000]))
        max_amp = np.max(np.abs(fft_data[(frequencies < 1000) & (frequencies > 150)]))
        max_amp_clean = raw_data.max()

        #print("Ratio:", max_amp/avg_amp, max_amp, avg_amp, "Max amp clean:", max_amp_clean)

        return max_amp/avg_amp > 50 and max_amp_clean > 2000


    def process_segments(self):
        """
        Looks for speech segments in buffered audio and sends them to the Whisper thread.
        """

        speech_is_detected = False
        seconds_without_speech = 0
         
        while True:
            audio_segment = self.audio_segment_queue.get()

            audio_segment, processed_raw_data = self.audio_segment_preprocess(audio_segment)

            is_speech = self.detect_speech3(processed_raw_data)
            speech_is_detected = speech_is_detected or is_speech

            if speech_is_detected:
                self.active_speech_buffer = np.concatenate((self.active_speech_buffer, processed_raw_data))
                self.active_speech_buffer2 += audio_segment

            if is_speech:
                seconds_without_speech = 0
            else:
                seconds_without_speech += len(processed_raw_data) / self.sample_rate

            if seconds_without_speech >= self.no_speech_treshold_secs and speech_is_detected:
                #print("Speech segment ended")
                if len(self.active_speech_buffer)/self.sample_rate > self.minimum_speech_duration:
                    self.audio_for_processing.put(np.array(normalize(self.active_speech_buffer2).get_array_of_samples(), dtype=np.int16))

                self.active_speech_buffer = np.zeros(0)
                self.active_speech_buffer2 = AudioSegment.empty()
                speech_is_detected = False
                seconds_without_speech = 0
                

    def process_audio(self):
        """
        Takes audio segments containing speech, and transcribes them using the Whisper model.
        """
        while True:
            audio_raw = self.audio_for_processing.get()
            audio_raw = torch.tensor(audio_raw, dtype=torch.float32) / 32768

            result = self.model.transcribe(audio_raw, verbose=None, fp16=False, language="en")
            #self.pub.publish(result['text'])
            text = result['text']
            if text.strip() != "":
                print("Transcription:", text)
                self.pub.publish(text)
            #print("Transcription:", result['text'])

if __name__ == '__main__':
    node = ROSWhisper()
    threading.Thread(target=node.process_segments, daemon=True).start()
    threading.Thread(target=node.process_audio, daemon=True).start()
    rospy.spin()  # This keeps your node active and listening for callbacks