import argparse
import os
import random
from collections import defaultdict

import cv2
import re

import numpy as np
from PIL import Image
import torch
import html
import gradio as gr

import torchvision.transforms as T
import torch.backends.cudnn as cudnn

from minigpt4.common.config import Config

from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Conversation, SeparatorStyle, Chat

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

import rospy
import ros_numpy
from std_msgs.msg import String
from sensor_msgs.msg import Image as Im


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default='eval_configs/minigptv2_eval.yaml',
                        help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
            "in xxx=yyy format will be merged into config file (deprecate), "
            "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args

class ChatNode:

    def __init__(self, args):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        cudnn.benchmark = False
        cudnn.deterministic = True

        rospy.init_node('chat_node', anonymous=True)

        self.sub = rospy.Subscriber('/user_input', String, self.call_back)

        #compressed image
        self.pic_sub = rospy.Subscriber('/usb_cam_simple/image_raw', Im, self.image_callback)
        self.pub = rospy.Publisher('/text_action', String, queue_size=10)
        self.pub_full = rospy.Publisher('/full_response', String, queue_size=10)
        print('Initializing Chat')
        cfg = Config(args)

        device = 'cuda:{}'.format(args.gpu_id)

        model_config = cfg.model_cfg
        model_config.device_8bit = args.gpu_id
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to(device)

        vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

        model = model.eval()

        self.CONV_VISION = Conversation(
            system="",
            roles=(r"<s>[INST] ", r" [/INST]"),
            messages=[],
            offset=2,
            sep_style=SeparatorStyle.SINGLE,
            sep="",
        )

        # self.curr_image = np.zeros((200,200,3))
        self.curr_image = Image.open('./examples_v2/glip_test.jpg').convert("RGB")
        print(self.curr_image.size, '\n\n\n\n\n\n\n\n\n\n\n\n\n')

        self.chat = Chat(model, vis_processor, device=device)


    def image_callback(self, img_msg):
        raw_image = ros_numpy.numpify(img_msg)
        self.curr_image = Image.fromarray(raw_image)

    def call_back(self, data):

        # First, identify what place the user wants you to go, then analyze where the goal is, and lastly, you should prioritize your current direction to line up with the goal over the distance. 
        max_new_tokens = 500
        num_beams = 1
        temperature = 0.75

        message = '''
        Prompt: You are a self-driving robot in a household. You need to navigate to a place by selecting from three possible actions. You will be given an egocentric view of the environment in front of you. You will be given a user verbal command. 
        First, identify what place the user wants you to go, then analyze where the goal is and what is the best route, third, you should prioritize your current direction to line up with the goal over the distance. 
        Possible actions: "move straight", "turn left", "turn right"
        User command: ""
        First, describe the locations of objects in the scene. Then, describe your thought process to accomplish user's command. Finally, select the most appropriate action. 
        Answer strictly with this format, do not reply anything other than replacing contents in [] with your answers:
        {Begin of Answer}
        describe the scene: [your description]
        describe the thinking process: [your thinking process]
        action: [action selected]
        {End of Answer}
        '''

        message = message.replace('User command: ""', 'User command: "' + data.data + '"')
        raw_image = self.curr_image
        chat = self.chat
        chat_state = self.CONV_VISION.copy()
        img_list = []
        llm_message = self.chat.upload_img(raw_image, chat_state, img_list)
        self.chat.encode_img(img_list)
        print("here's the prompt: ", message)

        self.chat.ask(message, chat_state)
        print('thinking......')
        llm_message = self.chat.answer(
            conv=chat_state,
            img_list=img_list,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            temperature=temperature
        )[0]
        print("here's the answer: ")
        print(llm_message)
        

        action=llm_message
        self.pub.publish( String(llm_message.split('action: ')[-1]))
        self.pub_full.publish( String(llm_message))

        # image = './examples_v2/glip_test.jpg'
        # message = '''
        # Prompt: You are a self-driving robot in a household. You need to navigate to a place by selecting from three possible actions. You will be given an egocentric view of the environment in front of you. You will be given a user verbal command. First, identify what place the user wants you to go, then analyze where the goal is, and lastly, you should prioritize your current direction to line up with the goal over the distance. 
        # User command: “navigate to the tv”
        # Possible actions: “move straight”, “turn left”, “turn right”
        # Describe your thought process. Then select the most appropriate action. 
        # '''
        # # message = 'where is the TV?'
        # raw_image = Image.open(image).convert("RGB")
        # chat_state = CONV_VISION.copy()
        # img_list = []
        # llm_message = chat.upload_img(raw_image, chat_state, img_list)
        # chat.encode_img(img_list)
        # chat.ask(message, chat_state)

        # llm_message = chat.answer(
        #     conv=chat_state,
        #     img_list=img_list,
        #     max_new_tokens=max_new_tokens,
        #     num_beams=num_beams,
        #     temperature=temperature
        # )[0]


if __name__ == '__main__':
    args = parse_args()
    chat_node = ChatNode(args)
    rospy.spin()