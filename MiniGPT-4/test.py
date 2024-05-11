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


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

cudnn.benchmark = False
cudnn.deterministic = True

print('Initializing Chat')
args = parse_args()
cfg = Config(args)

device = 'cuda:{}'.format(args.gpu_id)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to(device)
bounding_box_size = 100

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

model = model.eval()

CONV_VISION = Conversation(
    system="",
    roles=(r"<s>[INST] ", r" [/INST]"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="",
)

chat = Chat(model, vis_processor, device=device)

max_new_tokens = 500
num_beams = 1
temperature = 0.75

image = './examples_v2/glip_test.jpg'
message = '''
Prompt: You are a self-driving robot in a household. You need to navigate to a place by selecting from three possible actions. You will be given an egocentric view of the environment in front of you. You will be given a user verbal command. First, identify what place the user wants you to go, then analyze where the goal is, and lastly, you should prioritize your current direction to line up with the goal over the distance. 
User command: “navigate to the tv”
Possible actions: “move straight”, “turn left”, “turn right”
Describe your thought process. Then select the most appropriate action. 
'''
# message = 'where is the TV?'
raw_image = Image.open(image).convert("RGB")
chat_state = CONV_VISION.copy()
img_list = []
llm_message = chat.upload_img(raw_image, chat_state, img_list)
chat.encode_img(img_list)
chat.ask(message, chat_state)

llm_message = chat.answer(
    conv=chat_state,
    img_list=img_list,
    max_new_tokens=max_new_tokens,
    num_beams=num_beams,
    temperature=temperature
)[0]

print("here's the answer: ")
print(llm_message)

# copilot
