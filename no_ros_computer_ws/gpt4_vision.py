import base64
import os
from enum import Enum
from io import BytesIO
from typing import Iterable, List, Literal, Optional, Union, Callable

import fitz
from dotenv import load_dotenv

# Instructor is powered by Pydantic, which is powered by type hints.
# Schema validation, prompting is controlled by type annotations
import instructor
import matplotlib.pyplot as plt
import pandas as pd
#from IPython.display import display
from PIL import Image
from openai import OpenAI
from pydantic import BaseModel, Field

import redis
from queue import Queue
import threading
import json


# Based on the image decide how many degrees should turn before move
def encode_image(image_path: str):
    # check if the image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found {image_path}")
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def display_images(image_data: dict):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    for i, (key, value) in enumerate(image_data.items()):
        img = Image.open(BytesIO(base64.b64decode(value)))
        ax = axs[i]
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(key)
    plt.tight_layout()
    plt.show()


class CommandInput(BaseModel):
    user_speach_command: str = Field(
        ...,
        description="The command that determines your high level objective, and the subsequent actions you will take."
    )


class HandleRedis:
    """
    Just ha helper class to handel redis and hold the connection
    In scenario where ros runs on same version as the gpt4 interaction
    ie python >= 3.9 then this is not needed as function are called directly

    Its only ment to exist one version of this class
    """
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(HandleRedis, cls).__new__(cls)
            # any initialization here
        return cls._instance

    def __init__(self):
        self.redis_client = redis.Redis(host="localhost", port=6379, db=0)
        self.llm_cmd_input_channel = "llm_cmd_input"
        self.robot_cmd_output_channel = "robot_cmd_output"

        self.newest_image_base64: str = ""

        # Thread-safe queue
        self.llm_input_data_queue = Queue()
        llm_thread = self.start(channel_name=self.llm_cmd_input_channel,
                                queue=self.llm_input_data_queue)
        self.safe_data = None

    def subscribe_and_queue(self, channel_name, queue: Queue):
        pubsub = self.redis_client.pubsub()
        pubsub.subscribe(channel_name)
        print(f"Subscribed to {channel_name}")
        for message in pubsub.listen():
            if message['type'] == 'message':
                data = json.loads(message["data"])
                queue.put(data)

    def start(self, channel_name, queue : Queue):
        thread = threading.Thread(target=self.subscribe_and_queue, args=(channel_name, queue, ))
        thread.start()
        return thread

    def wait_for_new_command(self):
        while True:
            data = self.llm_input_data_queue.get_nowait() if not self.llm_input_data_queue.empty() else None
            if data:
                break
        self.newest_image_base64 = data["image"]
        self.safe_data = data

    def get_new_command(self) -> CommandInput:
        return CommandInput(
            user_speach_command=self.safe_data["user_input"]
        )

    def send_robot_cmd(self, distance: float, angle: float = 0.0):
        data = {
            "distance": distance,
            "angle": angle
        }
        self.redis_client.publish(self.robot_cmd_output_channel, json.dumps(data))
        print(f"Data published to Redis channel {self.robot_cmd_output_channel}")


# Global variable used by rest
redis_handler = HandleRedis()
def get_newest_image() -> str:
    return redis_handler.newest_image_base64
def get_newest_command() -> CommandInput:
    return redis_handler.get_new_command()
def go_straight(distance: float):
    redis_handler.send_robot_cmd(distance=distance, angle=0.0)

def go_left(distance: float, turn: float):
    redis_handler.send_robot_cmd(distance=distance, angle=turn)

def go_right(distance: float, turn: float):
    redis_handler.send_robot_cmd(distance=distance, angle=turn)


def stop(distance: float, turn: float):
    redis_handler.send_robot_cmd(distance=0, angle=0)



    def __call__(self):
        # Blocking code, should it be here???
        #command_input : CommandInput = get_newest_command()
        if self.action == "go_straight":
            return go_straight(self.est_distance)
        if self.action == "go_left":
            return go_left(self.est_distance, self.est_turn)
        if self.action == "go_right":
            return go_right(self.est_distance, self.est_turn)
        if self.action == "do_not_move":
            return stop(0, 0)

class GoStraight(MovementFunctionCallBase):
    pass
class GoLeft(MovementFunctionCallBase):
    pass
class GoRight(MovementFunctionCallBase):
    pass

class DoNotMove(MovementFunctionCallBase):
    pass
class SugestBetterPath(MovementFunctionCallBase):
    pass

class QuestionFunctionCallBase:
    pass

class SceneObject(BaseModel):
    pass
    
class SceneDescription_input(BaseModel):
    """ The input into the VLM to get a scene description """
    image_description: Optional[str] = Field(
        ...,
        description="The detailed description of the robot environment image."
    )
class SceneDescription_output(BaseModel):
    """ The Scene Description Output generated by the LLM """
    scene_description: Optional[str] = Field(
        ...,
        description="The detailed description of the environment based on all the images you have seen."
    )




def llm_input_and_output():
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    MODEL = "gpt-4-turbo-2024-04-09"
    INSTRUCTION_PROMPT = ("As a self-driving robot with advanced vision capabilities, your primary tasks include "
                          "navigating to specified objects in images and analyzing the environment. Decide to move "
                          "left, right, forward, or use the 'do_not_move' function if your path is obstructed by an "
                          "obstacle like a wall, or if the target object is not visible. Always assess the path's "
                          "clarity, calculate the distance to the object, and determine the necessary adjustments to "
                          "align properly. Utilize all your tools to ensure safe and accurate navigation. If a direct "
                          "path isn't possible, use 'suggest_better_path' to propose an alternative route.")

    payload = {
        "model": MODEL,
        # When receiving a response from gpt4 Instructor validates the response against the specified response model
        "response_model": Iterable[Union[GoStraight, GoLeft, GoRight, DoNotMove, SugestBetterPath]],
        "tool_choice": "auto",  # automatically select the tool based on the context
        "temperature": 0.0,  # for less diversity in responses
        "seed": 123,  # Set a seed for reproducibility
    }

    payload["messages"] = [
        {
            "role": "user",
            "content": INSTRUCTION_PROMPT,
        },
        {
            "role": "user",
            "content": get_newest_command().user_speach_command
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{get_newest_image()}"
                    }
                },
            ],
        }
    ]
    function_calls = instructor.from_openai(
        OpenAI(api_key=openai_api_key), mode=instructor.Mode.PARALLEL_TOOLS
    ).chat.completions.create(**payload)
    for tool in function_calls:
        print(f"- Tool call: {tool.action}")

        print(f"- Parameters: {tool}")
        print(f">> Action result: {tool()}")



if __name__ == "__main__":
    while True:
        redis_handler.wait_for_new_command()
        llm_input_and_output()