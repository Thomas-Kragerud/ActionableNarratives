import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer

import rospy
from std_msgs.msg import String



class ChatNode:

    def __init__(self):
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        #model_id = "meta-llama/Llama-2-7b-chat-hf"
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":1})
        model.config.use_cache = True
        model.eval()

        self.tokenizer = tokenizer
        self.model = model

        rospy.init_node('analysis_node', anonymous=True)
        self.sub = rospy.Subscriber('/full_response', String, self.callback)
        self.user_sub = rospy.Subscriber('/user_input', String, self.call_back)

        self.user_command = ""

    def call_back(self, data):
        self.user_command = data.data

    def callback(self, data):
        messages = [
            # {"role": "system", "content": 'You are a self-driving robot in a household. You need to navigate to a place by selecting from three possible actions: "move straight", "move forward and turn left", "move forward and turn right". You need to judge if the user command is achievable. If yes, plan the best route by selecting one of the actions. If not, reply why you cannot follow. '},
            {"role": "system", "content": 'You are a self-driving robot in a household. You need to navigate to a place by selecting from four possible actions: "move straight", "move forward and turn left", "move forward and turn right", "stay still". You need to judge if the user command is achievable. If yes, plan the best route by selecting one of the actions. If no, you can stay still. Reply with your action first, then repond to the user in a dirty way. '},
            {"role": "user", "content": "{}".format(data.data)},
            {"role": "user", "content": "{}".format(self.user_command)},
        ]

        print(messages)
        
        tokenizer = self.tokenizer
        model = self.model
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = model.generate(
            input_ids,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        response = outputs[0][input_ids.shape[-1]:]
        print(tokenizer.decode(response, skip_special_tokens=True))


if __name__ == '__main__':
    chat_node = ChatNode()
    rospy.spin()