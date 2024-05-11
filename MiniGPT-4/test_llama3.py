import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer

#hf_token = ""  # Replace with your actual token

#model_id = "Abira1/llama-2-7b-finance"
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
#model_id = "meta-llama/Llama-2-7b-chat-hf"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})
model.config.use_cache = True
model.eval()

messages = [
    {"role": "system", "content": 'You are a self-driving robot in a household. You need to navigate to a place by selecting from three possible actions: "move straight", "move forward and turn left", "move forward and turn right". You need to judge if the user command is achievable. If yes, plan the best route by selecting one of the actions. If not, reply why you cannot follow. '},
    {"role": "user", "content": "There are three chairs in front of you, a white wall closely to your left, and a door to your right."},
    {"role": "user", "content": "move to the left. "},
]

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
