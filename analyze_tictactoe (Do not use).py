# Author: Nathan McGugan
# This script is WIP and not working yet, I made this before I found out about the cli.py script in the LLaVA repo.
# cli_override.py is the same as cli.py but with some modifications to allow for command line prompting.

from transformers import AutoTokenizer
from llava import LlavaProcessor, LlavaModel
import torch
from PIL import Image

MODEL_NAME = 'liuhaotian/LLaVA-7b-delta-v1.5'
ADAPTER_PATH = 'checkpoints/llava-v1.5-7b-task-lora'

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

# Load the model
model = LlavaModel.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map='auto',
    offload_folder="offload/",
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)

# Prepare the prompt
prompt = (
    "<image>\nDetermine the tic-tac-toe game state from the following image, "
    "provide the game state as a 3x3 matrix using the X, O, and _ characters"
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate the output
output_ids = model.generate(
    input_ids=inputs["input_ids"],
    images="./training/augmented_images/image1.jpg",
    max_new_tokens=256,
    do_sample=True,
    temperature=0.7,
)

# Decode and print the output
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(output_text)