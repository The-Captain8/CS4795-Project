#!/bin/bash

python cli_override.py \
    --prompt "Determine the tic-tac-toe game state from the following image, provide the game state as a 3x3 matrix using the X, O, and _ characters" \
    --model-path liuhaotian/llava-v1.5-7b \
    --image-file ./training/augmented_images/image1.jpg \
    --load-4bit