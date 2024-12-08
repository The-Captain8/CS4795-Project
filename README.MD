# CS 4795: Tic-Tac-Toe Recognition Project

Author: Nathan McGugan

## Overview

This project is a neural network that can recognize tic-tac-toe boards from images, and a fine-tuned LLaVA model that can do the same.

## Files

- `NeuralNetwork.py`: Defines the neural network
- `evaluate.py`: Evaluates the performance of the neural network on the tic-tac-toe dataset.
- `evaluate_fine_tuned.py`: Evaluates the performance of a fine-tuned LLaVA model on the tic-tac-toe dataset.

## Fine-Tuning LLaVA

To fine-tune the LLaVA model:

1. Clone the LLaVA repository and CD into it:

```
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
```

2. Install the dependencies
3. Copy the `training/augmented_images/` and `training/augmented_annotations.json` to the LLaVA directory in a new folder called `training`
4. Run the following command to create the LORA weights:

```
deepspeed llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path liuhaotian/llava-v1.5-7b \
    --version v1 \
    --data_path ~/ImageAI/training/augmented_annotations.json \
    --image_folder ./ \
    --vision_tower openai/clip-vit-large-patch14 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-7b-task-lora-VT \
    --num_train_epochs 5 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
```

5. Merge the LORA weights with the base model by running:
```
./scripts/merge_lora_weights.py --model-base liuhaotian/llava-v1.5-7b --model-path ~/LLaVA/checkpoints/llava-v1.5-7b-task-lora-VT --save-model-path ./llava-newmodel
```

This will create a new model in the LLaVA directory called `llava-newmodel` which can be run with:

```
python -m llava.serve.cli \
    --model-path ./llava-newmodel \
    --image-file <image_path> \
    --load-4bit
```

Alternatively, you can run the `evaluate_fine_tuned.py` script to test the model on a directory of images and annotations. 
However you will need to modify the script since the model has a different path or just pass the path with `--model-path`.

## Neural Network

To train and evaluate the neural network, simply run `evaluate.py`.

The neural network uses a sigmoid activation function for the output layer and a ReLU activation function for the hidden layer. Some of the variables for the network can be changed when initiating the `NeuralNetwork` object, such as the input size, output shape, learning rate, number of hidden neurons, and number of iterations.

## Note

- The error rate for both the neural network and the fine-tuned LLaVA model is calculated by comparing each element of the predicted tic-tac-toe board to the expected tic-tac-toe board. If the predicted element is wrong, it is counted as an error. The total error is then divided by 9 to get a percentage error.
