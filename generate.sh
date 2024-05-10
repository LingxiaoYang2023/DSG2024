#!/bin/bash

# 1. Choose the root of Dataset
data_root="./data/samples/"

# 2. Choose the task
task_config=("configs/inpainting_config.yaml" "configs/gaussian_deblur_config.yaml" "configs/super_resolution_config.yaml")
#task_config=("configs/inpainting_config.yaml")

for yaml_file in "${task_config[@]}"
do
    python generate_measurement.py --model_config configs/model_config.yaml --diffusion_config configs/diffusion_config.yaml --task_config "$yaml_file" --gpu 0 --data_root "$data_root"
done

