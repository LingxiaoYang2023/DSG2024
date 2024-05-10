#!/bin/bash


#1. Choose The Task
#task_config=("configs/super_resolution_config.yaml" "configs/inpainting_config.yaml" "configs/gaussian_deblur_config.yaml")
task_config=("configs/inpainting_config.yaml")

#2. Choose the root of dataset
data_root="./data/samples/"

#3. Choose the hyperparameters
guidance_scales=(0.2)
intervals=(1)

#4. Choose the DDIM steps and GPU
DDIM=100
GPU=0

#5. Choose the diffusion model config
config="model_config.yaml"
#config="imagenet_model_config.yaml"

save_root="total_results_DSG_DDIM"$DDIM

for interval in "${intervals[@]}"; do
  for guidance_scale in "${guidance_scales[@]}"; do
    save_dir="./"$save_root"/DSG_interval_${interval}_guidance_${guidance_scale}"
    for yaml_file in "${task_config[@]}"; do
      python sample_condition_same_inputs.py --model_config configs/"$config" --diffusion_config configs/diffusion_ddim"$DDIM"_config.yaml --task_config "$yaml_file" --gpu "$GPU" --interval="$interval" --save_dir="$save_dir" --method "DSG" --data_root "$data_root" --guidance_scale "$guidance_scale"
    done
  done
done
