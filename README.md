# [ICML 2024] Guidance with Spherical Gaussian Constraint for Conditional Diffusion

Code release for "Guidance with Spherical Gaussian Constraint for Conditional Diffusion(DSG)". 

[[paper]](https://arxiv.org/abs/2402.03201)

![1](/figures/overview.jpg)

The code implementation is based on [https://github.com/DPS2022/diffusion-posterior-sampling](https://github.com/DPS2022/diffusion-posterior-sampling). 
This version only includes the Linear Inverse Problems; the code for Non-linear Problems is coming soon.

## 1) Set environment

Install dependencies:

```
conda create -n DSG python=3.8

conda activate DSG

pip install -r requirements.txt

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

```

## 2) Download checkpoints

Download checkpoint `ffhq_10m.pt` or `imagenet256.pt`  from https://github.com/DPS2022/diffusion-posterior-sampling and paste it to `./models/`. 


## 3) Generate noisy measurement

You could modify the parameters following the comment in `generate.sh` and run it.

```
bash generate.sh
```

## 4) Inference

You could modify the parameters following the comment in `run_DSG.sh` and run it using the hyperparameter in Table 3 in the Appendix of paper. 
The results are shown in `./total_results_DSG_DDIM"$DDIM"/DSG_interval_${interval}_ guidance_${guidance_scale}/{TASK}/recon`.

```
bash run_DSG.sh
```

## 5) Test in FFHQ/Imagenet

1. Change the **data_root** in `generate.sh` and `run_DSG.sh`.

2. Change the **self.fpaths** in `/data/dataloader.py`

    e.g. if image is `jpg` format, change it to:

```
self.fpaths = sorted(glob(root + '/*.jpg', recursive=False))
```

3. Change the **model config** in `run_DSG.sh`.

## 6) Citation

If you find our work useful in your research, please consider citing

```
@inproceedings{
yang2023dsg,
title={Guidance with Spherical Gaussian Constraint for Conditional Diffusion},
author={Lingxiao Yang and Shutong Ding and Yifan Cai and Jingyi Yu and Jingya Wang and Ye Shi},
booktitle={International Conference on Machine Learning},
year={2024}
}
```



