# DSG For Solving Non-Linear Problems

## Installation

### Environment

The code is based on [Freedom](https://github.com/vvictoryuki/FreeDoM/tree/main). You can use the conda 
environment from [Stable Diffusion](https://github.com/CompVis/stable-diffusion).

### FaceID Guidance

Download pretrained model  `celebahq.ckpt` and `model_ir_se50.pth` following Freedom 
[(https://github.com/vvictoryuki/FreeDoM/tree/main/Face-GD)](https://github.com/vvictoryuki/FreeDoM/tree/main/Face-GD), 
and place it to `./exp/celebahq.ckpt` and `./exp/model_ir_se50.pth`.

### Style/Text-Style Guidance

Download pretrained model SD-v1-4 from [(https://github.com/vvictoryuki/FreeDoM/tree/main/SD_style)](https://github.com/vvictoryuki/FreeDoM/tree/main/SD_style)
following Freedom and place it to `models/ldm/stable-diffusion-v1/model.ckpt`.

## Quick Start

### FaceID Guidance

`cd Face-GD/` and run `bash run_faceid.sh`.

### Style Guidance

`cd SD-style/` and run `bash run_style_guidance.sh`.

### Text-Style Guidance

`cd SD-style/` and run `bash run_text_style_guidance.sh`.

