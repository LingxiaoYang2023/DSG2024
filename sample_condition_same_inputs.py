from functools import partial
import os
import argparse
import yaml
import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from data.dataloader import get_dataset, get_dataloader
from util.img_utils import clear_color, mask_generator
from util.logger import get_logger


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--diffusion_config', type=str)
    parser.add_argument('--task_config', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--step_size', type=float, default=1.0)
    parser.add_argument('--dps_step_size', type=float, default=None)
    parser.add_argument('--interval', type=int, default=1)
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--method', type=str, default='ours')
    parser.add_argument('--guidance_scale', type=float, default=0.1)
    parser.add_argument("--use_shortcut", action='store_true')
    parser.add_argument('--shortcut_start_time', type=int, default=400)
    args = parser.parse_args()

    # logger
    logger = get_logger()

    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)

    # Load configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)
    task_config['data']['root'] = args.data_root

    # assert model_config['learn_sigma'] == diffusion_config['learn_sigma'], \
    # "learn_sigma must be the same for model and diffusion configuartion."

    # Load model
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()

    # Prepare Operator and noise
    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config)

    # Prepare conditioning method
    cond_config = task_config['conditioning']
    print(f'args.step_size:{args.step_size} args.interval:{args.interval}')
    cond_config['params']['step_size'] = args.step_size
    cond_config['params']['interval'] = args.interval
    cond_config['method'] = args.method
    # if cond_config['method'] != 'ps':
    cond_config['params']['guidance_scale'] = args.guidance_scale
    if args.dps_step_size is not None:
        cond_config['params']['scale'] = args.dps_step_size

    cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
    measurement_cond_fn = cond_method.conditioning
    logger.info(f"Conditioning method : {task_config['conditioning']['method']}")

    sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn,
                        use_shortcut=args.use_shortcut, shortcut_start_time=args.shortcut_start_time)

    # Working directory
    out_path = os.path.join(args.save_dir, measure_config['operator']['name'])
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'label']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # Prepare dataloader
    data_config = task_config['data']
    transform = transforms.Compose([transforms.Resize([256, 256]), transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = get_dataset(**data_config, transforms=transform, load_inputs=True,
                          task_name=measure_config['operator']['name'], device=device)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)

    # Exception) In case of inpainting, we need to generate a mask
    if measure_config['operator']['name'] == 'inpainting':
        mask_gen = mask_generator(
            **measure_config['mask_opt']
        )

    distance_lists = []
    # all_images = []
    # Do Inference
    for i, (ref_img, y_n) in enumerate(loader):
        logger.info(f"Inference for image {i}")
        fname = str(i).zfill(5) + '.png'
        ref_img = ref_img.to(device)
        os.makedirs(os.path.join(out_path, 'progress', str(i).zfill(5)), exist_ok=True)

        if measure_config['operator']['name'] == 'inpainting':
            y_n, mask = y_n

            y_n = y_n.to(device)
            mask = mask.to(device)
            y_n = y_n.squeeze(0)
            mask = mask.squeeze(0)
            measurement_cond_fn = partial(cond_method.conditioning, mask=mask)
            sample_fn = partial(sample_fn, measurement_cond_fn=measurement_cond_fn)
        else:
            y_n = y_n.to(device)
            y_n = y_n.squeeze(0)

        # Sampling
        x_start = torch.randn(ref_img.shape, device=device).requires_grad_()

        sample, distance_list = sample_fn(x_start=x_start, measurement=y_n, record=False,
                                          save_root=os.path.join(out_path, 'progress', str(i).zfill(5)))

        if not measure_config['operator']['name'] == 'faceid':
            plt.imsave(os.path.join(out_path, 'input', fname), clear_color(y_n))
        else:
            torch.save(y_n, os.path.join(out_path, 'input', fname))
        plt.imsave(os.path.join(out_path, 'label', fname), clear_color(ref_img))
        plt.imsave(os.path.join(out_path, 'recon', fname), clear_color(sample))

        distance_lists.append(distance_list)


if __name__ == '__main__':
    main()
