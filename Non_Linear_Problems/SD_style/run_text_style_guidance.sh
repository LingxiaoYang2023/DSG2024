method="DSG"
task="style"
outdir="outputs/txt2img-samples/DSG_text_style_guidance"

CUDA_VISIBLE_DEVICES=0 python txt2img.py --prompt "A bird standing on the tree." --style_ref_img_path "./style_images/jojo.jpeg" \
--ddim_steps 100 --n_iter 1 --n_samples 1 --seed 2023 --H 512 --W 512 --scale 5.0 --method "$method" --task "$task" --outdir "$outdir" --ddim_eta 1 --guidance_rate 0.1

