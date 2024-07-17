method="DSG"
task="style"
outdir="outputs/txt2img-samples/DSG_style_guidance"
image_paths=("./style_images/brice-marden_6-red-rock-1-2002.jpg")

for image_path in "${image_paths[@]}"; do
  # DSG
  CUDA_VISIBLE_DEVICES=0 python txt2img.py --prompt "a cat wearing glasses." --style_ref_img_path "$image_path" --ddim_steps 100 --n_iter 1 --n_samples 1 \
  --H 512 --W 512 --scale 5.0 --method "$method" --task "$task" --outdir "$outdir" --ddim_eta 1 --uncondition --guidance_rate 0.1
done