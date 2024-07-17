# face
method="dsg"
#method="freedom"

CUDA_VISIBLE_DEVICES=0 python main.py -i arcface_face -s arc_ddim --doc celeba_hq --timesteps 100 --rho_scale 100.0 \
--seed 1234 --stop 100 --ref_path  ./images/00005.jpg --batch_size 1 --gpu 0 --method "$method"




