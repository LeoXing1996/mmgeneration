PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python tools/sample_nerf.py \
    configs/cips_3d/ffhq256.py \
    work_dirs/ckpts/CIPS-3D-weights/ffhq_cvt.pt \
    --traj circle_near_far \
    --num_frames 10 --return_nerf --device cpu
