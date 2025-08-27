# CUDA_VISIBLE_DEVICES=5 python evaluate.py \
#   --data_root ../datasets/sa-1b_split/test \
#   --model slimsam_global \
#   --ckpt checkpoints/vit_b_slim_step2_.pth \
#   --eval_mode miou_point \
#   --out_dir ./eval_results/slimsam50_global_original_seed \
#   --viz_percent 5 \
#   --viz_metric miou

CUDA_VISIBLE_DEVICES=5 python evaluate.py \
  --data_root ../datasets/sa-1b_split/test \
  --model slimsam_global \
  --ckpt checkpoints/vit_b_slim_step2_minmax_r.pth \
  --eval_mode miou_point \
  --out_dir ./eval_results/slimsam50_global_minmax_conv \
  --viz_percent 5 \
  --viz_metric miou

# CUDA_VISIBLE_DEVICES=1 python evaluate.py \
#   --data_root ../datasets/sa-1b_split/test \
#   --model slimsam_global \
#   --ckpt checkpoints/vit_b_slim_step2_int8.pth \
#   --eval_mode miou_point \
#   --out_dir ./eval_results/slimsam50_global_dynamic \
#   --viz_percent 5 \
#   --viz_metric miou \
#   --device cpu