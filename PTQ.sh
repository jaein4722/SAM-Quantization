# python PTQ.py \
#   --checkpoint checkpoints/vit_b_slim_step2_.pth \
#   --use-torch-load \
#   --patch-forward \
#   --method dynamic \
#   --output checkpoints/vit_b_slim_step2_int8.pth \
#   --verify-image images/truck.jpg \
#   --verify-point 750 370

python PTQ.py \
  --checkpoint checkpoints/vit_b_slim_step2_.pth \
  --use-torch-load --patch-forward \
  --method minmax \
  --include-conv \
  --output checkpoints/vit_b_slim_step2_minmax_W4.pth \
  --minmax-verbose \
  --calib-dir ../datasets/sa-1b_split/val \
  --calib-size 50 \
  --verify-image images/truck.jpg \
  --verify-point 750 370 \
  --w-bits 4 \
  --a-bits 4 \

# CUDA_VISIBLE_DEVICES=1 python PTQ.py \
#   --checkpoint checkpoints/vit_b_slim_step2_.pth \
#   --use-torch-load --patch-forward \
#   --method minmax \
#   --output checkpoints/vit_b_slim_step2_minmax_calib.pth \
#   --minmax-verbose \
#   --do-calib \
#   --calib-dir ../datasets/sa-1b_split/val \
#   --calib-size 50 \
#   --lambda-reg 1e-2 \
#   --verify-image images/truck.jpg \
#   --verify-point 750 370 \
#   --w-bits 4 \