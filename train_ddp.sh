# 2 GPUs
# Attention, the following batch size is on single GPU, not all GPUs.
python -m torch.distributed.launch --nproc_per_node=2 train.py \
                                                    --cuda \
                                                    -dist \
                                                    -d coco \
                                                    --root ./data/ \
                                                    -v fcos50 \
                                                    -lr 0.01 \
                                                    -lr_bk 0.01 \
                                                    --batch_size 4 \
                                                    --grad_clip_norm 4.0 \
                                                    --num_workers 4 \
                                                    --schedule 1x
#                                                     --sybn
                                                    # --mosaic
