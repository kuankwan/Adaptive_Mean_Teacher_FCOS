#voc
python train_mt.py --cuda -d voc --num_workers 2 -v fcos50 -lr 0.01 -lr_bk 0.01 --batch_size 2 --schedule 1x --grad_clip_norm 4.0


#coco
python train_mt.py --cuda -d coco --num_workers 2 -v fcos50 -lr 0.01 -lr_bk 0.01 --batch_size 2 --schedule 1x --grad_clip_norm 4.0


#coco
python train_mt.py --cuda -d coco --root /root/autodl-tmp --num_workers 2 -v fcos50 -lr 0.01 -lr_bk 0.01 --batch_size 2 --schedule 1x --grad_clip_norm 4.0