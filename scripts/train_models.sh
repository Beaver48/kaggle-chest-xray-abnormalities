#!/bin/bash
for fold_id in {0..4}
do
PYTHONPATH=. python -m torch.distributed.launch --nproc_per_node 1 scripts/train_yolo.py --batch-size $2 --img $3 $3 --data configs/yolo/data_fold_$fold_id.yaml --cfg configs/yolo/yolov4-p6.yaml --weights /scaled_yolo_weights/yolov4-p6.pt --sync-bn --device $1 --hyp configs/yolo/yolo.yaml --logdir=runs --epoch=$4
done