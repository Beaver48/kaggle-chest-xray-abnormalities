export PYTHONPATH=.
python scripts/preprocess_vinbigdata.py
#python scripts/preprocess_nih.py
#python scripts/preprocess_chestxdet.py
#cat data/processed/vinbigdataVOC2012/image_sets/train_vin.txt <(echo) data/processed/vinbigdataVOC2012/image_sets/train_nih.txt <(echo) data/processed/vinbigdataVOC2012/image_sets/train_chestxdet.txt | shuf > data/processed/vinbigdataVOC2012/image_sets/train_full.txt
#cat data/processed/vinbigdataVOC2012/yolo_image_sets/train_vin.txt <(echo) data/processed/vinbigdataVOC2012/yolo_image_sets/train_nih.txt <(echo) data/processed/vinbigdataVOC2012/yolo_image_sets/train_chestxdet.txt | shuf > data/processed/vinbigdataVOC2012/yolo_image_sets/train_full.txt
