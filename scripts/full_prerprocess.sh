export PYTHONPATH=.
python scripts/preprocess_vinbigdata.py
python scripts/preprocess_nih.py
python scripts/preprocess_chestxdet.py
cat data/processed/vin_dataVOC2012/image_sets/train_vin.txt <(echo) data/processed/vin_dataVOC2012/image_sets/train_nih.txt <(echo) data/processed/vin_dataVOC2012/image_sets/train_chestxdet.txt | shuf > data/processed/vin_dataVOC2012/image_sets/train_full.txt
