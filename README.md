# VinBigData Chest X-ray Abnormalities Detection

## Brief description
Solution of [VinBigData Chest X-ray Abnormalities Detection](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/overview) that achieved 37 place out of 1277 participant. The main task was to construct an algorithm that can provide second opinion for radiologists about accurately identifing and localizing findings on chest X-rays.

Data: 15000 chest x-ray dicoms. Labels - 14 classes of abnormalities with bbox. ~3700 dicoms have some abnoramlities.
Metric: MAP@[IoU=0.4]


## My final results


## Solution overview



## Minimal hardware requirements
- 32 GB of RAM
- 1 NVIDIA GPU with at least 12 GB of GRAM
- 300 GB of disk space

## Hardware used
5-fold models training time is around 5 days with this configuration
- AMD Ryzen Threadripper 3960X 24 core
- 2 GPU - EVGA GeForce RTX 2080 TI FTW3 ULTRA HYBRID GAMING 12 GB
- 128 GB RAM
- Samsung 970 evo plus 2 TB


# Install and run instructions

## Repository structure

```text
data
|-raw               --- raw and exteranl data
|-processed         --- processed data
configs             --- configs for training models of detectorrs and scaled yolo nets, preprocessing and postprocessing scripts
dockerimage         --- docker image description and python dependencies
scripts             --- runnable scripts of pipeline
vinbigdata          --- python package with common code for this competition
results             --- folder with result models and prediction
.*                  --- config for linters and typing checkers
```

## Build and run container

Build image
```
docker build dockerimage/ -t cuda_image
pip install pre-commit
```
Run and enter to the container
```
docker run --gpus '"device=COMMA_SEPARATED_ID_LIST_OF_DEVICES"' --ipc=host -it -d -v /PATH_TO_LOCAL_REPOSITORY:/workdir -p 8888:8888 -p 6006:6006 cuda_image nohup jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root

docker exec -it RUNNING_CONTAINER_ID bash
```
If for you it's easy to work with notebooks every file inside scripts folder can be converted to jupyter notebook by running ipynb-py-convert {SCRIPT_NAME}.py {SCRIPT_NAME}.ipynb.

## Train pipeline
All commands of the pipeline should be executed inside container.
```
#download raw data
sh scripts/download.sh

# preprocess raw dicom and radiologists annotations to VOC 2012 format
sh scripts/full_prerprocess.sh

# Train models - you can skip this step and download trained models
bash scripts/train_models.sh {COMMA_SEPARATED_ID_LIST_OF_DEVICES} {BATCH_SIZE} {IMAGE_SHAPE} {EPOCH_NUMBER}
bash scripts/train_models.sh 0 3 1024 150 # run command for one GPU with 12 GB of VRAM

# Download final model for submission generation if you don't want to retrain it
sh scripts/download_pretrained_models.sh
```
## Evaluate and submit pipeline
All commands of the pipeline should be executed inside container.
```
# Evaluate models by MAP@[IoU=0.4]
PYTHONPATH=. python scripts/validate.py

# generate submission with output submition.csv file in results directory
PYTHONPATH=. python scripts/generate_submission.py
```
validate.py and generate_submission.py can be converted to jupyter notebook with additional visualization of results
