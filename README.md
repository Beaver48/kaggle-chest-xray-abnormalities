# VinBigData Chest X-ray Abnormalities Detection

Solution of [VinBigData Chest X-ray Abnormalities Detection](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/overview) that achieved 37 place out of 1277 participant. The main task was to construct an algorithm that can provide second opinion for radiologists about accurately identifing and localizing findings on chest X-rays.

## Solution Overview


## 


## Minimal requirements
32 GB of RAM
1 GPU with at least 12 GB of GRAM
250 GB of disk

## Hardware used
AMD Ryzen Threadripper 3960X 24 core
2 cards - EVGA GeForce RTX 2080 TI FTW3 ULTRA HYBRID GAMING 12 GB
128 GB RAM
Samsung 970 evo plus 2 TB



# Install and run instructions

```
docker build dockerimage/ -t cuda_image
pip install pre-commit

docker run --gpus '"device=0"' --ipc=host -it -d -v /PATH_TO_REPOSITORY:/workdir -p 8888:8888 -p 6006:6006 cuda_image nohup jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root

```

### Download data

```
sh scripts/download.sh

```
### Download final models for submission if you don't want to retrain it

```
sh scripts/download_pretrained_models.sh

```

