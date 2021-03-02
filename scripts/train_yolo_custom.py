# %%
import sys
sys.path.append('vinbigdata/scaled_yolov4/') # torch load bug fix
import argparse
import math
import os
import glob
import random
import time
from pathlib import Path

import numpy as np
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.optim.swa_utils import AveragedModel, SWALR
from vinbigdata.scaled_yolov4.models.yolo import Model
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from vinbigdata.scaled_yolov4.utils.datasets import create_dataloader
from vinbigdata.scaled_yolov4.utils.general import (check_anchors, check_file, check_img_size, compute_loss, fitness,
                           get_latest_run, increment_dir, labels_to_class_weights, labels_to_image_weights,
                           plot_images, plot_labels, plot_results, strip_optimizer,
                           torch_distributed_zero_first)
from vinbigdata.scaled_yolov4.test import evaluate
from vinbigdata.scaled_yolov4.utils.torch_utils import ModelEMA, init_seeds, intersect_dicts, select_device

from vinbigdata.utils import is_interactive

if is_interactive():
    get_ipython().magic('load_ext autoreload')
    get_ipython().magic('autoreload 2')

# %%
parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=str, default='/scaled_yolo_weights/yolov4-p6.pt', help='initial weights path')
parser.add_argument('--cfg', type=str, default='configs/yolo/yolov4-p6.yaml', help='model.yaml path')
parser.add_argument('--data', type=str, default='configs/yolo/data.yaml', help='data.yaml path')
parser.add_argument('--hyp', type=str, default='configs/yolo/yolo.yaml', help='hyperparameters path, i.e. data/hyp.scratch.yaml')
parser.add_argument('--epochs', type=int, default=150)
parser.add_argument('--swa_start', type=int, default=130, help='first epoch of swa')
parser.add_argument('--batch-size', type=int, default=3, help='total batch size for all GPUs')
parser.add_argument('--nominal-batch-size', type=int, default=64, help='nominal batch size with gradient accumulation')
parser.add_argument('--img-size', nargs='+', type=int, default=[1024, 1024], help='train,test sizes')
parser.add_argument('--rect', action='store_true', help='rectangular training')
parser.add_argument('--resume', action='store_true', help='resume training from optimizer conf')
parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
parser.add_argument('--logdir', type=str, default='runs/', help='logging directory')
parser.add_argument('-f', type=str, default='', help='miss this argument')
opt = parser.parse_args()

# %%
opt.resume = True
opt.weights = 'runs/exp85/weights/last.pt'

# %%
print(opt)
with open(opt.hyp) as f:
    hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps
log_dir = Path(increment_dir(Path(opt.logdir) / 'exp', opt.name))
tb_writer = SummaryWriter(log_dir=log_dir)
(log_dir / 'weights').mkdir(exist_ok=True)
assert opt.swa_start < opt.epochs

# %%
device = torch.device('cuda')
with open(opt.data) as reader:
    data_dict = yaml.load(reader, Loader=yaml.FullLoader)  # model dict

loaded_checkpoint = torch.load(opt.weights, map_location=device)# to FP32
num_classes, class_names = (int(data_dict['nc']), data_dict['names'])
model = Model(opt.cfg, ch=3, nc=num_classes).to(device)
model.nc = num_classes 
model.names = class_names
state_dict = loaded_checkpoint['model'].float().state_dict()
state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=['anchor'])
print(model.load_state_dict(state_dict, strict=False))
print('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), opt.weights))
max_stride = int(max(model.stride))
if opt.resume:
    start_epoch = 0 if loaded_checkpoint['epoch'] == -1 else loaded_checkpoint['epoch'] + 1
    best_fitness = loaded_checkpoint['best_fitness']
else:
    start_epoch = 0
    best_fitness = 0.0
model = torch.nn.DataParallel(model)

if opt.sync_bn and device.type != 'cpu':
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

# %%
imgsz, imgsz_test = [check_img_size(x, max_stride) for x in opt.img_size]

train_dataloader, train_dataset = create_dataloader(
    path=data_dict['train'],
    imgsz=imgsz,
    batch_size=opt.batch_size,
    stride=max_stride,
    opt=opt,
    hyp=hyp,
    augment=True,
    cache=opt.cache_images,
    pad=0.5,
    rect=opt.rect)

test_dataloader, test_dataset = create_dataloader(
    path=data_dict['val'],
    imgsz=imgsz_test,
    batch_size=opt.batch_size,
    stride=max_stride,
    opt=opt,
    hyp=hyp,
    augment=False,
    cache=opt.cache_images,
    pad=0.5,
    rect=True)

# %%
nbs = opt.nominal_batch_size
accumulate = max(round(nbs / opt.batch_size), 1)  # accumulate loss before optimizing
hyp['weight_decay'] *= opt.batch_size * accumulate / nbs  # scale weight_decay

pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
for k, v in model.named_parameters():
    v.requires_grad = True
    if '.bias' in k:
        pg2.append(v)  # biases
    elif '.weight' in k and '.bn' not in k:
        pg1.append(v)  # apply weight decay
    else:
        pg0.append(v)  # all else

if opt.adam:
    optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
else:
    optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
if opt.resume and loaded_checkpoint['optimizer'] is not None:
    optimizer.load_state_dict(loaded_checkpoint['optimizer'])
print('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
del pg0, pg1, pg2

# %%
lf = lambda x: (((1 + math.cos(x * math.pi / opt.swa_start)) / 2)**1.0) * 0.8 + 0.2  # cosine
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
scheduler.last_epoch = max(start_epoch - 1, 0)
swa_scheduler = SWALR(optimizer, swa_lr=hyp['lr0'] / 5)

# %%
ema_alpha = 0.9999
ema_avg_func = lambda averaged_model_parameter, model_parameter, num_averaged: \
                    (1 - ema_alpha) * averaged_model_parameter + ema_alpha * model_parameter
swa_model = AveragedModel(model.module).to(device)#, avg_fn=ema_avg_func)

# %%
if not opt.noautoanchor:
    check_anchors(train_dataset,
                  model=model, 
                  thr=hyp['anchor_t'], 
                  imgsz=imgsz)

# Start training
t0 = time.time()
nw = max(3 * len(train_dataloader), 1e3)  # number of warmup iterations, max(3 epochs, 1k iterations)
maps = np.zeros(num_classes)  # mAP per class
results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
scaler = amp.GradScaler(enabled=True)
print('Image sizes %g train, %g test' % (imgsz, imgsz_test))
print('Using %g dataloader workers' % train_dataloader.num_workers)
print('Starting training for %g epochs...' % opt.epochs)

# %%
hyp['cls'] *= num_classes / 80.  # scale coco-tuned hyp['cls'] to current dataset
model.nc = num_classes  # attach number of classes to model
model.hyp = hyp  # attach hyperparameters to model
model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
model.class_weights = labels_to_class_weights(train_dataset.labels, num_classes).to(device)  # attach class weights
model.names = class_names

# %%
def train_epoch(model, train_dataloader, optimize=True):
    model.train()
    mloss = torch.zeros(4, device=device)  # mean losses
    pbar = enumerate(train_dataloader)
    print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))
    pbar = tqdm(pbar, total=len(train_dataloader))  # progress bar
    optimizer.zero_grad()
    for i, (imgs, targets, paths, _) in pbar:  
        ni = i + len(train_dataloader) * epoch  # number integrated batches (since train start)
        imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

        # Warmup
        if ni <= nw:
            xi = [0, nw]  # x interp
            # model.gr = np.interp(ni, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
            accumulate = max(1, np.interp(ni, xi, [1, opt.nominal_batch_size / opt.batch_size]).round())
            for j, x in enumerate(optimizer.param_groups):
                # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                if 'momentum' in x:
                    x['momentum'] = np.interp(ni, xi, [0.9, hyp['momentum']])
        else:
            accumulate = max(round(nbs / opt.batch_size), 1)
        # Autocast
        with amp.autocast(enabled=True):
            # Forward
            pred = model(imgs)
            loss, loss_items = compute_loss(pred, targets.to(device), model)

        # Backward
        scaler.scale(loss).backward()

        # Optimize
        if ni % accumulate and optimize:
            scaler.step(optimizer)  # optimizer.step
            scaler.update()
            optimizer.zero_grad()
            #if ema is not None:
            #    ema.update(model)
    
        mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
        mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
        s = ('%10s' * 2 + '%10.4g' * 6) % ('%g/%g' %
                                           (epoch, opt.epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
        pbar.set_description(s)

        # Plot
        if ni < 3:
            f = str(Path(log_dir) / ('train_batch%g.jpg' % ni))  # filename
            result = plot_images(images=imgs, targets=targets, paths=paths, fname=f)
            if tb_writer and result is not None:
                tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
    return mloss

# %%
for epoch in range(start_epoch, opt.epochs): 
    mloss = train_epoch(model, train_dataloader)
    if epoch >= opt.swa_start:
        if epoch == opt.swa_start:
            swa_model = AveragedModel(model)
        swa_model.update_parameters(model.module)
        swa_scheduler.step()
    else:
        scheduler.step()
    
    results, maps, times = evaluate(
            opt.data,
            task='val',
            batch_size=opt.batch_size,
            imgsz=imgsz_test,
            save_json=False,
            model=model,
            dataloader=test_dataloader,
            save_dir=log_dir,
            verbose=True)
    
    if tb_writer:
        tags = [
            'train/giou_loss', 'train/obj_loss', 'train/cls_loss', 'metrics/precision', 'metrics/recall',
            'metrics/mAP_0.4', 'metrics/mAP_0.4:0.95', 'val/giou_loss', 'val/obj_loss', 'val/cls_loss'
        ]
        for x, tag in zip(list(mloss[:-1]) + list(results), tags):
            tb_writer.add_scalar(tag, x, epoch)
    
    fi = fitness(np.array(results).reshape(1, -1))  # fitness_i = weighted combination of [P, R, mAP, F1]
    if fi > best_fitness:
        best_fitness = fi
    
    if not opt.nosave:
        checkpoint = {
            'best_fitness': best_fitness,
            'epoch': epoch,
            'model': model.module,
            'optimizer': optimizer.state_dict()
        }
        # Save last, best and delete
        torch.save(checkpoint, log_dir / 'weights/last.pt')
        if epoch >= opt.swa_start:
            torch.save(checkpoint, str(log_dir / 'weights/last.pt').replace('.pt', '_{:03d}.pt'.format(epoch)))
        if best_fitness == fi:
            torch.save(checkpoint, log_dir / 'weights/best.pt')
        del checkpoint

# %%
@torch.no_grad()
def update_bn(loader, model, device=None):
    r"""Updates BatchNorm running_mean, running_var buffers in the model.
    It performs one pass over data in `loader` to estimate the activation
    statistics for BatchNorm layers in the model.
    Args:
        loader (torch.utils.data.DataLoader): dataset loader to compute the
            activation statistics on. Each data batch should be either a
            tensor, or a list/tuple whose first element is a tensor
            containing data.
        model (torch.nn.Module): model for which we seek to update BatchNorm
            statistics.
        device (torch.device, optional): If set, data will be transferred to
            :attr:`device` before being passed into :attr:`model`.
    Example:
        >>> loader, model = ...
        >>> torch.optim.swa_utils.update_bn(loader, model) 
    .. note::
        The `update_bn` utility assumes that each data batch in :attr:`loader`
        is either a tensor or a list or tuple of tensors; in the latter case it 
        is assumed that :meth:`model.forward()` should be called on the first 
        element of the list or tuple corresponding to the data batch.
    """
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0
swa_model = swa_model.module
update_bn(train_dataloader, swa_model, device=device)
epoch= 150
swa_model.nc = num_classes
swa_model.hyp = hyp
swa_model.names = class_names
swa_model.gr = 1.0
swa_model.avg_fn = None
mloss = train_epoch(swa_model, train_dataloader, optimize=False)
checkpoint = {
    'epoch': -1,
    'model': swa_model,
    'optimizer': None
}

results, maps, times = evaluate(
            opt.data,
            task='val',
            batch_size=opt.batch_size,
            imgsz=imgsz_test,
            save_json=False,
            model=swa_model,
            dataloader=test_dataloader,
            save_dir=log_dir,
            verbose=True)
torch.save(checkpoint, log_dir / 'weights/swa.pt')

# %%
#models = glob.glob('runs/exp91/weights/last_*')
#swa_model = AveragedModel(torch.load(models[0], map_location=device)['model'])
#for model_name in tqdm(models[1:]):
#    swa_model.update_parameters(torch.load(model_name, map_location=device)['model'])
#torch.cuda.empty_cache()