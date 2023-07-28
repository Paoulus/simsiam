#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import pathlib

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.utils

import simsiam.loader
import simsiam.builder

import data_utils.trueface_dataset as trueface_dataset
from data_utils.utils import AverageMeter, ProgressMeter
import data_utils.augmentations as augmentations
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plot

import wandb

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default=None, type=str,
                    help='path to dataset')
parser.add_argument('--train-prepost',action='store_true')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N',
                    help='mini-batch size (default: 512), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument("--save-checkpoints", action="store_true")
parser.add_argument('--real-amount', default=None, type=int)
parser.add_argument('--fake-amount', default=None, type=int)
parser.add_argument('--run-suffix', default="", type=str)
parser.add_argument("--save-dir",default="saved_checkpoints",type=str)
parser.add_argument("--image-size",default=None,type=int)
parser.add_argument("--augmentations", default="pad",choices=["pad","resize","identity"])
parser.add_argument("--crop-min",default=256,type=int)
parser.add_argument("--crop-max",default=512,type=int)

# simsiam specific configs:
parser.add_argument('--dim', default=2048, type=int,
                    help='feature dimension (default: 2048)')
parser.add_argument('--pred-dim', default=512, type=int,
                    help='hidden dimension of the predictor (default: 512)')
parser.add_argument('--fix-pred-lr', action='store_true',
                    help='Fix learning rate for the predictor')


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    # format for run name: dataset dataset_size batch_size suffix
    ds_size = args.real_amount + args.fake_amount
    
    ds_size_string = trueface_dataset.get_dataset_size_string(ds_size)

    dataset_compact_name = "Pre+post" if args.train_prepost else trueface_dataset.get_compact_name(args.data) 

    run_name = trueface_dataset.get_run_name(dataset_compact_name,
                                             ds_size_string,
                                             args.batch_size,
                                             args.run_suffix,
                                             args.lr)

    wandb.init(
        # set the wandb project where this run will be logged
        project="training-simsiam-trueface",
        
        name=run_name,

        # track hyperparameters and run metadata
        config={
        "architecture": "SimSiam",
        "optimizer":"SGD",
        "dataset": dataset_compact_name,
        "epochs": args.epochs,
        "batch_size":args.batch_size,
        "learning_rate": args.lr,
        "augmentations":args.augmentations,
        "real_samples_amount":args.real_amount,
        "fake_samples_amount":args.fake_amount,
        "is distributed":args.multiprocessing_distributed
        }
    )

    config = wandb.config

    if args.data == None and args.train_prepost == False:
        warnings.warn("No dataset speficied! Pass a path with data=PATH if you want to train on a single \
                      dataset slice with augmentation, or use --train_prepost to train with both pre and post slices")
        warnings.warn("Aborting since no dataset given")
        return

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args,config)
    
    wandb.finish()

def convert_and_plot(data,marker,axis,label):
    # adapt data for visualization on scatter plot
    x = []
    y = []
    for el in data:
        x.append(el[0])
        y.append(el[1])
    axis.scatter(x,y,marker=marker,label=label)

def main_worker(gpu, ngpus_per_node, args,config):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()
    # create model
    print("=> creating model '{}'".format(args.arch))
    model = simsiam.builder.SimSiam(
        models.__dict__[args.arch],
        args.dim, args.pred_dim)

    # infer learning rate before changing batch size
    init_lr = config["learning_rate"] * config["batch_size"] / 256
    #init_lr = config["learning_rate"]       # try with direct control of lr due to small batch size

    if args.distributed:
        # Apply SyncBN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            config["batch_size"] = int(config["batch_size"] / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        # raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    print(model) # print model after SyncBatchNorm

    # define loss function (criterion) and optimizer
    criterion = nn.CosineSimilarity(dim=1).cuda(args.gpu)

    if args.fix_pred_lr:
        optim_params = [{'params': model.module.encoder.parameters(), 'fix_lr': False},
                        {'params': model.module.predictor.parameters(), 'fix_lr': True}]
    else:
        optim_params = model.parameters()

    optimizer = torch.optim.SGD(optim_params, init_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    augmentation_convert = [
        transforms.ToTensor(),
        normalize
    ]

    if args.image_size != None:
        augmentation_convert.insert(0,transforms.Resize(args.image_size))

    augmentation_presoc = []
    if config["augmentations"] == "pad":
        augmentation_presoc.append(augmentations.ResizeAtRandomLocationAndPad(args.crop_min,args.crop_max))
    if config["augmentations"] == "resize":
        augmentation_presoc.append(transforms.RandomResizedCrop(512,(0.02,1.)))
    
    # custom augmentation meant to simulate the changes applied by image post processing
    augmentation_presoc.extend([
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([simsiam.loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        augmentations.CompressToJPEGWithRandomParams(),
        transforms.ToTensor(),
        normalize
    ])

    if args.image_size != None:
        augmentation_presoc.insert(6,transforms.Resize(args.image_size))

    print("Augmentations for second branch:")
    for aug in augmentation_presoc:
        print(aug)
    print("-"*8)


    total_dataset = None
    if args.train_prepost:
        total_dataset = trueface_dataset.PreAndPostDataset(
            ["Telegram","Facebook","Twitter","Whatsapp"],
            transforms.Compose(augmentation_convert),
            real_images_amount=args.real_amount,
            fake_images_amount=args.fake_amount
        )
    elif args.data != None:
        total_dataset = trueface_dataset.TruefaceTotal(
            args.data, augmentations.ApplyDifferentTransforms(
                transforms.Compose(augmentation_convert),
                transforms.Compose(augmentation_presoc)
            ),
            real_amount=args.real_amount,
            fake_amount=args.fake_amount
        )

    train_dataset, val_dataset = total_dataset.split_into_train_val(0.1)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=False, sampler=train_sampler, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,batch_size=4,num_workers=args.workers, pin_memory=False, drop_last=True)
    
    for epoch in range(args.start_epoch, config["epochs"]):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, init_lr, epoch, args, config)

        # train for one epoch
        train_wandb_metrics = train(train_loader, model, criterion, optimizer, epoch, args)

        # measure unsup loss on validation set
        val_wandb_metrics, true_fv, fake_fv = validate(val_loader,model,criterion,args)

        # put both metrics in the same database, as it's much cleaner if we log everything
        # with a single wandb.log() call
        train_wandb_metrics.update(val_wandb_metrics)

        print("Epoch metrics:")
        for key,value in train_wandb_metrics.items():
            print("{} : {}".format(key,value))
        print("-"*10)

        # scatter plot of validation transformed feature vectors
        fig, ax = plot.subplots()
        true_fv_transformed = pca_transform(true_fv,2)
        fake_fv_transformed = pca_transform(fake_fv,2)
        convert_and_plot(true_fv_transformed,'o',ax,"real")
        convert_and_plot(fake_fv_transformed,'x',ax,"fake")
        fig.suptitle("Feature fectors after PCA reduction")
        fig.legend()
        fig.savefig("real_fake_scatter.png")

        train_wandb_metrics.update({"validation/scatter_plot":wandb.Image(fig)})

        wandb.log(train_wandb_metrics)

        if (not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0)) and args.save_checkpoints:
            checkpoint_path = pathlib.Path(args.save_dir)
            checkpoint_path = checkpoint_path / 'checkpoint_{:04d}.pth.tar'.format(epoch)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename=checkpoint_path)

            if epoch == (config["epochs"] - 1):
                last_checkpoint_artifact = wandb.Artifact("last-checkpoint-model","model","Last checkpoint of the unsupervised training")
                last_checkpoint_artifact.add_file(checkpoint_path)
                wandb.log_artifact(last_checkpoint_artifact)



def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    corrects = 0
    for i, (images, labels,paths) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # print("Batch labels are {}".format(labels.numpy().tolist()))

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # compute output and loss
        p1, p2, z1, z2 = model(x1=images[0], x2=images[1])
        loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

        losses.update(loss.item(), images[0].size(0))

        if i == 1:
            torchvision.utils.save_image(images[0][0],"x0_{}_{}.png".format(i,epoch))
            torchvision.utils.save_image(images[1][0],"x1_{}_{}.png".format(i,epoch))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            progress.display(i)

    return {"train/mean_loss":losses.avg,}

def pca_transform(data,desired_components):
    pca = PCA(n_components=desired_components)
    pca.fit(data)

    result = pca.transform(data)
    return result

def validate(val_loader,model,criterion,args):
    model.eval()
    feature_vectors_true = torch.empty(0).cuda(args.gpu,non_blocking=True)
    feature_vectors_fake = torch.empty(0).cuda(args.gpu,non_blocking=True)
    with torch.no_grad():
        data_real = []
        data_fake = []
        validation_losses = AverageMeter("Val loss")
        for i , (images, labels, paths) in enumerate(val_loader):
            if args.gpu is not None:
                images[0] = images[0].cuda(args.gpu, non_blocking=True)
                images[1] = images[1].cuda(args.gpu, non_blocking=True)
            
            # compute output and loss
            p1, p2, z1, z2 = model(x1=images[0], x2=images[1])
            loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
            validation_losses.update(loss.item(),images[0].size(0))

            labels = labels.cuda(args.gpu,non_blocking=True)

            for index, el in enumerate(p1):
                if labels[index].item() == 0:
                    feature_vectors_true = torch.cat((feature_vectors_true,el),dim=0)
                else:
                    feature_vectors_fake = torch.cat((feature_vectors_fake,el),dim=0)

        pca_true_reshaped = np.reshape(feature_vectors_true.tolist(),(-1,2048))
        pca_fake_reshaped = np.reshape(feature_vectors_fake.tolist(),(-1,2048))

        metrics = {"validation/mean_loss":validation_losses.avg}
        return metrics , pca_true_reshaped, pca_fake_reshaped

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def adjust_learning_rate(optimizer, init_lr, epoch, args,config):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / config["epochs"]))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


if __name__ == '__main__':
    main()
