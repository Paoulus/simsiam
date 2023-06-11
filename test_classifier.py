import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

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

import data_utils.trueface_dataset as trueface_dataset

def main():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50')
    parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=4096, type=int,
                        metavar='N',
                        help='mini-batch size (default: 4096), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial (base) learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                        metavar='W', help='weight decay (default: 0.)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
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
    parser.add_argument('--finetuned',type=str)
    parser.add_argument('--real-amount', default=None, type=int)
    parser.add_argument('--fake-amount', default=None, type=int)
    parser.add_argument('--run-suffix', default="", type=str)

    args = parser.parse_args()

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](num_classes=2)

    print("=> loading checkpoint '{}'".format(args.finetuned))
    checkpoint = torch.load(args.finetuned, map_location="cpu")

    # rename moco pre-trained keys
    state_dict = checkpoint['state_dict']

    args.start_epoch = 0
    msg = model.load_state_dict(state_dict, strict=False)
#    assert set(msg.missing_keys)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]) 

    test_dataset = trueface_dataset.TruefaceTotal(args.data,
                                                  transforms.Compose([transforms.ToTensor(),normalize]),
                                                  real_amount=args.real_amount,
                                                  fake_amount=args.fake_amount)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=None,
        num_workers=args.workers, pin_memory=False, sampler=None, drop_last=True) 

    model.eval()
    model.cuda(args.gpu)

    corrects  = 0
    for i, (images, labels) in enumerate(test_loader):
        images = images.cuda(args.gpu)
        labels = labels.cuda(args.gpu)
        output = model(images)
        corrects += torch.sum(torch.eq(output.argmax(1),labels)).item()
    
    accuracy = corrects / len(test_dataset)
    print(accuracy)

if __name__ == '__main__':
    main()