import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import wandb

from pathlib import Path

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

from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np

import data_utils.trueface_dataset as trueface_dataset
from data_utils.mnist_dataset import prepare_mnist_dataset

def compute_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

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
    parser.add_argument("--image-size",default=1024,type=int)
    parser.add_argument("--save-precise-report",action="store_true")
    parser.add_argument("--report-unprocessed-paths",action="store_true")
    parser.add_argument("--conf-matrix",action="store_true")

    args = parser.parse_args()

    run_name = "testing {} {}"
    run_suffix = args.run_suffix
    run_name = run_name.format(args.batch_size, run_suffix)

    wandb.init(project="testing simsiam",
        name=run_name,

        # track hyperparameters and run metadata
            config={
            "learning_rate": args.lr,
            "architecture": "SimSiam",
            "optimizer":"SGD",
            "dataset": "MNIST",
            "epochs": args.epochs,
            "batch_size":args.batch_size,
            "real_samples_amount":args.real_amount,
            "fake_samples_amount":args.fake_amount,
            "finetuned model":args.finetuned
            }
        )

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](num_classes=10)

    print("=> loading checkpoint '{}'".format(args.finetuned))
    checkpoint = torch.load(args.finetuned, map_location="cpu")

    # rename moco pre-trained keys
    state_dict = checkpoint['state_dict']

    args.start_epoch = 0
    msg = model.load_state_dict(state_dict, strict=False)
#    assert set(msg.missing_keys)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]) 

    test_dataset,_ = prepare_mnist_dataset("../datasets/mnist",
                                                  transforms.Compose([
                                                    transforms.Lambda(lambda x : x.convert('RGB')),
                                                    transforms.ToTensor(),
                                                    normalize]),test=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=None,
        num_workers=args.workers, pin_memory=False, sampler=None, drop_last=True) 

    model.eval()
    model.cuda(args.gpu)

    y_true = torch.empty(0, dtype=torch.int).to(args.gpu)
    y_pred = torch.empty(0, dtype=torch.int).to(args.gpu)

    acc2_average = AverageMeter("Acc5",":6.2f")
    with open("output-testing.log","w") as output_testing:
        corrects  = 0
        for i, (elements) in enumerate(test_loader):
            images = elements[0]
            labels = elements[1]

            images = images.cuda(args.gpu)
            labels = labels.cuda(args.gpu)
            output = model(images)
            y_true = torch.cat((y_true,labels))
            labels_as_list = labels.tolist()
            if args.save_precise_report:
                for i in range(len(labels_as_list)):
                    pred = output.argmax(1).tolist()
                    """
                    processed_path = path[i]
                    if not args.report_unprocessed_paths:
                        processed_path = Path(processed_path)
                        processed_path = "/".join(processed_path.parts[5:])
                    print("{}, label {} output {}".format(processed_path,labels_as_list[i],pred[i]),file=output_testing)
                    """
            y_pred = torch.cat((y_pred,output.argmax(1)))
            corrects += torch.sum(torch.eq(output.argmax(1),labels)).item()
            acc1, acc2 = compute_accuracy(output,labels, (1,2))
            acc2_average.update(acc2[0],images.size(0))
        
        if args.conf_matrix: 
            cf_matrix = confusion_matrix(y_true.cpu().numpy(), y_pred.cpu().numpy())
            # do not create dataFrame if there are nan in the cf_matrix
            if cf_matrix.shape == (2,2):
                df_cm = pd.DataFrame((cf_matrix.T/np.sum(cf_matrix,axis=1)).T *100, index = [i for i in ['real','fake']],
                            columns = [i for i in ['real','fake']])
                print('Confusion_Matrix:\n {}\n'.format(df_cm))
                print('Confusion_Matrix:\n {}\n'.format(df_cm),file=output_testing)
    
    accuracy = corrects / len(test_dataset)
    print(accuracy)
    print(acc2_average.avg)

    test_stats = {
        "acc1":accuracy,
        "acc2":acc2_average.avg.item()
    }

    wandb.log(test_stats)

    wandb.finish()

if __name__ == '__main__':
    main()