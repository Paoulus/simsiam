import argparse
from pathlib import Path

import torch
from torchvision import transforms
from torchvision import utils
import numpy as np
import matplotlib.pyplot as plot

import data_utils.trueface_dataset as trueface_dataset
import data_utils.augmentations as augmentations
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.utils
import simsiam.loader
import simsiam.builder
import sklearn.cluster as cluster
import sklearn.metrics.pairwise as pairwise

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default=None, type=str,
                    help='path to dataset')
parser.add_argument("--checkpoint", type=str, help="path to checkpoint to use for feature vector stats")
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N',
                    help='mini-batch size (default: 512), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
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
parser.add_argument("--output", default="feature-vector-stats.png")

# simsiam specific configs:
parser.add_argument('--dim', default=2048, type=int,
                    help='feature dimension (default: 2048)')
parser.add_argument('--pred-dim', default=512, type=int,
                    help='hidden dimension of the predictor (default: 512)')
parser.add_argument('--fix-pred-lr', action='store_true',
                    help='Fix learning rate for the predictor')

def main():
    args = parser.parse_args()
    
    print("=> creating model '{}'".format(args.arch))
    model = simsiam.builder.SimSiam(
        models.__dict__[args.arch],
        args.dim, args.pred_dim)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    augmentation = [
        transforms.Resize(args.image_size),
        transforms.ToTensor(),
        normalize
    ]

    total_dataset = trueface_dataset.TruefaceTotal(
            args.data,
            augmentations.ApplyDifferentTransforms(
                transforms.Compose(augmentation),
                transforms.Compose(augmentation)
            ),
            real_amount=args.real_amount,
            fake_amount=args.fake_amount
        )
    
    _ , val_dataset = total_dataset.split_into_train_val(0.1)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,batch_size=args.batch_size,num_workers=args.workers, pin_memory=False, drop_last=True)
    

    model.cuda(args.gpu)
    model.eval()
    with torch.no_grad():
        feature_vectors = []
        labels_list = []
        for i , (images, labels, paths) in enumerate(val_loader):
            if args.gpu is not None:
                images[0] = images[0].cuda(args.gpu, non_blocking=True)
                images[1] = images[1].cuda(args.gpu, non_blocking=True)
            
            p1, p2, z1, z2 = model(x1=images[0], x2=images[1])
    
            assert p1.equal(p2) # sanity check: the two feature vectors should always be the same, as we're using the same transforms on both sides

            # 'unroll' the result of the image application to the model, and do the same for the labels
            for el in p1:
                feature_vectors.append(el.cpu().tolist())
            for label in labels:
                labels_list.append(label.cpu())

        fv_numpy = np.array(feature_vectors)
        labels_numpy = np.array(labels_list)

        # compute k means and kmeans centroids
        centroids = cluster.k_means(fv_numpy,n_clusters=2,n_init='auto')
        print(centroids)

        # compute feature vectors means
        means = np.mean(fv_numpy,axis=1)
        print(means)

        # compute euclidian distance between each f.v.
        distances = pairwise.euclidean_distances(fv_numpy)
        print(distances)

        fig, axs = plot.subplots(4,figsize=(14, 14))
        fig.suptitle('Feature vectors stats')

        distances_real=[]
        distances_fake=[]
        index_reals = []
        index_fake = []
        for index, row in enumerate(distances):
            if labels_numpy[index] == 0:
                distances_real.append(row[0])
                index_reals.append(index)
            else :
                distances_fake.append(row[0])
                index_fake.append(index)
        
        axs[0].scatter(distances_real,index_reals,marker='x',c='#dd2222',label='real')
        axs[0].scatter(distances_fake,index_fake,marker='o',c='#0909ee',label='fake')
        axs[0].set_title("Distances of other vectors from vector 0")
        axs[0].set_xlabel("distance")
        axs[0].set_ylabel("vector")
        axs[0].legend()

        means_real=[]
        means_fake=[]
        for index, el in enumerate(means):
            if labels_numpy[index] == 0:
                means_real.append(el)
            else :
                means_fake.append(el)

        axs[1].scatter(index_reals,means_real,marker='x',c='#dd2222',label='real')
        axs[1].scatter(index_fake,means_fake,marker='o',c='#0909ee',label='fake')
        axs[1].set_title("Means")
        axs[1].set_xlabel("vector")
        axs[1].set_ylabel("mean")
        axs[1].legend()
        
        for index, el in enumerate(centroids[0][0]):
            axs[2].bar(index,el)
        axs[2].set_title("Centroids of cluster 1")
        axs[2].set_xlabel("feature")
        axs[2].set_ylabel("value")

        
        for index, el in enumerate(centroids[0][1]):
            axs[3].bar(index,el)
        axs[3].set_title("Centroids of cluster 2")
        axs[3].set_xlabel("feature")
        axs[3].set_ylabel("value")

        fig.tight_layout()
        fig.savefig(args.output)


if __name__ == '__main__':
    main()
