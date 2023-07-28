import argparse

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import data_utils.trueface_dataset as trueface_dataset
import simsiam
import matplotlib.pyplot as plot
import sklearn.cluster as clustering
import numpy as np

from main_simsiam import convert_and_plot, validate, pca_transform
from data_utils import augmentations as augmentations

parser = argparse.ArgumentParser(description='scatter plot of pretrained model')

parser.add_argument('--data', metavar='DIR', default=None, type=str,
                    help='path to dataset')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument("--checkpoint",default="",type=str)
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50')
parser.add_argument('--real-amount', default=None, type=int)
parser.add_argument('--fake-amount', default=None, type=int)
parser.add_argument('--batch-size', default=None, type=int)
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument("--image-size",type=int)
parser.add_argument("--crop-min",type=int)
parser.add_argument("--crop-max",type=int)
parser.add_argument("--output-filename",type=str,default="real_fake_scatter.png")

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

    print("=> loading checkpoint '{}'".format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    # rename moco pre-trained keys
    state_dict = checkpoint['state_dict']

    msg = model.load_state_dict(state_dict, strict=False)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]) 

    augmentation_convert = [
        transforms.ToTensor(),
        normalize
    ]

    augmentation_convert.insert(0,transforms.Resize(args.image_size))

    augmentation_presoc = []
    augmentation_presoc.append(augmentations.ResizeAtRandomLocationAndPad(args.crop_min,args.crop_max))
    
    # custom augmentation meant to simulate the changes applied by image post processing
    augmentation_presoc.extend([
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([simsiam.loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        augmentations.CompressToJPEGWithRandomParams(),
        transforms.Resize(args.image_size),
        transforms.ToTensor(),
        normalize
    ])

    test_dataset = trueface_dataset.TruefaceTotal(args.data,
                                                augmentations.ApplyDifferentTransforms(
                                                        transforms.Compose(augmentation_convert),
                                                        transforms.Compose(augmentation_presoc)
                                                    ),
                                                real_amount=args.real_amount,
                                                fake_amount=args.fake_amount)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=None,
        num_workers=args.workers, pin_memory=False, sampler=None, drop_last=True) 

    model.eval()
    model.cuda(args.gpu)
    
    criterion = nn.CosineSimilarity(dim=1).cuda(args.gpu)

    val_wandb_metrics, true_fv, fake_fv = validate(test_loader,model,criterion,args)
    fig, ax = plot.subplots()

    kmeans = clustering.KMeans(n_clusters=2).fit(np.concatenate((true_fv,fake_fv)))
    print("norm between the centroid clusters:")
    centroid_clusters_distance = np.linalg.norm(kmeans.cluster_centers_[1] - kmeans.cluster_centers_[0])
    print(centroid_clusters_distance)

    true_fv_transformed = pca_transform(true_fv,2)
    fake_fv_transformed = pca_transform(fake_fv,2)
    convert_and_plot(true_fv_transformed,'o',ax,'real')
    convert_and_plot(fake_fv_transformed,'x',ax,'fake')
    fig.suptitle("Feature fectors after PCA reduction")
    fig.legend()
    fig.text(0.1,0.01,"norm between centroid clusters: {:.3f}".format(centroid_clusters_distance))
    fig.savefig("{}_{}".format(args.image_size,args.output_filename))
    
if __name__ == '__main__':
    main()