from torchvision.datasets import CocoDetection,VOCDetection
from torch.utils.data import random_split
import torchvision.transforms as transforms


def prepare_coco_detection(root_dir,annotations_file,transform):
    coco = CocoDetection(root_dir,annotations_file,transform
                         )
    val_size = int(len(coco) * 0.2)
    train_size = len(coco) - val_size
    train, val = random_split(coco,[train_size,val_size])
    return train,val

def prepare_voc_detection(root_dir,set,transform):
    dataset = VOCDetection(root_dir, year='2007', 
                                      image_set=set, 
                                      download=True, 
                                      transform=transform)
    return dataset