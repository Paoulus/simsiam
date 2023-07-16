import pickle
import gzip
import matplotlib as pyplot
from torch.utils.data import TensorDataset
from torchvision.datasets import MNIST
import torch.utils.data

def prepare_mnist_dataset(path,transform,test=False):    
    mnist_dataset = MNIST(path,transform=transform,download=True,train=not test)
    train_length = int(len(mnist_dataset) * 0.8)
    val_length = len(mnist_dataset) - train_length
    return torch.utils.data.random_split(mnist_dataset,[train_length,val_length])
