import torch
import torchvision

if __name__=="__main__":
    train_set = torchvision.datasets.CIFAR10("CIFAR10", 'train', download=True)
    test_set = torchvision.datasets.CIFAR10("CIFAR10", 'test', download=True)