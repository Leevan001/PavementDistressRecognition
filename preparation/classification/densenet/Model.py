from torchvision.models import densenet121
import torch
from torch.nn import Linear, Module

DenseNetModel = densenet121()
DenseNetModel.add_module('classifier', Linear(1024, 2))

if __name__ == '__main__':
    img = torch.ones((1, 3, 200, 300))
    print(DenseNetModel(img).shape)
    print(DenseNetModel)