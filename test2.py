import os
import torch
import shutil
import numpy as np
from torchvision import datasets, transforms
import csv




apply_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.4, 0.4, 0.4], [0.2, 0.2, 0.2])
])
GTSRB_data = datasets.ImageFolder('./data/GTSRB/train', transform=apply_transform)
GTSRB_test_data = datasets.ImageFolder('./data/GTSRB/test', transform=apply_transform)

print(GTSRB_data)
print(GTSRB_data.classes)
print(GTSRB_data[0][0])
print(GTSRB_data[0][0].size())
print(len(GTSRB_data))

print(len(GTSRB_test_data))



if torch.is_tensor(GTSRB_data.targets):
    labels = GTSRB_data.targets.numpy()
else:
    labels = np.array(GTSRB_data.targets)

labels = np.array(labels)
print(len(labels))