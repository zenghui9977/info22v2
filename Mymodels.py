from torch import nn
import torch.nn.functional as F
from torchvision import models
import torchvision


class CNNMnist(nn.Module):
    def __init__(self):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(self.conv3(x))
        x = x.view(in_size, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class FMnist_model(nn.Module):
    def __init__(self):
        super(FMnist_model, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def Cifar_model(model_type, pre = False):
    # model type can be --> alexnet, vgg11, inception_v3, resnet18, resnet34, densenet121
    if model_type == 'vgg11':
        model = torchvision.models.vgg11(pretrained = pre)
    elif model_type == 'resnet18':
        model = torchvision.models.resnet18(pretrained = pre)
    elif model_type == 'resnet34':
        model = torchvision.models.resnet34(pretrained = pre)
    elif model_type == 'densenet121':
        model = torchvision.models.densenet121(pretrained = pre)

    return model
    