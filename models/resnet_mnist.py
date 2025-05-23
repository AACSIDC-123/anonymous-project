
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, bn=False, stride=1):
        super(ResBlock, self).__init__()
        self.bn = bn
        if bn:
            self.bn0 = nn.BatchNorm2d(in_planes)

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1)
        if bn:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1)
        self.shortcut = nn.Sequential()

        if stride > 1:
            self.shortcut = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1)

    def forward(self, x):

        if self.bn:
            out = F.leaky_relu(self.bn1(self.conv1(x)))
        else:
            out = F.leaky_relu(self.conv1(x))

        out = self.conv2(out)
        out += self.shortcut(x)
        return out

class ResBlock_s(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, level, hidden, num_class, bn=False, stride=1):
        super(ResBlock_s, self).__init__()
        self.bn = bn
        if bn:
            self.bn0 = nn.BatchNorm2d(in_planes)

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1)
        
        if bn:
            self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.shortcut = nn.Sequential()

        if stride > 1:
            self.shortcut = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1)

        if level == 1:
            self.fc1 = nn.Sequential(
                nn.Linear(25088, num_class),
            )
        if level ==2:
            self.fc1 = nn.Sequential(
                nn.Linear(12544, num_class),
            )
        if level ==3:
            self.fc1 = nn.Sequential(
                nn.Linear(3136, num_class),
            )
        if level ==4:
            self.fc1 = nn.Sequential(
                nn.Linear(2048, num_class),
            )

    def forward(self, x):
        if self.bn:
            out = F.leaky_relu(self.bn1(self.conv1(x)))
        else:
            out = F.leaky_relu(self.conv1(x))

        out = self.conv2(out)
        out += self.shortcut(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out
    
class ResNet4(nn.Module):
    def __init__(self, input_shape, level, hidden, num_classes):
        super(ResNet4, self).__init__()
        
        self.level = level
        net = []
        
        # Initial layers
        net += [nn.Conv2d(input_shape[0], 32, 3, 1, 1)]
        net += [nn.BatchNorm2d(32)]
        net += [nn.LeakyReLU()]

        # Add ResBlock layers based on the level
        net += [ResBlock(32, 32)]
        
        if level >= 2:
            net += [ResBlock(32, 64, stride=2)]
        if level >= 3:
            net += [ResBlock(64, 64, stride=2)]
        if level == 4:
            net += [ResBlock_s(64, 128, level=level, hidden=hidden, num_class=num_classes, stride=2)]
        elif level > 4:
            raise Exception('No level %d' % level)
        
        # Use nn.Sequential to define the network layers
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)

class ResNet3(nn.Module):
    def __init__(self, input_shape, level, hidden, num_classes):
        super(ResNet3, self).__init__()
        net = []
        net += [nn.Conv2d(input_shape[0], 32, 3, 1, 1)]
        net += [nn.BatchNorm2d(32)]
        net += [nn.LeakyReLU()]
        net += [ResBlock(32, 32)]
        if level >= 2:
            net += [ResBlock(32, 64, stride=2)]
        if level == 3:
            net += [ResBlock_s(64, 64, level=level, hidden=hidden, num_class=num_classes, stride=2)]
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)

class ResNet2(nn.Module):
    def __init__(self, input_shape, level, hidden, num_classes):
        super(ResNet2, self).__init__()
        net = []
        net += [nn.Conv2d(input_shape[0], 32, 3, 1, 1)]
        net += [nn.BatchNorm2d(32)]
        net += [nn.LeakyReLU()]
        net += [ResBlock(32, 32)]
        if level >= 2:
            net += [ResBlock_s(32, 64, level=level, hidden=hidden, num_class=num_classes, stride=2)]
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)

    
class ResNet1(nn.Module):
    def __init__(self, input_shape, level, hidden, num_classes):
        super(ResNet1, self).__init__()
        net = []
        net += [nn.Conv2d(input_shape[0], 32, 3, 1, 1)]
        net += [nn.BatchNorm2d(32)]
        net += [nn.LeakyReLU()]
        net += [ResBlock_s(32, 32, level=level, hidden=hidden, num_class=num_classes, stride=1)]
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class ClientResNet18Mnist(nn.Module):
    def __init__(self, level, hidden, num_classes):
        super(ClientResNet18Mnist, self).__init__()
        self.level = level

        if level == 1:
            self.model = ResNet1(input_shape=(1, 28, 28), level=level, hidden=hidden, num_classes=num_classes)
        elif level == 2:
            self.model = ResNet2(input_shape=(1, 28, 28), level=level, hidden=hidden, num_classes=num_classes)
        elif level == 3:
            self.model = ResNet3(input_shape=(1, 28, 28), level=level, hidden=hidden, num_classes=num_classes)
        elif level == 4:
            self.model = ResNet4(input_shape=(1, 28, 28), level=level, hidden=hidden, num_classes=num_classes)
        else:
            raise ValueError(f"No level {level} defined")

    def GetModel(self):
        return self.model
