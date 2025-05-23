
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
                nn.Linear(32768, num_class),
            )
        if level == 2:
            self.fc1 = nn.Sequential(
                nn.Linear(16384, num_class),
            )
        if level == 3:
            self.fc1 = nn.Sequential(
                nn.Linear(16384, num_class),
            )
        if level == 4:
            self.fc1 = nn.Sequential(
                nn.Linear(8192, num_class),
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

def resnet4(input_shape, level, hidden, num_classes):
    net = []

    net += [nn.Conv2d(input_shape[0], 32, 3, 1, 1)]
    net += [nn.BatchNorm2d(32)]
    net += [nn.LeakyReLU()]
    net += [ResBlock(32, 32)]

    if level == 1:
        return nn.Sequential(*net)

    net += [ResBlock(32, 64, stride=2)]

    if level == 2:
        return nn.Sequential(*net)
    
    net += [ResBlock(64, 64)]

    if level == 3:
        return nn.Sequential(*net)

    net += [ResBlock_s(64, 128, level=level,  hidden=hidden, num_class=num_classes, stride=2)]

    if level <= 4:
        return nn.Sequential(*net)
    else:
        raise Exception('No level %d' % level)

def resnet3(input_shape, level, hidden, num_classes):
    net = []

    net += [nn.Conv2d(input_shape[0], 32, 3, 1, 1)]
    net += [nn.BatchNorm2d(32)]
    net += [nn.LeakyReLU()]
    net += [ResBlock(32, 32)]

    if level == 1:
        return nn.Sequential(*net)

    net += [ResBlock(32, 64, stride=2)]

    if level == 2:
        return nn.Sequential(*net)
    
    net += [ResBlock_s(64, 64, level=level,  hidden=hidden, num_class=num_classes)]

    if level == 3:
        return nn.Sequential(*net)

def resnet2(input_shape, level, hidden, num_classes):
    net = []
    net += [nn.Conv2d(input_shape[0], 32, 3, 1, 1)]
    net += [nn.BatchNorm2d(32)]
    net += [nn.LeakyReLU()]
    net += [ResBlock(32, 32)]

    if level == 1:
        return nn.Sequential(*net)

    net += [ResBlock_s(32, 64, level=level,  hidden=hidden, num_class=num_classes,stride=2)]

    if level == 2:
        return nn.Sequential(*net)
    
def resnet1(input_shape, level, hidden, num_classes):
    net = []
    net += [nn.Conv2d(input_shape[0], 32, 3, 1, 1)]
    net += [nn.BatchNorm2d(32)]
    net += [nn.LeakyReLU()]
    net += [ResBlock_s(32, 32, level=level,  hidden=hidden, num_class=num_classes, stride=1)]

    return nn.Sequential(*net)
    
class ClientResNet18(nn.Module):
    def __init__(self, level, hidden, num_classes):
        super(ClientResNet18, self).__init__()
        self.level = level

        if level == 1:
            self.model = resnet1(input_shape=(3, 32, 32), level=level, hidden=hidden, num_classes=num_classes)
        elif level == 2:
            self.model = resnet2(input_shape=(3, 32, 32), level=level, hidden=hidden, num_classes=num_classes)
        elif level == 3:
            self.model = resnet3(input_shape=(3, 32, 32), level=level, hidden=hidden, num_classes=num_classes)
        elif level == 4:
            self.model = resnet4(input_shape=(3, 32, 32), level=level, hidden=hidden, num_classes=num_classes)
        else:
            raise ValueError(f"No level {level} defined")

    def GetModel(self):
        return self.model


# Server_ResNet_leakyRelu for two clients
class Server_ResNet_cat(nn.Module):
    def __init__(self, hidden2=128, num_classes=2):
        super(Server_ResNet_cat, self).__init__()
        act = nn.LeakyReLU
        self.fc1 = nn.Sequential(
            nn.Linear(hidden2, hidden2),
            act(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden2, 1024),
            act(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(1024, 512),
            act(),
        )
        self.fc4 = nn.Sequential(
            nn.Linear(512, 256),
            act(),
        )
        self.fc5 = nn.Sequential(
            nn.Linear(256, 128),
            act(),
        )
        self.fc6 = nn.Sequential(
            nn.Linear(128, num_classes),
        )

    def forward(self, x1, x2):
        x= torch.cat([x1, x2], dim=1)
        out1 = self.fc1(x)
        out2 = self.fc2(out1)
        out3 = self.fc3(out2)
        out4 = self.fc4(out3)
        out5 = self.fc5(out4)
        out6 = self.fc6(out5)

        return out5, out6

# Server_ResNet_leakyRelu for 4 clients
class Server_ResNet_cat_4clients(nn.Module):
    def __init__(self, hidden=128, num_classes=2):
        super(Server_ResNet_cat_4clients, self).__init__()
        act = nn.LeakyReLU

        self.fc1 = nn.Sequential(
            nn.Linear(hidden, hidden),
            act(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden, 1024),
            act(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(1024, 512),
            act(),
        )
        self.fc4 = nn.Sequential(
            nn.Linear(512, 256),
            act(),
        )
        self.fc5 = nn.Sequential(
            nn.Linear(256, 128),
            act(),
        )
        self.fc6 = nn.Sequential(
            nn.Linear(128, num_classes),
        )

    def forward(self, x1, x2, x3, x4):
        x = torch.cat([x1, x2, x3, x4], dim=1)
        out1 = self.fc1(x)
        out2 = self.fc2(out1)
        out3 = self.fc3(out2)
        out4 = self.fc4(out3)
        out5 = self.fc5(out4)
        out6 = self.fc6(out5)

        return out5, out6
# 2clients
class Server_ResNet_sum(nn.Module):
    def __init__(self,  hidden2=128, num_classes=2):
        super(Server_ResNet_sum, self).__init__()
        act = nn.LeakyReLU
        self.fc2 = nn.Sequential(
            nn.Linear(hidden2, hidden2),
            act(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(hidden2, 1024),
            act(),
        )
        self.fc4 = nn.Sequential(
            nn.Linear(1024, 512),
            act(),
        )
        self.fc5 = nn.Sequential(
            nn.Linear(512, 256),
            act(),
        )
        self.fc6 = nn.Sequential(
            nn.Linear(256, 128),
            act(),
        )
        self.fc7 = nn.Sequential(
            nn.Linear(128, num_classes),
        )

    
    def forward(self, x1, x2):
        x= (x1+ x2)/2
        out1 = self.fc2(x)
        out2 = self.fc3(out1)
        out3 = self.fc4(out2)
        out4 = self.fc5(out3)
        out5 = self.fc6(out4)
        out6 = self.fc7(out5)

        return out5, out6

# 4clients
class Server_ResNet_sum_4clients(nn.Module):
    def __init__(self, hidden=128, num_classes=2):
        super(Server_ResNet_sum_4clients, self).__init__()
        act = nn.LeakyReLU
        self.fc1 = nn.Sequential(
            nn.Linear(hidden, hidden),
            act(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden, 1024),
            act(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(1024, 512),
            act(),
        )
        self.fc4 = nn.Sequential(
            nn.Linear(512, 256),
            act(),
        )
        self.fc5 = nn.Sequential(
            nn.Linear(256, 128),
            act(),
        )
        self.fc6 = nn.Sequential(
            nn.Linear(128, num_classes),
        )

    def forward(self, x1, x2, x3, x4):
        x= (x1 + x2 + x3 + x4)/4
        out1 = self.fc1(x)
        out2 = self.fc2(out1)
        out3 = self.fc3(out2)
        out4 = self.fc4(out3)
        out5 = self.fc5(out4)
        out6 = self.fc6(out5)

        return out5, out6

#2clients
class Server_ResNet_standalone(nn.Module):
    def __init__(self, hidden2=128, num_classes=2):
        super(Server_ResNet_standalone, self).__init__()
        act = nn.LeakyReLU
        self.fc2 = nn.Sequential(
            nn.Linear(hidden2, hidden2),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(hidden2, 1024),
            act(),
        )
        self.fc4 = nn.Sequential(
            nn.Linear(1024, 512),
            act(),
        )
        self.fc5 = nn.Sequential(
            nn.Linear(512, 256),
            act(),
        )
        self.fc6 = nn.Sequential(
            nn.Linear(256, 128),
            act(),
        )
        self.fc7 = nn.Sequential(
            nn.Linear(128, num_classes),
        )

    def forward(self, x1, x2):
        out1 = self.fc2(x1)
        out2 = self.fc3(out1)
        out3 = self.fc4(out2)
        out4 = self.fc5(out3)
        out5 = self.fc6(out4)
        out6 = self.fc7(out5)

        return out5, out6

#4clients
class Server_ResNet_standalone_4clients(nn.Module):
    def __init__(self, hidden=128, num_classes=2):
        super(Server_ResNet_standalone_4clients, self).__init__()
        act = nn.LeakyReLU
        self.fc1 = nn.Sequential(
            nn.Linear(hidden, hidden),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden, 1024),
            act(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(1024, 512),
            act(),
        )
        self.fc4 = nn.Sequential(
            nn.Linear(512, 256),
            act(),
        )
        self.fc5 = nn.Sequential(
            nn.Linear(256, 128),
            act(),
        )
        self.fc6 = nn.Sequential(
            nn.Linear(128, num_classes),
        )

    def forward(self, x1):
        out1 = self.fc1(x1)
        out2 = self.fc2(out1)
        out3 = self.fc3(out2)
        out4 = self.fc4(out3)
        out5 = self.fc5(out4)
        out6 = self.fc6(out5)

        return out5, out6
