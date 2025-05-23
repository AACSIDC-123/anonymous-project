
import torch
import torch.nn as nn
import numpy as np
from parse import args
import os

os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"
device = torch.device(f"cuda:0") 

class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc / 2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])

class a_1(nn.Module):
    def __init__(self):
        super(a_1, self).__init__()

    def forward(self, output, x_true, x_true_ratio):
        a = torch.ones(size=output.shape).to(device)
        num_zeros = int(x_true.shape[-1]/2)
        a[:,:,:,:num_zeros]=0 
        
        output = output*a
        if x_true_ratio != 0:
            output[:,:,:, -int(num_zeros*x_true_ratio): ] = x_true[:,:,:, -int(num_zeros*x_true_ratio):]
            
        return output
    

class a_2(nn.Module):
    def __init__(self):
        super(a_2, self).__init__()
    def forward(self, output, x_true, x_true_ratio):
        a = torch.ones(size=output.shape).to(device)
        num_zeros = int(x_true.shape[-1]/2)
        
        a[:,:,:,num_zeros:]=0 
        output = output*a
        if x_true_ratio != 0:
            output[:,:,:, :int(num_zeros*x_true_ratio) ] = x_true[:,:,:, : int(num_zeros*x_true_ratio) ]
        return output

class s_1(nn.Module):
    def __init__(self):
        super(s_1, self).__init__()
    def forward(self, output, x_true, x_true_ratio):
        
        a = torch.ones(size=output.shape).to(device)
        
        half_width = int(x_true.shape[-1]/2)
        half_height = int(x_true.shape[-2]/2)
        
        a[:,:,:,half_width:] = 0 
        a[:,:,half_height:,:] = 0
        
        output = output*a
        
        if x_true_ratio != 0:
            
            output[:,:,:, :int(half_width*x_true_ratio)] = x_true[:,:,:, :int(half_width*x_true_ratio)]
            output[:,:,:int(half_height*x_true_ratio),:] = x_true[:,:,:int(half_height*x_true_ratio),:]
            
        return output

class s_2(nn.Module):
    def __init__(self):
        super(s_2, self).__init__()
    def forward(self, output, x_true, x_true_ratio):
        
        a = torch.ones(size=output.shape).to(device)
        
        half_width = int(x_true.shape[-1]/2)
        half_height = int(x_true.shape[-2]/2)
        
        a[:,:,:,:half_width] = 0 
        a[:,:,half_height:,:] = 0
        
        output = output*a
        
        if x_true_ratio != 0:
            
            output[:,:,:,half_width + int(half_width*(1-x_true_ratio)):] = x_true[:,:,:,half_width + int(half_width*(1-x_true_ratio)):]
            output[:,:,:int(half_height*x_true_ratio),:] = x_true[:,:,:int(half_height*x_true_ratio),:]
            
        return output

class s_3(nn.Module):
    def __init__(self):
        super(s_3, self).__init__()
    def forward(self, output, x_true, x_true_ratio):
        
        a = torch.ones(size=output.shape).to(device)
        
        half_width = int(x_true.shape[-1]/2)
        half_height = int(x_true.shape[-2]/2)
        
        a[:,:,:,half_width:] = 0 
        a[:,:,:half_height,:] = 0
        
        output = output*a
        
        if x_true_ratio != 0:
            
            output[:,:,:,:int(half_width*x_true_ratio)] = x_true[:,:,:,:int(half_width*x_true_ratio)]
            output[:,:,half_height+int(half_height*(1-x_true_ratio)):,:] = x_true[:,:,half_height+int(half_height*(1-x_true_ratio)):,:]
            
        return output

class s_4(nn.Module):
    def __init__(self):
        super(s_4, self).__init__()
    def forward(self, output, x_true, x_true_ratio):
        
        a = torch.ones(size=output.shape).to(device)
        
        half_width = int(x_true.shape[-1]/2)
        half_height = int(x_true.shape[-2]/2)
        
        a[:,:,:,:half_width] = 0 
        a[:,:,:half_height,:] = 0
        
        output = output*a
        
        if x_true_ratio != 0:
            
            output[:,:,:,half_width + int(half_width*(1-x_true_ratio)):] = x_true[:,:,:,half_width + int(half_width*(1-x_true_ratio)):]
            output[:,:,half_height+int(half_height*(1-x_true_ratio)):,:] = x_true[:,:,half_height+int(half_height*(1-x_true_ratio)):,:]
            
        return output

class b_1(nn.Module):
    def __init__(self):
        super(b_1, self).__init__()

    def forward(self, output):
        a = torch.ones(size=output.shape).to(device)
        b =[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 27, 28, 29, 30, 31]
        for i in range(len(b)):
            a[:,:,:,b[i]]=0 

        return output*a


class c_1(nn.Module):
    def __init__(self):
        super(c_1, self).__init__()

    def forward(self, output):
        a = torch.ones(size=output.shape).to(device)
        b = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31]
        for i in range(len(b)):
            a[:,:,:,b[i]]=0 

        return output*a


class Generator_cifar(nn.Module):
    def __init__(self, channel=3, shape_img=32, n_hidden_1=2048, batchsize=6, g_in=128, d = 32, iters=0):
        super(Generator_cifar, self).__init__()
        self.g_in = g_in
        self.batchsize = batchsize
        self.iters = iters

        self.layer1 = nn.Sequential(nn.Linear(g_in, n_hidden_1), )
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, g_in), )

        block_num = int(np.log2(shape_img) - 3)
        self.block0 = nn.Sequential(
            nn.ConvTranspose2d(g_in, d*pow(2,block_num) * 2, 4, 1, 0),
            GLU()
        )
        self.blocks = nn.ModuleList()
        for bn in reversed(range(block_num)):
            self.blocks.append(self.upBlock(pow(2, bn + 1) * d, pow(2, bn) * d))
        self.deconv_out = self.upBlock(d, channel)

        self.a_output_1 = a_2()
        self.a_output_2 = a_1()

    @staticmethod
    def upBlock(in_planes, out_planes):
        def conv3x3(in_planes, out_planes):
            return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                             padding=1, bias=False)
        block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            conv3x3(in_planes, out_planes*2),
            nn.BatchNorm2d(out_planes*2),
            GLU()       )
        return block

    # forward method
    def forward(self, x, x_true, x_true_ratio, batch_size, iters):
        x = self.layer1(x)
        x = self.layer2(x)

        x = x.view(batch_size, self.g_in, 1, 1)
        output = self.block0(x)
        for block in self.blocks:
            output = block(output)
        output = self.deconv_out(output)
        output = torch.sigmoid(output)
        
        if iters%2 ==0:
            output = self.a_output_1(output, x_true, x_true_ratio)
        else:
            output = self.a_output_2(output, x_true, x_true_ratio)

        return output



class Generator_cifar_4clients(nn.Module):
    def __init__(self, channel=3, shape_img=32, n_hidden_1=2048, batchsize=6, g_in=128, d = 32, iters=0):
        super(Generator_cifar_4clients, self).__init__()
        self.g_in = g_in
        self.batchsize = batchsize
        self.iters = iters

        self.layer1 = nn.Sequential(nn.Linear(g_in, n_hidden_1), )
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, g_in), )

        block_num = int(np.log2(shape_img) - 3)
        self.block0 = nn.Sequential(
            nn.ConvTranspose2d(g_in, d*pow(2,block_num) * 2, 4, 1, 0),
            GLU()
        )
        self.blocks = nn.ModuleList()
        for bn in reversed(range(block_num)):
            self.blocks.append(self.upBlock(pow(2, bn + 1) * d, pow(2, bn) * d))
        self.deconv_out = self.upBlock(d, channel)

        self.a_output_1 = s_1()
        self.a_output_2 = s_2()
        self.a_output_3 = s_3()
        self.a_output_4 = s_4()

    @staticmethod
    def upBlock(in_planes, out_planes):
        def conv3x3(in_planes, out_planes):
            return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                             padding=1, bias=False)
        block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            conv3x3(in_planes, out_planes*2),
            nn.BatchNorm2d(out_planes*2),
            GLU()       )
        return block

    # forward method
    def forward(self, x, x_true, x_true_ratio, batch_size, iters):
        x = self.layer1(x)
        x = self.layer2(x)

        x = x.view(batch_size, self.g_in, 1, 1)
        output = self.block0(x)
        for block in self.blocks:
            output = block(output)
        output = self.deconv_out(output)
        output = torch.sigmoid(output)
        
        if iters == 1: 
            output = self.a_output_1(output, x_true, x_true_ratio)
        elif iters == 2: 
            output = self.a_output_2(output, x_true, x_true_ratio)
        elif iters == 3: 
            output = self.a_output_3(output, x_true, x_true_ratio)
        elif iters == 4: 
            output = self.a_output_4(output, x_true, x_true_ratio)
        return output


class Generator_stl10(nn.Module):
    def __init__(self, channel=3, shape_img=32, n_hidden_1=2048, batchsize=6, g_in=128, d = 32, iters=0):
        super(Generator_stl10, self).__init__()
        self.g_in = g_in
        self.batchsize = batchsize
        self.iters = iters

        self.layer1 = nn.Sequential(nn.Linear(g_in, n_hidden_1), )
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, g_in), )

        block_num = int(np.log2(shape_img) - 3)
        self.block0 = nn.Sequential(
            nn.ConvTranspose2d(g_in, d*pow(2,block_num) * 2, 6, 1, 0),
            GLU()
        )
        self.blocks = nn.ModuleList()
        for bn in reversed(range(block_num)):
            self.blocks.append(self.upBlock(pow(2, bn + 1) * d, pow(2, bn) * d))
        self.deconv_out = self.upBlock(d, channel)

        self.a_output_1 = a_1()
        self.a_output_2 = a_2()

    @staticmethod
    def upBlock(in_planes, out_planes):
        def conv3x3(in_planes, out_planes):
            return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                             padding=1, bias=False)
        block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            conv3x3(in_planes, out_planes*2),
            nn.BatchNorm2d(out_planes*2),
            GLU()       )
        return block

    # forward method
    def forward(self, x, x_true, x_true_ratio, batch_size, iters):
        x = self.layer1(x)
        x = self.layer2(x)

        x = x.view(batch_size, self.g_in, 1, 1)
        output = self.block0(x)
        for block in self.blocks:
            output = block(output)
        output = self.deconv_out(output)
        output = torch.sigmoid(output)
        
        if iters%2 ==0:
            output = self.a_output_1(output, x_true, x_true_ratio)
        else:
            output = self.a_output_2(output, x_true, x_true_ratio)

        return output

class Generator_mnist(nn.Module):
    def __init__(self, channel=3, shape_img=32, n_hidden_1=2048, batchsize=6, g_in=128, d = 32, iters=0):
        super(Generator_mnist, self).__init__()
        self.g_in = g_in
        self.batchsize = batchsize
        self.iters = iters

        self.layer1 = nn.Sequential(nn.Linear(g_in, n_hidden_1), )
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, g_in), )

        block_num = int(np.log2(shape_img) - 3)
        self.block0 = nn.Sequential(
            nn.ConvTranspose2d(g_in, d*pow(2,block_num) * 2, 7, 1, 0),
            GLU()
        )
        self.blocks = nn.ModuleList()
        for bn in reversed(range(block_num)):
            self.blocks.append(self.upBlock(pow(2, bn + 1) * d, pow(2, bn) * d))
        self.deconv_out = self.upBlock(d, channel)

        
        self.a_output_1 = a_1()
        self.a_output_2 = a_2()

    @staticmethod
    def upBlock(in_planes, out_planes):
        def conv3x3(in_planes, out_planes):
            return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                             padding=1, bias=False)
        block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            conv3x3(in_planes, out_planes*2),
            nn.BatchNorm2d(out_planes*2),
            GLU()       )
        return block

    # forward method
    def forward(self, x, x_true, x_true_ratio, batch_size, iters):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(batch_size, self.g_in, 1, 1)
        
        output = self.block0(x)
        for block in self.blocks:
            output = block(output)
        output = self.deconv_out(output)
        output = torch.sigmoid(output)
        
        if iters%2 ==0:
            output = self.a_output_1(output, x_true, x_true_ratio)
        else:
            output = self.a_output_2(output, x_true, x_true_ratio)

        return output


class Generator_client3(nn.Module):
    def __init__(self, channel=3, shape_img=32, n_hidden_1=2048, batchsize=6, g_in=128, d = 32, iters=0):
        super(Generator_client3, self).__init__()
        self.g_in = g_in
        self.batchsize = batchsize
        self.iters = iters

        self.layer1 = nn.Sequential(nn.Linear(g_in, n_hidden_1),
                                   )
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, g_in),
                                  )

        block_num = int(np.log2(shape_img) - 3)
        self.block0 = nn.Sequential(
            nn.ConvTranspose2d(g_in, d*pow(2,block_num) * 2, 4, 1, 0),
            GLU()
        )
        self.blocks = nn.ModuleList()
        for bn in reversed(range(block_num)):
            self.blocks.append(self.upBlock(pow(2, bn + 1) * d, pow(2, bn) * d))
        self.deconv_out = self.upBlock(d, channel)

        self.a_output_1 = nn.Sequential(b_1())


    @staticmethod
    def upBlock(in_planes, out_planes):
        def conv3x3(in_planes, out_planes):
            return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                             padding=1, bias=False)

        block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            conv3x3(in_planes, out_planes*2),
            nn.BatchNorm2d(out_planes*2),
            GLU()
        )
        return block


    # forward method
    def forward(self, x, batch_size, iters):
        x = self.layer1(x)
        x = self.layer2(x)

        x = x.view(batch_size, self.g_in, 1, 1)
        output = self.block0(x)
        for block in self.blocks:
            output = block(output)
        output = self.deconv_out(output)
        output = torch.sigmoid(output)
    
        return output

class Generator_client4(nn.Module):
    def __init__(self, channel=3, shape_img=32, n_hidden_1=2048, batchsize=6, g_in=128, d = 32, iters=0):
        super(Generator_client4, self).__init__()
        self.g_in = g_in
        self.batchsize = batchsize
        self.iters = iters

        self.layer1 = nn.Sequential(nn.Linear(g_in, n_hidden_1),
                                   )
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, g_in),
                                  )

        block_num = int(np.log2(shape_img) - 3)
        self.block0 = nn.Sequential(
            nn.ConvTranspose2d(g_in, d*pow(2,block_num) * 2, 4, 1, 0),
            GLU()
        )
        self.blocks = nn.ModuleList()
        for bn in reversed(range(block_num)):
            self.blocks.append(self.upBlock(pow(2, bn + 1) * d, pow(2, bn) * d))
        self.deconv_out = self.upBlock(d, channel)

        self.a_output_1 = nn.Sequential(c_1())


    @staticmethod
    def upBlock(in_planes, out_planes):
        def conv3x3(in_planes, out_planes):
            return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                             padding=1, bias=False)

        block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            conv3x3(in_planes, out_planes*2),
            nn.BatchNorm2d(out_planes*2),
            GLU()
        )
        return block


    # forward method
    def forward(self, x, batch_size, iters):
        x = self.layer1(x)
        x = self.layer2(x)

        x = x.view(batch_size, self.g_in, 1, 1)
        output = self.block0(x)
        for block in self.blocks:
            output = block(output)
        output = self.deconv_out(output)
        output = torch.sigmoid(output)

        return output

