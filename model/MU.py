import torch
import torch.nn as nn
from torch.nn import functional as F
from model.block import *
from torchvision import models

class MUNet(nn.Module): 
    def __init__(self):
        super(MUNet, self).__init__()
        self.convMU = nn.Sequential(
            nn.Conv2d(8, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        self.convPU = nn.Sequential(
            nn.Conv2d(8, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        
        self.blk1_m = ResBlk( 64,  64, stride=1)
        self.blk2_m = ResBlk( 64, 128, stride=1)
        self.blk3_m = ResBlk(128, 128, stride=1)        
        self.blk4_m = ResBlk(128,   8, stride=1)
        
        self.blk1_p = ResBlk( 64,  64, stride=1)
        self.blk2_p = ResBlk( 64, 128, stride=1)
        self.blk3_p = ResBlk(128, 128, stride=1)        
        self.blk4_p = ResBlk(128,   8, stride=1)
        
    def forward(self,m,p):
        m = F.relu(self.convMU(m))
        m = self.blk1_m(m)
        m = self.blk2_m(m)
        m = self.blk3_m(m)
        m = self.blk4_m(m)
        
        p = F.relu(self.convPU(p))
        p = self.blk1_p(p)
        p = self.blk2_p(p)
        p = self.blk3_p(p)
        p = self.blk4_p(p)
        
        return m,p

class UNet(nn.Module):
    def __init__(self, bilinear=True):
        super(UNet, self).__init__()
        self.bilinear = bilinear
        self.inc = nn.Conv2d( 8, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # 本地模型
        self.vgg16 = models.vgg16(pretrained=True)
        self.vgg16.eval()
        for param in self.vgg16.parameters():
            param.requires_grad_(False)
        self.vgg16.load_state_dict(torch.load('vgg16.pth'))
        # VGG 的预训练加载
        self.down0 =  self.vgg16.features[1:4]
        self.down1 = self.vgg16.features[4:9]
        self.down2 = self.vgg16.features[9:16]
        self.down3 = self.vgg16.features[16:23]
        
    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.down0(x0)  # [1, 64, 256, 256]  1/1
        x2 = self.down1(x1)  # [1, 128, 128, 128] 1/2
        x3 = self.down2(x2)  # [1, 256, 64, 64]   1/4
        x4 = self.down3(x3)  # [1, 512, 32, 32]   1/8
        output = [x1, x2, x3, x4]
        return  output

class Perceptual_loss(nn.Module):
    def __init__(self):
        super(Perceptual_loss, self).__init__()
        self.vgg = UNet().to(DEVICE)
        self.criterion = nn.MSELoss()
        self.weights = [1.0, 1.0, 1.0, 1.0]
    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss =  self.weights[0]* self.criterion(x_vgg[0], y_vgg[0].detach())+\
               self.weights[1]* self.criterion(x_vgg[1], y_vgg[1].detach())+\
               self.weights[2]*self.criterion(x_vgg[2], y_vgg[2].detach())+\
               self.weights[3]*self.criterion(x_vgg[3], y_vgg[3].detach())
        return loss 
dist_loss = nn.MSELoss()
perceptual_loss = Perceptual_loss()