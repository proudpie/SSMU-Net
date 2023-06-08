import torch.nn as nn
from torch.nn import functional as F
from model.block import *

class Gen(nn.Module):
    def __init__(self):
        super(Gen, self).__init__()
        self.convMS4 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32)
        )
        self.convPAN = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32)
        )
        
        self.blk1_mgan = ResBlk( 32,  64, stride=1)
        self.blk2_mgan = ResBlk( 64,  64, stride=1)
        self.blk3_mgan = ResBlk( 64, 128, stride=1)
        self.blk4_mgan = ResBlk(128, 128, stride=1)
        self.blk5_mgan = ResBlk(128,  12, stride=1)
        
        self.blk1_pgan = ResBlk( 32,  64, stride=2)
        self.blk2_pgan = ResBlk( 64,  64, stride=1)
        self.blk3_pgan = ResBlk( 64, 128, stride=1)
        self.blk4_pgan = ResBlk(128, 128, stride=1)
        self.blk5_pgan = ResBlk(128,   3, stride=1) 
        
    def forward(self,x,y):
        x = F.relu(self.convMS4(x))
        #print(x.size())
        x = self.blk1_mgan(x)
        x = self.blk2_mgan(x)
        x = self.blk3_mgan(x)
        x = self.blk4_mgan(x)
        x = self.blk5_mgan(x)
        #print(x.size())
        
        y = F.relu(self.convPAN(y))
        #print(y.size())
        y = self.blk1_pgan(y)
        y = self.blk2_pgan(y)
        y = self.blk3_pgan(y)
        y = self.blk4_pgan(y)
        y = self.blk5_pgan(y)
        #print(y.size())
        
        return x,y

class Discriminator_1(nn.Module):
    def __init__(self):
        super(Discriminator_1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        self.blk1 = ResBlk( 64,  64, stride=1)
        self.blk2 = ResBlk( 64,  64, stride=1)
        self.blk3 = ResBlk( 64, 256, stride=1)
        self.outlayer = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = F.adaptive_avg_pool2d(x, [1, 1])
        x = x.view(x.size()[0],  -1)
        x = self.outlayer(x)
        x = self.sigmoid(x)
        return x

class Discriminator_2(nn.Module):
    def __init__(self):
        super(Discriminator_2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        self.blk1 = ResBlk( 64,  64, stride=1)
        self.blk2 = ResBlk( 64,  64, stride=1)
        self.blk3 = ResBlk( 64, 256, stride=1)
        self.outlayer = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = F.adaptive_avg_pool2d(x, [1, 1])
        x = x.view(x.size()[0],  -1)
        x = self.outlayer(x)
        x = self.sigmoid(x)
        return x