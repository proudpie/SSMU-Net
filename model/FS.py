import torch.nn as nn
from torch.nn import functional as F
from model.block import *

class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(24, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        self.blk1 = ResBlk(64, 64, stride=1)
        self.blk2 = ResBlk(64, 64, stride=1)
        self.blk3 = ResBlk(64, 128, stride=1)
        self.blk4 = ResBlk(128, 128, stride=1)
        self.blk5 = ResBlk(128, 256, stride=2)
        self.blk6 = ResBlk(256, 512, stride=1)
        self.outlayer = nn.Linear(512, 13)#(512,categories)
        self.sigmoid = nn.Sigmoid()

    def forward(self, ms, mc, ps, pc):
        output = torch.cat([ms, ps], 1)
        content = torch.max(mc, pc)
        output = torch.cat([output, content], 1)
        output = F.relu(self.conv(output))
        output = self.blk1(output)
        output = self.blk2(output)
        output = self.blk3(output)
        output = self.blk4(output)
        output = self.blk5(output)
        output = self.blk6(output)
        output = F.adaptive_avg_pool2d(output, [1, 1])
        # print(output.size())
        output = output.view(output.size()[0], -1)
        output = self.outlayer(output)
        # print(output.size())
        return output