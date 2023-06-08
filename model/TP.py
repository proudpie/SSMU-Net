import torch.nn as nn
from torch.nn import functional as F
from model.block import *

class TripleNet(nn.Module): 
    def __init__(self):
        super(TripleNet, self).__init__()
        self.convMS4s = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        self.convMS4c = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        self.convMS4a = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        self.convPANs = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        self.convPANc = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        self.convPANa = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        
        self.blk1_ma = ResBlk( 64,  64, stride=1)
        self.blk2_ma = ResBlk( 64,  64, stride=1)
        self.blk3_ma = ResBlk( 64, 128, stride=1)
        
        self.blk1_pa = ResBlk( 64,  64, stride=1)
        self.blk2_pa = ResBlk( 64,  64, stride=1)
        self.blk3_pa = ResBlk( 64, 128, stride=1)
        
        self.blk1_ms = ResBlk( 64,  64, stride=1)
        self.blk2_ms = ResBlk( 64,  64, stride=1)
        self.blk3_ms = ResBlk( 64, 128, stride=1)
        self.blk4_ms = ResBlk(128, 128, stride=1)
        self.blk5_ms = ResBlk(128,   8, stride=2)
        
        self.blk1_mc = ResBlk( 64,  64, stride=1)
        self.blk2_mc = ResBlk( 64,  64, stride=1)
        self.blk3_mc = ResBlk( 64, 128, stride=1)
        self.blk4_mc = ResBlk(128, 128, stride=1)
        self.blk5_mc = ResBlk(128,   8, stride=2)
        
        self.blk1_ps = ResBlk( 64,  64, stride=1)
        self.blk2_ps = ResBlk( 64,  64, stride=1)
        self.blk3_ps = ResBlk( 64, 128, stride=2)
        self.blk4_ps = ResBlk(128, 128, stride=2)
        self.blk5_ps = ResBlk(128,   8, stride=2)
        
        self.blk1_pc = ResBlk( 64,  64, stride=1)
        self.blk2_pc = ResBlk( 64,  64, stride=1)
        self.blk3_pc = ResBlk( 64, 128, stride=2)
        self.blk4_pc = ResBlk(128, 128, stride=1)
        self.blk5_pc = ResBlk(128,   8, stride=2)
        self.sigmoid = nn.Sigmoid()


    def forward(self, ms, mc, ps, pc, ma, pa):
        ma = F.relu(self.convMS4a(ma))
        ma = self.blk1_ma(ma)
        ma = self.blk2_ma(ma)
        ma = self.blk3_ma(ma)
        
        pa = F.relu(self.convPANa(pa))
        pa = self.blk1_pa(pa)
        pa = self.blk2_pa(pa)
        pa = self.blk3_pa(pa)
        
        ms = F.relu(self.convMS4s(ms))
        ms = self.blk1_ms(ms)
        ms = self.blk2_ms(ms)
        ms = self.blk3_ms(ms)
        ms = self.sigmoid(ms * ma)
        ms = self.blk4_ms(ms)
        ms = self.blk5_ms(ms)
        #print(ms.size())
        
        mc = F.relu(self.convMS4c(mc))
        mc = self.blk1_mc(mc)
        mc = self.blk2_mc(mc)
        mc = self.blk3_mc(mc)
        mc = self.blk4_mc(mc)
        mc = self.blk5_mc(mc)
        #print(mc.size())

        ps = F.relu(self.convPANs(ps))
        ps = self.blk1_ps(ps)
        ps = self.blk2_ps(ps)
        ps = self.blk3_ps(ps)
        ps = self.sigmoid(ps * pa)
        ps = self.blk4_ps(ps)
        ps = self.blk5_ps(ps)
        #print(ps.size())
        
        pc = F.relu(self.convPANc(pc))
        pc = self.blk1_pc(pc)
        pc = self.blk2_pc(pc)
        pc = self.blk3_pc(pc)
        pc = self.blk4_pc(pc)
        pc = self.blk5_pc(pc)
        #print(pc.size())
        
        return ms,mc,ps,pc