import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import lr_scheduler
from torchvision import models
DEVICE ="cuda:1" if torch.cuda.is_available() else "cpu"

class ResBlk(nn.Module):
    def __init__(self, ch_in, ch_out, stride=1):
        super(ResBlk, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.extra = nn.Sequential()
        if ch_out != ch_in or stride != 1:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.extra(x) + out
        return out 

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
    def __init__(self, bilinear=True, inchannel = 8):
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

gen = Gen().to(DEVICE)
dis1 = Discriminator_1().to(DEVICE)
dis2 = Discriminator_2().to(DEVICE)
TP = TripleNet().to(DEVICE)
MU = MUNet().to(DEVICE)
FS = FusionNet().to(DEVICE)

g_optim = torch.optim.Adam(gen.parameters(),lr=0.001)
d1_optim = torch.optim.Adam(dis1.parameters(),lr=0.001)
d2_optim = torch.optim.Adam(dis2.parameters(),lr=0.001)
t_optim = torch.optim.Adam(TP.parameters(),lr=0.001)
m_optim = torch.optim.Adam(MU.parameters(),lr=0.001)

lr_f = 0.001
f_optim = torch.optim.Adam(FS.parameters(),lr=lr_f)
scheduler = lr_scheduler.StepLR(f_optim, 6, gamma=0.03, last_epoch=-1)
loss_fn = torch.nn.BCELoss()

d1_loss = 0.0
d2_loss = 0.0
g_loss = 0.0
t_loss = 0.0
f_loss = 0.0