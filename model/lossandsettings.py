from model.GAN import *
from model.TP import *
from model.MU import *
from model.FS import *
import torch
from torch.optim import lr_scheduler
DEVICE ="cuda:1" if torch.cuda.is_available() else "cpu"


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