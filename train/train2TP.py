import torch
import sys
sys.path.append(...)
from preprocessing.preprocessing_image10 import *
from model.lossandsettings import*
from model.GAN import *
from model.TP import *

gen = torch.load('./gen_image10.pkl')
gen = gen.to(DEVICE)
#训练模块2，三元组损失
triplet_loss = nn.TripletMarginLoss(margin=0.3, p=2)
dist_loss = nn.MSELoss()
lamda = [0.1,0.1,1]
FS_1 = FusionNet().to(DEVICE)
f1_loss = 0.0
f1_optim = torch.optim.Adam(FS_1.parameters(),lr=lr_f)
for epoch in range(30):
    gen.eval()
    TP.train()
    FS_1.train()
    for step, (MS4, PAN, _, _, MS4_LL, PAN_LL, label, _, _) in enumerate(train_loader):
        MS4, PAN = MS4.to(DEVICE), PAN.to(DEVICE)
        MS4_LL, PAN_LL = MS4_LL.to(DEVICE), PAN_LL.to(DEVICE)
        label = label.to(DEVICE)
        ms4s, pans = gen(MS4, PAN)
        ms4s, ms4c, pans, panc = TP(MS4, MS4_LL, PAN, PAN_LL, ms4s, pans)
        output = FS_1(ms4s, ms4c, pans, panc)
        

        ms4loss = triplet_loss(ms4c, panc, ms4s)
        panloss = triplet_loss(panc, ms4c, pans)
        f1_loss = F.cross_entropy(output, label.long())#定义反向传播
        t_loss = lamda[0] * ms4loss + lamda[1] * panloss + lamda[2] * f1_loss
        
        f1_optim.zero_grad()
        t_optim.zero_grad()
        if (step % 500 == 0):
            print(epoch, step)
        f1_loss.backward(retain_graph=True)
        t_loss.backward()
        f1_optim.step()
        t_optim.step()

torch.save(TP, './TP_image10_0.3.pkl')