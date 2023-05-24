import torch
from mycode_0net import *
from mycode_0pre10 import *
gen = torch.load('./gen_image10.pkl')
gen = gen.to(DEVICE)
TP = torch.load('./TP_image10_0.3.pkl')
TP = TP.to(DEVICE)
#训练模块3，模态统一损失
for epoch in range(5):
    gen.eval()
    TP.eval()
    MU.train()
    for step, (MS4, PAN, _, _, MS4_LL, PAN_LL, label, _, _) in enumerate(train_loader):
        MS4, PAN = MS4.to(DEVICE), PAN.to(DEVICE)
        MS4_LL, PAN_LL, label = MS4_LL.to(DEVICE), PAN_LL.to(DEVICE), label.to(DEVICE)
        ms4s, pans = gen(MS4, PAN)
        ms4s, ms4c, pans, panc = TP(MS4, MS4_LL, PAN, PAN_LL, ms4s, pans)
        ms4_, pan_ = MU(ms4c, panc)
        loss0 = perceptual_loss(ms4_, ms4c)
        loss1 = perceptual_loss(pan_, panc)
        loss2 = max(dist_loss(ms4_, pan_)-0.3, 0)
        m_loss = lamda[0]*loss0 + lamda[1]*loss1 + lamda[2]*loss2
        
        m_optim.zero_grad()
        if(step%2000==0):
            print(step,m_loss,loss0,loss1,loss2)
        m_loss.backward()
        m_optim.step()

torch.save(MU, './MU_image10.pkl')