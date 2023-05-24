import torch
from mycode_0net import *
from mycode_0pre5 import *
from torch.optim import lr_scheduler
gen = torch.load('./gen_image10.pkl')
gen = gen.to(DEVICE)
TP = torch.load('./TP_image10_0.3.pkl')
TP = TP.to(DEVICE)
MU = torch.load('./MU_image10.pkl')
MU = MU.to(DEVICE)
train_acc = []
for epoch in range(15):
    gen.eval()
    TP.eval()
    MU.eval()
    FS.train()
    correct = 0.0
    train_loss = 0.0
    for step, (MS4, PAN, _, _, MS4_LL, PAN_LL, label, _, _) in enumerate(train_loader):
        MS4, PAN = MS4.to(DEVICE), PAN.to(DEVICE)
        MS4_LL, PAN_LL, label = MS4_LL.to(DEVICE), PAN_LL.to(DEVICE), label.to(DEVICE)
        ms4s, pans = gen(MS4, PAN)
        ms4s, ms4c, pans, panc = TP(MS4, MS4_LL, PAN, PAN_LL, ms4s, pans)
        ms4s, pans = MU(ms4s, pans)
        ms4c, panc = MU(ms4c, panc)
        output = FS(ms4s, ms4c, pans, panc)
        f_loss = F.cross_entropy(output, label.long())#定义反向传播
        train_loss += F.cross_entropy(output, label.long()).item()
        pred = output.max(1, keepdim=True)[1]#返回最大结果对应的位置
        correct += pred.eq(label.view_as(pred).long()).sum().item()
        
        f_optim.zero_grad()
        if(step%1000==0):
            print(epoch,step,f_loss)
        f_loss.backward()
        f_optim.step()
    train_loss = train_loss / len(train_loader)
    acc1 = 100.0 * correct / len(train_loader.dataset)
    print("train-average loss: {:.4f}, Accuracy:{:.3f} \n".format(train_loss, acc1))
    train_acc.append(acc1)
torch.save(FS, './FS_image10_0.3.pkl')