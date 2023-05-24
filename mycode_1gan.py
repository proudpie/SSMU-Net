import torch
from mycode_0net import *
from mycode_0pre10 import *

#训练模块1，对抗损失
for epoch in range(5):
    gen.train()
    dis1.train()
    dis2.train()
    for step, (MS4, PAN, MS4s, PANs, _, _, _, _, _) in enumerate(train_loader):
        MS4, PAN = MS4.to(DEVICE), PAN.to(DEVICE)
        MS4s, PANs = MS4s.to(DEVICE), PANs.to(DEVICE)
        
        output_ms4s, output_pans = gen(MS4, PAN)
        fakeoutput_ms4s = dis1(output_ms4s.detach())
        realoutput_ms4s = dis1(MS4s)
        fakeoutput_pans = dis2(output_pans.detach())
        realoutput_pans = dis2(PANs)
        
        d1_real_loss=loss_fn(realoutput_ms4s,torch.ones_like(realoutput_ms4s))#和全一的比较，计算真实损失
        d1_fake_loss=loss_fn(fakeoutput_ms4s,torch.ones_like(fakeoutput_ms4s))#得到在生成预测上的损失
        d1_loss=(d1_real_loss+d1_fake_loss)/2#分类器损失
        d2_real_loss=loss_fn(realoutput_pans,torch.ones_like(realoutput_pans))#和全一的比较，计算真实损失
        d2_fake_loss=loss_fn(fakeoutput_pans,torch.ones_like(fakeoutput_pans))#得到在生成预测上的损失
        d2_loss=(d2_real_loss+d2_fake_loss)/2#分类器损失
        gfake1=dis1(output_ms4s)
        gfake2=dis2(output_pans)
        g1_loss=loss_fn(gfake1,torch.ones_like(gfake1))#生成器损失
        g2_loss=loss_fn(gfake2,torch.ones_like(gfake2))#生成器损失
        g_loss = g1_loss + g2_loss
        
        g_optim.zero_grad()
        d1_optim.zero_grad()
        d2_optim.zero_grad()
        if(step%500==0):
            print(step,g_loss)
        g_loss.backward(retain_graph=True)    
        d1_loss.backward(retain_graph=True)
        d2_loss.backward()
        g_optim.step()
        d1_optim.step()
        d2_optim.step()
torch.save(gen, './gen_image10.pkl')