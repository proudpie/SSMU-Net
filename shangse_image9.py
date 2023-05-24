from mycode_0net import *
from mycode_0pre10 import *
import time

gen = torch.load('gen_image9.pkl')
gen = gen.to(DEVICE)
TP = torch.load('TP_image9_0.3.pkl')
TP = TP.to(DEVICE)
MU = torch.load('MU_image9.pkl')
MU = MU.to(DEVICE)
FS = torch.load('FS_image9_0.3.pkl')
FS = FS.to(DEVICE)
color = [[0, 0, 0], [0, 255, 255], [0, 0, 255], [237,145,33], [189,252,201], [255, 0, 0],
        [0, 255, 0], [160, 32, 240], [221, 160, 221], [240, 230, 140], [255, 255, 0]]
print(len(all_data_loader))

t1=time.time()
class_count = np.zeros(11)
out_clour = np.zeros((6905, 7300, 3))
def clour_model(dataloader):
    with torch.no_grad():
        gen.eval()
        TP.eval()
        MU.eval()
        FS.eval()
        for step, (MS4, PAN, _, _, MS4_LL, PAN_LL, label, x, y) in enumerate(dataloader):
            MS4, PAN = MS4.to(DEVICE), PAN.to(DEVICE)
            MS4_LL, PAN_LL, label = MS4_LL.to(DEVICE), PAN_LL.to(DEVICE), label.to(DEVICE)
            ms4s, pans = gen(MS4, PAN)
            ms4s, ms4c, pans, panc = TP(MS4, MS4_LL, PAN, PAN_LL, ms4s, pans)
            ms4s, pans = MU(ms4s, pans)
            ms4c, panc = MU(ms4c, panc)
            output = FS(ms4s, ms4c, pans, panc)
            pred_y = torch.max(output, 1)[1].cuda().data.squeeze()
            pred_y_numpy = pred_y.cpu().numpy()
            x = x.numpy()
            y = y.numpy()
            for k in range(len(x)):
                class_count[pred_y_numpy[k]] = class_count[pred_y_numpy[k]] + 1
                out_clour[x[k]][y[k]] = color[pred_y_numpy[k]]
            if step % 2000 == 0:
                print(step, class_count)
        cv2.imwrite(f"Beijing.png", out_clour)
clour_model(all_data_loader)
t2 = time.time()
print(t2-t1)