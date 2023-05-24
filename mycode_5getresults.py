from mycode_0net import *
from mycode_0pre10 import *
import time
gen = torch.load('./gen_image10.pkl')
gen = gen.to(DEVICE)
TP = torch.load('./TP_image10_0.3.pkl')
TP = TP.to(DEVICE)
MU = torch.load('./MU_image10.pkl')
MU = MU.to(DEVICE)
FS = torch.load('./FS_image10_0.3.pkl')
FS = FS.to(DEVICE)
def test_model(test_loader):
    gen.eval()
    TP.eval()
    FS.eval()
    correct = 0.0
    test_loss = 0.0
    test_matrix = np.zeros([Categories_Number, Categories_Number])
    with torch.no_grad():
        for step, (MS4, PAN, _, _, MS4_LL, PAN_LL, label, _, _) in enumerate(test_loader):
            MS4, PAN = MS4.to(DEVICE), PAN.to(DEVICE)
            MS4_LL, PAN_LL, label = MS4_LL.to(DEVICE), PAN_LL.to(DEVICE), label.to(DEVICE)
            ms4s, pans = gen(MS4, PAN)
            ms4s, ms4c, pans, panc = TP(MS4, MS4_LL, PAN, PAN_LL, ms4s, pans)
            ms4s, pans = MU(ms4s, pans)
            ms4c, panc = MU(ms4c, panc)
            output = FS(ms4s, ms4c, pans, panc)
            test_loss += F.cross_entropy(output, label.long()).item()
            pred = output.max(1, keepdim=True)[1]#返回最大结果对应的位置
            for i in range(len(label)):
                test_matrix[int(pred[i].item())][int(label[i].item())] += 1
            correct += pred.eq(label.view_as(pred).long()).sum().item()
            
        test_loss = test_loss / len(test_loader)
        acc = 100.0 * correct / len(test_loader.dataset)
        print("test-average loss: {:.4f}, Accuracy:{:.3f} \n".format(test_loss, acc))
    return test_matrix

def kappa(matrix):
    n = np.sum(matrix)
    sum_po = 0
    sum_pe = 0
    for i in range(len(matrix[0])):
        sum_po += matrix[i][i]
        row = np.sum(matrix[i, :])
        col = np.sum(matrix[:, i])
        sum_pe += row * col
    po = sum_po / n
    pe = sum_pe / (n * n)
    return (po - pe) / (1 - pe)


def aa_oa(matrix):
    accuracy = []
    b = np.sum(matrix, axis=0)
    p = np.sum(matrix, axis=1)
    c = 0
    on_display = []
    for i in range(1, matrix.shape[0]):
        a = matrix[i][i]/b[i]
        pre = matrix[i][i]/p[i]
        c += matrix[i][i]
        accuracy.append(a)
        on_display.append([b[i], matrix[i][i], a])
        print("Category:{}. Overall:{}. Correct:{}. Accuracy:{:.6f}".format(i, p[i], matrix[i][i], pre))
    aa = np.mean(accuracy)
    oa = c / np.sum(b, axis=0)
    k = kappa(matrix)
    print("OA:{:.6f} AA:{:.6f} Kappa:{:.6f}".format(oa, aa, k))
    return [aa, oa, k, on_display]

result = test_model(test_loader)
kappa(result)
aa_oa(result)