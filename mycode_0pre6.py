import numpy as np
import torch
import torch.nn as nn
from libtiff import TIFF
import pywt
import cv2
from torch.utils.data import Dataset, DataLoader, Subset

# 读取ms4图
ms4_tif = TIFF.open('/home/gpu/Experiment/Remote data/image6/ms4.tif', mode='r')
ms4_np = ms4_tif.read_image()
ms4_np = ms4_np.astype(np.uint8)
# plt.imshow(ms4_np)
# plt.show()
print(ms4_np.shape)
ms4_1 = ms4_np[:, :, 0]
ms4_2 = ms4_np[:, :, 1]
ms4_3 = ms4_np[:, :, 2]
ms4_4 = ms4_np[:, :, 3]
# plt.figure(figsize=(8,8))
# plt.subplot(2,2,1)
# plt.imshow(ms4_1)
# plt.subplot(2,2,2)
# plt.imshow(ms4_2)
# plt.subplot(2,2,3)
# plt.imshow(ms4_3)
# plt.subplot(2,2,4)
# plt.imshow(ms4_4)
# plt.show()
# 对img进行haar小波变换,变量分别是低频，水平高频，垂直高频，对角线高频
coeffs = pywt.dwt2(ms4_1, 'haar')
cA1, (cH1, cV1, cD1) = coeffs
coeffs = pywt.dwt2(ms4_2, 'haar')
cA2, (cH2, cV2, cD2) = coeffs
coeffs = pywt.dwt2(ms4_3, 'haar')
cA3, (cH3, cV3, cD3) = coeffs
coeffs = pywt.dwt2(ms4_4, 'haar')
cA4, (cH4, cV4, cD4) = coeffs
print(cA1.shape)

cA1 = cv2.resize(cA1, (2101, 2001))
cA2 = cv2.resize(cA2, (2101, 2001))
cA3 = cv2.resize(cA3, (2101, 2001))
cA4 = cv2.resize(cA4, (2101, 2001))
cH1 = cv2.resize(cH1, (2101, 2001))
cH2 = cv2.resize(cH2, (2101, 2001))
cH3 = cv2.resize(cH3, (2101, 2001))
cH4 = cv2.resize(cH4, (2101, 2001))
cV1 = cv2.resize(cV1, (2101, 2001))
cV2 = cv2.resize(cV2, (2101, 2001))
cV3 = cv2.resize(cV3, (2101, 2001))
cV4 = cv2.resize(cV4, (2101, 2001))
cD1 = cv2.resize(cD1, (2101, 2001))
cD2 = cv2.resize(cD2, (2101, 2001))
cD3 = cv2.resize(cD3, (2101, 2001))
cD4 = cv2.resize(cD4, (2101, 2001))
print(cA1.shape)
# plt.figure(figsize=(8,8))
# plt.subplot(2,2,1)
# plt.imshow(cA1)
# plt.subplot(2,2,2)
# plt.imshow(cH2)
# plt.subplot(2,2,3)
# plt.imshow(cV3)
# plt.subplot(2,2,4)
# plt.imshow(cD4)
# plt.show()
cV1s = np.expand_dims(cV1, axis=2)  # 二维数据进网络前要加一维
cV2s = np.expand_dims(cV2, axis=2)
cV3s = np.expand_dims(cV3, axis=2)
cV4s = np.expand_dims(cV4, axis=2)
cH1s = np.expand_dims(cH1, axis=2)
cH2s = np.expand_dims(cH2, axis=2)
cH3s = np.expand_dims(cH3, axis=2)
cH4s = np.expand_dims(cH4, axis=2)
cD1s = np.expand_dims(cD1, axis=2)
cD2s = np.expand_dims(cD2, axis=2)
cD3s = np.expand_dims(cD3, axis=2)
cD4s = np.expand_dims(cD4, axis=2)
ms4_self = np.concatenate((cV1s, cV2s, cV3s, cV4s, cH1s, cH2s, cH3s, cH4s, cD1s, cD2s, cD3s, cD4s), axis=2)
print(ms4_self.shape)
cA1s = np.expand_dims(cA1, axis=2)
cA2s = np.expand_dims(cA2, axis=2)
cA3s = np.expand_dims(cA3, axis=2)
cA4s = np.expand_dims(cA4, axis=2)
ms4_LL = np.concatenate((cA1s, cA2s, cA3s, cA4s), axis=2)
print(ms4_LL.shape)

# 读取pan图
pan_tif = TIFF.open('/home/gpu/Experiment/Remote data/image6/pan.tif', mode='r')
pan_np = pan_tif.read_image()
pan_np = pan_np.astype(np.uint8)
# plt.imshow(pan_np)
# plt.show()
pan_np
# 对pan进行haar小波变换,变量分别是低频，水平高频，垂直高频，对角线高频
coeffs = pywt.dwt2(pan_np, 'haar')
cApan, (cHpan, cVpan, cDpan) = coeffs
# plt.figure(figsize=(8,8))
# plt.subplot(2,2,1)
# plt.imshow(cApan)
# plt.subplot(2,2,2)
# plt.imshow(cHpan)
# plt.subplot(2,2,3)
# plt.imshow(cVpan)
# plt.subplot(2,2,4)
# plt.imshow(cDpan)
# plt.show()
cVpans = np.expand_dims(cVpan, axis=2)  # 二维数据进网络前要加一维
cHpans = np.expand_dims(cHpan, axis=2)
cDpans = np.expand_dims(cDpan, axis=2)
pan_self = np.concatenate((cVpans, cHpans, cDpans), axis=2)
print(pan_self.shape)
cApans = np.expand_dims(cApan, axis=2)  # 二维数据进网络前要加一维
pan_LL = cApans
print(pan_LL.shape)

label_np = np.load('/home/gpu/Experiment/Remote data/image6/label.npy')
print('label数组形状：', np.shape(label_np))

# ms4与pan图补零  (给图片加边框）
Ms4_patch_size = 16  # ms4截块的边长
Pan_patch_size = Ms4_patch_size * 4  # pan截块的边长
Interpolation = cv2.BORDER_REFLECT_101
# cv2.BORDER_REPLICATE： 进行复制的补零
# cv2.BORDER_REFLECT:  进行翻转的补零操作:gfedcba|abcdefgh|hgfedcb;
# cv2.BORDER_REFLECT_101： 进行翻转的补零操作:gfedcb|abcdefgh|gfedcb;
# cv2.BORDER_WRAP: 进行上下边缘调换的外包复制操作:bcdefgh|abcdefgh|abcdefg;

top_size, bottom_size, left_size, right_size = (int(Ms4_patch_size / 2 - 1), int(Ms4_patch_size / 2),
                                                int(Ms4_patch_size / 2 - 1), int(Ms4_patch_size / 2))
ms4_np = cv2.copyMakeBorder(ms4_np, top_size, bottom_size, left_size, right_size, Interpolation)
print('补零后的ms4图的形状：', np.shape(ms4_np))
top_size, bottom_size, left_size, right_size = (int(Ms4_patch_size / 2 - 1), int(Ms4_patch_size / 2),
                                                int(Ms4_patch_size / 2 - 1), int(Ms4_patch_size / 2))
ms4_self = cv2.copyMakeBorder(ms4_self, top_size, bottom_size, left_size, right_size, Interpolation)
print('补零后的ms4_self图的形状：', np.shape(ms4_self))
ms4_LL = cv2.copyMakeBorder(ms4_LL, top_size, bottom_size, left_size, right_size, Interpolation)
print('补零后的ms4_LL图的形状：', np.shape(ms4_LL))
top_size, bottom_size, left_size, right_size = (int(Pan_patch_size / 2 - 4), int(Pan_patch_size / 2),
                                                int(Pan_patch_size / 2 - 4), int(Pan_patch_size / 2))
pan_np = cv2.copyMakeBorder(pan_np, top_size, bottom_size, left_size, right_size, Interpolation)
print('补零后的pan图的形状：', np.shape(pan_np))
top_size, bottom_size, left_size, right_size = (int(Pan_patch_size / 4 - 2), int(Pan_patch_size / 4),
                                                int(Pan_patch_size / 4 - 2), int(Pan_patch_size / 4))
pan_self = cv2.copyMakeBorder(pan_self, top_size, bottom_size, left_size, right_size, Interpolation)
print('补零后的pan_self图的形状：', np.shape(pan_self))
pan_LL = cv2.copyMakeBorder(pan_LL, top_size, bottom_size, left_size, right_size, Interpolation)
print('补零后的pan_LL图的形状：', np.shape(pan_LL))

label_np = label_np  # 标签中0类标签是未标注的像素

label_element, element_count = np.unique(label_np, return_counts=True)  # 返回类别标签与各个类别所占的数量
print('类标：', label_element)
print('各类样本数：', element_count)
Categories_Number = len(label_element)  # 数据的类别数
print('标注的类别数：', Categories_Number)
label_row, label_column = np.shape(label_np)  # 获取标签图的行、列

'''归一化图片'''


def to_tensor(image):
    max_i = np.max(image)
    min_i = np.min(image)
    image = (image - min_i) / (max_i - min_i)
    return image


ms4 = to_tensor(ms4_np)
pan = to_tensor(pan_np)
ms4_self = to_tensor(ms4_self)
pan_self = to_tensor(pan_self)
ms4_LL = to_tensor(ms4_LL)
pan_LL = to_tensor(pan_LL)

ms4 = np.array(ms4).transpose((2, 0, 1))  # 调整通道
ms4_self = np.array(ms4_self).transpose((2, 0, 1))  # 调整通道
ms4_LL = np.array(ms4_LL).transpose((2, 0, 1))  # 调整通道
pan = np.expand_dims(pan, axis=0)  # 二维数据进网络前要加一维
pan_self = np.array(pan_self).transpose((2, 0, 1))  # 调整通道
pan_LL = np.expand_dims(pan_LL, axis=0)  # 二维数据进网络前要加一维

ms4 = torch.from_numpy(ms4).type(torch.FloatTensor)
pan = torch.from_numpy(pan).type(torch.FloatTensor)
ms4_self = torch.from_numpy(ms4_self).type(torch.FloatTensor)
ms4_LL = torch.from_numpy(ms4_LL).type(torch.FloatTensor)
pan_self = torch.from_numpy(pan_self).type(torch.FloatTensor)
pan_LL = torch.from_numpy(pan_LL).type(torch.FloatTensor)
print(ms4.size())
print(pan.size())
print(ms4_self.size())
print(pan_self.size())
print(ms4_LL.size())
print(pan_LL.size())

the_matrix = [[] for i in range(3)]
temp = 0
matrix_ = [[] for i in range(2)]
for i in range(label_row):
    for j in range(label_column):
        the_matrix[0].append(i)
        the_matrix[1].append(j)
        the_matrix[2].append(label_np[i][j])
        if label_np[i][j] == 0:
            matrix_[0].append(temp)
        else:
            matrix_[1].append(temp)
        temp += 1
for i in range(2):
    print('标签为{}的标签集合大小为{}'.format(i, len(matrix_[i])))


class MyData(Dataset):
    def __init__(self, MS4, PAN, MS4s, PANs, MS4_LL, PAN_LL, Label, x, y, cut_size):
        self.train_data1 = MS4
        self.train_data2 = PAN
        self.train_data3 = MS4s
        self.train_data4 = PANs
        self.train_data5 = MS4_LL
        self.train_data6 = PAN_LL
        self.train_labels = Label
        self.x = x
        self.y = y
        self.cut_ms_size = cut_size
        self.cut_pan_size = cut_size * 4

    def __getitem__(self, index):
        x_ms = self.x[index]
        y_ms = self.y[index]
        x_pan = int(4 * x_ms)
        y_pan = int(4 * y_ms)
        image_ms4 = self.train_data1[:, x_ms:x_ms + self.cut_ms_size,
                    y_ms:y_ms + self.cut_ms_size]
        image_pan = self.train_data2[:, x_pan:x_pan + self.cut_pan_size,
                    y_pan:y_pan + self.cut_pan_size]
        image_ms4s = self.train_data3[:, x_ms:x_ms + self.cut_ms_size,
                     y_ms:y_ms + self.cut_ms_size]
        image_pans = self.train_data4[:, int(x_pan / 2):int(x_pan / 2 + self.cut_pan_size / 2),
                     int(y_pan / 2):int(y_pan / 2 + self.cut_pan_size / 2)]
        image_ms4_LL = self.train_data5[:, x_ms:x_ms + self.cut_ms_size,
                       y_ms:y_ms + self.cut_ms_size]
        image_pan_LL = self.train_data6[:, int(x_pan / 2):int(x_pan / 2 + self.cut_pan_size / 2),
                       int(y_pan / 2):int(y_pan / 2 + self.cut_pan_size / 2)]
        label = self.train_labels[index]
        return image_ms4, image_pan, image_ms4s, image_pans, image_ms4_LL, image_pan_LL, label, x_ms, y_ms

    def __len__(self):
        return len(self.x)


all_data = MyData(ms4, pan, ms4_self, pan_self, ms4_LL, pan_LL, the_matrix[2], the_matrix[0], the_matrix[1],
                  Ms4_patch_size)
all_data_loader = DataLoader(dataset=all_data, batch_size=8, shuffle=True, num_workers=0)

train_data = Subset(all_data, indices=matrix_[1])
train_size = int(0.02 * len(train_data))
test_size = len(train_data) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(train_data, [train_size, test_size])
train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True, num_workers=0)
test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False, num_workers=0)

verify_data = Subset(all_data, indices=matrix_[0])
verify_loader = DataLoader(dataset=verify_data, batch_size=128, shuffle=False, num_workers=0)

print(len(train_loader))
print(len(test_loader))