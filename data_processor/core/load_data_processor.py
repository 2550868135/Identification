import sys

import numpy as np
import os
import torch
from torch.utils.data import Dataset
import cv2
# 用于归一化数据
import torchvision.transforms as transforms
from PIL import Image

# 图像类别映射
FILE_LABEL_MAPPING = {"positive_id_card": 0, "negative_id_card": 1,
                      "positive_social_security_card": 2, "negative_social_security_card": 3,
                      "name": 4, "other": 5}


# 训练数据预处理器
class ICDARDataset(Dataset):
    def __init__(self,
                 file_path):
        '''

        :param file_path: 输入的文件所在的根目录  identification/data
        :param label_category: 图片所在的目录
        :param labelsdir: 对应的标签名称
        '''
        if not os.path.isdir(file_path):
            raise Exception('[ERROR] {} is not a directory'.format(file_path))

        # 训练数据的图片名,用于保存所有的图片名到img_names[]中，在getitem方法中得到相对应的文件。保存的图片名为完整路径：路径+图片名
        self.img_names = []
        self.label = []
        # 读取/data目录下所有的文件分类文件加到 file_path_child
        for file_path_child in os.listdir(file_path):
            # 读取/data/子目录下的图片名
            for img_Name in os.listdir(file_path + '/' + file_path_child):
                # 保存完整的图片名称
                img_name = str(file_path + '/' + file_path_child + '/' + img_Name)
                # 将得到的图片加入到img_names列表中
                self.img_names.append(img_name)
                self.label.append(FILE_LABEL_MAPPING[file_path_child])

    # 得到img_names[]的长度
    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        '''
        修改通过下标访问列表数据的getitem方法
        :param self:
        :param idx: 传入的下标
        :return:
        '''
        # 通过传入的下标获得图像样本的名称
        img_name = self.img_names[idx]
        img_category = self.label[idx]
        # 通过图片名称读入图片数据
        img = cv2.imdecode(np.fromfile(os.path.join(img_name), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        # 如果读入图片为png图片，将图片改为3通道。JPG图片不用修改，本身就是3通道
        if img.shape[-1] == 4 :
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)  # 4通道转化为rgb三通道
        height = img.shape[0]
        width = img.shape[1]
        # 填充图片高度到与宽度相等
        if width > height:
            padding_size = (width - height) // 2
            pad_img = cv2.copyMakeBorder(img, padding_size, padding_size, 0, 0, cv2.BORDER_CONSTANT, value=(127,127,127))
        elif height > width:
            padding_size = (height - width) // 2
            pad_img = cv2.copyMakeBorder(img, 0, 0, padding_size, padding_size, cv2.BORDER_CONSTANT, value=(127,127,127))
        # 裁剪图片大小 512*512
        img = cv2.resize(img, [256,256])
        # m_img = img - IMAGE_MEAN  #均值化处理，用归一化代替
        # 通过ToTensor类的call方法将 img 转成Tensor形式的过程中1.会自动正则化到0~1范围内 2.自动将shape改为3*512*512
        img = transforms.ToTensor()(img)
        # 均值0.5，方差0.5 正则化到-1~1之间
        # img.sub_(0.5).div_(0.5)


        return img, img_category


if __name__ == '__main__':
    icd = ICDARDataset("../../data")
    for i in range(0,len(icd)):
        if icd[i][0].shape[0] == 1:
            print(i, icd[i][0].shape, icd[i][1], sep=' ')
