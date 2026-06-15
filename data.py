import os

import numpy as np
import torch
from torch.utils.data import Dataset
from utils import *
from torchvision import transforms

transform = transforms.Compose([
    # 将 PIL 图像转成张量，并把像素值归一化到 [0, 1]
    transforms.ToTensor()
])


class MyDataset(Dataset):
    def __init__(self, path):
        # 数据集根目录，例如 data/
        self.path = path
        # 以标签图文件名作为样本索引
        self.name = os.listdir(os.path.join(path, 'SegmentationClass'))

    def __len__(self):
        # 返回数据集样本总数
        return len(self.name)

    # 根据索引取出一组“输入图像 + 分割标签”
    def __getitem__(self, index):
        segment_name = self.name[index]  # 例如 0001.png
        segment_path = os.path.join(self.path, 'SegmentationClass', segment_name)
        image_path = os.path.join(self.path, 'JPEGImages', segment_name)

        # 读取标签图，并补齐/缩放到统一尺寸
        segment_image = keep_image_size_open(segment_path)
        # 读取原图，并补齐/缩放到统一尺寸
        image = keep_image_size_open_rgb(image_path)

        # 返回：
        # 1. 输入图像张量，形状通常为 [3, 256, 256]
        # 2. 标签图张量，形状通常为 [256, 256]
        #return transform(image), torch.Tensor(np.array(segment_image))

        # 返回：
        # 1. 输入图像张量，形状通常为 [3, 256, 256]
        # 2. 标签图张量，形状通常为 [1，256, 256]
        return transform(image), transform(segment_image)


if __name__ == '__main__':
    from torch.nn.functional import one_hot

    # 简单测试数据集输出的形状是否符合预期
    data = MyDataset('data')
    #data[0][0] 是原图 image，形状应该是 [3, 256, 256]
    print(data[0][0].shape)
    #data[0][1] 是标签图 segment_image，形状应该是 [256, 256]
    print(data[0][1].shape)
    # 对标签图进行 one-hot 编码，形状应该是 [256, 256, num_classes]
    #one-hot 编码会把每个像素的类别索引转换成一个长度为 num_classes 的向量，只有对应类别的位置是 1，其余位置是 0
    out = one_hot(data[0][1].long())
    print(out.shape)
