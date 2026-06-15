import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from utils import *

transform = transforms.Compose([
    # 将 PIL 图像转成张量，并把像素值归一化到 [0, 1]
    transforms.ToTensor()
])


class MyDataset(Dataset):
    def __init__(self, path, train_type=VOC_TRAIN_TYPE, binary_target_ids=None):
        # 数据集根目录，例如 data/
        self.path = path
        # 训练模式：
        # 1. voc    : 使用 VOC 原始多类别标签
        # 2. binary : 使用二值分割标签
        self.train_type = train_type
        # 二值分割时哪些 VOC 类别会被视为前景
        if binary_target_ids is None:
            binary_target_ids = DEFAULT_BINARY_TARGET_IDS
        self.binary_target_ids = tuple(binary_target_ids)
        # 以标签图文件名作为样本索引
        self.name = os.listdir(os.path.join(path, 'SegmentationClass'))
        self.image_suffixes = ['.jpg', '.jpeg', '.png', '.bmp']

    def __len__(self):
        # 返回数据集样本总数
        return len(self.name)

    # 根据索引取出一组“输入图像 + 分割标签”
    def __getitem__(self, index):
        segment_name = self.name[index]  # 例如 0001.png
        segment_path = os.path.join(self.path, 'SegmentationClass', segment_name)
        image_name = self._find_image_name(segment_name)
        image_path = os.path.join(self.path, 'JPEGImages', image_name)

        # 读取标签图，并补齐/缩放到统一尺寸
        segment_image = keep_image_size_open(segment_path)
        # 读取原图，并补齐/缩放到统一尺寸
        image = keep_image_size_open_rgb(image_path)

        # 标签图不能像普通图像一样直接做 ToTensor()。
        # 因为分割标签保存的是“类别编号”，不是像素强度，必须单独按类别规则处理。
        segment_array = np.array(segment_image, dtype=np.uint8)
        segment_tensor = self._build_segment_tensor(segment_array)

        # 返回：
        # 1. 输入图像张量，形状通常为 [3, 256, 256]
        # 2. 标签图张量，形状通常为 [256, 256]
        return transform(image), segment_tensor

    def _build_segment_tensor(self, segment_array):
        # 根据 train_type 选择不同的标签处理方法。
        if self.train_type == VOC_TRAIN_TYPE:
            return self._build_voc_mask(segment_array)
        if self.train_type == BINARY_TRAIN_TYPE:
            return self._build_binary_mask(segment_array)
        raise ValueError(f'Unsupported train_type: {self.train_type}')

    def _build_voc_mask(self, segment_array):
        # VOC 模式下直接保留原始类别编号：
        # 0~20 是有效类别，255 是忽略区域。
        return torch.from_numpy(segment_array.astype(np.int64))

    def _build_binary_mask(self, segment_array):
        # 二值模式下把指定类别映射为前景 1，其余有效类别映射为背景 0。
        # 255 仍然保留为忽略区域，方便继续配合 ignore_index 使用。
        binary_mask = np.zeros_like(segment_array, dtype=np.int64)
        positive_mask = np.isin(segment_array, self.binary_target_ids)
        ignore_mask = segment_array == IGNORE_INDEX
        binary_mask[positive_mask] = 1
        binary_mask[ignore_mask] = IGNORE_INDEX
        return torch.from_numpy(binary_mask)

    def _find_image_name(self, segment_name):
        # 标签图通常是 .png，而原图通常是 .jpg。
        # 这里用同名主文件名自动匹配常见图片后缀，避免再出现找不到原图的问题。
        image_stem = os.path.splitext(segment_name)[0]
        for suffix in self.image_suffixes:
            image_name = image_stem + suffix
            image_path = os.path.join(self.path, 'JPEGImages', image_name)
            if os.path.exists(image_path):
                return image_name
        raise FileNotFoundError(f'No matching image found for mask {segment_name}')


if __name__ == '__main__':
    from torch.nn.functional import one_hot

    # 简单测试数据集输出的形状是否符合预期
    data = MyDataset('data', train_type=VOC_TRAIN_TYPE)
    # data[0][0] 是原图 image，形状应该是 [3, 256, 256]
    print(data[0][0].shape)
    # data[0][1] 是标签图 segment_image，形状应该是 [256, 256]
    print(data[0][1].shape)
    # 对标签图进行 one-hot 编码前，需要确保标签值都在有效类别范围内
    valid_mask = data[0][1].clone()
    valid_mask[valid_mask == IGNORE_INDEX] = 0
    out = one_hot(valid_mask.long(), num_classes=VOC_NUM_CLASSES)
    print(out.shape)
