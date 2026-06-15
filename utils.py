import numpy as np
import torch
from PIL import Image

# 支持的训练模式：
# 1. voc    : 保留 VOC 的 21 类语义分割标签（0~20），255 表示忽略区域
# 2. binary : 把指定类别合并成“前景”，其余类别视为“背景”
VOC_TRAIN_TYPE = 'voc'
BINARY_TRAIN_TYPE = 'binary'
IGNORE_INDEX = 255

# VOC 一共 21 个有效类别（20 个目标类 + 1 个背景类）
VOC_NUM_CLASSES = 21
# 二值分割固定只有 2 类：背景 0、前景 1
BINARY_NUM_CLASSES = 2
# 默认把 VOC 中的 person（类别 15）当成二值分割的前景类
DEFAULT_BINARY_TARGET_IDS = (15,)

# VOC 常用的可视化颜色表。
# 下标就是类别编号，例如 0 是背景，15 是 person。
VOC_COLORMAP = np.array([
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128]
], dtype=np.uint8)


def keep_image_size_open(path, size=(256, 256)):
    # 读取标签图（mask）
    img = Image.open(path)
    # 取原图的最长边，用它来构造一个正方形画布
    temp = max(img.size)
    # 'P' 模式常用于调色板图像，适合分割标签图
    mask = Image.new('P', (temp, temp))
    # 将原图贴到左上角，较短边方向会自动补空白
    mask.paste(img, (0, 0))
    # 标签图必须使用最近邻插值，避免把类别编号插值成新的脏值
    mask = mask.resize(size, Image.NEAREST)
    return mask


def keep_image_size_open_rgb(path, size=(256, 256)):
    # 读取原始 RGB 图像
    img = Image.open(path)
    # 同样先补成正方形，避免直接拉伸导致形变过大
    temp = max(img.size)
    mask = Image.new('RGB', (temp, temp))
    mask.paste(img, (0, 0))
    # 原图是自然图像，使用双线性插值更平滑
    mask = mask.resize(size, Image.BILINEAR)
    return mask


def build_voc_config():
    # VOC 训练时保留全部类别编号，255 仍然表示忽略区域
    return {
        'num_classes': VOC_NUM_CLASSES,
        'ignore_index': IGNORE_INDEX
    }


def build_binary_config(binary_target_ids=None):
    # 二值分割时只保留“背景/前景”两类。
    # binary_target_ids 里列出的类别会被映射为前景 1，其余有效类别映射为背景 0。
    if binary_target_ids is None:
        binary_target_ids = DEFAULT_BINARY_TARGET_IDS
    return {
        'num_classes': BINARY_NUM_CLASSES,
        'ignore_index': IGNORE_INDEX,
        'binary_target_ids': tuple(binary_target_ids)
    }


def get_train_config(train_type, binary_target_ids=None):
    # 根据 train_type 返回对应的训练配置。
    # 这样 train.py / test.py / data.py 都可以共用同一套规则。
    if train_type == VOC_TRAIN_TYPE:
        return build_voc_config()
    if train_type == BINARY_TRAIN_TYPE:
        return build_binary_config(binary_target_ids)
    raise ValueError(f'Unsupported train_type: {train_type}')


def mask_to_visual_tensor(mask_tensor, train_type):
    # 把类别编号图转换成可直接保存/显示的 RGB 图。
    # 训练时保存效果图、测试时保存预测图都可以复用这个方法。
    if isinstance(mask_tensor, torch.Tensor):
        mask_array = mask_tensor.detach().cpu().numpy()
    else:
        mask_array = np.asarray(mask_tensor)

    if mask_array.ndim == 3 and mask_array.shape[0] == 1:
        mask_array = mask_array[0]

    mask_array = mask_array.astype(np.int64)

    if train_type == BINARY_TRAIN_TYPE:
        # 二值分割：背景显示为黑色，前景显示为白色，忽略区域显示为灰色。
        visual = np.zeros((mask_array.shape[0], mask_array.shape[1], 3), dtype=np.uint8)
        visual[mask_array == 1] = [255, 255, 255]
        visual[mask_array == IGNORE_INDEX] = [127, 127, 127]
    elif train_type == VOC_TRAIN_TYPE:
        # VOC 多类别分割：按类别编号映射到固定颜色表。
        valid_mask = np.clip(mask_array, 0, VOC_NUM_CLASSES - 1)
        visual = VOC_COLORMAP[valid_mask]
        visual[mask_array == IGNORE_INDEX] = [224, 224, 224]
    else:
        raise ValueError(f'Unsupported train_type: {train_type}')

    return torch.from_numpy(visual).permute(2, 0, 1).float() / 255.0
