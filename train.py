import os

import torch
import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from data import *
from net import *
from utils import *

# 自动选择训练设备：优先使用 GPU，没有 GPU 时退回 CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 训练过程中会用到的几个路径
# weight_path: 模型参数保存位置
# data_path:   数据集根目录
# save_path:   训练时保存预测效果图的位置
weight_path = 'params/unet.pth'
data_path = r'data'
save_path = 'train_image'

# 训练模式：
# 1. 'voc'    : 使用 VOC 原始 21 类分割
# 2. 'binary' : 使用二值分割（背景/前景）
train_type = VOC_TRAIN_TYPE

# 二值分割时要当成“前景”的类别编号。
# 例如 VOC 中 person 的编号是 15，如果想分割人，可以保持默认值不改。
binary_target_ids = DEFAULT_BINARY_TARGET_IDS


if __name__ == '__main__':
    # 根据 train_type 自动选择类别数、忽略标签等训练配置。
    # 这样切换训练模式时，不需要再手动改 num_classes 和 loss。
    train_config = get_train_config(train_type, binary_target_ids)
    num_classes = train_config['num_classes']
    ignore_index = train_config['ignore_index']

    # 构造数据加载器、网络、优化器和损失函数
    # 如果显存不够，可以优先调小 batch_size
    # DataLoader 会自动把数据集分成一个个 batch，并在训练时按需加载
    # shuffle=True 每个 epoch 都会打乱样本顺序，减少训练偏差。
    data_loader = DataLoader(
        MyDataset(data_path, train_type=train_type, binary_target_ids=binary_target_ids),
        batch_size=1,
        shuffle=True
    )
    # 输出形状是 [B, num_classes, H, W]
    net = UNet(num_classes).to(device)

    # 如果之前训练过，就读取已有权重继续训练；否则从头开始训练
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path, map_location=device))
        print('successful load weight!')
    else:
        print('not successful load weight')

    # 优化器： Adam 用于根据梯度更新网络参数
    opt = optim.Adam(net.parameters())
    # 交叉熵损失用于按像素做分类。
    # 标签中等于 ignore_index 的位置会被跳过，不参与损失计算。
    loss_fun = nn.CrossEntropyLoss(ignore_index=ignore_index)

    # 确保保存效果图的目录存在，避免第一次训练时报路径不存在。
    os.makedirs(save_path, exist_ok=True)

    # 从第 1 轮开始训练，一直训练到第 199 轮
    epoch = 1
    while epoch < 200:
        # 每一轮都会完整遍历一次数据集
        for i, (image, segment_image) in enumerate(tqdm.tqdm(data_loader)):
            # 把输入图像和标签图像都移动到训练设备上。
            # CrossEntropyLoss 要求标签为 long 类型，并且形状是 [B, H, W]。
            image = image.to(device)
            segment_image = segment_image.long().to(device)

            # 前向传播：得到每个像素在各类别上的预测分数
            out_image = net(image)

            # 计算预测结果和真实标签之间的损失
            train_loss = loss_fun(out_image, segment_image)

            # 反向传播与参数更新
            opt.zero_grad()
            train_loss.backward()
            opt.step()

            # 打印当前 batch 的训练损失
            if i % 1 == 0:
                print(f'{epoch}-{i}-train_loss===>>{train_loss.item()}')

            # 取出 batch 中第一张图的原图、标签和预测结果，保存用于观察训练效果。
            # 不同训练模式下，标签的可视化方式不同，所以统一走公共转换函数。
            _image = image[0].detach().cpu()
            _segment_image = mask_to_visual_tensor(segment_image[0], train_type)
            _out_image = mask_to_visual_tensor(torch.argmax(out_image[0], dim=0), train_type)

            # 按“原图 | 真实标签 | 预测结果”的顺序拼成三联图，便于和视频里的效果一致。
            img = torch.stack([_image, _segment_image, _out_image], dim=0)
            save_image(img, f'{save_path}/{i}.png')

        # 每 50 轮保存一次模型参数
        if epoch % 50 == 0:
            torch.save(net.state_dict(), weight_path)
            print('save successfully!')

        # 进入下一轮训练
        epoch += 1
