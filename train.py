import os

import torch
import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from data import *
from net import *

# 自动选择训练设备：优先使用 GPU，没有 GPU 时退回 CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 训练过程中会用到的几个路径
# weight_path: 模型参数保存位置
# data_path:   数据集根目录
# save_path:   训练时保存预测效果图的位置
weight_path = 'params/unet.pth'
data_path = r'data'
save_path = 'train_image'

if __name__ == '__main__':
    # 分割类别数。这里的 +1 一般表示把背景也作为一个类别
    # 例如，如果要分割的对象有 2 类，那么 num_classes 就是 3（背景 + 类别1 + 类别2）
    # 这个值需要与数据集中的标签类别数一致，否则训练时会报错
    num_classes = 2 + 1

    # 构造数据加载器、网络、优化器和损失函数
    # 如果显存不够，可以优先调小 batch_size
    # DataLoader 会自动把数据集分成一个个 batch，并在训练时按需加载
    # shuffle=True 每个 epoch 都会打乱样本顺序，减少训练偏差。
    data_loader = DataLoader(MyDataset(data_path), batch_size=1, shuffle=True)
    # 输出形状是 [B, num_classes, H, W]
    net = UNet(num_classes).to(device)

    # 如果之前训练过，就读取已有权重继续训练；否则从头开始训练
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print('successful load weight!')
    else:
        print('not successful load weight')

    # 优化器： Adam 用于根据梯度更新网络参数
    opt = optim.Adam(net.parameters())
    # 损失函数： 交叉熵损失常用于多分类语义分割
    loss_fun = nn.CrossEntropyLoss()

    # 从第 1 轮开始训练，一直训练到第 199 轮
    epoch = 1
    while epoch < 200:
        # 每一轮都会完整遍历一次数据集
        for i, (image, segment_image) in enumerate(tqdm.tqdm(data_loader)):
            # 把输入图像和标签图像都移动到训练设备上
            image, segment_image = image.to(device), segment_image.to(device)

            # 前向传播：得到每个像素在各类别上的预测分数
            out_image = net(image)

            # 计算预测结果和真实标签之间的损失
            # CrossEntropyLoss 要求标签为 long 类型
            train_loss = loss_fun(out_image, segment_image.long())

            # 反向传播与参数更新
            opt.zero_grad()
            train_loss.backward()
            opt.step()

            # 打印当前 batch 的训练损失
            if i % 1 == 0:
                print(f'{epoch}-{i}-train_loss===>>{train_loss.item()}')

            # 取出 batch 中第一张图的标签和预测结果，保存用于观察训练效果
            _image = image[0]
            # 在指定位置插入一个长度为 1 的新维度
            _segment_image = torch.unsqueeze(segment_image[0], 0) * 255

            # 在类别维上取最大值，得到每个像素最终预测的类别编号
            _out_image = torch.argmax(out_image[0], dim=0).unsqueeze(0) * 255

            # 把真实标签与预测结果堆叠在一起保存成图片，便于对比观察
            img = torch.stack([_segment_image, _out_image], dim=0)

            save_image(img, f'{save_path}/{i}.png')

        # 每 20 轮保存一次模型参数
        if epoch % 20 == 0:
            torch.save(net.state_dict(), weight_path)
            print('save successfully!')

        # 进入下一轮训练
        epoch += 1
