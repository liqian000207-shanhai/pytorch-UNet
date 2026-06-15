import torch
from torch import nn
from torch.nn import functional as F


# 基础卷积块（特征提取模块）：
# 连续做两次 3x3 卷积，每次卷积后都接 BatchNorm、Dropout 和 LeakyReLU。
#第一次把输入变成更丰富的特征，第二次继续提炼这些特征。

#Conv2d：提取图像特征
#BatchNorm2d：让训练更稳定  
#Dropout2d(0.3)：随机丢掉一部分特征，减少过拟合
#LeakyReLU()：加入非线性，避免网络只会做线性变换
# 这是 U-Net 编码器和解码器中最常用的特征提取模块。
#Conv_Block(1, 64)：[B, 1, 256, 256] -> [B, 64, 256, 256]
class Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv_Block, self).__init__()
        self.layer = nn.Sequential(
            #卷积 3x3、步长 1、padding 1。
            # padding_mode='reflect' 可以减少边界效应
            # bias=False 因为后面有 BatchNorm2d 会有偏置项了，这里卷积层就不需要了
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


# 下采样模块：
# 使用步长为 2 的卷积将特征图宽高减半，通道数保持不变。
# 例如 [B, 64, 256, 256] -> [B, 64, 128, 128]
class DownSample(nn.Module):
    def __init__(self, channel):
        super(DownSample, self).__init__()
        self.layer = nn.Sequential(
            #步长stride=2：宽高减半
            #这里它和很多教材里的 MaxPool2d(2) 下采样不一样，MaxPool2d没有特征提取能力，丢特征太多。
            #MaxPool2d(2) ：把每个 2x2 小块里“最大的那个数”留下，其余 3 个直接丢掉。
            #主要作用是“下采样、缩小尺寸”，不是“学习特征”。
            #因为它没有可学习参数，不会像卷积核那样自己学“边缘、纹理、形状”这些模式。
            #一个 2x2 区域原来有 4 个值，池化后只剩 1 个值，很多细节信息直接没了。
            #对分类任务这通常还能接受，因为分类更关心“有没有这个东西”；
            #但对分割任务，边缘、位置、细小结构很重要，池化太狠时容易把这些细节抹掉。
            #这份代码用的是“3*3卷积，步长为 2”来做下采样，而不是最大池化。

            nn.Conv2d(channel, channel, 3, 2, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


# 上采样模块：
# 1. 先用插值法把特征图放大 2 倍（不用原版的转置卷积，因为转置卷积会有棋盘效应）
# 2. 再用 1x1 卷积把通道数减半
# 3. 最后和编码器对应层的特征图做拼接（跳跃连接）
class UpSample(nn.Module):
    def __init__(self, channel):
        super(UpSample, self).__init__()
        #// 表示整除（向下取整除法）
        #channel // 2：上采样后通道数减半，和编码器对应层的通道数一致，方便拼接
        #上采样后通道数是 channel，编码器对应层的通道数是 channel // 2，所以这里卷积后通道数也要变成 channel // 2。
        #1x1 卷积：改变通道数，但不改变空间尺寸
        self.layer = nn.Conv2d(channel, channel // 2, 1, 1)

    def forward(self, x, feature_map):
        # 例如 [B, 1024, 16, 16] -> [B, 1024, 32, 32]
        # mode='nearest'（最近邻插值），也就是直接把相邻的像素复制一遍来放大图像。
        # scale_factor=2：空间尺寸扩大了两倍（通道数没变）。
        up = F.interpolate(x, scale_factor=2, mode='nearest')

        # 例如 [B, 1024, 32, 32] -> [B, 512, 32, 32]
        out = self.layer(up)

        # 与编码器同尺度特征拼接，恢复空间细节信息
        # out 是我们刚刚从深层网络放大上来的特征，通道数是 512。
        # feature_map 是从左侧编码器（Encoder）原封不动“抄”过来的浅层特征图，通道数也是 512。
        # 例如和 R4 [B, 512, 32, 32] 拼接后得到 [B, 1024, 32, 32]
        return torch.cat((out, feature_map), dim=1)


# U-Net 主体：
# 左边是编码器，负责逐步下采样并提取更强的语义特征；
# 右边是解码器，负责逐步上采样恢复分辨率；
# 中间通过跳跃连接把浅层细节信息传给解码器。
class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()

        # 编码器部分
        self.c1 = Conv_Block(3, 64)
        self.d1 = DownSample(64)
        self.c2 = Conv_Block(64, 128)
        self.d2 = DownSample(128)
        self.c3 = Conv_Block(128, 256)
        self.d3 = DownSample(256)
        self.c4 = Conv_Block(256, 512)
        self.d4 = DownSample(512)
        self.c5 = Conv_Block(512, 1024)

        # 解码器部分
        self.u1 = UpSample(1024)
        self.c6 = Conv_Block(1024, 512)
        self.u2 = UpSample(512)
        self.c7 = Conv_Block(512, 256)
        self.u3 = UpSample(256)
        self.c8 = Conv_Block(256, 128)
        self.u4 = UpSample(128)
        self.c9 = Conv_Block(128, 64)

        # 最后一层将通道数映射到类别数
        # 输出形状为 [B, num_classes, H, W]
        self.out = nn.Conv2d(64, num_classes, 3, 1, 1)

    def forward(self, x):
        # R1 ~ R5 是编码器各层输出
        # 尺寸变化：
        # x   : [B, 3,   256, 256]
        # R1  : [B, 64,  256, 256]
        R1 = self.c1(x)

        # R2  : [B, 128, 128, 128]
        R2 = self.c2(self.d1(R1))

        # R3  : [B, 256, 64, 64]
        R3 = self.c3(self.d2(R2))

        # R4  : [B, 512, 32, 32]
        R4 = self.c4(self.d3(R3))

        # R5  : [B, 1024, 16, 16]
        R5 = self.c5(self.d4(R4))

        # O1 ~ O4 是解码器各层输出
        # R5 上采样后与 R4 拼接，再卷积
        # O1  : [B, 512, 32, 32]
        O1 = self.c6(self.u1(R5, R4))

        # O2  : [B, 256, 64, 64]
        O2 = self.c7(self.u2(O1, R3))

        # O3  : [B, 128, 128, 128]
        O3 = self.c8(self.u3(O2, R2))

        # O4  : [B, 64, 256, 256]
        O4 = self.c9(self.u4(O3, R1))

        # 输出每个像素在各个类别上的预测分数
        return self.out(O4)


if __name__ == '__main__':
    # 构造一个 batch=2 的假输入，验证网络输出形状
    x = torch.randn(2, 3, 256, 256)
    net = UNet(num_classes=2)
    print(net(x).shape)
