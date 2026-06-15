# pytorch-unet

## 项目简介

这是一个基于 PyTorch 实现的 U-Net 语义分割项目，当前项目支持两种训练方式：

1. `VOC` 多类别分割
2. `binary` 二值分割

项目默认使用 Pascal VOC 风格的数据目录：

- 原图目录：`data/JPEGImages`
- 标签目录：`data/SegmentationClass`

目前代码已经支持通过 `train_type` 在两种训练方式之间切换，不需要再手动到多个文件里分别改类别数、损失函数或标签处理逻辑。

## 当前功能

- 使用 U-Net 做语义分割训练与测试
- 自动匹配标签图与原图同名但不同后缀的文件
- 支持 `VOC` 21 类分割
- 支持二值分割
- 统一处理 `ignore_index=255`
- 训练过程中保存标签/预测可视化结果
- 测试时保存预测结果图到 `result/result.png`

## 项目结构

```text
pytorch-unet/
├─ data/
│  ├─ JPEGImages/              # 原图
│  ├─ SegmentationClass/       # 分割标签
│  └─ make_mask_data.py        # 可能的数据处理脚本
├─ params/                     # 保存训练权重
├─ result/                     # 保存测试结果
├─ train_image/                # 保存训练过程中的可视化结果
├─ data.py                     # 数据集读取与标签处理
├─ net.py                      # U-Net 网络结构
├─ test.py                     # 单张图片测试
├─ train.py                    # 训练入口
├─ utils.py                    # 公共工具函数与训练模式配置
└─ README.md                   # 项目说明文档
```

## 环境要求

建议环境：

- Python 3.9 或更高
- PyTorch
- torchvision
- numpy
- pillow
- opencv-python
- tqdm

如果你使用 Conda，可以先创建环境后再安装依赖。

示例：

```bash
conda create -n torch_study python=3.10
conda activate torch_study
pip install torch torchvision numpy pillow opencv-python tqdm
```

## 数据集格式要求

项目按 VOC 风格读取数据，要求目录结构如下：

```text
data/
├─ JPEGImages/
│  ├─ 2007_000027.jpg
│  ├─ 2007_000032.jpg
│  └─ ...
├─ SegmentationClass/
│  ├─ 2007_000027.png
│  ├─ 2007_000032.png
│  └─ ...
```

注意：

1. 标签图文件名和原图文件名的主文件名必须一致。
2. 标签图通常是 `.png`，原图通常是 `.jpg`。
3. 当前代码会自动根据主文件名去 `JPEGImages` 中匹配 `.jpg/.jpeg/.png/.bmp`。

例如：

- 标签：`2010_005700.png`
- 原图：`2010_005700.jpg`

这类情况已经可以正常匹配。

## 两种训练模式

### 1. VOC 多类别分割

在这种模式下，标签图保留原始 VOC 类别编号：

- 有效类别：`0 ~ 20`
- 忽略区域：`255`

这时：

- `num_classes = 21`
- `ignore_index = 255`
- 预测结果会保存为彩色类别图

适合场景：

- 想直接使用 VOC 原始标签训练
- 想做多类别语义分割

### 2. 二值分割

在这种模式下，项目会把指定类别映射成“前景”，其它有效类别映射成“背景”：

- 背景：`0`
- 前景：`1`
- 忽略区域：`255`

这时：

- `num_classes = 2`
- `ignore_index = 255`
- 预测结果会保存为黑白图

默认设置：

```python
DEFAULT_BINARY_TARGET_IDS = (15,)
```

这里的 `15` 是 VOC 里的 `person` 类。  
也就是说，默认二值分割是“人 vs 非人”。

如果你想分割别的类别，可以修改 `binary_target_ids`。

## 如何切换训练模式

### 训练脚本

在 [train.py](/f:/读研日常/Pytorch/pytorch-unet/train.py:22) 中修改：

```python
train_type = VOC_TRAIN_TYPE
```

可选值：

- `VOC_TRAIN_TYPE`
- `BINARY_TRAIN_TYPE`

如果使用二值分割，还要设置前景类别：

```python
binary_target_ids = DEFAULT_BINARY_TARGET_IDS
```

例如如果你想把多个 VOC 类别同时当成前景，可以写成：

```python
binary_target_ids = (7, 15)
```

### 测试脚本

在 [test.py](/f:/读研日常/Pytorch/pytorch-unet/test.py:11) 中也要做同样设置：

```python
train_type = VOC_TRAIN_TYPE
binary_target_ids = DEFAULT_BINARY_TARGET_IDS
```

注意：

1. `test.py` 中的 `train_type` 必须和训练时一致
2. `binary_target_ids` 也必须和训练时一致

否则测试结果会和训练权重不匹配。

## 训练流程

### 1. 准备数据

把原图放到：

```text
data/JPEGImages
```

把标签图放到：

```text
data/SegmentationClass
```

### 2. 选择训练模式

在 `train.py` 中设置：

```python
train_type = VOC_TRAIN_TYPE
```

或者：

```python
train_type = BINARY_TRAIN_TYPE
```

### 3. 开始训练

```bash
python train.py
```

训练时：

- 权重保存在 `params/unet.pth`
- 可视化结果保存在 `train_image/`

### 4. 观察训练效果图

训练过程中会自动保存标签图和预测图的对比结果。

不同模式下显示方式不同：

- `VOC` 模式：彩色类别图
- `binary` 模式：黑白分割图

## 测试流程

运行：

```bash
python test.py
```

然后输入一张图片路径，例如：

```text
data/JPEGImages/2007_000032.jpg
```

测试结果会：

- 弹窗显示
- 保存到 `result/result.png`

## 核心文件说明

### `net.py`

定义 U-Net 网络结构，主要包括：

- `Conv_Block`：双卷积特征提取模块
- `DownSample`：下采样模块
- `UpSample`：上采样模块
- `UNet`：完整网络

当前网络输入为：

```text
[B, 3, 256, 256]
```

输出为：

```text
[B, num_classes, 256, 256]
```

### `data.py`

负责：

- 读取原图和标签图
- 自动匹配原图后缀
- 根据 `train_type` 生成不同的标签张量

关键方法：

- `__getitem__()`：读取一对图像与标签
- `_build_voc_mask()`：生成 VOC 多类别标签
- `_build_binary_mask()`：生成二值分割标签
- `_find_image_name()`：自动匹配原图文件名

### `utils.py`

负责：

- 图像预处理
- 标签图缩放
- 训练模式配置
- 掩码可视化

关键常量：

- `VOC_TRAIN_TYPE`
- `BINARY_TRAIN_TYPE`
- `IGNORE_INDEX`
- `VOC_NUM_CLASSES`
- `BINARY_NUM_CLASSES`
- `DEFAULT_BINARY_TARGET_IDS`

关键方法：

- `get_train_config()`
- `mask_to_visual_tensor()`
- `keep_image_size_open()`
- `keep_image_size_open_rgb()`

### `train.py`

负责：

- 根据 `train_type` 构造训练配置
- 创建数据集、模型、优化器、损失函数
- 训练循环
- 保存训练过程可视化结果
- 定期保存模型权重

### `test.py`

负责：

- 加载训练权重
- 读取单张测试图片
- 推理得到类别图
- 按训练模式保存可视化结果

## 标签处理说明

### 为什么标签图不能直接 `ToTensor()`

普通图像的 `ToTensor()` 会把像素值归一化到 `[0,1]`，这适合自然图像，但不适合分割标签图。

因为分割标签图里保存的不是颜色强度，而是“类别编号”。

例如：

- `0` 代表背景
- `15` 代表 `person`
- `255` 代表忽略区域

如果直接对标签做普通归一化，类别编号会被破坏。

所以当前项目对标签图采用了单独处理：

- 先转成 `numpy`
- 再根据 `train_type` 做类别映射
- 最后转成 `long` 类型张量

### 为什么标签图要用最近邻插值

标签图在缩放时不能用双线性插值。

原因是双线性插值会把离散类别编号“混合”出新的中间值，导致标签脏掉。

例如本来只有：

```text
0, 15, 255
```

插值后可能产生根本不存在的新数值。

所以当前项目在处理标签图时使用：

```python
Image.NEAREST
```

## 常见问题

### 1. 找不到图片

典型报错：

```text
FileNotFoundError: No such file or directory: 'data\\JPEGImages\\2010_005700.png'
```

原因：

- 标签是 `.png`
- 原图是 `.jpg`
- 旧代码直接拿标签名去找原图

当前项目已经通过 `_find_image_name()` 解决这个问题。

### 2. `CrossEntropyLoss` 提示 target 维度不对

典型报错：

```text
only batches of spatial targets supported (3D tensors)
```

原因：

- `CrossEntropyLoss` 需要标签形状是 `[B, H, W]`
- 旧写法把标签处理成了 `[B, 1, H, W]`

当前项目已经把标签改成返回二维类别图，训练时再组成 `[B, H, W]`。

### 3. CUDA 报 `t >= 0 && t < n_classes`

典型报错：

```text
Assertion `t >= 0 && t < n_classes` failed
```

原因通常有两个：

1. `num_classes` 配置小于标签真实类别数
2. 标签中有 `255`，但损失函数没有设置 `ignore_index=255`

当前项目已统一从 `train_type` 自动生成：

- `num_classes`
- `ignore_index`

### 4. 视频里是黑白图，我这里是彩色图

这是正常现象。

原因是任务不同：

- `VOC` 多类别分割：输出彩色类别图
- `binary` 二值分割：输出黑白图

如果你想得到视频里那种黑白图，请使用：

```python
train_type = BINARY_TRAIN_TYPE
```

### 5. 中文注释乱码

项目现在默认按 `UTF-8` 维护。

如果你看到终端输出乱码，很多时候只是终端编码显示问题，不一定是文件本身坏了。  
实际编辑和保存时应继续保持 UTF-8。

## 当前默认配置

当前代码默认配置是：

- `train_type = VOC_TRAIN_TYPE`
- `binary_target_ids = DEFAULT_BINARY_TARGET_IDS`
- 输入尺寸：`256 x 256`
- 权重文件：`params/unet.pth`

如果你想改成二值分割，最少要同步修改：

1. `train.py` 中的 `train_type`
2. `test.py` 中的 `train_type`
3. 如果需要，修改 `binary_target_ids`

## 可继续扩展的方向

- 增加验证集评估
- 增加 mIoU、Pixel Accuracy 等指标
- 增加配置文件支持
- 支持多张图片批量测试
- 支持自定义颜色表
- 支持保存原图、标签、预测三联图

## 维护约定

从当前版本开始，本项目采用以下维护约定：

1. 只要项目中的训练逻辑、数据处理逻辑、测试逻辑、可视化逻辑发生变更，就必须同步更新 `README.md`
2. 新增或修改中文注释、中文文档时，统一使用 `UTF-8` 编码，避免乱码
3. 如果新增新的训练模式，也要在 `README.md` 中补充：
   - 使用场景
   - 配置方法
   - 训练方式
   - 测试方式
   - 标签处理规则
4. 如果修改了 `train.py` 和 `test.py` 的关键开关，README 中的对应说明也要一起改

这条约定是当前项目的一部分，后续继续维护这个仓库时默认遵守。

## 参考数据集

Pascal VOC 2012：

```text
http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
```

## 参考视频

B 站：

```text
https://www.bilibili.com/video/BV11341127iK
```
