import os

import cv2
import numpy as np
import torch

from data import *
from net import *
from utils import *

# 测试时的训练模式需要和训练权重保持一致。
# 例如：如果训练时用的是 VOC 全类别，这里也要写 VOC_TRAIN_TYPE。
train_type = VOC_TRAIN_TYPE

# 二值分割时要当成前景的类别编号，需要和训练时保持一致。
binary_target_ids = DEFAULT_BINARY_TARGET_IDS

# 根据 train_type 自动选择模型输出类别数。
test_config = get_train_config(train_type, binary_target_ids)
num_classes = test_config['num_classes']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = UNet(num_classes).to(device)

weights = 'params/unet.pth'
if os.path.exists(weights):
    net.load_state_dict(torch.load(weights, map_location=device))
    print('successfully')
else:
    print('no loading')

_input = input('please input JPEGImages path:')

# 推理时的原图预处理要和训练时保持一致。
img = keep_image_size_open_rgb(_input)
img_data = transform(img).to(device)
img_data = torch.unsqueeze(img_data, dim=0)

net.eval()
with torch.no_grad():
    out = net(img_data)
    out = torch.argmax(out, dim=1)
    out = torch.squeeze(out, dim=0)

# 输出这张图里实际预测到了哪些类别编号，便于快速排查结果是否合理。
print(set(out.reshape(-1).tolist()))

# 根据训练模式把类别图转换成可视化 RGB 图。
visual = mask_to_visual_tensor(out, train_type)
visual = (visual.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

os.makedirs('result', exist_ok=True)
cv2.imwrite('result/result.png', cv2.cvtColor(visual, cv2.COLOR_RGB2BGR))
cv2.imshow('out', cv2.cvtColor(visual, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
