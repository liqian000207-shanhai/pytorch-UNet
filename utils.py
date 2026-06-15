from PIL import Image
# 处理图像的工具函数，主要用于数据预处理阶段

#处理标签图： 把原来长宽不一致的图统一成模型输入尺寸，避免后面训练时尺寸对不上
def keep_image_size_open(path, size=(256, 256)):
    # 读取标签图（mask）
    img = Image.open(path)
    # 取原图的最长边，用它来构造一个正方形画布
    temp = max(img.size)
    # 'P' 模式常用于调色板图像，适合分割标签图
    mask = Image.new('P', (temp, temp))
    # 将原图贴到左上角，较短边方向会自动补空白
    mask.paste(img, (0, 0))
    # 缩放到训练时需要的固定尺寸
    mask = mask.resize(size)
    return mask

##处理原图：把原来长宽不一致的图统一成模型输入尺寸，避免后面训练时尺寸对不上
def keep_image_size_open_rgb(path, size=(256, 256)):
    # 读取原始 RGB 图像
    img = Image.open(path)
    # 同样先补成正方形，避免直接拉伸导致形变过大
    temp = max(img.size)
    mask = Image.new('RGB', (temp, temp))
    mask.paste(img, (0, 0))
    mask = mask.resize(size)
    return mask
