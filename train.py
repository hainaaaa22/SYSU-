# ========== 库 ===========
import os                #处理文件路径
import json              #读取标签映射
import pandas as pd      #读取csv
from PIL import Image    #读取图片

#pytorch 核心
import torch
import torch.nn as nn                              #神经网络层，涉及卷积，全连接，损失函数
import torch.optim as optim                        #优化器，用来更新模型参数
from torch.utils.data import Dataset, DataLoader   #数据加载的标准接口

#计算机视觉工具
from torchvision import models, transforms         #CV相关，包括预训练模型，图像预处理/数据增强，常用数据集

#进度条
from tqdm import tqdm


# ========== 配置路径 ===========
train_csv = "数据集/train.csv"
test_csv = "数据集/test.csv"
label_map = "数据集/label_map.json"
image_root = "数据集/images"

# ========== 设备配置 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备：", device)
print(torch.cuda.is_available())  # True=有N卡，False=没有
print("torch.__version__:", torch.__version__)      # 有没有 +cu128？
print("torch.version.cuda:", torch.version.cuda)    # 是不是 12.8？
print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.cuda.device_count():", torch.cuda.device_count())