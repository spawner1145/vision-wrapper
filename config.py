# config.py

import torch

# --- 数据集和路径设置 ---
DATA_DIR = "./data"  # 您的数据集根目录
CHECKPOINT_DIR = "./checkpoints"  # 模型保存路径
LOG_DIR = "./logs" # 日志保存路径

# --- 模型设置 ---
# 'vit_base_patch16_224', 'efficientnet_b0', 'resnet50' 等, 从timm库中选择
MODEL_NAME = 'vit_base_patch16_224'
PRETRAINED = True  # 是否使用预训练权重
IMAGE_SIZE = 224

# --- 训练目标控制 ---
# 在此列表中添加 'pooled' 或 'mask' 来忽略对应的目标
# 例如: ['pooled'] 表示不训练pooled embed
IGNORE_TARGETS = []

# --- 训练参数 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 50
BATCH_SIZE = 32
NUM_WORKERS = 4  # 数据加载器的工作进程数
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# --- 损失函数权重 ---
# MSE损失 (用于对齐embed和pooled) 和 对比损失 (用于学习风格) 的权重
LOSS_WEIGHT_ALIGNMENT = 1.0  # 对齐损失权重
LOSS_WEIGHT_CONTRASTIVE = 0.5  # 风格对比损失权重

# --- 对比学习参数 ---
# SupConLoss 的温度参数
CONTRASTIVE_TEMPERATURE = 0.07

# --- 其他 ---
SAVE_EVERY_EPOCH = 5 # 每隔多少个epoch保存一次模型