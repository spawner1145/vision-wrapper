# vision-wrapper
# 1. 准备工作
## 步骤 1.1: 检查目录结构
```
项目目录/
├── data/
│   ├── style_A/
│   │   ├── image01_img.png
│   │   ├── image01_embed.pt
│   │   ├── image01_pooled.pt
│   │   ├── image01_mask.pt
│   │   └── ...
│   └── style_B/
│       └── ...
... (其他所有.py文件)
```
## 步骤 1.2

`pip install -r requirements.txt`

# 2. 训练风格编码器

## 步骤 2.1: 修改配置文件 (config.py)

在开始训练前，打开 config.py，检查并修改以下关键参数：

DATA_DIR: 确认它指向数据集文件夹 ("./data")。

MODEL_NAME: 可以尝试不同的 timm 模型，'vit_base_patch16_224' 是一个很好的起点。

EPOCHS: 训练的总轮数，可以先设为50，后续根据效果调整。

BATCH_SIZE: 根据显卡的显存大小调整。如果遇到“Out of Memory”错误，就调小这个值（例如16或8）。

LEARNING_RATE: 学习率，1e-4 是一个常用的初始值。

IGNORE_TARGETS: 如果数据集中没有 _pooled.pt 或 _mask.pt 文件，可以在这里将其忽略，例如：IGNORE_TARGETS = ['pooled', 'mask']。

## 步骤 2.2: 开始训练

准备就绪后，在终端中运行主训练脚本：

`python train.py`

会看到一个进度条开始滚动，显示训练进度和loss

## 步骤 2.3: 查看训练结果

训练完成后：

模型权重: 会发现在 checkpoints/ 文件夹下生成了 .pth 文件，例如 model_epoch_50.pth。这就是训练好的模型

训练日志: 在 logs/ 文件夹下会生成日志文件。可以在终端中运行以下命令来启动TensorBoard，从而可视化训练过程（例如查看损失曲线）：

`tensorboard --logdir=logs`

然后在浏览器中打开显示的网址（通常是 `http://localhost:6006/`）。

# 3. 工具与应用：使用你训练好的模型

当拥有了 .pth 模型文件后，就可以使用我们准备好的工具脚本来发挥它的作用了。

## 功能 3.1: 比较两张图片的风格相似度
这个功能用来检验模型对“风格”的理解能力。

使用脚本: `style_comparator.py`

`python style_comparator.py <图片1路径> <图片2路径> --weights <你的模型权重路径>`

示例:
```
# 比较同一风格下的两张图片
python style_comparator.py ./data/style_A/image01_img.png ./data/style_A/image02_img.png --weights ./checkpoints/model_epoch_50.pth

# 比较不同风格的两张图片
python style_comparator.py ./data/style_A/image01_img.png ./data/style_B/image03_img.png --weights ./checkpoints/model_epoch_50.pth
```
预期输出:

脚本会输出一个-1到1之间的“风格相似度”分数。分数越接近1，说明模型认为两张图的风格越相似。

## 功能 3.2: 检查单张图片的详细输出
这个功能用于调试或深入分析模型对某一张图片的具体输出。

使用脚本: `inspect_image.py`

`python inspect_image.py <图片路径> --weights <你的模型权重路径>`

示例:
```
python inspect_image.py ./data/style_A/image01_img.png --weights ./checkpoints/model_epoch_50.pth
```
预期输出:
脚本会打印出一个详细的报告，告诉你模型为这张图片生成的 embed, pooled embed, 和 attention mask 的形状、数据类型和所在设备。

## 功能 3.3 (进阶): 在其他项目中使用工具函数
`multimodal_utils.py` 文件为提供了未来将此编码器集成到LLM项目中的核心“零件”。

使用场景: 当开始构建VLM（视觉语言模型）时。

如何使用: 在VLM代码中，可以像这样导入和使用这些函数：
```python
from multimodal_utils import project_sequence_embed, create_attention_mask_from_embed

# ... 加载视觉编码器和LLM ...

# 假设 vision_embed 是编码器输出
# 假设 llm_projector 是一个 nn.Linear(D_vision, D_llm) 层

# 将视觉embed投影到LLM的维度空间
projected_vision_embed = project_sequence_embed(vision_embed, llm_projector)

# 为其创建对应的mask
vision_mask = create_attention_mask_from_embed(projected_vision_embed)

# ... 然后将它们与文本部分拼接 ...
```
