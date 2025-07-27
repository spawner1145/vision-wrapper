# style_comparator.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import argparse
from typing import Callable

try:
    from model import VisionStyleEncoder
except ImportError:
    print("错误：无法导入 VisionStyleEncoder。请确保 model.py 文件位于同一目录中。")
    exit()

def get_style_vector(model: nn.Module, image_path: str, transform: Callable, device: str) -> torch.Tensor:
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"错误：找不到图片文件 {image_path}")
        return None

    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        if output.get('pred_pooled') is not None:
            style_vector = output['pred_pooled']
        elif output.get('pred_embed') is not None:
            print("警告：未使用pooled_embed，将使用embed的平均值作为风格向量。")
            style_vector = output['pred_embed'].mean(dim=1)
        else:
            print("错误：模型输出中无 'pred_pooled' 或 'pred_embed'。")
            return None
    return F.normalize(style_vector, p=2, dim=1)

def main():
    parser = argparse.ArgumentParser(description="比较两张图片的风格相似度。")
    parser.add_argument("image1", type=str, help="第一张图片的路径。")
    parser.add_argument("image2", type=str, help="第二张图片的路径。")
    parser.add_argument("--weights", type=str, required=True, help="训练好的模型权重文件(.pth)的路径。")
    parser.add_argument("--model-name", type=str, default="vit_base_patch16_224", help="训练时使用的timm模型名称。")
    parser.add_argument("--image-size", type=int, default=224, help="训练时使用的图像尺寸。")
    args = parser.parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"正在加载模型 {args.model_name}...")
    try:
        model = VisionStyleEncoder(model_name=args.model_name, pretrained=False)
        model.load_state_dict(torch.load(args.weights, map_location=DEVICE))
        model.to(DEVICE).eval()
    except Exception as e:
        print(f"加载模型失败: {e}")
        return
    print("模型加载成功！")

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("\n正在处理第一张图片...")
    style_vec1 = get_style_vector(model, args.image1, transform, DEVICE)
    print("正在处理第二张图片...")
    style_vec2 = get_style_vector(model, args.image2, transform, DEVICE)

    if style_vec1 is None or style_vec2 is None:
        print("\n因图片处理失败，无法进行比较。")
        return

    similarity = F.cosine_similarity(style_vec1, style_vec2).item()
    
    print("\n--- 风格相似度分析结果 ---")
    print(f"图片 1: {args.image1}")
    print(f"图片 2: {args.image2}")
    print(f"风格相似度: {similarity:.4f}")
    print("\n(说明: 分数越接近1.0，表示风格越相似；越接近-1.0表示风格越对立；接近0.0表示风格不相关。)")

if __name__ == '__main__':
    main()