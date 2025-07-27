import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import argparse
from typing import Callable

try:
    from model import VisionStyleEncoder
    from multimodal_utils import create_attention_mask_from_embed
except ImportError as e:
    print(f"错误：无法导入必要的模块: {e}")
    print("请确保 model.py 和 multimodal_utils.py 文件与此脚本位于同一目录中。")
    exit()

def get_all_outputs(model: nn.Module, image_path: str, transform: Callable, device: str) -> dict:
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"错误：找不到图片文件 {image_path}")
        return None

    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        model_output = model(image_tensor)
        final_embed = model_output.get('pred_embed')
        if final_embed is None:
            print("错误：模型输出中未找到'pred_embed'。")
            return None
        final_pooled = model_output.get('pred_pooled')
        final_mask = create_attention_mask_from_embed(final_embed)
    return {"embed": final_embed, "pooled": final_pooled, "mask": final_mask}

def main():
    parser = argparse.ArgumentParser(description="输入一张图片，检查并输出模型生成的三个值。")
    parser.add_argument("image", type=str, help="要检查的单张图片的路径。")
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

    print(f"\n正在处理图片: {args.image}...")
    outputs = get_all_outputs(model, args.image, transform, DEVICE)
    if outputs is None:
        print("\n处理失败。")
        return

    print("\n" + "="*40 + "\n           模型输出详细信息\n" + "="*40)
    print("\n[1] Embed (序列特征)")
    print("-" * 25)
    print(f"形状 (Shape): {outputs['embed'].shape}")
    print(f"数据类型 (dtype): {outputs['embed'].dtype}")
    print(f"设备 (Device): {outputs['embed'].device}")

    print("\n[2] Pooled Embed (池化/汇总特征)")
    print("-" * 25)
    if outputs['pooled'] is not None:
        print(f"形状 (Shape): {outputs['pooled'].shape}")
        print(f"数据类型 (dtype): {outputs['pooled'].dtype}")
        print(f"设备 (Device): {outputs['pooled'].device}")
    else:
        print("未生成 (模型可能配置为忽略此项)")

    print("\n[3] Attention Mask (注意力遮罩)")
    print("-" * 25)
    print(f"形状 (Shape): {outputs['mask'].shape}")
    print(f"数据类型 (dtype): {outputs['mask'].dtype}")
    print(f"设备 (Device): {outputs['mask'].device}")
    print(f"(此mask是根据embed形状生成的，全为1)")
    print("\n" + "="*40)

if __name__ == '__main__':
    main()