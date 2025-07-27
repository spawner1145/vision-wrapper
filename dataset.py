from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class StyleDataset(Dataset):
    def __init__(self, root_dir, image_size=224, ignore_targets=None):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.ignore_targets = ignore_targets if ignore_targets is not None else []
        
        self.samples = []
        self.style_map = {}

        if not self.root_dir.exists():
            print(f"警告：数据集目录 {self.root_dir} 不存在")
            return
            
        style_dirs = [d for d in self.root_dir.iterdir() if d.is_dir()]
        self.style_names = sorted([d.name for d in style_dirs])
        
        self.style_map = {name: i for i, name in enumerate(self.style_names)}
        
        for style_name in self.style_names:
            style_id = self.style_map[style_name]
            style_dir = self.root_dir / style_name
            
            prefixes = sorted(list(set([f.stem.replace('_img', '').replace('_embed', '').replace('_pooled', '').replace('_mask', '') 
                                        for f in style_dir.iterdir()])))
            
            for prefix in prefixes:
                img_path = style_dir / f"{prefix}_img.png"
                embed_path = style_dir / f"{prefix}_embed.pt"
                
                if img_path.exists() and embed_path.exists():
                    self.samples.append({
                        "prefix": prefix,
                        "style_name": style_name,
                        "style_id": style_id,
                        "img_path": img_path,
                        "embed_path": embed_path,
                        "pooled_path": style_dir / f"{prefix}_pooled.pt",
                        "mask_path": style_dir / f"{prefix}_mask.pt"
                    })

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        image = Image.open(sample_info["img_path"]).convert("RGB")
        image_tensor = self.transform(image)
        target_embed = torch.load(sample_info["embed_path"], map_location='cpu')

        target_pooled = torch.tensor([])
        if 'pooled' not in self.ignore_targets and sample_info["pooled_path"].exists():
            target_pooled = torch.load(sample_info["pooled_path"], map_location='cpu')

        target_mask = torch.tensor([])
        if 'mask' not in self.ignore_targets and sample_info["mask_path"].exists():
            target_mask = torch.load(sample_info["mask_path"], map_location='cpu')

        return {
            "image": image_tensor,
            "target_embed": target_embed,
            "target_pooled": target_pooled,
            "target_mask": target_mask,
            "style_id": torch.tensor(sample_info["style_id"], dtype=torch.long)
        }