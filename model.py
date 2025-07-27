import torch
import torch.nn as nn
import timm
import config
from dataset import StyleDataset

class VisionStyleEncoder(nn.Module):
    def __init__(self, model_name, pretrained=True, ignore_targets=None):
        super().__init__()
        self.ignore_targets = ignore_targets if ignore_targets is not None else []
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        
        if 'vit' in model_name:
            self.backbone.head = nn.Identity()
            embed_dim = self.backbone.embed_dim
        else:
            embed_dim = self.backbone.get_classifier().in_features
            self.backbone.reset_classifier(0)
            
        self.embed_dim = embed_dim

        # 通过加载一个样本数据, 让模型自动适应目标维度
        dummy_dataset = StyleDataset(config.DATA_DIR, ignore_targets=self.ignore_targets)
        if len(dummy_dataset) == 0:
            raise ValueError(f"数据集为空或路径 '{config.DATA_DIR}' 不正确! 无法推断模型输出维度。")
        dummy_sample = dummy_dataset[0]
        
        target_embed_dim = dummy_sample['target_embed'].shape[-1]
        self.embed_projector = nn.Linear(self.embed_dim, target_embed_dim)
        
        print(f"Vision model feature dim: {self.embed_dim}")
        print(f"Target embed dim: {target_embed_dim}")

        self.pooled_projector = None
        if 'pooled' not in self.ignore_targets and dummy_sample['target_pooled'].numel() > 0:
            target_pooled_dim = dummy_sample['target_pooled'].shape[-1]
            self.pooled_projector = nn.Linear(self.embed_dim, target_pooled_dim)
            print(f"Target pooled_embed dim: {target_pooled_dim}")
        else:
            print("Ignoring pooled embed output.")

    def forward(self, x):
        features = self.backbone.forward_features(x)
        pred_embed, pred_pooled = None, None

        if 'vit' in self.backbone.default_cfg['architecture']:
            cls_token = features[:, 0]
            patch_tokens = features[:, 1:]
            pred_embed = self.embed_projector(patch_tokens)
            if self.pooled_projector:
                pred_pooled = self.pooled_projector(cls_token)
        else:
            pooled_features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
            if self.pooled_projector:
                pred_pooled = self.pooled_projector(pooled_features)
            b, c, h, w = features.shape
            seq_features = features.view(b, c, h * w).permute(0, 2, 1) 
            pred_embed = self.embed_projector(seq_features)

        return {"pred_embed": pred_embed, "pred_pooled": pred_pooled}