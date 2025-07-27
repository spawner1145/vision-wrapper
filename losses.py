# losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels=None):
        device = torch.device('cuda') if features.is_cuda else torch.device('cpu')
        batch_size = features.shape[0]

        if labels is None:
            labels = torch.arange(batch_size, device=device).view(-1, 1)

        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        
        equals_mask = torch.eq(labels, labels.T).float().to(device)
        
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.ones_like(equals_mask) - torch.eye(batch_size).to(device)
        equals_mask = equals_mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)
        
        mask_sum = equals_mask.sum(1)
        mask_sum[mask_sum == 0] = 1
        mean_log_prob_pos = (equals_mask * log_prob).sum(1) / mask_sum
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(batch_size).mean()
        return loss

class CombinedLoss(nn.Module):
    def __init__(self, temperature, weight_align, weight_contrastive, ignore_targets=None):
        super().__init__()
        self.ignore_targets = ignore_targets if ignore_targets is not None else []
        self.weight_align = weight_align
        self.weight_contrastive = weight_contrastive
        self.mse_loss = nn.MSELoss()
        self.contrastive_loss = SupConLoss(temperature=temperature)

    def forward(self, model_output, targets):
        total_loss = 0.0
        loss_dict = {}

        align_loss = 0.0
        pred_embed = model_output['pred_embed']
        target_embed = targets['target_embed']
        
        min_len = min(pred_embed.shape[1], target_embed.shape[1])
        pred_embed_c = pred_embed[:, :min_len, :]
        target_embed_c = target_embed[:, :min_len, :]
        
        target_mask = targets.get('target_mask')
        if 'mask' not in self.ignore_targets and target_mask is not None and target_mask.numel() > 0:
            target_mask_c = target_mask[:, :min_len].unsqueeze(-1)
            mask_sum = target_mask_c.sum()
            if mask_sum > 0:
                masked_mse = F.mse_loss(pred_embed_c, target_embed_c, reduction='none')
                align_loss_embed = (masked_mse * target_mask_c).sum() / mask_sum
            else:
                align_loss_embed = torch.tensor(0.0, device=pred_embed.device)
        else:
            align_loss_embed = self.mse_loss(pred_embed_c, target_embed_c)
            
        align_loss += align_loss_embed
        loss_dict['align_embed'] = align_loss_embed.item()
        
        if 'pooled' not in self.ignore_targets and model_output.get('pred_pooled') is not None and targets.get('target_pooled').numel() > 0:
            pred_pooled = model_output['pred_pooled']
            target_pooled = targets['target_pooled']
            align_loss_pooled = self.mse_loss(pred_pooled, target_pooled)
            align_loss += align_loss_pooled
            loss_dict['align_pooled'] = align_loss_pooled.item()

        total_loss += self.weight_align * align_loss

        if self.weight_contrastive > 0:
            if model_output.get('pred_pooled') is not None:
                contrastive_features = model_output['pred_pooled']
            else:
                contrastive_features = model_output['pred_embed'].mean(dim=1)
            
            contrastive_features = F.normalize(contrastive_features, p=2, dim=1)
            style_labels = targets['style_id']
            contrast_loss = self.contrastive_loss(contrastive_features, style_labels)
            
            loss_dict['contrastive'] = contrast_loss.item()
            total_loss += self.weight_contrastive * contrast_loss

        loss_dict['total'] = total_loss.item()
        return total_loss, loss_dict