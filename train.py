# train.py

import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import config
from dataset import StyleDataset
from model import VisionStyleEncoder
from losses import CombinedLoss

def main():
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    writer = SummaryWriter(log_dir=config.LOG_DIR)
    device = torch.device(config.DEVICE)

    print("Loading dataset...")
    train_dataset = StyleDataset(
        root_dir=config.DATA_DIR, 
        image_size=config.IMAGE_SIZE,
        ignore_targets=config.IGNORE_TARGETS
    )
    if len(train_dataset) == 0:
        print(f"错误: 在目录 '{config.DATA_DIR}' 中未找到数据。请检查路径和数据格式。")
        return

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=True
    )
    print(f"数据集加载完毕，共 {len(train_dataset)} 个样本，{len(train_dataset.style_map)} 种风格。")

    print("Initializing model...")
    model = VisionStyleEncoder(
        model_name=config.MODEL_NAME, pretrained=config.PRETRAINED,
        ignore_targets=config.IGNORE_TARGETS
    ).to(device)

    criterion = CombinedLoss(
        temperature=config.CONTRASTIVE_TEMPERATURE, weight_align=config.LOSS_WEIGHT_ALIGNMENT,
        weight_contrastive=config.LOSS_WEIGHT_CONTRASTIVE, ignore_targets=config.IGNORE_TARGETS
    )
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    print(f"开始训练，共 {config.EPOCHS} 个 epochs，设备: {device}...")
    for epoch in range(config.EPOCHS):
        model.train()
        epoch_loss_sum = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}", leave=False)

        for batch in progress_bar:
            images = batch['image'].to(device)
            targets_dict = {
                'target_embed': batch['target_embed'].to(device),
                'target_pooled': batch['target_pooled'].to(device),
                'target_mask': batch['target_mask'].to(device),
                'style_id': batch['style_id'].to(device)
            }

            optimizer.zero_grad()
            model_output = model(images)
            loss, loss_breakdown = criterion(model_output, targets_dict)
            loss.backward()
            optimizer.step()

            epoch_loss_sum += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            
            global_step = epoch * len(train_loader) + progress_bar.n
            for k, v in loss_breakdown.items():
                writer.add_scalar(f'Loss/{k}', v, global_step)

        avg_loss = epoch_loss_sum / len(train_loader)
        print(f"Epoch {epoch+1:02d}/{config.EPOCHS} | 平均损失: {avg_loss:.4f}")
        writer.add_scalar('Loss/epoch_avg_train', avg_loss, epoch)

        if (epoch + 1) % config.SAVE_EVERY_EPOCH == 0 or epoch == config.EPOCHS - 1:
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f'model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"模型已保存至: {checkpoint_path}")

    writer.close()
    print("训练完成。")

if __name__ == '__main__':
    main()