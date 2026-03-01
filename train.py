"""
训练脚本 - v2 100帧版本
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from lighting_speech_shield.model_v2 import ComplexMaskNet
from lighting_speech_shield.dataset import SpeechNoiseDataset


def complex_mse_loss(pred, target):
    return nn.MSELoss()(pred[..., 0], target[..., 0]) + nn.MSELoss()(pred[..., 1], target[..., 1])


def train_model(data_dir, epochs=1, batch_size=2, num_frames=100, save_dir='checkpoints'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dataset = SpeechNoiseDataset(data_dir, num_frames=num_frames)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print(f"Dataset: {len(dataset)} samples, Frames: {num_frames}")
    
    model = ComplexMaskNet(base_channels=12).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for input_spec, target_mask in pbar:
            input_spec = input_spec.to(device)
            target_mask = target_mask.to(device)
            
            pred_mask = model(input_spec)
            
            loss = complex_mse_loss(pred_mask, target_mask)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        
        print(f"  Epoch {epoch+1}: avg_loss={avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'loss': avg_loss,
            }, save_path / 'best_model_v2.pth')
            print(f"  [OK] Saved best model")
    
    print(f"\nTraining complete! Best loss: {best_loss:.4f}")
    return model


if __name__ == "__main__":
    import sys
    
    # 默认参数
    data_dir = "data/synthetic"
    epochs = 1
    batch_size = 2
    num_frames = 100
    
    # 解析命令行参数
    if len(sys.argv) >= 2:
        data_dir = sys.argv[1]
    if len(sys.argv) >= 3:
        epochs = int(sys.argv[2])
    if len(sys.argv) >= 4:
        batch_size = int(sys.argv[3])
    if len(sys.argv) >= 5:
        num_frames = int(sys.argv[4])
    
    train_model(data_dir, epochs=epochs, batch_size=batch_size, num_frames=num_frames)
