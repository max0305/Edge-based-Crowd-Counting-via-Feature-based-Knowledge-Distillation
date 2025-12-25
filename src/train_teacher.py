import os
import time
import json
import shutil
import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

from data.dataset import ShanghaiTechDataset
from models.teacher import CSRNet
from config import Config
from utils.vis import visualize_sample, visualize_loss_curve

import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

def train_teacher():
    # 1. Setup Device
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Preparation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ShanghaiTechDataset(
        root_path=Config.DATASET_PATH, 
        phase='train', 
        main_transform=transform,
        fixed_size=Config.INPUT_SIZE,
        augment=True
    )

    # Load full test set first
    # IMPORTANT: For validation/testing, we should NOT resize the image (fixed_size=None).
    # Training uses RandomCrop (preserving scale), but if we resize validation images,
    # the people will look smaller, causing a "Scale Mismatch" and poor performance.
    # Since batch_size=1 for validation, variable image sizes are fine.
    full_test_dataset = ShanghaiTechDataset(
        root_path=Config.DATASET_PATH, 
        phase='test', 
        main_transform=transform,
        fixed_size=None 
    )

    # Split Test set into Validation (20%) and Test (80%)
    test_size = len(full_test_dataset)
    val_size = int(test_size * 0.2)
    real_test_size = test_size - val_size
    
    val_dataset, test_dataset = random_split(
        full_test_dataset, 
        [val_size, real_test_size],
        generator=torch.Generator().manual_seed(42) # Ensure reproducibility
    )

    print(f"Dataset Split:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val:   {len(val_dataset)} (from Test set)")

    img, density_map = train_dataset[random.randint(0, len(train_dataset)-1)]
    visualize_sample(img, density_map)
    
    # Use a smaller batch size for VGG16 as it consumes more memory
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=Config.NUM_WORKERS)

    # 3. Model Setup
    model = CSRNet(load_weights=False).to(device)
    
    # 4. Optimizer & Loss
    # Baseline Configuration (CSRNet Paper):
    optimizer = optim.SGD(model.parameters(), lr=Config.LR, momentum=0.95, weight_decay=5e-4)
    
    # Scheduler: Disabled for baseline
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    criterion = nn.MSELoss(size_average=False).to(device) # Sum of squared errors

    # 5. Training Loop
    best_mae = float('inf')
    save_dir = "./checkpoints"
    exp_dir = os.path.join("./exp", "teacher", time.strftime("%m%d_%H%M", time.localtime()))
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(exp_dir, exist_ok=True)
    print(f"Result will be saved to: {exp_dir}")

    # History tracking
    history = {
        'train_loss': [], 
        'train_mae': [], 'train_mse': [],
        'val_mae': [], 'val_mse': []
    }

    print("Starting Teacher Training...")
    for epoch in range(Config.EPOCHS):
        model.train()
        epoch_loss = 0.0
        train_mae = 0.0
        train_mse = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} [Train]", leave=False)
        for img, target in pbar:
            img = img.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            
            # Forward
            output, _ = model(img) # CSRNet returns (density_map, features)
            
            # CSRNet output is 1/8 size of input due to pooling layers in VGG16 frontend
            # We need to downsample the target density map to match output size
            if output.shape[2:] != target.shape[2:]:
                # Use sum pooling (adaptive avg pool * area) to preserve count
                # Or simply interpolate. For density maps, sum pooling is theoretically better but interpolation is common.
                # Let's use interpolation for simplicity and consistency with common implementations.
                # Note: To preserve count, we must scale by the area ratio.
                ratio = (target.shape[2] * target.shape[3]) / (output.shape[2] * output.shape[3])
                target = torch.nn.functional.interpolate(target, size=output.shape[2:], mode='bicubic', align_corners=False)
                target = target * ratio

            # Loss
            loss = criterion(output, target)
            
            # Backward
            loss.backward()
            
            # Gradient Clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Calculate Train Metrics (MAE/MSE)
            with torch.no_grad():
                # Note: output and target are density maps. Sum them to get counts.
                # Since batch_size > 1, we need to sum over (H, W) dimensions
                pred_counts = output.view(output.shape[0], -1).sum(1)
                gt_counts = target.view(target.shape[0], -1).sum(1)
                
                diff = pred_counts - gt_counts
                train_mae += torch.abs(diff).sum().item()
                train_mse += (diff ** 2).sum().item()
            
            pbar.set_postfix({'Loss': loss.item()})
        print("", end="\r") # Clear tqdm line

        avg_train_loss = epoch_loss / len(train_loader)
        avg_train_mae = train_mae / len(train_dataset)
        avg_train_mse = (train_mse / len(train_dataset)) ** 0.5
        
        history['train_loss'].append(avg_train_loss)
        history['train_mae'].append(avg_train_mae)
        history['train_mse'].append(avg_train_mse)
        
        # --- Validation Phase ---
        model.eval()
        val_mae = 0.0
        val_mse = 0.0
        
        with torch.no_grad():
            for img, target in tqdm(val_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} [Val]", leave=False):
                img = img.to(device)
                target = target.to(device)
                
                output, _ = model(img)
                
                # CSRNet output is 1/8 size. For counting, sum is invariant to spatial resolution if we handled it correctly.
                # But for visualization or pixel-wise metrics, we might want to upsample output or downsample target.
                # For MAE/MSE (Count), we just sum, so shape mismatch doesn't matter for count calculation.
                
                # Calculate Count
                pred_count = output.sum().item()
                gt_count = target.sum().item()
                
                # Metrics
                val_mae += abs(pred_count - gt_count)
                val_mse += (pred_count - gt_count) ** 2

        avg_val_mae = val_mae / len(val_dataset)
        avg_val_mse = (val_mse / len(val_dataset)) ** 0.5
        
        history['val_mae'].append(avg_val_mae)
        history['val_mse'].append(avg_val_mse)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}: Loss: {avg_train_loss:.4f} | Train MAE: {avg_train_mae:.2f} | Val MAE: {avg_val_mae:.2f} | Val MSE: {avg_val_mse:.2f} | LR: {current_lr:.2e}")
        
        scheduler.step()

        # Save Best Model (based on MAE)
        if avg_val_mae < best_mae:
            best_mae = avg_val_mae
            save_path = os.path.join(save_dir, "teacher_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"New Best MAE: {best_mae:.4f}. Saved to {save_path}")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(save_dir, f"teacher_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)

    # Save final model
    torch.save(model.state_dict(), os.path.join(save_dir, "teacher_final.pth"))

    # Save best weight and loss curve to exp dir
    shutil.copy(os.path.join(save_dir, 'teacher_best.pth'), exp_dir)
    print(f"Saved {os.path.join(exp_dir, 'teacher_best.pth')}")
    visualize_loss_curve(history, save_path=os.path.join(exp_dir, "teacher_loss_curve.png"))
    with open(os.path.join(exp_dir, "history.json"), 'w') as f:
        json.dump(history, f)
    print("Training Complete.")

    
if __name__ == "__main__":
    train_teacher()