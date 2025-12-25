import os
import time
import json
import torch
import random
import shutil
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

from data.dataset import ShanghaiTechDataset
from models.teacher import CSRNet
from models.student import CrowdResNet18
from models.distiller import DistillationModel, FeatureConnector, distillation_loss
from config import Config
from utils.vis import visualize_sample, visualize_loss_curve
from utils.utils import adjust_target_size

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

def train_student(distill=False):
    if distill:
        print("Training student with distillation.")
    else:
        print("Training student without distillation (baseline).")

    model_name = "student" if distill else "student_baseline"
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

    full_test_dataset = ShanghaiTechDataset(
        root_path=Config.DATASET_PATH, 
        phase='test', 
        main_transform=transform,
        fixed_size=None 
    )

    test_size = len(full_test_dataset)
    val_size = int(test_size * 0.2)
    real_test_size = test_size - val_size
    
    val_dataset, test_dataset = random_split(
        full_test_dataset, 
        [val_size, real_test_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Dataset Split:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val:   {len(val_dataset)} (from Test set)")


    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=Config.NUM_WORKERS)

    # 3. Model Setup
    if not distill:
        model = CrowdResNet18().to(device)
        target_student = model
    else:
        # A. Teacher
        teacher = CSRNet(load_weights=False)
        teacher_ckpt = "./checkpoints/teacher_best.pth"
        if os.path.exists(teacher_ckpt):
            teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device))
            print(f"Loaded teacher weights from {teacher_ckpt}")
        else:
            raise FileNotFoundError(f"Teacher checkpoint not found at {teacher_ckpt}")
        
        # B. Student
        student = CrowdResNet18()
        target_student = student
        
        # C. Connector (Student: 256 channels -> Teacher: 512 channels)
        # ResNet18 Layer3 output has 256 channels
        # CSRNet Frontend (VGG16) output has 512 channels
        connector = FeatureConnector(student_channels=256, teacher_channels=512)
        
        # D. Distiller
        model = DistillationModel(teacher, student, connector).to(device)

    # 4. Optimizer
    #optimizer = optim.SGD([
    #    {'params': student.frontend.parameters(), 'lr': Config.STUDENT_LR},
    #    {'params': student.backend.parameters(), 'lr': Config.STUDENT_LR * 10},
    #    {'params': connector.parameters(), 'lr': Config.STUDENT_LR * 10}
    #], lr=Config.STUDENT_LR, momentum=0.95, weight_decay=5e-4)
    optimizer = optim.SGD(model.parameters(), lr=Config.STUDENT_LR, momentum=0.95, weight_decay=5e-4)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    # Helper function for forward pass and loss
    def forward_loss(img, target):
        if distill:
            output, proj_feat, teacher_feat = model(img)
            target_down = adjust_target_size(output, target)
            loss, task_loss, dist_loss = distillation_loss(
                output, target_down, proj_feat, teacher_feat, lambda_kd=Config.DISTILLATION_LAMBDA
            )
            return loss, task_loss, dist_loss, output
        else:
            output, _ = model(img)
            target_down = adjust_target_size(output, target)
            loss = nn.MSELoss(reduction='sum')(output, target_down)
            return loss, loss, torch.tensor(0.0), output

    # 5. Training Loop
    best_mae = float('inf')
    save_dir = "./checkpoints"
    exp_dir = os.path.join("./exp", model_name, time.strftime("%m%d_%H%M", time.localtime()))
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(exp_dir, exist_ok=True)
    print(f"Result will be saved to: {exp_dir}")

    # History tracking
    history = {
        'train_loss': [], 'task_loss': [], 'dist_loss': [],
        'train_mae': [], 'train_mse': [],
        'val_mae': [], 'val_mse': []
    }

    for epoch in range(Config.EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_task_loss = 0.0
        epoch_dist_loss = 0.0
        train_mae = 0.0
        train_mse = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}", leave=False)
        for img, target in pbar:
            img = img.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            
            # Forward & Loss
            loss, task_loss, dist_loss, output = forward_loss(img, target)
            
            loss.backward()
            
            # Gradient Clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            optimizer.step()

            
            epoch_loss += loss.item()
            epoch_task_loss += task_loss.item()
            epoch_dist_loss += dist_loss.item()
            
            # Calculate Train Metrics
            with torch.no_grad():
                # Need to adjust target size for metric calculation if not already done
                # But forward_loss already computed loss against adjusted target.
                # However, we need the adjusted target for metrics.
                # Let's just re-adjust or return it from forward_loss.
                # Simpler: just re-adjust here, it's cheap.
                target_down = adjust_target_size(output, target)
                
                pred_counts = output.view(output.shape[0], -1).sum(1)
                gt_counts = target_down.view(target_down.shape[0], -1).sum(1)
                diff = pred_counts - gt_counts
                train_mae += torch.abs(diff).sum().item()
                train_mse += (diff ** 2).sum().item()
            
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}", 
                'Task': f"{task_loss.item():.4f}", 
                'Dist': f"{dist_loss.item():.4f}"
            })
            
        # Validation
        model.eval()
        val_mae = 0.0
        val_mse = 0.0
        with torch.no_grad():
            for img, target in tqdm(val_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} [Val]", leave=False):
                img = img.to(device)
                target = target.to(device)
                
                output, _ = target_student(img)
                
                # Downsample target for validation comparison
                target_down =  adjust_target_size(output, target)
                
                pred_count = output.sum().item()
                gt_count = target_down.sum().item()
                
                mae = abs(pred_count - gt_count)
                mse = (pred_count - gt_count) ** 2
                val_mae += mae
                val_mse += mse
                
        val_mae /= len(val_dataset)
        val_mse = (val_mse / len(val_dataset)) ** 0.5
        
        # Update History
        history['train_loss'].append(epoch_loss / len(train_loader))
        history['task_loss'].append(epoch_task_loss / len(train_loader))
        history['dist_loss'].append(epoch_dist_loss / len(train_loader))
        history['train_mae'].append(train_mae / len(train_dataset))
        history['train_mse'].append((train_mse / len(train_dataset)) ** 0.5)
        history['val_mae'].append(val_mae)
        history['val_mse'].append(val_mse)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}: Loss: {epoch_loss / len(train_loader):.4f} | Train MAE: {train_mae / len(train_dataset):.2f} | Val MAE: {val_mae:.2f} | Val MSE: {val_mse:.2f} | LR: {current_lr:.2e}")

        scheduler.step()

        visualize_loss_curve(history=history, save_path=os.path.join(exp_dir, "loss_curve.png"), print_path=False)
        
        # Save Best
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(target_student.state_dict(), os.path.join(save_dir, model_name + "_best.pth"))
            print(f"New Best MAE: {best_mae:.4f}. Saved to {os.path.join(save_dir, model_name + '_best.pth')}")
            
    # Save Latest
    torch.save(target_student.state_dict(), os.path.join(exp_dir, model_name + "_final.pth"))
    # Save best weight and loss curve to exp dir
    shutil.copy(os.path.join(save_dir, model_name + '_best.pth'), exp_dir)
    print(f"Saved {os.path.join(exp_dir, model_name + '_best.pth')}")
    with open(os.path.join(exp_dir, "history.json"), 'w') as f:
        json.dump(history, f)
    print("Training Complete.")

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--distill', action='store_true', help='Use distillation during student training')
    args = parse.parse_args()
    train_student(distill=args.distill)