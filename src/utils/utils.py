import torch

def denormalize(tensor):
    """Reverses the ImageNet normalization for visualization."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor * std + mean
    return torch.clamp(tensor, 0, 1)


def evaluate(model, dataset, device):
    print("Evaluating on full test set...")
    mae = 0.0
    mse = 0.0
    
    with torch.no_grad():
        for i in range(len(dataset)):
            img, gt_density = dataset[i]
            img_tensor = img.unsqueeze(0).to(device)
            
            pred_density, _ = model(img_tensor)
            
            pred_count = pred_density.sum().item()
            gt_count = gt_density.sum().item()
            
            mae += abs(pred_count - gt_count)
            mse += (pred_count - gt_count) ** 2
            
            if (i + 1) % 20 == 0:
                print(f"Processed {i + 1}/{len(dataset)}...", end='\r')

    avg_mae = mae / len(dataset)
    avg_mse = (mse / len(dataset)) ** 0.5
    
    print(f"\nFull Test Set Results: MAE: {avg_mae:.2f} | MSE: {avg_mse:.2f}")
    print("-" * 50)

def adjust_target_size(output, target):
    if output.shape[2:] != target.shape[2:]:
        ratio = (target.shape[2] * target.shape[3]) / (output.shape[2] * output.shape[3])
        target = torch.nn.functional.interpolate(target, size=output.shape[2:], mode='bicubic', align_corners=False)
        target = target * ratio
        
    return target