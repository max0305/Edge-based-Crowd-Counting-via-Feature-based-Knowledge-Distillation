import os
import torch
import random
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from models.teacher import CSRNet
from data.dataset import ShanghaiTechDataset
from config import Config

def denormalize(tensor):
    """Reverses the ImageNet normalization for visualization."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor * std + mean
    return torch.clamp(tensor, 0, 1)

def predict(model_path=None):
    # 1. Setup Device
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Model
    model = CSRNet(load_weights=False).to(device)
    
    if model_path is None:
        # Default to best teacher checkpoint
        model_path = "./checkpoints/teacher_best.pth"
    
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Warning: Model path {model_path} not found. Using random weights.")

    model.eval()

    # 3. Prepare Dataset (Test Set)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Note: For prediction/visualization, we MUST use the original size (fixed_size=None).
    # If we resize the image (e.g. to 512x512), the scale of people changes, 
    # and the model (trained on original scale crops) will fail to recognize them.
    test_dataset = ShanghaiTechDataset(
        root_path=Config.DATASET_PATH, 
        phase='test', 
        main_transform=transform,
        fixed_size=None, # Use original size
        augment=False 
    )
    
    print(f"Test dataset size: {len(test_dataset)}")

    # 4. Pick a Random Sample
    idx = random.randint(0, len(test_dataset) - 1)
    img, gt_density = test_dataset[idx]
    
    # Add batch dimension
    img_tensor = img.unsqueeze(0).to(device)
    gt_density = gt_density.unsqueeze(0).to(device)

    # 5. Inference
    with torch.no_grad():
        pred_density, _ = model(img_tensor)
    
    # 6. Process Results
    # CSRNet output is 1/8 of input size. 
    # For visualization, we can upsample prediction to match input size.
    pred_density_upsampled = torch.nn.functional.interpolate(
        pred_density, size=img.shape[1:], mode='bilinear', align_corners=False
    )
    
    # Calculate Counts
    # Note: The model is trained such that sum(pred_density) approx count.
    # However, if we upsample, the sum increases by factor of 8*8=64 if we don't scale.
    # But wait, density map values represent density per pixel. 
    # If we resize the map, the sum changes.
    # The standard way to get count from CSRNet output (1/8 size) is just to sum it.
    pred_count = pred_density.sum().item()
    gt_count = gt_density.sum().item()
    
    print(f"Sample Index: {idx}")
    print(f"Ground Truth Count: {gt_count:.2f}")
    print(f"Predicted Count:    {pred_count:.2f}")
    print(f"Error:              {abs(pred_count - gt_count):.2f}")

    # 7. Visualization
    img_vis = denormalize(img).permute(1, 2, 0).numpy()
    gt_vis = gt_density.squeeze().cpu().numpy()
    pred_vis = pred_density_upsampled.squeeze().cpu().numpy()
    
    # Scale predicted density map for visualization purposes (to match GT range roughly)
    # Or just normalize them independently for heatmap visualization
    
    plt.figure(figsize=(15, 5))
    
    # Input Image
    plt.subplot(1, 3, 1)
    plt.imshow(img_vis)
    plt.title(f"Input Image")
    plt.axis('off')
    
    # Ground Truth
    plt.subplot(1, 3, 2)
    plt.imshow(gt_vis, cmap='jet')
    plt.title(f"GT Density (Count: {gt_count:.2f})")
    plt.axis('off')
    
    # Prediction
    plt.subplot(1, 3, 3)
    plt.imshow(pred_vis, cmap='jet')
    plt.title(f"Pred Density (Count: {pred_count:.2f})")
    plt.axis('off')
    
    save_path = "prediction_result.png"
    plt.savefig(save_path)
    print(f"Result saved to {save_path}")
    # plt.show() # Uncomment if running in a GUI environment

if __name__ == "__main__":
    predict()
