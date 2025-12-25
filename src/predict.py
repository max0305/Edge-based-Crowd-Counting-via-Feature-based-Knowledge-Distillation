import os
import torch
import random
import matplotlib.pyplot as plt
import numpy as np
import argparse
from torchvision import transforms
from models.teacher import CSRNet
from models.student import CrowdResNet18
from data.dataset import ShanghaiTechDataset
from config import Config
from utils.utils import denormalize, evaluate
from utils.vis import visualize_prediction
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

def predict(model_path=None, eval=False, type=None):
    # 1. Setup Device
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Model
    if type == 'teacher':
        model = CSRNet(load_weights=False).to(device)
    else:
        model = CrowdResNet18().to(device)
    
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

    if eval:
        evaluate(model, test_dataset, device)

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
    
    # Calculate Counts
    # The standard way to get count from CSRNet output (1/8 size) is just to sum it.
    pred_count = pred_density.sum().item()
    gt_count = gt_density.sum().item()
    
    print(f"Sample Index: {idx}")
    print(f"Ground Truth Count: {gt_count:.2f}")
    print(f"Predicted Count:    {pred_count:.2f}")
    print(f"Error:              {abs(pred_count - gt_count):.2f}")

    # 7. Visualization
    img_vis = denormalize(img).permute(1, 2, 0).numpy()
    
    # Generate Downsampled GT (The actual training target)
    # CSRNet output is 1/8 size, so we downsample GT density map accordingly.
    # Need to scale up the sum according to resize ratio to preserve count.
    ratio = (gt_density.shape[2] * gt_density.shape[3]) / (pred_density.shape[2] * pred_density.shape[3])
    gt_density_downsampled = torch.nn.functional.interpolate(
        gt_density, size=pred_density.shape[2:], mode='bicubic', align_corners=False
    ) * ratio
    
    gt_down_vis = gt_density_downsampled.squeeze().cpu().numpy() # (1/8 size)
    pred_vis = pred_density.squeeze().cpu().numpy() # Show raw prediction (1/8 size)
    
    visualize_prediction(img=img_vis, gt_density=gt_down_vis, pred_density=pred_vis, save_path="prediction_result.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict using trained CSRNet model.")
    parser.add_argument('--model_path', type=str, default=None, help='Path to the trained model weights.')
    parser.add_argument('--teacher', action='store_true', help='Use teacher model')
    parser.add_argument('--student', action='store_true', help='Use student model')
    parser.add_argument('--baseline', action='store_true', help='Use student baseline model')
    parser.add_argument('--eval', action='store_true', help='Evaluate on full test set')
    args = parser.parse_args()

    if args.model_path is not None:
        if args.teacher:
            print(f"Using teacher model at {args.model_path} for prediction.")
            predict(model_path=args.model_path, eval=args.eval, type='teacher')
        elif args.student:
            print(f"Using student model at {args.model_path} for prediction.")
            predict(model_path=args.model_path, eval=args.eval, type='student')
        elif args.baseline:
            print(f"Using student baseline model at {args.model_path} for prediction.")
            predict(model_path=args.model_path, eval=args.eval, type='student_baseline')
        else:
            print("Please specify --teacher or --student to indicate model type.")
            exit(1)
    elif args.teacher:
        print("Using teacher model for prediction.")
        predict(model_path='./checkpoints/teacher_best.pth', eval=args.eval, type='teacher')
    elif args.student:
        print("Using student model for prediction.")
        predict(model_path='./checkpoints/student_best.pth', eval=args.eval, type='student')
    elif args.baseline:
        print("Using student baseline model for prediction.")
        predict(model_path='./checkpoints/student_baseline_best.pth', eval=args.eval, type='student_baseline')
    else:
        print("No model specified.")
        exit(1)
