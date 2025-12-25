import matplotlib.pyplot as plt
import torch

def visualize_sample(img, density_map, save_path="sample_visualization.png"):

    # Denormalize for visualization
    # Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    vis_img = img * std + mean
    vis_img = torch.clamp(vis_img, 0, 1) # Ensure valid range
    
    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(vis_img.permute(1, 2, 0).numpy())
    ax1.set_title("Input Image")
    ax1.axis('off')
    ax2 = fig.add_subplot(1,2,2)
    ax2.imshow(density_map.squeeze().numpy(), cmap='jet')
    ax2.set_title(f"Density Map (Count: {density_map.sum().item():.2f})")
    ax2.axis('off')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")


def visualize_loss_curve(history, save_path="loss_curve.png", print_path=True):
    # Plot History
    plt.figure(figsize=(15, 5))
    
    # 1. Loss Curve
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss (Pixel MSE)')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 2. MAE Curve (Train vs Val)
    plt.subplot(1, 3, 2)
    plt.plot(history['train_mae'], label='Train MAE')
    plt.plot(history['val_mae'], label='Val MAE')
    plt.title('Mean Absolute Error (Count)')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    # 3. MSE Curve (Train vs Val)
    plt.subplot(1, 3, 3)
    plt.plot(history['train_mse'], label='Train MSE')
    plt.plot(history['val_mse'], label='Val MSE')
    plt.title('Root Mean Squared Error (Count)')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    if print_path:
        print(f"Saved {save_path}")

def visualize_prediction(img, gt_density, pred_density, save_path="prediction_visualization.png"):
    plt.figure(figsize=(15, 5))
    
    # Input Image
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title(f"Input Image (Count: {gt_density.sum().item():.2f})")
    plt.axis('off')
        
    # Downsampled GT
    plt.subplot(1, 3, 2)
    plt.imshow(gt_density, cmap='jet')
    plt.title(f"Training Target (1/8 Scale)")
    plt.axis('off')
    
    # Prediction
    plt.subplot(1, 3, 3)
    plt.imshow(pred_density, cmap='jet')
    plt.title(f"Prediction (Count: {pred_density.sum().item():.2f})")
    plt.axis('off')
    
    plt.savefig(save_path)
    plt.close()
    print(f"Result saved to {save_path}")