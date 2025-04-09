import torch
import torch.nn as nn
from torchvision import models
import os
import sys

# Add the project directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
from features.convnext_features import convnext_tiny_26_features
import torchvision.transforms as transforms
from PIL import Image

INPUT_SIZE = 192

def compute_effective_receptive_field(model, input_tensor, feature_idx=None):
    """
    Compute and visualize the effective receptive field using gradients.
    
    Args:
        model: The model to analyze
        input_tensor: Input tensor of shape [1, 3, H, W]
        feature_idx: Index of feature map to analyze (None = all features)
    
    Returns:
        Gradient heatmap showing the effective receptive field
    """
    # Create a copy with gradient tracking
    input_tensor = input_tensor.clone().detach().requires_grad_(True)
    
    # Forward pass
    output = model(input_tensor)
    
    # Get dimensions
    batch_size, num_channels, height, width = output.shape
    
    # Choose channel to analyze
    channel_idx = 0 if feature_idx is None else feature_idx
    
    # Center point in feature map
    center_h, center_w = height // 2, width // 2
    
    # Create gradient target (1.0 at center, 0 elsewhere)
    grad_target = torch.zeros_like(output)
    if feature_idx is None:
        # Use middle 25% of channels for better signal
        start_c = num_channels // 4
        end_c = start_c * 3
        for c in range(start_c, end_c):
            grad_target[0, c, center_h, center_w] = 1.0
    else:
        grad_target[0, channel_idx, center_h, center_w] = 1.0
    
    # Backward pass to get gradients
    output.backward(grad_target)
    
    # Get gradient on input (absolute value, sum across channels)
    grad_input = input_tensor.grad.abs().sum(dim=1).squeeze(0)
    
    # Normalize for visualization
    grad_min = grad_input.min()
    grad_max = grad_input.max()
    norm_grad = (grad_input - grad_min) / (grad_max - grad_min + 1e-8)
    
    return norm_grad.detach().cpu().numpy()

def visualize_all_stages(save_dir="receptive_field_viz", stage_configs = [3, 5, 7], threshold=0.1):
    """
    Visualize receptive fields for different stage configurations.
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Sample image (or use zeros)
    # Using zeros is cleaner for pure receptive field visualization
    input_tensor = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE)
    
    # Optionally use a real image for context
    # transform = transforms.Compose([
    #     transforms.Resize(INPUT_SIZE),
    #     transforms.CenterCrop(INPUT_SIZE),
    #     transforms.ToTensor(),
    # ])
    # sample_img = Image.open("sample_image.jpg")
    # input_tensor = transform(sample_img).unsqueeze(0)
    
    # Analyze each stage configuration
    
    for num_stages in stage_configs:
        print(f"Analyzing model with {num_stages} stages...")
        
        # Load model with specific number of stages
        model = convnext_tiny_26_features(
            pretrained=True, 
            use_mid_layers=True, 
            num_stages=num_stages
        )
        model.eval()
        
        # Compute receptive field
        with torch.no_grad():
            # Forward pass to check output shape
            sample_output = model(input_tensor)
            print(f"Output shape for {num_stages} stages: {sample_output.shape}")
        
        # Compute the effective receptive field
        receptive_field = compute_effective_receptive_field(model, input_tensor)
        
        # Visualize
        plt.figure(figsize=(10, 10))
        plt.imshow(receptive_field, cmap='hot')
        plt.colorbar(label='Gradient Magnitude')
        plt.title(f'Effective Receptive Field: {num_stages} Stages')
        
        # Add grid lines at intervals of 16 pixels
        plt.grid(which='major', color='w', linestyle='-', linewidth=0.5, alpha=0.3)
        plt.xticks(np.arange(0, INPUT_SIZE, 16))
        plt.yticks(np.arange(0, INPUT_SIZE, 16))
        
        # Calculate receptive field size (thresholded)
        threshold = threshold   # 10% of maximum gradient value
        active_pixels = (receptive_field > threshold).sum()
        rf_size = int(np.sqrt(active_pixels))
        plt.suptitle(f'Estimated RF Size: ~{rf_size}x{rf_size} pixels')
        
        # Save the figure
        plt.savefig(os.path.join(save_dir, f"receptive_field_stage_{num_stages}.png"))
        plt.close()
        
        # Also save a zoomed version focusing on the receptive field
        center = INPUT_SIZE // 2
        crop_size = min(rf_size * 2, INPUT_SIZE)  # Double the RF size but cap at image size
        half_crop = crop_size // 2
        
        plt.figure(figsize=(10, 10))
        plt.imshow(receptive_field[center-half_crop:center+half_crop, center-half_crop:center+half_crop], cmap='hot')
        plt.colorbar(label='Gradient Magnitude')
        plt.title(f'Zoomed Effective Receptive Field: {num_stages} Stages')
        plt.grid(which='major', color='w', linestyle='-', linewidth=0.5, alpha=0.3)
        plt.savefig(os.path.join(save_dir, f"receptive_field_stage_{num_stages}_zoomed.png"))
        plt.close()
        
        print(f"Estimated receptive field size for {num_stages} stages: ~{rf_size}x{rf_size} pixels")
        
    print(f"Visualizations saved to {save_dir}/")

def visualize_specific_features(num_stages=3, num_features=5, save_dir="feature_rf_viz", threshold=0.1):
    """
    Visualize receptive fields for specific features in a given stage.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Load model with specific number of stages
    model = convnext_tiny_26_features(
        pretrained=True, 
        use_mid_layers=True, 
        num_stages=num_stages
    )
    model.eval()
    
    # Sample input
    input_tensor = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE)
    
    # Check output channels
    with torch.no_grad():
        output = model(input_tensor)
        num_channels = output.shape[1]
        print(f"Model has {num_channels} output channels")
    
    # Sample random feature indices
    if num_features > num_channels:
        num_features = num_channels
        
    feature_indices = np.linspace(0, num_channels-1, num_features, dtype=int)
    
    # Create a grid figure
    rows = int(np.ceil(np.sqrt(num_features)))
    cols = int(np.ceil(num_features / rows))
    
    plt.figure(figsize=(cols*5, rows*5))
    
    for i, feature_idx in enumerate(feature_indices):
        print(f"Analyzing feature {feature_idx}/{num_channels}...")
        
        # Compute receptive field for this feature
        receptive_field = compute_effective_receptive_field(model, input_tensor, feature_idx)
        
        # Plot
        plt.subplot(rows, cols, i+1)
        plt.imshow(receptive_field, cmap='hot')
        plt.title(f'Feature {feature_idx}')
        plt.colorbar()
        
        # Calculate receptive field size
        threshold = threshold 
        active_pixels = (receptive_field > threshold).sum()
        rf_size = int(np.sqrt(active_pixels))
        plt.xlabel(f'RF Size: ~{rf_size}x{rf_size}')
        
    plt.suptitle(f'Receptive Fields for {num_features} Features (Stage {num_stages})')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"feature_receptive_fields_stage_{num_stages}.png"))
    plt.close()

if __name__ == "__main__":
    THRESHOLD = 0.1

    # Visualize receptive fields for different stage configurations
    visualize_all_stages(threshold=THRESHOLD)
    
    # Optionally visualize specific features for stage 3
    visualize_specific_features(num_stages=3, num_features=9, threshold=THRESHOLD)