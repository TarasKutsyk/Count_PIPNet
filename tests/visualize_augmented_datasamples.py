#!/usr/bin/env python
"""
Visualization tool for augmented data samples in PIPNet.

This script helps visualize how data augmentation affects images during PIPNet training.
It shows both the original images and their augmented versions to assess whether the
transformations preserve human-recognizable features while providing enough variety
for the model to learn robust prototype representations.

Usage:
    python visualize_augmented_datasamples.py --dataset [mnist_counting/geometric_shapes] 
                                             --num_samples 5
                                             --num_per_class 3
                                             --max_classes 8
                                             --output_dir ./augmentation_viz

"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import random
import torchvision

# Add the root directory to the path so we can import the util modules
sys.path.append(str(Path(__file__).parent.parent))

from util.data import get_data


def denormalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    Denormalize a tensor image with mean and standard deviation.
    
    Args:
        tensor: Normalized input tensor image
        mean: Mean used for normalization
        std: Standard deviation used for normalization
        
    Returns:
        Denormalized image tensor
    """
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def visualize_augmented_datasamples(dataset_name, num_samples=5, output_dir=None):
    """
    Visualize augmented data samples from the dataset as they would be seen by PIPNet
    during prototype pre-training.
    
    This function shows original images alongside their augmented versions that are
    used as training pairs in PIPNet prototype pre-training.
    
    Args:
        dataset_name: Name of the dataset ('mnist_counting' or 'geometric_shapes')
        num_samples: Number of samples to visualize
        output_dir: Directory to save visualization images
    """
    # Create an argument namespace for get_data
    class Args:
        def __init__(self):
            self.dataset = dataset_name
            self.image_size = 224
            self.seed = 42
            self.validation_size = 0.2
            self.disable_cuda = True
            self.weighted_loss = False
            self.num_workers = 0
            self.batch_size = num_samples
    
    args = Args()
    
    # Set random seeds for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Get the data
    trainset, trainset_pretraining, trainset_normal, trainset_normal_augment, projectset, testset, testset_projection, classes, _, _, _ = get_data(args)
    
    # Create a dataloader for the training set (which has paired augmentations)
    dataloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False
    )
    
    # Get a batch of data
    for view1, view2, labels in dataloader:
        break
    
    # Find original (unaugmented) images for each sample
    originals = []
    for label in labels:
        found = False
        for img, orig_label in trainset_normal:
            if orig_label == label:
                originals.append(img)
                found = True
                break
        if not found:
            # If we couldn't find a matching image, use a blank one
            originals.append(torch.zeros_like(view1[0]))
    
    # Create a figure
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    
    # For each sample
    for i in range(min(num_samples, len(labels))):
        if num_samples > 1:
            ax_orig, ax_view1, ax_view2 = axes[i]
        else:
            ax_orig, ax_view1, ax_view2 = axes
        
        # Original image
        img_np = denormalize(originals[i]).permute(1, 2, 0).numpy()
        ax_orig.imshow(img_np)
        ax_orig.set_title(f"Original Image\nClass: {classes[labels[i]]}")
        ax_orig.axis('off')
        
        # View 1 (first augmentation)
        img_np = denormalize(view1[i]).permute(1, 2, 0).numpy()
        ax_view1.imshow(img_np)
        ax_view1.set_title(f"Augmented View 1\n(First Training Input)")
        ax_view1.axis('off')
        
        # View 2 (second augmentation)
        img_np = denormalize(view2[i]).permute(1, 2, 0).numpy()
        ax_view2.imshow(img_np)
        ax_view2.set_title(f"Augmented View 2\n(Second Training Input)")
        ax_view2.axis('off')
    
    plt.suptitle(f"Data Augmentation in {dataset_name}\nPIPNet sees these as representing the same visual concept", fontsize=16)
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"{dataset_name}_augmented_datasamples.png"), dpi=150, bbox_inches='tight')
    
    plt.show()


def visualize_class_samples(dataset_name, num_per_class=3, max_classes=8, output_dir=None):
    """
    Visualize samples from each class in the dataset.
    
    Args:
        dataset_name: Name of the dataset ('mnist_counting' or 'geometric_shapes')
        num_per_class: Number of samples to show per class
        max_classes: Maximum number of classes to show
        output_dir: Directory to save visualization images
    """
    # Create an argument namespace for get_data
    class Args:
        def __init__(self):
            self.dataset = dataset_name
            self.image_size = 224
            self.seed = 42
            self.validation_size = 0.2
            self.disable_cuda = True
            self.weighted_loss = False
            self.num_workers = 0
            self.batch_size = 1
    
    args = Args()
    
    # Set random seeds for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Get the data
    trainset, trainset_pretraining, trainset_normal, trainset_normal_augment, projectset, testset, testset_projection, classes, _, _, _ = get_data(args)
    
    # Group samples by class
    class_samples = {}
    for i, (img, label) in enumerate(trainset_normal):
        if label not in class_samples:
            class_samples[label] = []
        if len(class_samples[label]) < num_per_class:
            class_samples[label].append(img)
    
    # Limit the number of classes to show
    num_classes = min(len(classes), max_classes)
    
    # Create a figure
    fig, axes = plt.subplots(num_classes, num_per_class, figsize=(3 * num_per_class, 3 * num_classes))
    
    # For each class
    for class_idx in range(num_classes):
        if class_idx not in class_samples:
            continue
            
        # For each sample
        for j, img in enumerate(class_samples[class_idx]):
            if num_classes > 1 and num_per_class > 1:
                ax = axes[class_idx, j]
            elif num_classes > 1:
                ax = axes[class_idx]
            elif num_per_class > 1:
                ax = axes[j]
            else:
                ax = axes
                
            # Display the image
            img_np = denormalize(img).permute(1, 2, 0).numpy()
            ax.imshow(img_np)
            ax.set_title(f"Class: {classes[class_idx]}")
            ax.axis('off')
    
    plt.suptitle(f"Sample Images from {dataset_name} Dataset", fontsize=16)
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"{dataset_name}_class_samples.png"), dpi=150, bbox_inches='tight')
    
    plt.show()


def visualize_augmentation_components(dataset_name, num_samples=5, output_dir=None):
    """
    Visualize the components of the augmentation pipeline to understand
    how each transformation affects the image.
    
    Args:
        dataset_name: Name of the dataset ('mnist_counting' or 'geometric_shapes')
        num_samples: Number of samples to visualize
        output_dir: Directory to save visualization images
    """
    # Create an argument namespace for get_data
    class Args:
        def __init__(self):
            self.dataset = dataset_name
            self.image_size = 224
            self.seed = 42
            self.validation_size = 0.2
            self.disable_cuda = True
            self.weighted_loss = False
            self.num_workers = 0
            self.batch_size = num_samples
    
    args = Args()
    
    # Set random seeds for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Get the data with all components
    trainset, trainset_pretraining, trainset_normal, trainset_normal_augment, projectset, testset, testset_projection, classes, _, _, _ = get_data(args)
    
    # Access the transform components
    transform1 = trainset.dataset.transform1
    transform2 = trainset.dataset.transform2
    
    # Create a figure
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    
    # For each sample
    sample_indices = random.sample(range(len(trainset_normal)), num_samples)
    
    for i, idx in enumerate(sample_indices):
        if num_samples > 1:
            ax_orig, ax_t1, ax_t1t2 = axes[i]
        else:
            ax_orig, ax_t1, ax_t1t2 = axes
        
        # Get original image
        img, label = trainset_normal[idx]
        
        # Display original image
        img_np = denormalize(img).permute(1, 2, 0).numpy()
        ax_orig.imshow(img_np)
        ax_orig.set_title(f"Original\nClass: {classes[label]}")
        ax_orig.axis('off')
        
        # Apply transform1 and display
        # Since transform1 doesn't return a tensor, we need to get the PIL image
        pil_img = torchvision.transforms.ToPILImage()(denormalize(img))
        img_t1 = transform1(pil_img)
        ax_t1.imshow(img_t1)
        ax_t1.set_title("After Transform1\n(Geometric Transforms)")
        ax_t1.axis('off')
        
        # Apply transform2 to the result of transform1
        img_t2 = transform2(img_t1)
        img_t2_np = denormalize(img_t2).permute(1, 2, 0).numpy()
        ax_t1t2.imshow(img_t2_np)
        ax_t1t2.set_title("After Transform2\n(Final Augmentation)")
        ax_t1t2.axis('off')
    
    plt.suptitle(f"Augmentation Pipeline Components for {dataset_name}", fontsize=16)
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"{dataset_name}_augmentation_components.png"), dpi=150, bbox_inches='tight')
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize augmented data samples for PIPNet datasets')
    parser.add_argument('--dataset', type=str, default='mnist_counting',
                      choices=['mnist_counting', 'geometric_shapes'], 
                      help='Dataset to visualize')
    parser.add_argument('--num_samples', type=int, default=5, 
                      help='Number of samples to visualize augmentations for')
    parser.add_argument('--num_per_class', type=int, default=3,
                      help='Number of samples to show per class')
    parser.add_argument('--max_classes', type=int, default=8,
                      help='Maximum number of classes to show')
    parser.add_argument('--output_dir', type=str, default='./augmentation_viz',
                      help='Directory to save visualization images')
    
    args = parser.parse_args()
    
    # Create output directory if needed
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Run the visualization functions
    print(f"Visualizing class samples for {args.dataset}...")
    visualize_class_samples(args.dataset, args.num_per_class, args.max_classes, args.output_dir)
    
    print(f"Visualizing augmented data samples for {args.dataset}...")
    visualize_augmented_datasamples(args.dataset, args.num_samples, args.output_dir)
    
    print(f"Visualizing augmentation pipeline components for {args.dataset}...")
    try:
        visualize_augmentation_components(args.dataset, args.num_samples, args.output_dir)
    except Exception as e:
        print(f"Warning: Could not visualize augmentation components: {e}")
        print("This is normal if using TrivialAugmentWide transforms.")


if __name__ == "__main__":
    main()