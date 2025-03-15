"""
MNIST Digit Counting Dataset Generator

This module provides a generator for creating synthetic datasets for testing CountPIPNet's
ability to count instances of specific MNIST digits. The generator creates images with
varying numbers of digits from the MNIST dataset, organized into classes based on both
digit identity and count.

The dataset is specifically designed to test counting capabilities while also requiring 
recognition of specific digit identities, preventing the model from using simplistic
shortcuts for classification.
"""

import os
import random
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

# Default configuration
CONFIG = {
    # Dataset parameters
    
    'output_dir': './data/mnist_counting/dataset', # Directory to save the generated images

    'img_size': 224,                            # Size of output images (square)
    'train_samples_per_class': 200,             # Number of training samples per class
    'test_samples_per_class': 50,               # Number of test samples per class
    'seed': 42,                                 # Random seed for reproducibility
    
    # Object parameters - based on estimated receptive field size
    'min_object_size': 24,                  # Minimum size for a digit (pixels)
    'max_object_size': 32,                  # Maximum size for a digit (pixels)
    'size_mean': 28,                        # Mean size of objects - based on receptive field estimate
    'size_std': 4,                          # Standard deviation of object size
    
    # Difficulty parameters
    'max_rotation': 15,                     # Maximum rotation angle (degrees)
    'max_overlap': 0.15,                    # Maximum allowed overlap between objects (0-1)
    'noise_level': 50,                      # Background noise level (0-255)
    
    # Class parameters - each tuple defines a class by (digit, count)
    # This creates a balanced dataset with equal focus on digit identity and count
    'class_definitions': [
        # Classes with 1 object
        (1, 1),   # Class 1: One digit '1'
        (9, 1),   # Class 2: One digit '9'
        
        # Classes with 2 objects
        (1, 2),   # Class 3: Two digits '1'
        (9, 2),   # Class 4: Two digits '9'
        
        # Classes with 3 objects
        (1, 3),   # Class 5: Three digits '1'
        (9, 3),   # Class 6: Three digits '9'
        
        # Classes with 4 objects
        (1, 4),   # Class 7: Four digits '1'
        (9, 4),   # Class 8: Four digits '9'
    ]
}


class MNISTCountingGenerator:
    """
    Generates a dataset of images with specific MNIST digits and counts.
    Classes are defined based on both digit identity and count.
    
    This generator creates synthetic images with varying numbers of specific MNIST digits,
    with class labels determined by both the identity of the digit and how many instances
    are present in the image. The generator ensures proper balancing between classes and
    controls digit placement to avoid excessive overlap.
    
    Attributes:
        config (Dict): Configuration parameters for the generator
        output_dir (str): Directory to save the generated images
        img_size (int): Size of output images (square)
        size_mean (int): Mean size of objects - based on receptive field estimate
        size_std (int): Standard deviation of object size
        min_object_size (int): Minimum size for a digit (pixels)
        max_object_size (int): Maximum size for a digit (pixels)
        max_rotation (int): Maximum rotation angle (degrees)
        max_overlap (float): Maximum allowed overlap between objects (0-1)
        noise_level (int): Background noise level (0-255)
        class_definitions (List[Tuple[int, int]]): List of (digit, count) tuples defining classes
        mnist (Dataset): MNIST dataset loaded from torchvision
        digit_indices (Dict[int, List[int]]): Dictionary mapping digit labels to indices in MNIST dataset
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the generator with the given configuration.
        
        Args:
            config: Dictionary of configuration parameters. If None, default CONFIG is used.
        """
        self.config = config or CONFIG
        
        # Extract parameters from config for easier access
        self.output_dir = self.config['output_dir']
        self.img_size = self.config['img_size']
        self.seed = self.config['seed']
        self.min_object_size = self.config['min_object_size']
        self.max_object_size = self.config['max_object_size']
        self.size_mean = self.config['size_mean']
        self.size_std = self.config['size_std']
        self.max_rotation = self.config['max_rotation']
        self.max_overlap = self.config['max_overlap']
        self.noise_level = self.config['noise_level']
        self.class_definitions = self.config['class_definitions']
        
        # Set random seed for reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        # Load MNIST dataset
        self.mnist = MNIST(root='./data', train=True, download=True,
                          transform=transforms.ToTensor())
        
        # Create output directories
        self._create_output_directories()
        
        # Group MNIST digits by label for efficient retrieval
        self.digit_indices = self._group_mnist_by_label()
    
    def _create_output_directories(self) -> None:
        """
        Create the necessary directory structure for saving generated images.
        
        Creates:
            - Main output directory
            - Train and test subdirectories for each class
        """
        os.makedirs(self.output_dir, exist_ok=True)
        for i, _ in enumerate(self.class_definitions, 1):
            os.makedirs(os.path.join(self.output_dir, 'train', f'class_{i}'), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, 'test', f'class_{i}'), exist_ok=True)
    
    def _group_mnist_by_label(self) -> Dict[int, List[int]]:
        """
        Group MNIST dataset indices by digit label for efficient retrieval.
        
        Returns:
            Dictionary mapping digit labels (0-9) to lists of indices in the MNIST dataset
        """
        digit_indices = {}
        for i, (_, label) in enumerate(self.mnist):
            if label not in digit_indices:
                digit_indices[label] = []
            digit_indices[label].append(i)
        return digit_indices
    
    def generate_dataset(self, train_samples_per_class=None, test_samples_per_class=None):
        # Use config values if not provided
        train_samples_per_class = train_samples_per_class or self.config['train_samples_per_class']
        test_samples_per_class = test_samples_per_class or self.config['test_samples_per_class']
        
        print(f"Generating dataset with {len(self.class_definitions)} classes...")
        print(f"{train_samples_per_class} training and {test_samples_per_class} test images per class")
        
        # Generate training images
        for class_idx, (digit, count) in enumerate(self.class_definitions, 1):
            print(f"Generating class {class_idx}: {count} instances of digit {digit} (train)...")
            for i in range(train_samples_per_class):
                img = self._generate_image(digit, count)
                img.save(os.path.join(self.output_dir, 'train', f'class_{class_idx}', 
                                    f'img_{i:04d}.png'))
        
        # Generate test images
        for class_idx, (digit, count) in enumerate(self.class_definitions, 1):
            print(f"Generating class {class_idx}: {count} instances of digit {digit} (test)...")
            for i in range(test_samples_per_class):
                img = self._generate_image(digit, count)
                img.save(os.path.join(self.output_dir, 'test', f'class_{class_idx}', 
                                    f'img_{i:04d}.png'))
        
        print("Dataset generation complete!")
    
    def _preprocess_digit(self, digit_tensor: torch.Tensor, target_size: int, 
                         rotation: float = 0.0) -> Image.Image:
        """
        Process MNIST digit to create a clean black digit on transparent background.
        
        Args:
            digit_tensor: PyTorch tensor containing MNIST digit
            target_size: Target size for the digit (pixels)
            rotation: Rotation angle in degrees
            
        Returns:
            Processed PIL Image with digit as black on transparent background
        """
        # Convert tensor to PIL Image
        digit_img = transforms.ToPILImage()(digit_tensor)
        
        # Invert to make digit black on white background (MNIST is white on black)
        digit_img = ImageOps.invert(digit_img)
        
        # Convert to RGBA for transparency handling
        digit_img = digit_img.convert("RGBA")
        
        # Threshold to make digit solid black while keeping background transparent
        # This approach uses numpy array operations for precise control
        data = np.array(digit_img)
        r, g, b, a = data.T
        
        # Create mask for the digit (where it's significantly dark)
        # The threshold value 200 works well for MNIST digits - pixels below this
        # are considered part of the digit
        digit_mask = (r < 200) & (g < 200) & (b < 200)
        
        # Set the digit to solid black (0,0,0) and preserve full opacity (255)
        data[..., :3][digit_mask.T] = (0, 0, 0)  # Solid black for digit
        
        # Set the background to fully transparent (alpha=0)
        data[..., 3][~digit_mask.T] = 0  # Transparent for background
        
        # Convert back to PIL image
        digit_img = Image.fromarray(data)
        
        # Resize after thresholding to maintain clean edges
        digit_img = digit_img.resize((target_size, target_size), Image.LANCZOS)
        
        # Rotate if needed - using bicubic resampling for smoother rotation
        if rotation != 0:
            digit_img = digit_img.rotate(rotation, expand=True, resample=Image.BICUBIC)
        
        return digit_img
    
    def _generate_image(self, digit: int, count: int) -> Image.Image:
        """
        Generate an image with the specified number of instances of the given digit.
        
        Args:
            digit: The digit to place in the image (0-9)
            count: How many instances of the digit to place
            
        Returns:
            PIL Image object with the generated image
        """
        # Create a blank canvas
        canvas = Image.new('RGB', (self.img_size, self.img_size), color=(255, 255, 255))
        
        # Add some background noise if configured
        if self.noise_level > 0:
            # Generate random noise with the specified intensity
            noise = np.random.randint(0, self.noise_level, 
                                     (self.img_size, self.img_size, 3), dtype=np.uint8)
            noise_img = Image.fromarray(255 - noise)
            canvas = Image.blend(canvas, noise_img, 0.05)
        
        # Track occupied regions to control overlap between digits
        occupied_regions = []
        
        # Place digits on the canvas
        for _ in range(count):
            # Get a random instance of the specified digit from MNIST
            idx = random.choice(self.digit_indices[digit])
            digit_tensor, _ = self.mnist[idx]
            
            # Determine the size for this digit using a normal distribution
            # within the configured constraints
            size = int(np.clip(
                np.random.normal(self.size_mean, self.size_std),
                self.min_object_size, 
                self.max_object_size
            ))
            
            # Apply random rotation if configured
            rotation = random.uniform(-self.max_rotation, self.max_rotation) if self.max_rotation > 0 else 0
            
            # Process the digit to get a clean black digit on transparent background
            digit_img = self._preprocess_digit(digit_tensor, size, rotation)
            
            # Find position with minimal overlap with existing digits
            position = self._find_position(digit_img.width, digit_img.height, occupied_regions)
            x, y = position
            
            # Update occupied regions
            occupied_regions.append((x, y, x + digit_img.width, y + digit_img.height))
            
            # Convert canvas to RGBA for transparent compositing
            if canvas.mode != 'RGBA':
                canvas = canvas.convert('RGBA')
                
            # Paste digit onto canvas with transparency
            canvas.paste(digit_img, (x, y), digit_img)
        
        # Convert back to RGB for final output
        canvas = canvas.convert('RGB')
        
        return canvas
    
    def _find_position(self, width: int, height: int, 
                      occupied_regions: List[Tuple[int, int, int, int]], 
                      max_attempts: int = 50) -> Tuple[int, int]:
        """
        Find a position for an object with controlled overlap.
        
        This method attempts to place an object on the canvas while respecting
        the maximum allowed overlap with existing objects.
        
        Args:
            width: Width of the object to place
            height: Height of the object to place
            occupied_regions: List of (x1, y1, x2, y2) tuples representing existing objects
            max_attempts: Maximum number of attempts to find a suitable position
            
        Returns:
            Tuple of (x, y) coordinates for the top-left corner of the object
        """
        for attempt in range(max_attempts):
            # Random position within canvas boundaries
            x = random.randint(0, self.img_size - width)
            y = random.randint(0, self.img_size - height)
            
            # Check overlap with existing objects
            region = (x, y, x + width, y + height)
            overlap_too_much = False
            
            for other_region in occupied_regions:
                # Calculate intersection area between this position and existing object
                intersection = self._get_intersection(region, other_region)
                if intersection:
                    ix1, iy1, ix2, iy2 = intersection
                    intersection_area = (ix2 - ix1) * (iy2 - iy1)
                    region_area = (region[2] - region[0]) * (region[3] - region[1])
                    overlap_ratio = intersection_area / region_area
                    
                    # Check if overlap exceeds the maximum allowed
                    if overlap_ratio > self.max_overlap:
                        overlap_too_much = True
                        break
            
            # If this position works or we've reached max attempts, use it
            if not overlap_too_much or attempt == max_attempts - 1:
                return (x, y)
        
        # If all attempts failed, return a random position
        return (random.randint(0, self.img_size - width), 
                random.randint(0, self.img_size - height))
    
    def _get_intersection(self, region1: Tuple[int, int, int, int], 
                         region2: Tuple[int, int, int, int]) -> Optional[Tuple[int, int, int, int]]:
        """
        Calculate intersection between two rectangular regions.
        
        Args:
            region1: First region as (x1, y1, x2, y2)
            region2: Second region as (x1, y1, x2, y2)
            
        Returns:
            Intersection region as (x1, y1, x2, y2) or None if no intersection
        """
        # Find the intersection boundaries
        x1 = max(region1[0], region2[0])
        y1 = max(region1[1], region2[1])
        x2 = min(region1[2], region2[2])
        y2 = min(region1[3], region2[3])
        
        # Return the intersection region if it exists
        if x1 < x2 and y1 < y2:
            return (x1, y1, x2, y2)
        return None
    
    def visualize_samples(self, num_samples: int = 2) -> plt.Figure:
        """
        Generate and visualize sample images from each class.
        
        Creates a grid of sample images showing examples from each class in the dataset.
        
        Args:
            num_samples: Number of samples to generate per class
            
        Returns:
            Matplotlib Figure object containing the visualization
        """
        num_classes = len(self.class_definitions)
        fig, axes = plt.subplots(num_classes, num_samples, 
                                figsize=(num_samples * 3, num_classes * 3))
        
        for i, (digit, count) in enumerate(self.class_definitions):
            class_desc = f"{count} digit{'s' if count>1 else ''} '{digit}'"
            
            for j in range(num_samples):
                img = self._generate_image(digit, count)
                
                if num_classes > 1 and num_samples > 1:
                    axes[i, j].imshow(img)
                    if j == 0:  # Only show class description on first sample
                        axes[i, j].set_title(f"Class {i+1}\n{class_desc}", fontsize=10)
                    axes[i, j].axis('off')
                else:
                    axes[i].imshow(img)
                    axes[i].set_title(f"Class {i+1}\n{class_desc}")
                    axes[i].axis('off')
        
        plt.tight_layout()
        return fig


# Example usage
if __name__ == "__main__":
    # Customize config if needed
    my_config = CONFIG.copy()
    # Example of modifying the class definitions:
    # my_config['class_definitions'] = [
    #     (0, 1), (1, 1), (2, 1), (3, 1),  # Four classes with 1 object
    #     (0, 2), (1, 2), (2, 2), (3, 2),  # Four classes with 2 objects
    #     (0, 3), (1, 3), (2, 3), (3, 3),  # Four classes with 3 objects
    # ]
    
    # Create generator with the configuration
    generator = MNISTCountingGenerator(my_config)
    
    # Generate a small dataset for testing
    # Using smaller values for testing; use larger values for real dataset
    generator.generate_dataset(train_samples_per_class=100, test_samples_per_class=25)
    
    # Visualize samples and save the visualization
    fig = generator.visualize_samples(num_samples=3)
    plt.savefig("mnist_counting_samples.png")
    plt.show()