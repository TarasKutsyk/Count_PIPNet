"""
Geometric Shapes Dataset Generator

This module provides a generator for creating synthetic datasets specifically designed
to test CountPIPNet's ability to count objects and recognize different shapes. The 
generator creates images with varying numbers of geometric shapes, organized into classes
based on both shape type and count.

This dataset is ideal for evaluating prototype counting mechanisms since it provides
clear, distinct shapes with configurable complexity levels, allowing for controlled
testing of both counting and classification capabilities.
"""

import os
import random
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import matplotlib.pyplot as plt
import argparse


# Default configuration
CONFIG = {
    # Dataset parameters
    # Directory to save the generated images
    'output_dir': './data/geometric_shapes_no_noise_test/dataset',

    'img_size': 192,                          # Size of output images (square)
    'train_samples_per_class': 100,           # Number of training samples per class
    'test_samples_per_class': 0,             # Number of test samples per class
    'seed': 42,                               # Random seed for reproducibility
    
    # Object parameters - based on estimated receptive field size
    'min_object_size': 12,                    # Minimum size for a shape (pixels)
    'max_object_size': 20,                    # Maximum size for a shape (pixels)
    'size_mean': 16,                          # Mean size of objects - based on receptive field estimate
    'size_std': 4,                            # Standard deviation of object size
    
    # Difficulty parameters
    'max_rotation': 15,                       # Maximum rotation angle (degrees)
    'max_overlap': 0.15,                      # Maximum allowed overlap between objects (0-1)
    'noise_level': 0,                        # Background noise level (0-255)
    'outline_width': 2,                       # Width of shape outlines (pixels)
    
    # Shape types available
    'shape_types': ['circle', 'square', 'triangle', 'hexagon'],
    
    # Class definitions - each tuple defines (shape_type, count)
    # This creates a balanced dataset with different shapes at each count
    'class_definitions': [
        # Classes with 1 object
        ('circle', 1),      # Class 1: One circle
        # ('square', 1),      # Class 2: One square
        ('triangle', 1),    # Class 3: One triangle
        ('hexagon', 1),     # Class 4: One hexagon
        
        # Classes with 2 objects
        ('circle', 2),      # Class 5: Two circles
        # ('square', 2),      # Class 6: Two squares
        ('triangle', 2),    # Class 7: Two triangles
        ('hexagon', 2),     # Class 8: Two hexagons
        
        # Classes with 3 objects
        ('circle', 3),      # Class 9: Three circles
        # ('square', 3),      # Class 10: Three squares
        ('triangle', 3),    # Class 11: Three triangles
        ('hexagon', 3),     # Class 12: Three hexagons
    ]
}


class GeometricShapesGenerator:
    """
    Generates a dataset of images with geometric shapes.
    Classes are defined based on both shape type and count.
    
    This generator creates synthetic images with varying numbers of geometric shapes,
    with class labels determined by both the type of shape and how many instances
    are present in the image. The generator ensures proper balancing between classes
    and controls shape placement to avoid excessive overlap.
    
    Attributes:
        config (Dict): Configuration parameters for the generator
        output_dir (str): Directory to save the generated images
        img_size (int): Size of output images (square)
        seed (int): Random seed for reproducibility
        min_object_size (int): Minimum size for a shape (pixels)
        max_object_size (int): Maximum size for a shape (pixels)
        size_mean (int): Mean size of objects - based on receptive field estimate
        size_std (int): Standard deviation of object size
        max_rotation (int): Maximum rotation angle (degrees)
        max_overlap (float): Maximum allowed overlap between objects (0-1)
        noise_level (int): Background noise level (0-255)
        outline_width (int): Width of shape outlines (pixels)
        shape_types (List[str]): Available shape types to generate
        class_definitions (List[Tuple[str, int]]): List of (shape_type, count) tuples defining classes
        shape_colors (Dict[str, Tuple[int, int, int]]): Color mapping for different shape types
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
        self.outline_width = self.config['outline_width']
        self.shape_types = self.config['shape_types']
        self.class_definitions = self.config['class_definitions']
        
        # Set random seed for reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Define color mapping for shape types to ensure consistent colors
        self.shape_colors = {
            'circle': (50, 50, 200),      # Blue-ish
            'square': (200, 50, 50),      # Red-ish
            'triangle': (50, 200, 50),    # Green-ish
            'hexagon': (200, 150, 50),    # Orange-ish
            'star': (150, 50, 200),       # Purple-ish
            'cross': (50, 200, 200),      # Cyan-ish
        }
    
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
    
    def generate_dataset(self, train_samples_per_class=None, test_samples_per_class=None):
        # Create output directories before generating
        self._create_output_directories()
        
        # Use config values if not provided
        train_samples_per_class = train_samples_per_class or self.config['train_samples_per_class']
        test_samples_per_class = test_samples_per_class or self.config['test_samples_per_class']
        
        print(f"Generating dataset with {len(self.class_definitions)} classes...")
        print(f"{train_samples_per_class} training and {test_samples_per_class} test images per class")
        
        # Generate training images
        for class_idx, (shape_type, count) in enumerate(self.class_definitions, 1):
            print(f"Generating class {class_idx}: {count} {shape_type}(s) (train)...")
            for i in range(train_samples_per_class):
                img = self._generate_image(shape_type, count)
                img.save(os.path.join(self.output_dir, 'train', f'class_{class_idx}', 
                                    f'img_{i:04d}.png'))
        
        # Generate test images
        for class_idx, (shape_type, count) in enumerate(self.class_definitions, 1):
            print(f"Generating class {class_idx}: {count} {shape_type}(s) (test)...")
            for i in range(test_samples_per_class):
                img = self._generate_image(shape_type, count)
                img.save(os.path.join(self.output_dir, 'test', f'class_{class_idx}', 
                                    f'img_{i:04d}.png'))
        
        print("Dataset generation complete!")
    
    def _generate_image(self, shape_type: str, count: int) -> Image.Image:
        """
        Generate an image with the specified number of instances of the given shape.
        
        Args:
            shape_type: The type of shape to place in the image (from shape_types list)
            count: How many instances of the shape to place
            
        Returns:
            PIL Image object with the generated image
        """
        # Create a blank canvas
        canvas = Image.new('RGB', (self.img_size, self.img_size), color=(255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        
        # Add some background noise if configured
        if self.noise_level > 0:
            # Generate random noise with the specified intensity
            noise = np.random.randint(0, self.noise_level, 
                                     (self.img_size, self.img_size, 3), dtype=np.uint8)
            noise_img = Image.fromarray(255 - noise)
            canvas = Image.blend(canvas, noise_img, 0.05)
            draw = ImageDraw.Draw(canvas)  # Recreate draw object after blending
        
        # Track occupied regions to control overlap between shapes
        occupied_regions = []
        
        # Get base color for this shape type
        color = self.shape_colors.get(shape_type, (100, 100, 100))
        
        # Add color variation to make images more diverse
        # This adds subtle variations to the shape colors while maintaining type recognizability
        color_variation = 30
        color = tuple(max(0, min(255, c + random.randint(-color_variation, color_variation))) 
                     for c in color)
        
        # Place shapes on the canvas
        for _ in range(count):
            # Determine the size for this shape using a normal distribution
            # within the configured constraints
            size = int(np.clip(
                np.random.normal(self.size_mean, self.size_std),
                self.min_object_size, 
                self.max_object_size
            ))
            
            # Find position and place the shape
            x, y, shape_bbox = self._place_shape(shape_type, size, occupied_regions, draw, color)
            
            # Update occupied regions
            occupied_regions.append(shape_bbox)
        
        return canvas
    
    def _place_shape(self, shape_type: str, size: int, 
                    occupied_regions: List[Tuple[int, int, int, int]], 
                    draw: ImageDraw.Draw, color: Tuple[int, int, int]) -> Tuple[int, int, Tuple[int, int, int, int]]:
        """
        Place a shape on the canvas and return its bounding box.
        
        This method finds a suitable position for the shape, draws it on the canvas,
        and returns information about its location.
        
        Args:
            shape_type: Type of shape to draw
            size: Size of the shape in pixels
            occupied_regions: List of bounding boxes for already placed shapes
            draw: PIL ImageDraw object for drawing
            color: RGB color tuple for the shape
            
        Returns:
            Tuple of (x, y, bounding_box) where:
                x, y: Top-left coordinates of the shape
                bounding_box: (x1, y1, x2, y2) coordinates of shape bounding box
        """
        max_attempts = 50
        
        # Calculate shape height based on shape type
        # Different shapes have different aspect ratios
        if shape_type == 'triangle':
            height = int(size * 0.866)  # height of equilateral triangle
            bbox_width, bbox_height = size, height
        elif shape_type == 'hexagon':
            height = int(size * 0.866 * 2)  # height of regular hexagon
            bbox_width, bbox_height = size, height
        else:
            # For circle, square, etc.
            bbox_width, bbox_height = size, size
        
        # Find a position with acceptable overlap
        for attempt in range(max_attempts):
            # Random position within canvas boundaries
            x = random.randint(0, self.img_size - bbox_width)
            y = random.randint(0, self.img_size - bbox_height)
            
            # Calculate bounding box for this position
            bbox = (x, y, x + bbox_width, y + bbox_height)
            
            # Check overlap with existing shapes
            overlap_too_much = False
            for other_bbox in occupied_regions:
                # Calculate intersection area
                intersection = self._get_intersection(bbox, other_bbox)
                if intersection:
                    ix1, iy1, ix2, iy2 = intersection
                    intersection_area = (ix2 - ix1) * (iy2 - iy1)
                    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    overlap_ratio = intersection_area / bbox_area
                    
                    # Check if overlap exceeds the maximum allowed
                    if overlap_ratio > self.max_overlap:
                        overlap_too_much = True
                        break
            
            # If this position works or we've reached max attempts, use it
            if not overlap_too_much or attempt == max_attempts - 1:
                break
        
        # Now draw the shape at the selected position
        self._draw_shape(shape_type, x, y, size, draw, color)
        
        return x, y, bbox
    
    def _draw_shape(self, shape_type: str, x: int, y: int, size: int, 
                   draw: ImageDraw.Draw, color: Tuple[int, int, int]) -> None:
        """
        Draw a shape at the specified position with the given size.
        
        Different shape types are handled separately with appropriate drawing methods.
        
        Args:
            shape_type: Type of shape to draw
            x: X-coordinate of top-left corner
            y: Y-coordinate of top-left corner
            size: Size of the shape in pixels
            draw: PIL ImageDraw object for drawing
            color: RGB color tuple for the shape
        """
        outline_color = (0, 0, 0)  # Black outline for all shapes
        
        if shape_type == 'circle':
            # Simple circle drawing
            draw.ellipse((x, y, x + size, y + size), fill=color, 
                         outline=outline_color, width=self.outline_width)
            
        elif shape_type == 'square':
            # For squares, we can optionally apply rotation
            if self.max_rotation > 0 and random.random() < 0.5:
                # Create a temporary image for the rotated square to handle rotation properly
                temp = Image.new('RGBA', (size+20, size+20), (255, 255, 255, 0))
                temp_draw = ImageDraw.Draw(temp)
                temp_draw.rectangle((10, 10, 10 + size, 10 + size), 
                                   fill=color, outline=outline_color, width=self.outline_width)
                
                # Rotate and paste with proper alpha handling
                angle = random.uniform(-self.max_rotation, self.max_rotation)
                temp = temp.rotate(angle, expand=True)
                canvas = draw._image
                canvas.paste(temp, (x - 10, y - 10), temp)
            else:
                # Draw a normal square if no rotation
                draw.rectangle((x, y, x + size, y + size), 
                              fill=color, outline=outline_color, width=self.outline_width)
            
        elif shape_type == 'triangle':
            # Equilateral triangle
            height = int(size * 0.866)  # height of equilateral triangle
            points = [(x, y + height), (x + size//2, y), (x + size, y + height)]
            
            # Optional rotation for the triangle
            if self.max_rotation > 0:
                # Rotate points around center using trigonometry
                center_x = x + size//2
                center_y = y + height//2
                angle_rad = random.uniform(-self.max_rotation, self.max_rotation) * (np.pi / 180)
                
                rotated_points = []
                for px, py in points:
                    # Translate point to origin
                    tx, ty = px - center_x, py - center_y
                    # Apply rotation transform
                    rx = tx * np.cos(angle_rad) - ty * np.sin(angle_rad)
                    ry = tx * np.sin(angle_rad) + ty * np.cos(angle_rad)
                    # Translate back to original position
                    rotated_points.append((rx + center_x, ry + center_y))
                
                points = rotated_points
            
            # Draw the triangle using the calculated points
            draw.polygon(points, fill=color, outline=outline_color, width=self.outline_width)
            
        elif shape_type == 'hexagon':
            # Regular hexagon
            center_x = x + size//2
            center_y = y + size//2
            radius = size//2
            
            # Calculate vertices using trigonometry
            points = []
            for i in range(6):
                angle_rad = (np.pi / 3) * i
                px = center_x + radius * np.cos(angle_rad)
                py = center_y + radius * np.sin(angle_rad)
                points.append((px, py))
            
            # Optional rotation for the hexagon
            if self.max_rotation > 0:
                # Rotate points around center
                angle_rad = random.uniform(-self.max_rotation, self.max_rotation) * (np.pi / 180)
                
                rotated_points = []
                for px, py in points:
                    # Translate point to origin
                    tx, ty = px - center_x, py - center_y
                    # Apply rotation transform
                    rx = tx * np.cos(angle_rad) - ty * np.sin(angle_rad)
                    ry = tx * np.sin(angle_rad) + ty * np.cos(angle_rad)
                    # Translate back to original position
                    rotated_points.append((rx + center_x, ry + center_y))
                
                points = rotated_points
            
            # Draw the hexagon using the calculated points
            draw.polygon(points, fill=color, outline=outline_color, width=self.outline_width)
        else:
            raise ValueError(f'Unsupported shape type: {shape_type}')
    
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

    def visualize_class_grid(self) -> plt.Figure:
        """
        Generates and visualizes one sample image for each class in a grid.
        Best suited for a number of classes that fits a grid, like 9 for a 3x3.
        """
        num_classes = len(self.class_definitions)
        if num_classes == 0:
            print("No classes to visualize.")
            return plt.figure()
            
        # Determine grid size, aiming for a square-like layout
        cols = int(np.ceil(np.sqrt(num_classes)))
        rows = int(np.ceil(num_classes / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
        axes = axes.flatten()  # Flatten to make iteration easy

        for i, (shape_type, count) in enumerate(self.class_definitions):
            class_desc = f"{count} {shape_type}{'s' if count > 1 else ''}"
            
            # Generate one sample image for the class
            img = self._generate_image(shape_type, count)

            border_size = 2  # The width of the border in pixels
            img_with_border = ImageOps.expand(img, border=border_size, fill='black')
            
            ax = axes[i]
            ax.imshow(img_with_border) # Display the image with the border
            ax.set_title(f"Class {i+1}\n{class_desc}", fontsize=10)
            ax.axis('off')
            
        # Hide any unused subplots
        for i in range(num_classes, len(axes)):
            axes[i].axis('off')

        # fig.suptitle("Dataset Samples (One Per Class)", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle
        return fig

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
        if num_classes == 1: # Ensure axes is a numpy array for consistent indexing
            axes = np.array([axes])
        if num_samples == 1:
            axes = axes.reshape(-1, 1)

        for i, (shape_type, count) in enumerate(self.class_definitions):
            class_desc = f"{count} {shape_type}{'s' if count>1 else ''}"
            
            for j in range(num_samples):
                img = self._generate_image(shape_type, count)
                
                ax = axes[i, j]
                ax.imshow(img)
                if j == 0:  # Only show class description on first sample
                    ax.set_title(f"Class {i+1}\n{class_desc}", fontsize=10)
                ax.axis('off')

        plt.tight_layout()
        return fig


# Main execution block
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate or visualize a geometric shapes dataset.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--viz_only',
        action='store_true',
        help="If set, only visualize a grid of samples (one per class)\ninstead of generating the full dataset."
    )
    args = parser.parse_args()
    
    # Create a clean 9-class configuration for the 3x3 grid visualization
    my_config = CONFIG.copy()
    my_config['class_definitions'] = [
        ('circle', 1),   ('triangle', 1),   ('hexagon', 1),
        ('circle', 2),   ('triangle', 2),   ('hexagon', 2),
        ('circle', 3),   ('triangle', 3),   ('hexagon', 3),
    ]

    # Create generator with the updated configuration
    generator = GeometricShapesGenerator(my_config)

    if args.viz_only:
        # --- Visualization Only Mode ---
        print("--- Visualization Only Mode ---")
        print("Generating a grid of samples (one per class)...")
        
        # Use the new method to show one sample from each of the 9 classes
        fig = generator.visualize_class_grid()
        
        save_path = "geometric_shapes_grid_visualization.png"
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
        plt.show()

    else:
        # --- Full Dataset Generation Mode (Original functionality) ---
        print("--- Full Dataset Generation Mode ---")
        generator.generate_dataset()
        
        # After generating, visualize a few samples from each class
        print("\nVisualizing a few generated samples...")
        fig = generator.visualize_samples(num_samples=3)
        
        save_path = "geometric_shapes_samples.png"
        plt.savefig(save_path, dpi=300)
        print(f"Dataset generated successfully in: {generator.output_dir}")
        print(f"A sample visualization has been saved to {save_path}")
        plt.show()