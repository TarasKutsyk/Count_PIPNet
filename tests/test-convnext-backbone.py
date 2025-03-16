import torch
import torch.nn as nn
from torchvision import models
import os
import sys

# Add the project directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.convnext_features import convnext_tiny_26_features, MidLayerConvNeXt, replace_convlayers_convnext


def analyze_convnext_stages():
    """Analyze what CountPIPNet receives for different num_stages configurations"""
    print("\n" + "="*80)
    print("CONVNEXT BACKBONE ANALYSIS FOR DIFFERENT STAGE CONFIGURATIONS")
    print("="*80 + "\n")
    
    # Original ConvNeXt structure
    print("First, let's understand the original ConvNeXt Tiny architecture:")
    model = models.convnext_tiny(pretrained=False)
    
    print("\nOriginal ConvNeXt features structure:")
    for i, layer in enumerate(model.features):
        layer_type = layer.__class__.__name__
        print(f"  Layer {i}: {layer_type}")
        
        # Check for sequential to understand stage structure
        if isinstance(layer, nn.Sequential):
            num_blocks = len(list(layer.children()))
            print(f"    Contains {num_blocks} blocks")
            
            # Examine first block to get channel info
            first_block = next(iter(layer.children()))
            if hasattr(first_block, 'block'):
                try:
                    in_channels = first_block.block[0].in_channels
                    print(f"    Input channels: {in_channels}")
                except (AttributeError, IndexError):
                    pass
    
    # Now test with different stage counts
    for num_stages in [1, 2, 3, 4]:
        print("\n" + "="*80)
        print(f"ANALYZING NUM_STAGES = {num_stages}")
        print("="*80)
        
        # Create the backbone with specific number of stages
        backbone = convnext_tiny_26_features(
            pretrained=False,
            use_mid_layers=True,
            num_stages=num_stages
        )
        
        # Print the structure of this backbone
        print(f"\nMidLayerConvNeXt with {num_stages} stages structure:")
        print("  features:")
        for stage_idx in sorted([int(k) for k in backbone.features._modules.keys()]):
            stage = backbone.features._modules[str(stage_idx)]
            if isinstance(stage, nn.Sequential):
                print(f"    Stage {stage_idx}: Sequential with {len(list(stage.children()))} modules")
            else:
                print(f"    Stage {stage_idx}: {stage.__class__.__name__}")
        
        # Trace tensor flow through the network
        backbone.eval()
        dummy_input = torch.zeros(1, 3, 224, 224)
        
        # Register hooks to capture both inputs and outputs
        stage_data = {}
        def get_io_hook(name):
            def hook(module, input, output):
                stage_data[name] = {
                    'input': input[0].detach(),  # input is a tuple, first element is the tensor
                    'output': output.detach()
                }
            return hook
        
        # Register hooks on each stage
        for stage_idx in backbone.features._modules.keys():
            backbone.features._modules[stage_idx].register_forward_hook(
                get_io_hook(f"stage_{stage_idx}")
            )
        
        # Forward pass
        with torch.no_grad():
            final_output = backbone(dummy_input)
        
        # Show each stage's input and output shapes
        print("\nInput and output tensor shapes at each stage:")
        for name in sorted(stage_data.keys()):
            input_shape = stage_data[name]['input'].shape
            output_shape = stage_data[name]['output'].shape
            print(f"  {name}:")
            print(f"    Input:  {input_shape}")
            print(f"    Output: {output_shape}")
            
            # Analyze transformation
            if input_shape != output_shape:
                changes = []
                if input_shape[1] != output_shape[1]:
                    changes.append(f"Channels: {input_shape[1]} → {output_shape[1]}")
                if input_shape[2:] != output_shape[2:]:
                    changes.append(f"Spatial: {input_shape[2]}×{input_shape[3]} → {output_shape[2]}×{output_shape[3]}")
                if changes:
                    print(f"    Transforms: {', '.join(changes)}")
        
        # Final output analysis
        print("\nFinal output tensor (what CountPIPNet receives):")
        print(f"  Shape: {final_output.shape}")
        print(f"  Channels: {final_output.size(1)}")
        print(f"  Spatial dimensions: {final_output.size(2)}×{final_output.size(3)}")
        
        # Find which stage's output matches the final output
        matching_stage = None
        for name, data in stage_data.items():
            if data['output'].shape == final_output.shape and torch.allclose(data['output'], final_output):
                matching_stage = name
        
        if matching_stage:
            print(f"  ✓ Final output is identical to {matching_stage} output")
        else:
            print(f"  ✗ Final output doesn't match any individual stage output")
        
        # Memory cleanup
        del backbone
        del stage_data
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

if __name__ == "__main__":
    analyze_convnext_stages()