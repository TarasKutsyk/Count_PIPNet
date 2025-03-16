import torch
import torch.nn as nn
from torchvision import models

def replace_convlayers_convnext(model, threshold):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_convlayers_convnext(module, threshold)
        if isinstance(module, nn.Conv2d):
            if module.stride[0] == 2:
                if module.in_channels > threshold: #replace bigger strides to reduce receptive field, skip some 2x2 layers. 
                    # >100 gives output size (26, 26). >300 gives (13, 13)
                    module.stride = tuple(s//2 for s in module.stride)
                    
    return model

class MidLayerConvNeXt(nn.Module):
    """ConvNeXt model that only uses the middle layers for feature extraction"""
    def __init__(self, original_model, num_stages=2):
        super().__init__()
        
        # In torchvision's ConvNeXt, 'features' contains the main network
        # The 'features' module has structure: [0]=downsample, [1...4]=stages
        self.features = nn.Sequential()
        
        # Add the downsample layer (what functions as the stem)
        if hasattr(original_model, 'features') and len(original_model.features) > 0:
            self.features.add_module('0', original_model.features[0])
            
            # Add the desired number of stages (1-indexed in ConvNeXt)
            for i in range(min(num_stages, len(original_model.features) - 1)):  # ConvNeXt has 4 stages max
                if i+1 < len(original_model.features):
                    self.features.add_module(str(i+1), original_model.features[i+1])
            
    def forward(self, x):
        return self.features(x)

def convnext_tiny_26_features(pretrained=False, use_mid_layers=False, num_stages=2, **kwargs):
    """
    ConvNeXt Tiny model with modified strides for better spatial resolution.
    
    Args:
        pretrained: Whether to use pretrained weights
        use_mid_layers: Whether to use only middle layers
        num_stages: Number of stages to use if use_mid_layers is True
    
    Returns:
        ConvNeXt feature extractor
    """
    weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    model = models.convnext_tiny(weights=weights)
    
    with torch.no_grad():
        # Remove classification head
        model.avgpool = nn.Identity()
        model.classifier = nn.Identity()    
        
        # Apply stride modifications
        model = replace_convlayers_convnext(model, 100) 
        
        # Optionally use only the middle layers
        if use_mid_layers:
            model = MidLayerConvNeXt(model, num_stages=num_stages)
    
    return model

def convnext_tiny_13_features(pretrained=False, use_mid_layers=False, num_stages=2, **kwargs):
    """
    ConvNeXt Tiny model with modified strides for lower spatial resolution.
    
    Args:
        pretrained: Whether to use pretrained weights
        use_mid_layers: Whether to use only middle layers
        num_stages: Number of stages to use if use_mid_layers is True
    
    Returns:
        ConvNeXt feature extractor
    """
    weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    model = models.convnext_tiny(weights=weights)
    
    with torch.no_grad():
        # Remove classification head
        model.avgpool = nn.Identity()
        model.classifier = nn.Identity()    
        
        # Apply stride modifications
        model = replace_convlayers_convnext(model, 300)
        
        # Optionally use only the middle layers
        if use_mid_layers:
            model = MidLayerConvNeXt(model, num_stages=num_stages)
    
    return model

# Helper function to get feature map dimensions for a given configuration
def get_feature_dimensions(use_mid_layers=False, num_stages=2, input_size=224):
    """
    Returns the expected feature dimensions for the current configuration.
    Useful for debugging and configuring subsequent layers.
    """
    model = convnext_tiny_26_features(pretrained=False, use_mid_layers=use_mid_layers, num_stages=num_stages)
    model.eval()
    
    with torch.no_grad():
        dummy_input = torch.zeros(1, 3, input_size, input_size)
        output = model(dummy_input)
        return output.shape