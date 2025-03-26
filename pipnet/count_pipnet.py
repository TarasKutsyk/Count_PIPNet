import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from features.convnext_features import convnext_tiny_26_features, convnext_tiny_13_features
from typing import List, Tuple, Dict, Optional, Union, Callable
from torch.types import Tensor
import random

from .count_pipnet_utils import GumbelSoftmax, STE_Round, OneHotEncoder

class CountPIPNet(nn.Module):
    """
    Count-aware PIP-Net: Patch-based Intuitive Prototypes Network with prototype counting.
    This version extends the original PIP-Net by replacing max-pooling with a counting
    mechanism that keeps track of how many times each prototype appears in an image.
    """
    def __init__(self,
                 num_classes: int,
                 num_prototypes: int,
                 feature_net: nn.Module,
                 args: argparse.Namespace,
                 add_on_layers: nn.Module,
                 classification_layer: nn.Module,
                 max_count: int = 3,
                 use_ste: bool = False):
        """
        Initialize the CountPIPNet model.
        
        Args:
            num_classes: Number of output classes
            num_prototypes: Number of prototypical parts
            feature_net: Backbone network for feature extraction
            args: Command line arguments
            add_on_layers: Layers applied after feature extraction
            classification_layer: Final classification layer
            max_count: Maximum count value to consider (counts >= max_count get mapped to max_count)
            use_ste: Whether to use Straight-Through Estimators for non-differentiable operations
        """
        super().__init__()
        assert num_classes > 0
            
        self._num_features = args.num_features
        self._num_classes = num_classes
        self._num_prototypes = num_prototypes
        self._net = feature_net
        self._add_on = add_on_layers
        self._classification = classification_layer
        self._max_count = max_count
        self._use_ste = use_ste
        self._multiplier = classification_layer.normalization_multiplier
        
        # STE function for rounding (only used if use_ste=True)
        self.ste_round = STE_Round.apply
        
        # Create unified one-hot encoder (handles both normal and STE modes)
        self.onehot_encoder = OneHotEncoder(self._max_count, use_ste=use_ste)
    
    def forward(self, xs, inference=False):
        """
        Forward pass of CountPIPNet.
        
        Args:
            xs: Input tensor of shape [batch_size, channels, height, width]
            inference: Whether to run in inference mode
            
        Returns:
            Tuple of (proto_features, pooled_counts, output_logits)
        """
        # Get features from backbone network
        features = self._net(xs)  # [batch_size, channels, height, width]
        
        # Apply add-on layers (including Gumbel-Softmax)
        proto_features = self._add_on(features)  # [batch_size, num_prototypes, height, width]
        
        # Sum over spatial dimensions to count prototype occurrences
        # Each position contributes 1 if the prototype is fully active, or a fraction for partial activation
        counts = proto_features.sum(dim=(2, 3))  # [batch_size, num_prototypes]
        
        # During inference, we may want to round to nearest integer for cleaner counts
        if self._use_ste:
            # During training with STE, we round in forward pass but pass gradients through
            rounded_counts = self.ste_round(counts)
        else:
            rounded_counts = counts.round()
        
        # Clamp rounded_counts to max value (integrated into one-hot encoding)
        # The encoder automatically clamps values to [0, max_count]
        
        # Convert rounded_counts to one-hot vectors
        encoded_counts = self.onehot_encoder(rounded_counts)  # [batch_size, num_prototypes, num_bins]
        flattened_counts = encoded_counts.reshape(encoded_counts.size(0), -1)
        
        # Flatten for the classifier
        # Since our classifier now expects [batch_size, num_prototypes, max_count]
        # we can just pass the encoded_counts directly if the classifier is StructuredCountClassifier
        # Otherwise, we need to flatten
        if isinstance(self._classification, StructuredCountClassifier):
            # The StructuredCountClassifier expects the count dimension to be just max_count (not max_count+1)
            # Make sure this matches your encoder's output dimension
            out = self._classification(encoded_counts)
        else:
            # Flatten for backwards compatibility with old NonNegLinear
            out = self._classification(flattened_counts)
        
        # In inference mode, return flattened_counts for interpretability
        if inference:
            return proto_features, flattened_counts, out
        # In training, return original counts for input to L_T loss term
        return proto_features, counts, out
    
    def update_temperature(self, new_temperature):
        """
        Update the Gumbel-Softmax temperature parameter during training.
        
        Args:
            current_epoch: Current training epoch
            total_epochs: Total number of training epochs
        """
        # Find the Gumbel-Softmax layer
        for module in self._add_on.modules():
            if isinstance(module, GumbelSoftmax):
                module.tau = new_temperature
                break

# Base architectures mapping (ConvNeXt only)
base_architecture_to_features = {
    'convnext_tiny_26': convnext_tiny_26_features,
    'convnext_tiny_13': convnext_tiny_13_features
}

class NonNegLinear(nn.Module):
    """Applies a linear transformation to the incoming data with non-negative weights`
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(NonNegLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.normalization_multiplier = nn.Parameter(torch.ones((1,), requires_grad=True))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize parameters using Kaiming uniform initialization"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def initialize_count_weights(self, num_prototypes, max_count):
        """Initialize weights to prefer one count value per prototype"""
        if self.in_features != num_prototypes * max_count:
            raise ValueError(f"Expected in_features to be {num_prototypes * max_count}, got {self.in_features}")
        
        with torch.no_grad():
            # Reset weights to a low value
            self.weight.fill_(0.1)
            
            # For each prototype, randomly select one count value to have higher initial weight
            for p in range(num_prototypes):
                # Skip count=0 (which is handled differently in the modified one-hot encoding)
                best_count = random.randint(1, max_count-1)  # -1 because 0-indexed
                
                for c in range(self.out_features):  # for each class
                    # Compute the index in the flattened weights
                    idx = p * max_count + best_count
                    # Set a higher initial weight
                    self.weight[c, idx] = 1.0

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, torch.relu(self.weight), self.bias)
    
class StructuredCountClassifier(nn.Module):
    """
    A classification layer that maintains the structure of prototype-count pairs.
    Weights are organized as [num_classes, num_prototypes, max_count] to explicitly
    model the relationship between different count values of the same prototype.
    """
    def __init__(self, num_prototypes: int, num_classes: int, max_count: int, bias: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(StructuredCountClassifier, self).__init__()
        self.num_prototypes = num_prototypes
        self.num_classes = num_classes
        self.max_count = max_count
        
        # Create weight parameter with explicit prototype-count structure
        self.weight = nn.Parameter(torch.empty((num_classes, num_prototypes, max_count), **factory_kwargs))
        
        # Normalization multiplier (same as in NonNegLinear)
        self.normalization_multiplier = nn.Parameter(torch.ones((1,), requires_grad=True))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(num_classes, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        """Initialize parameters using Kaiming uniform initialization."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def initialize_count_weights(self):
        """Initialize weights to prefer one count value per prototype"""
        with torch.no_grad():
            # Reset weights to a low value
            self.weight.fill_(0.1)
            
            # For each prototype, randomly select one count value to have higher initial weight
            for p in range(self.num_prototypes):
                # Skip count=0 (which is handled differently in the modified one-hot encoding)
                best_count = random.randint(1, self.max_count-1)  # -1 because 0-indexed
                
                for c in range(self.num_classes):  # for each class
                    # Set a higher initial weight for the selected count
                    self.weight[c, p, best_count] = 1.0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the structured classification layer.
        
        Args:
            x: Input tensor of shape [batch_size, num_prototypes, max_count]
                This is the one-hot encoded count representations
                
        Returns:
            Output tensor of shape [batch_size, num_classes]
        """
        batch_size = x.shape[0]
        
        # Apply non-negative constraint to weights using ReLU
        non_neg_weight = F.relu(self.weight)
        
        # Compute class scores with broadcasting
        # [batch_size, num_prototypes, max_count] * [num_classes, num_prototypes, max_count]
        # → [batch_size, num_classes, num_prototypes, max_count]
        weighted = x.unsqueeze(1) * non_neg_weight.unsqueeze(0)
        
        # Sum over prototype and count dimensions
        # [batch_size, num_classes, num_prototypes, max_count] → [batch_size, num_classes]
        output = weighted.sum(dim=(2, 3))
        
        # Add bias if present
        if self.bias is not None:
            output = output + self.bias
            
        return output

# Cleaner channel detection for get_count_network function
def get_count_network(num_classes: int, args: argparse.Namespace, max_count: int = 3, 
                      use_ste: bool = False):
    """
    Create a CountPIPNet model with the specified parameters.
    
    Args:
        num_classes: Number of output classes
        args: Command line arguments
        max_count: Maximum count value to consider
        use_ste: Whether to use Straight-Through Estimators
        
    Returns:
        Tuple of (model, num_prototypes)
    """
    # Validate network architecture is supported
    if args.net not in base_architecture_to_features:
        supported = list(base_architecture_to_features.keys())
        raise ValueError(f"Network '{args.net}' is not supported. Supported networks: {supported}")
    
    # Get the feature extractor backbone (ConvNeXt only)
    use_mid_layers = getattr(args, 'use_mid_layers', False)
    num_stages = getattr(args, 'num_stages', 2)
    
    features = base_architecture_to_features[args.net](
        pretrained=not args.disable_pretrained,
        use_mid_layers=use_mid_layers,
        num_stages=num_stages)
    
    # Determine the number of output channels using the most reliable method
    first_add_on_layer_in_channels = detect_output_channels(features)

    # In get_count_network function
    activation = getattr(args, 'activation', 'gumbel_softmax')

    if activation == 'softmax':
        activation_layer = nn.Softmax(dim=1)
    else:  # Default to gumbel_softmax
        activation_layer = GumbelSoftmax(dim=1, tau=1.0)
    
    # Determine the number of prototypes
    if args.num_features == 0:
        num_prototypes = first_add_on_layer_in_channels
        print(f"Number of prototypes: {num_prototypes}", flush=True)
        
        # Use Gumbel-Softmax for better discretization
        add_on_layers = nn.Sequential(
            activation_layer
        )
    else:
        num_prototypes = args.num_features
        print(f"Number of prototypes set from {first_add_on_layer_in_channels} to {num_prototypes}. Extra 1x1 conv layer added.", flush=True)
        
        # Add 1x1 convolution to adjust the number of prototypes
        add_on_layers = nn.Sequential(
            nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=num_prototypes, 
                     kernel_size=1, stride=1, padding=0, bias=True),
            activation_layer
        )
    
    # The expanded dimensionality after one-hot encoding of count values
    expanded_dim = num_prototypes * max_count
    
    # Create the classification layer
    classification_layer = NonNegLinear(expanded_dim, num_classes, bias=getattr(args, 'bias', False))
    # classification_layer = StructuredCountClassifier(
    #     num_prototypes=num_prototypes,
    #     num_classes=num_classes,
    #     max_count=max_count,
    #     bias=getattr(args, 'bias', False)
    # )
    
    # Create the full CountPIPNet model
    model = CountPIPNet(
        num_classes=num_classes,
        num_prototypes=num_prototypes,
        feature_net=features,
        args=args,
        add_on_layers=add_on_layers,
        classification_layer=classification_layer,
        max_count=max_count,
        use_ste=use_ste
    )
    
    return model, num_prototypes

def detect_output_channels(features):
    """
    Detect the number of output channels from a feature extractor.
    
    Args:
        features: Feature extractor model
        
    Returns:
        Number of output channels
    """
    # For MidLayerConvNeXt, check the last stage directly
    if hasattr(features, 'features') and len(features.features) > 0:
        # Get the last module in the features Sequential
        last_stage = features.features[-1]
        
        # Find the last convolutional layer
        last_conv = None
        for module in last_stage.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        
        if last_conv is not None:
            channels = last_conv.out_channels
            print(f"Detected {channels} output channels from last conv layer", flush=True)
            return channels
    
    raise RuntimeError("Could not detect output channels from the feature extractor.")