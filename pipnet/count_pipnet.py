import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from features.convnext_features import convnext_tiny_26_features, convnext_tiny_13_features
from typing import List, Tuple, Dict, Optional, Union, Callable
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical


class GumbelSoftmax(nn.Module):
    """
    Applies the Gumbel-Softmax function to the input tensor.
    This helps produce more one-hot-like distributions compared to regular softmax.
    Uses PyTorch's built-in gumbel_softmax function.
    """
    def __init__(self, dim: int = 1, tau: float = 1.0):
        """
        Args:
            dim: Dimension along which to apply the softmax
            tau: Temperature parameter controlling discreteness (lower = more discrete)
        """
        super().__init__()
        self.dim = dim
        self.tau = tau
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Gumbel-Softmax layer.
        
        Args:
            x: Input tensor of shape [batch_size, num_features, height, width]
            
        Returns:
            Tensor after Gumbel-Softmax, same shape as input
        """
        if self.training:
            # During training use soft samples for gradient flow
            return F.gumbel_softmax(x, tau=self.tau, hard=False, dim=self.dim)
        else:
            # During inference use hard samples for discrete prototype activations
            return F.gumbel_softmax(x, tau=self.tau, hard=True, dim=self.dim)


class StraightThroughEstimator(torch.autograd.Function):
    """
    Straight-Through Estimator for non-differentiable operations.
    In forward pass, rounds values to the nearest integer.
    In backward pass, passes gradients through unmodified.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        """Round to nearest integer in forward pass"""
        return x.round()
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """Pass gradients straight through in backward pass"""
        return grad_output


def create_onehot_encoding(x: torch.Tensor, num_bins: int) -> torch.Tensor:
    """
    Helper function to create one-hot encodings from count values.
    
    Args:
        x: Input tensor of counts [batch_size, num_prototypes]
        num_bins: Number of count bins (0, 1, 2, 3+)
        
    Returns:
        One-hot encoded tensor [batch_size, num_prototypes, num_bins]
    """
    # Get shape information
    batch_size, num_prototypes = x.shape
    
    # Clamp counts to valid indices (0 to num_bins-1)
    indices = torch.clamp(x.long(), 0, num_bins-1)
    
    # Create one-hot vectors
    onehot = torch.zeros(batch_size, num_prototypes, num_bins, device=x.device)
    
    # Create batch and prototype indices for scatter operation
    batch_indices = torch.arange(batch_size, device=x.device).view(-1, 1).repeat(1, num_prototypes)
    proto_indices = torch.arange(num_prototypes, device=x.device).view(1, -1).repeat(batch_size, 1)
    
    # Fill the one-hot vectors
    # This fancy code is equivalent to the following loop:
    # for i in range(batch_size):
    #     for j in range(num_prototypes):
    #         onehot[i, j, indices[i, j]] = 1.0
    onehot[batch_indices, proto_indices, indices] = 1.0
    
    return onehot


class OneHotEncoder(nn.Module):
    """
    Converts count values to one-hot encoded vectors.
    Can operate with or without Straight-Through Estimator for backpropagation.
    """
    def __init__(self, num_bins: int = 4, use_ste: bool = False):
        """
        Args:
            num_bins: Number of count bins (0, 1, 2, 3+)
            use_ste: Whether to use Straight-Through Estimator for gradient computation
        """
        super().__init__()
        self.num_bins = num_bins
        self.use_ste = use_ste
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - converts counts to one-hot encodings.
        
        Args:
            x: Input tensor of counts [batch_size, num_prototypes]
            
        Returns:
            One-hot encoded tensor [batch_size, num_prototypes, num_bins]
        """
        if self.use_ste and self.training:
            return OneHotSTEFunction.apply(x, self.num_bins)
        else:
            return create_onehot_encoding(x, self.num_bins)


class OneHotSTEFunction(torch.autograd.Function):
    """
    Straight-Through Estimator for one-hot encoding.
    In the forward pass, performs actual one-hot encoding.
    In the backward pass, passes gradients through to the corresponding count values.
    """
    @staticmethod
    def forward(ctx, counts: torch.Tensor, num_bins: int) -> torch.Tensor:
        """
        Forward pass: Create one-hot vectors.
        
        Args:
            counts: Tensor of prototype counts [batch_size, num_prototypes]
            num_bins: Number of count bins (0, 1, 2, 3+)
            
        Returns:
            One-hot encoded tensor [batch_size, num_prototypes, num_bins]
        """
        # Save inputs for backward pass
        ctx.save_for_backward(counts)
        ctx.num_bins = num_bins
        
        # Use the shared function to create one-hot encodings
        return create_onehot_encoding(counts, num_bins)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """
        Backward pass: Pass gradients to the specific activated bin.
        
        Args:
            grad_output: Gradient from subsequent layers [batch_size, num_prototypes, num_bins]
            
        Returns:
            Gradient for counts, None for num_bins parameter
        """
        counts, = ctx.saved_tensors
        
        # Clamp counts to valid indices
        indices = torch.clamp(counts.long(), 0, ctx.num_bins-1)
        
        # Get shape information
        batch_size, num_prototypes = counts.shape
        
        # Get batch and prototype indices
        batch_indices = torch.arange(batch_size, device=counts.device).view(-1, 1).repeat(1, num_prototypes)
        proto_indices = torch.arange(num_prototypes, device=counts.device).view(1, -1).repeat(batch_size, 1)
        
        # Pass gradients through to the counts
        counts_grad = grad_output[batch_indices, proto_indices, indices]
        
        return counts_grad, None


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
                 freeze_mode: str = 'none',
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
            freeze_mode: Freezing strategy - 'none': train all parameters, 
                                            'backbone': freeze only backbone,
                                            'all_except_classification': freeze everything except classification layer
            use_ste: Whether to use Straight-Through Estimators for non-differentiable operations
        """
        super().__init__()
        assert num_classes > 0
        assert freeze_mode in ['none', 'backbone', 'all_except_classification'], \
            "freeze_mode must be one of ['none', 'backbone', 'all_except_classification']"
            
        self._num_features = args.num_features
        self._num_classes = num_classes
        self._num_prototypes = num_prototypes
        self._net = feature_net
        self._add_on = add_on_layers
        self._classification = classification_layer
        self._max_count = max_count
        self._num_bins = max_count + 1  # +1 for count=0
        self._use_ste = use_ste
        self._freeze_mode = freeze_mode
        
        # Apply freezing strategy
        self._apply_freezing()

        # STE function for rounding (only used if use_ste=True)
        self.ste_round = StraightThroughEstimator.apply
        
        # Create unified one-hot encoder (handles both normal and STE modes)
        self.onehot_encoder = OneHotEncoder(self._num_bins, use_ste=use_ste)
    
    def _apply_freezing(self):
        """Apply the selected freezing strategy to model parameters."""
        if self._freeze_mode == 'none':
            # Train all parameters
            return
                
        elif self._freeze_mode == 'backbone':
            # Freeze only backbone
            for name, param in self.named_parameters():
                if name.startswith('_net'):
                    param.requires_grad = False
                    
        elif self._freeze_mode == 'all_except_classification':
            # Freeze all except classification parameters
            for name, param in self.named_parameters():
                if not name.startswith('_classification'):
                    param.requires_grad = False

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
        if inference:
            counts = counts.round()
        elif self._use_ste:
            # During training with STE, we round in forward pass but pass gradients through
            counts = self.ste_round(counts)
        
        # Clamp counts to max value (integrated into one-hot encoding)
        # The encoder automatically clamps values to [0, max_count]
        
        # Convert counts to one-hot vectors
        encoded_counts = self.onehot_encoder(counts)  # [batch_size, num_prototypes, num_bins]
        
        # Flatten the prototype-count combinations
        # This converts from [batch_size, num_prototypes, num_bins] to [batch_size, num_prototypes * num_bins]
        flattened_counts = encoded_counts.reshape(encoded_counts.size(0), -1)
        
        # Apply classification layer
        out = self._classification(flattened_counts)  # [batch_size, num_classes]
        
        # Return intermediate representations for interpretability
        return proto_features, flattened_counts, out
    
    def update_temperature(self, current_epoch, total_epochs):
        """
        Update the Gumbel-Softmax temperature parameter during training.
        
        Args:
            current_epoch: Current training epoch
            total_epochs: Total number of training epochs
        """
        # Find the Gumbel-Softmax layer
        for module in self._add_on.modules():
            if isinstance(module, GumbelSoftmax):
                # Anneal from 1.0 to 0.1
                module.tau = max(0.1, 1.0 - 0.9 * (current_epoch / total_epochs))
                print(f"Updated Gumbel-Softmax temperature to {module.tau:.3f}", flush=True)
                break


# Base architectures mapping (ConvNeXt only)
base_architecture_to_features = {
    'convnext_tiny_26': convnext_tiny_26_features,
    'convnext_tiny_13': convnext_tiny_13_features
}

class NonNegLinear(nn.Module):
    """
    Linear layer with non-negative weights.
    Ensures that prototype presence can only add positive evidence for a class.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        """
        Initialize non-negative linear layer.
        
        Args:
            in_features: Size of input features
            out_features: Size of output features
            bias: Whether to include bias parameters
            device: Device to place tensor on
            dtype: Data type of tensor
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(NonNegLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the non-negative linear layer.
        Applies ReLU to weights to ensure they are non-negative.
        
        Args:
            input: Input tensor [batch_size, in_features]
            
        Returns:
            Output tensor [batch_size, out_features]
        """
        return F.linear(input, torch.relu(self.weight), self.bias)


# Cleaner channel detection for get_count_network function
def get_count_network(num_classes: int, args: argparse.Namespace, max_count: int = 3, 
                     freeze_mode: str = 'none', use_ste: bool = False):
    """
    Create a CountPIPNet model with the specified parameters.
    
    Args:
        num_classes: Number of output classes
        args: Command line arguments
        max_count: Maximum count value to consider
        freeze_mode: Freezing strategy - 'none': train all parameters, 
                                        'backbone': freeze only backbone,
                                        'all_except_classification': freeze everything except classification layer
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
    
    # Determine the number of prototypes
    if args.num_features == 0:
        num_prototypes = first_add_on_layer_in_channels
        print(f"Number of prototypes: {num_prototypes}", flush=True)
        
        # Use Gumbel-Softmax for better discretization
        add_on_layers = nn.Sequential(
            GumbelSoftmax(dim=1, tau=1.0)
        )
    else:
        num_prototypes = args.num_features
        print(f"Number of prototypes set from {first_add_on_layer_in_channels} to {num_prototypes}. Extra 1x1 conv layer added.", flush=True)
        
        # Add 1x1 convolution to adjust the number of prototypes
        add_on_layers = nn.Sequential(
            nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=num_prototypes, 
                     kernel_size=1, stride=1, padding=0, bias=True),
            GumbelSoftmax(dim=1, tau=1.0)
        )
    
    # The expanded dimensionality after one-hot encoding of count values
    expanded_dim = num_prototypes * (max_count + 1)  # +1 for count=0
    
    # Create the classification layer
    classification_layer = NonNegLinear(expanded_dim, num_classes, bias=getattr(args, 'bias', False))
    
    # Create the full CountPIPNet model
    model = CountPIPNet(
        num_classes=num_classes,
        num_prototypes=num_prototypes,
        feature_net=features,
        args=args,
        add_on_layers=add_on_layers,
        classification_layer=classification_layer,
        max_count=max_count,
        freeze_mode=freeze_mode,
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