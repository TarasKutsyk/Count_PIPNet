import argparse
import math
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from features.convnext_features import convnext_tiny_26_features, convnext_tiny_13_features
from typing import List, Tuple, Dict, Optional, Union, Callable
from tqdm import tqdm
import matplotlib.pyplot as plt

from .count_pipnet_utils import *

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
                 intermediate_layer: nn.Module, 
                 classification_layer: nn.Module,
                 max_count: int = 3,
                 use_ste: bool = True):
        """
        Initialize the CountPIPNet model.
        
        Args:
            num_classes: Number of output classes
            num_prototypes: Number of prototypical parts
            feature_net: Backbone network for feature extraction
            args: Command line arguments
            add_on_layers: Layers applied after feature extraction
            intermediate_layer: Layer that maps counts to classification input
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
        self._intermediate = intermediate_layer

        self._max_count = max_count
        self._use_ste = use_ste
        self._multiplier = classification_layer.normalization_multiplier
        
        # STE function for rounding (only used if use_ste=True)
        self.ste_round = STE_Round.apply
        
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
        counts = proto_features.sum(dim=(2, 3))  # [batch_size, num_prototypes]
        
        if self._use_ste:
            # During training with STE, we round in forward pass but pass gradients through
            rounded_counts = self.ste_round(counts)
        else:
            rounded_counts = counts.round() if inference else counts
        
        # Clamp to max_count
        clamped_counts = torch.clamp(rounded_counts, 0, self._max_count)
            
        # Process through intermediate layer
        intermediate_features = self._intermediate(clamped_counts)
        # E.g., if intermediate layer is OneHotEncoder, this will be [batch_size, num_prototypes * max_count]
        
        # Apply classification layer
        out = self._classification(intermediate_features)  # [batch_size, num_classes]
        
        # In inference mode, return clamped_counts for interpretability
        if inference:
            return proto_features, clamped_counts, out
        # In training, return original counts for input to L_T loss term
        return proto_features, counts, out
    
    def _calculate_counts_for_testing(self, proto_features):
        """
        Helper method for testing that isolates the count calculation logic.
        
        Args:
            proto_features: Tensor of prototype feature maps [batch_size, num_prototypes, height, width]
            
        Returns:
            Tensor of counts [batch_size, num_prototypes]
        """
        # Sum over spatial dimensions to count prototype occurrences
        counts = proto_features.sum(dim=(2, 3))
        return counts
    
    def get_prototype_importance_per_class(self, prototype_idx, classifier_input_scalars = None):
        intermediate_layer = self._intermediate

        # Get the mapping from the given prototype to its "influence" on the classifiers' input X
        # this will return the weights of the same shape as X: typically [num_prototypes * max_count]
        classifier_input_weights = intermediate_layer.prototype_to_classifier_input_weights(prototype_idx) 

        if classifier_input_scalars is not None:
            assert classifier_input_scalars.shape == classifier_input_weights.shape, "Classifier input scalars must have the same shape as the classifier input weights" + \
                  "\nclassifier_input_weights.shape: " + str(classifier_input_weights.shape) + \
                  "\nclassifier_input_scalars.shape: " + str(classifier_input_scalars.shape)
            
            classifier_input_weights = classifier_input_weights * classifier_input_scalars

        # Get the absolute weights to avoid cancellation effects
        classifier_input_weights = torch.abs(classifier_input_weights)

        # Now compute the per-class importance for a given prototype by taking a dot product
        # between the classifier_input_weights and the classifier weights themselves for each class
        prototype_importance_per_class = einops.einsum(classifier_input_weights, self._classification.weight,
                                                       "input_dim, n_classes input_dim -> n_classes")
        return prototype_importance_per_class
    

    def get_prototype_importance(self, prototype_idx):
        prototype_importance_per_class = self.get_prototype_importance_per_class(prototype_idx)
        
        # The total importance is the sum of all per-class importances
        return prototype_importance_per_class.sum().item()
    
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
        self.normalization_multiplier = nn.Parameter(torch.ones((1,),requires_grad=True))
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

def calculate_virtual_weights(net, dataloader, device, custom_onehot_scale=False):
    def plot_mean_intermediate_features(mean_intermediate_features, num_prototypes, max_count):
        assert mean_intermediate_features.shape[0] == num_prototypes * max_count, \
            f"Expected shape ({num_prototypes * max_count},), but got {mean_intermediate_features.shape}"

        # Reshape to a grid [num_prototypes x max_count]
        feature_grid = mean_intermediate_features.view(num_prototypes, max_count).cpu().numpy()

        plt.figure(figsize=(max(6, max_count * 0.5), max(4, num_prototypes * 0.4)))
        im = plt.imshow(feature_grid, aspect='auto', cmap='coolwarm')
        plt.colorbar(im, label='Mean Feature Value')
        plt.xlabel('Count index')
        plt.ylabel('Prototype index')
        plt.title('Mean Intermediate Features Grid')
        plt.tight_layout()
        plt.show()

    is_intermediate_onehot = isinstance(net.module._intermediate, OneHotEncoder)

    if is_intermediate_onehot and custom_onehot_scale:
        print("Intermediate is onehot, computing mean intermediate features...")

        # Estimate mean intermediate features from clamped counts across the entire dataset
        pbar_collect = tqdm(dataloader, desc="Estimating mean intermediate features", ncols=100, leave=False)
        all_clamped_counts = []

        with torch.no_grad():
            for xs, ys in pbar_collect:
                xs = xs.to(device)

                # Perform the forward pass to get activations
                try:
                    _, clamped_counts, _ = net(xs, inference=True) # inference=True to get clamped counts
                except Exception as e:
                    print(f"\nError during forward pass: {e}. Skipping batch.")
                    continue

                # Store the results from the current batch
                all_clamped_counts.append(clamped_counts)
        
        # Cleanly close the progress bar
        pbar_collect.close()

        # Concatenate all clamped counts over batches
        all_clamped_counts = torch.cat(all_clamped_counts, dim=0)
        print("all_clamped_counts.shape: ", all_clamped_counts.shape)

        # Compute intermediate features
        intermediate_features = net.module._intermediate(all_clamped_counts)
        print("intermediate_features.shape: ", intermediate_features.shape)
        # Average over batches
        mean_intermediate_features = intermediate_features.mean(dim=0)

        plot_mean_intermediate_features(mean_intermediate_features, net.module._num_prototypes, net.module._max_count)

        classifier_input_scalars = mean_intermediate_features
    else:
        classifier_input_scalars = None

    # Construct the virtual classification matrix
    classification_weights = torch.zeros((net.module._num_classes, net.module._num_prototypes), device=device)

    for i in range(net.module._num_prototypes):
        # Get importance of this prototype for each class
        prototype_importance_per_class = net.module.get_prototype_importance_per_class(i, classifier_input_scalars)
        classification_weights[:, i] = prototype_importance_per_class.to(device)

    # To compute the importance, multiply the mean intermediate features by the classification weights
    return classification_weights

# Cleaner channel detection for get_count_network function
def get_count_network(num_classes: int, args: argparse.Namespace, max_count: int = 3, 
                      use_ste: bool = True, device=None):
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
    
    # Determine the number of output channels
    first_add_on_layer_in_channels = detect_output_channels(features)

    # Get activation type
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
    
    # Get intermediate layer type
    intermediate_type = getattr(args, 'intermediate_layer', 'onehot')
    # Get positive gradient strategy
    positive_grad_strategy = getattr(args, 'positive_grad_strategy', None)
    print(f"Using positive gradient strategy: {positive_grad_strategy}", flush=True)
    
    # Create the appropriate intermediate layer
    if intermediate_type == 'linear':
        # Use linear intermediate layer
        intermediate_layer = LinearIntermediate(num_prototypes, max_count)
        expanded_dim = num_prototypes * max_count
    elif intermediate_type == 'linear_full':
        # Use full linear intermediate layer
        intermediate_layer = LinearFull(num_prototypes, max_count)
        expanded_dim = num_prototypes * max_count
    # Create the appropriate intermediate layer
    elif intermediate_type == 'bilinear':
        # Use bilinear intermediate layer
        intermediate_layer = BilinearIntermediate(num_prototypes, max_count)
        expanded_dim = num_prototypes * max_count
    elif intermediate_type == 'onehot':
        # Use one-hot encoder (original approach)
        intermediate_layer = OneHotEncoder(max_count, use_ste=use_ste, respect_active_grad=False, 
                                           num_prototypes=num_prototypes, device=device,
                                           positive_grad_strategy=positive_grad_strategy)
        expanded_dim = num_prototypes * max_count
    elif intermediate_type == 'identity':
        # Identity intermediate layer
        intermediate_layer = IdentityIntermediate(num_prototypes, device=device)
        expanded_dim = num_prototypes
    else:
        raise ValueError(f"Unknown intermediate layer type: {intermediate_type}")
    
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
        intermediate_layer=intermediate_layer,
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