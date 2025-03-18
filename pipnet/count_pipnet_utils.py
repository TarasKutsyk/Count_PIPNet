import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional, Union, Callable


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


class STE_Round(torch.autograd.Function):
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