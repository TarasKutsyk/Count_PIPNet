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


class OneHotEncoder(nn.Module):
    """
    Converts count values to modified encodings where count 0 maps to all zeros.
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
        Forward pass - converts counts to modified encodings where 0 count → all zeros.
        
        Args:
            x: Input tensor of counts [batch_size, num_prototypes]
            
        Returns:
            Encoded tensor [batch_size, num_prototypes, num_bins]
        """
        if self.use_ste:
            return ModifiedSTEFunction.apply(x, self.num_bins)
        else:
            return create_modified_encoding(x, self.num_bins)


def create_modified_encoding(x: torch.Tensor, max_count: int) -> torch.Tensor:
    """
    Helper function to create modified count encodings where:
    - count 0 → (0, 0, 0) (for max_count = 3)
    - count 1 → (1, 0, 0)
    - count 2 → (0, 1, 0)
    - count 3 → (0, 0, 1)
    
    Args:
        x: Input tensor of counts [batch_size, num_prototypes]
        max_count: Maximum count value
        
    Returns:
        Encoded tensor [batch_size, num_prototypes, max_count]
    """

    # Get shape information
    batch_size, num_prototypes = x.shape
    
    # Create zero-filled tensor
    encoded = torch.zeros(batch_size, num_prototypes, max_count, device=x.device)

    # Create mask for non-zero counts
    non_zero_mask = x > 0.1 # 0.1 threshold should work because x is obtained via rounding
    
    # Skip processing if no non-zero counts
    if not torch.any(non_zero_mask):
        return encoded
    
    # Create indices for scatter operation
    batch_indices = torch.arange(batch_size, device=x.device).view(-1, 1).repeat(1, num_prototypes)
    proto_indices = torch.arange(num_prototypes, device=x.device).view(1, -1).repeat(batch_size, 1)
    
    # Get only indices for non-zero counts
    batch_idx = batch_indices[non_zero_mask]
    proto_idx = proto_indices[non_zero_mask]
    
    # Get count values for non-zero counts (adjusted for index)
    # count 1 → index 0, count 2 → index 1, etc.
    count_idx = torch.clamp(x[non_zero_mask].long() - 1, 0, max_count - 1)
    
    # Set the appropriate positions to 1.0
    encoded[batch_idx, proto_idx, count_idx] = 1.0
    
    return encoded


class ModifiedSTEFunction(torch.autograd.Function):
    """
    Straight-Through Estimator for modified count encoding.
    """
    @staticmethod
    def forward(ctx, counts: torch.Tensor, max_count: int) -> torch.Tensor:
        """
        Forward pass: Create modified encodings.
        """
        # Save inputs for backward pass
        ctx.save_for_backward(counts)
        ctx.max_count = max_count
        
        # Use the shared function to create encodings
        return create_modified_encoding(counts, max_count)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """
        Backward pass focusing on activation preferences.
        """
        counts, = ctx.saved_tensors
        max_count = ctx.max_count
        batch_size, num_prototypes = counts.shape
        
        # Initialize gradient tensor
        counts_grad = torch.zeros_like(counts)
        
        # Use threshold for numerical stability
        zero_threshold = 0.1
        zero_mask = counts < zero_threshold
        non_zero_mask = ~zero_mask
        
        # Part 1: Handle zero counts
        if torch.any(zero_mask):
            batch_indices = torch.arange(batch_size, device=counts.device).view(-1, 1).repeat(1, num_prototypes)
            proto_indices = torch.arange(num_prototypes, device=counts.device).view(1, -1).repeat(batch_size, 1)
            
            batch_idx_zeros = batch_indices[zero_mask]
            proto_idx_zeros = proto_indices[zero_mask]
            
            # Get gradient for position 0 (count=1)
            # For zeros, we only care if increasing to 1 would be beneficial
            pos0_grad = grad_output[batch_idx_zeros, proto_idx_zeros, 0]
            
            # If position 0 has positive gradient, increase count
            counts_grad[zero_mask] = pos0_grad
        
        # Part 2: Handle non-zero counts
        if torch.any(non_zero_mask):
            batch_indices = torch.arange(batch_size, device=counts.device).view(-1, 1).repeat(1, num_prototypes)
            proto_indices = torch.arange(num_prototypes, device=counts.device).view(1, -1).repeat(batch_size, 1)
            
            batch_idx_nonzeros = batch_indices[non_zero_mask]
            proto_idx_nonzeros = proto_indices[non_zero_mask]
            
            # Current counts (adjusted for 0-indexing)
            current_counts = torch.clamp(counts[non_zero_mask].long() - 1, 0, max_count - 1)
            
            # Extract current position gradients
            current_pos_grad = grad_output[batch_idx_nonzeros, proto_idx_nonzeros, current_counts]
            
            # Initialize with current gradient (preserving current activation if positive)
            final_gradient = current_pos_grad.clone()
            
            # Now consider neighboring positions' preferences
            
            # # 1. If count+1 position has positive gradient, we want to increase
            # # (but only if not already at max)
            # next_counts = torch.clamp(current_counts + 1, 0, max_count - 1)
            # can_increase_mask = next_counts != current_counts
            
            # if torch.any(can_increase_mask):
            #     batch_sub = batch_idx_nonzeros[can_increase_mask]
            #     proto_sub = proto_idx_nonzeros[can_increase_mask]
            #     next_c = next_counts[can_increase_mask]
                
            #     # Get gradient at next position
            #     next_pos_grad = grad_output[batch_sub, proto_sub, next_c]
                
            #     # Only consider positive preferences for activation
            #     increase_preference = torch.clamp(next_pos_grad, min=0.0)
                
            #     # Add to final gradient for affected positions
            #     idx_in_nonzero = torch.where(can_increase_mask)[0]
            #     final_gradient[idx_in_nonzero] += increase_preference
            
            # # 2. If count-1 position has positive gradient, we want to decrease
            # # (but only if not already at min)
            # prev_counts = torch.clamp(current_counts - 1, 0, max_count - 1)
            # can_decrease_mask = prev_counts != current_counts
            
            # if torch.any(can_decrease_mask):
            #     batch_sub = batch_idx_nonzeros[can_decrease_mask]
            #     proto_sub = proto_idx_nonzeros[can_decrease_mask]
            #     prev_c = prev_counts[can_decrease_mask]
                
            #     # Get gradient at previous position
            #     prev_pos_grad = grad_output[batch_sub, proto_sub, prev_c]
                
            #     # Only consider positive preferences for activation
            #     decrease_preference = torch.clamp(prev_pos_grad, min=0.0)
                
            #     # Subtract from final gradient for affected positions
            #     idx_in_nonzero = torch.where(can_decrease_mask)[0]
            #     final_gradient[idx_in_nonzero] -= decrease_preference
            
            # Assign computed gradients for non-zero counts
            counts_grad[non_zero_mask] = final_gradient
        
        return counts_grad, None