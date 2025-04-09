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
    def __init__(self, num_bins: int = 4, use_ste: bool = False, respect_active_grad: bool = False,
                 num_prototypes: int = None, device: Optional[torch.device] = None):
        """
        Args:
            num_bins: Number of count bins (0, 1, 2, 3+)
            use_ste: Whether to use Straight-Through Estimator for gradient computation
            respect_active_grad: Whether to respect negative gradients at current position
        """
        super().__init__()
        self.num_bins = num_bins
        self.num_prototypes = num_prototypes
        self.device = device
        self.use_ste = use_ste
        self.respect_active_grad = respect_active_grad
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - converts counts to modified encodings where 0 count → all zeros.
        
        Args:
            x: Input tensor of counts [batch_size, num_prototypes]
            
        Returns:
            Encoded tensor [batch_size, num_prototypes * num_bins]
        """
        if self.use_ste:
            encodings = ModifiedSTEFunction.apply(x, self.num_bins, self.respect_active_grad)
        else:
            encodings = create_modified_encoding(x, self.num_bins)

        encodings_flattened = encodings.view(encodings.size(0), -1)
        return encodings_flattened

    def prototype_to_classifier_input_weights(self, prototype_idx):
        """
        Returns a vector of length (num_prototypes * num_bins) where the contiguous segment corresponding
        to the given prototype (i.e. indices [prototype_idx * num_bins, (prototype_idx+1) * num_bins))
        is filled with ones, and all other entries are zeros.
        """
        total_length = self.num_prototypes * self.num_bins
        relevance_vector = torch.zeros(total_length, device=self.device)

        start_idx = prototype_idx * self.num_bins
        end_idx = start_idx + self.num_bins

        relevance_vector[start_idx:end_idx] = 1.0
        return relevance_vector

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
    Backward pass implements the "follow the max gradient" principle.
    If max gradient is at the current position, the resulting grad is zero.
    """
    @staticmethod
    def forward(ctx, counts: torch.Tensor, max_count: int, respect_active_grad: bool) -> torch.Tensor:
        """
        Forward pass: Create modified encodings based on rounded counts.
        The output tensor is NOT flattened here; flattening happens in OneHotEncoder.

        Args:
            counts: Input count values [batch_size, num_prototypes].
            max_count: Number of bins (dimension of the one-hot encoding).
            respect_active_grad: Flag for backward pass behavior.
        """
        # Round counts for encoding determination
        rounded_counts = counts.round()
        # Save original counts and rounded counts for backward pass
        ctx.save_for_backward(counts, rounded_counts)
        ctx.max_count = max_count
        ctx.respect_active_grad = respect_active_grad
        # No need to store dims, shape is known from saved tensors

        # Use the shared function to create encodings based on rounded values
        # Return the 3D tensor here. Flattening is done *after* this function in OneHotEncoder.
        return create_modified_encoding(rounded_counts, max_count)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        """
        Backward pass implementing "follow the minimum gradient" principle with
        optional stability logic based on negative gradient at current position.

        Args:
            ctx: Context object from forward pass.
            grad_output: Gradient shape [batch_size, num_prototypes, max_count].

        Returns:
            Gradient w.r.t original counts, None for max_count, None for respect_active_grad.
        """
        counts, rounded_counts = ctx.saved_tensors
        max_count = ctx.max_count
        # Retrieve the renamed flag
        respect_active_grad = ctx.respect_active_grad
        batch_size, num_prototypes = counts.shape

        # --- grad_output arrives with shape [batch_size, num_prototypes, max_count] ---
        if grad_output.shape[0] != batch_size or grad_output.shape[1] != num_prototypes or grad_output.shape[2] != max_count:
             raise ValueError(f"Unexpected grad_output shape. Expected [{batch_size}, {num_prototypes}, {max_count}], got {grad_output.shape}")

        # Initialize gradient tensor for the original counts
        counts_grad = torch.zeros_like(counts)

        # --- Calculate current activation position index ---
        current_pos_idx = torch.clamp(rounded_counts.long() - 1, 0, max_count - 1)

        # --- Identify zero vs non-zero counts based on rounded counts ---
        zero_threshold = 0.1
        zero_mask = rounded_counts < zero_threshold
        non_zero_mask = ~zero_mask

        # --- Find the index and value of the MINIMUM *signed* gradient ---
        min_signed_grad_val, min_signed_grad_idx = torch.min(grad_output, dim=2)

        # --- Process Non-Zero Count Cases ---
        if torch.any(non_zero_mask):
            # Get relevant values for non-zero elements
            current_pos_idx_nz = current_pos_idx[non_zero_mask]
            min_signed_grad_val_nz = min_signed_grad_val[non_zero_mask]
            min_signed_grad_idx_nz = min_signed_grad_idx[non_zero_mask]

            # --- Apply "follow the MINIMUM gradient" logic ---
            final_grad_nz = torch.zeros_like(min_signed_grad_val_nz)
            gradient_magnitude = torch.abs(min_signed_grad_val_nz)

            # Case 1: Minimum gradient index is lower than current (decrease counts)
            decrease_mask = min_signed_grad_idx_nz < current_pos_idx_nz
            final_grad_nz[decrease_mask] = gradient_magnitude[decrease_mask] # Positive grad -> decrease counts

            # Case 2: Minimum gradient index is higher than current (increase counts)
            increase_mask = min_signed_grad_idx_nz > current_pos_idx_nz
            final_grad_nz[increase_mask] = -gradient_magnitude[increase_mask] # Negative grad -> increase counts

            # Case 3: Minimum gradient index is AT the current position (stable)
            # Gradient remains zero.

            # --- Apply Optional Stability Logic (Reversed Condition) ---
            # Zero out the gradient if the current position's gradient is NEGATIVE
            # (meaning the current state is already desirable according to the loss)
            if respect_active_grad:
                grad_at_current_pos = torch.gather(
                    grad_output[non_zero_mask], 1, current_pos_idx_nz.unsqueeze(1)
                ).squeeze(1)
                # Find where the gradient at the current position is negative
                negative_current_grad_mask = grad_at_current_pos < 0
                # Zero out the calculated gradient (final_grad_nz) where the current gradient was negative.
                # This stabilizes the count if the current state is already good.
                final_grad_nz[negative_current_grad_mask] = 0.0

            # Assign the calculated gradients back
            counts_grad[non_zero_mask] = final_grad_nz

        # --- Process Zero Count Cases ---
        # If count is zero, we only activate count '1' if grad at index 0 is negative.
        if torch.any(zero_mask):
            grad_for_count_1 = grad_output[:, :, 0][zero_mask]
            negative_grad_mask = grad_for_count_1 < 0
            # Pass the negative gradient back to increase counts from 0 towards 1.
            counts_grad[zero_mask][negative_grad_mask] = grad_for_count_1[negative_grad_mask]

        # Return gradient w.r.t counts, and None for max_count and the flag
        return counts_grad, None, None

class BilinearIntermediate(nn.Module):
    """
    A bilinear intermediate layer that applies bilinear transformation 
    after embedding prototype counts into a higher-dimensional space.
    """
    def __init__(self, num_prototypes, max_count, expanded_dim=None,
                 custom_init=False):
        """
        Args:
            num_prototypes: Number of prototypes in the model
            max_count: Maximum count value to consider
            expanded_dim: Size of the expanded feature space (defaults to num_prototypes * max_count)
        """
        super().__init__()
        self.num_prototypes = num_prototypes
        self.max_count = max_count
        self.expanded_dim = num_prototypes * max_count if expanded_dim is None else expanded_dim
        
        # Embedding layer to map from prototype counts to expanded dimension
        self.embed = nn.Linear(num_prototypes, self.expanded_dim, bias=False)
        
        # Two projection matrices for the bilinear transformation
        self.W = nn.Linear(self.expanded_dim, self.expanded_dim, bias=False)
        self.V = nn.Linear(self.expanded_dim, self.expanded_dim, bias=False)
        
        # Initialize the embedding to create a meaningful mapping
        with torch.no_grad():
            # First, zero out all weights
            self.embed.weight.zero_()
            
            # Then initialize the embedding so each prototype affects max_count consecutive dimensions
            for p in range(num_prototypes):
                for c in range(max_count):
                    idx = p * max_count + c
                    # Each output dimension corresponds to a specific count of a specific prototype
                    self.embed.weight[idx, p] = c + 1  # Scale by count value

        if custom_init:
            # Initialize W and V matrices for stable bilinear interaction
            nn.init.normal_(self.W.weight, mean=0.0, std=0.1)
            nn.init.normal_(self.V.weight, mean=0.0, std=0.1)
            
            # Make W and V slightly tend toward identity matrix to start
            # This encourages preserving the semantic structure from embedding
            for i in range(self.expanded_dim):
                self.W.weight[i, i] += 0.1
                self.V.weight[i, i] += 0.1
    
    def forward(self, x):
        """
        Forward pass - maps count values to expanded feature space and applies bilinear transformation.
        
        Args:
            x: Input tensor of counts [batch_size, num_prototypes]
            
        Returns:
            Expanded tensor [batch_size, expanded_dim]
        """
        # Map to expanded dimension
        embedded = self.embed(x)
        
        # Apply bilinear transformation
        return self.W(embedded) * self.V(embedded)

class LinearFull(nn.Module):
    """
    A full linear intermediate layer that maps prototype counts to an expanded dimension
    using the full parameter space (num_prototypes * max_count parameters).
    Unlike LinearIntermediate which only uses max_count parameters per prototype,
    this version allows for more complex interactions between different prototypes.
    """
    def __init__(self, num_prototypes, max_count, expanded_dim=None):
        """
        Args:
            num_prototypes: Number of prototypes in the model
            max_count: Maximum count value to consider
            expanded_dim: Size of the expanded feature space (defaults to num_prototypes * max_count)
        """
        super().__init__()
        self.num_prototypes = num_prototypes
        self.max_count = max_count
        self.expanded_dim = num_prototypes * max_count if expanded_dim is None else expanded_dim
        
        # Full linear projection from prototype counts to expanded dimension
        self.linear = nn.Linear(num_prototypes, self.expanded_dim, bias=False)
        
        # Initialize with a structured pattern that's more interpretable
        with torch.no_grad():
            # First zero out weights
            self.linear.weight.zero_()
            
            # Then initialize with a block-diagonal-like pattern
            # but with full connectivity to allow learning more complex relationships
            for p in range(num_prototypes):
                for c in range(max_count):
                    idx = p * max_count + c
                    # Primary connection - stronger weight to corresponding prototype
                    self.linear.weight[idx, p] = c + 1  # Scale by count value
                    
                    # Secondary connections - weaker weights to other prototypes
                    # This allows the model to learn interactions while still maintaining
                    # some interpretable structure
                    for other_p in range(num_prototypes):
                        if other_p != p:
                            self.linear.weight[idx, other_p] = 0.1 * (c + 1) / num_prototypes
    
    def forward(self, x):
        """
        Forward pass - maps count values to expanded feature space.
        
        Args:
            x: Input tensor of counts [batch_size, num_prototypes]
            
        Returns:
            Expanded tensor [batch_size, expanded_dim]
        """
        return self.linear(x)

    def prototype_to_classifier_input_weights(self, prototype_idx):
        # Return the full weight vector corresponding to the given prototype index.
        # self.linear.weight has shape [expanded_dim, num_prototypes]
        return self.linear.weight[:, prototype_idx]

class IdentityIntermediate(nn.Module):
    def __init__(self, num_prototypes, device):
        """
        A wrapper around nn.Identity that implements the uniform interface for intermediate layers.
        For the identity mapping, each prototype maps to a unique output position, so the relevance vector
        is one-hot encoded.
        
        Args:
            num_prototypes: The number of prototypes (i.e., the dimension of the identity mapping)
        """
        super().__init__()
        self.identity = nn.Identity()
        self.num_prototypes = num_prototypes
        self.device = device

    def forward(self, x):
        return self.identity(x)

    def prototype_to_classifier_input_weights(self, prototype_idx):
        """
        Returns a one-hot encoded vector of length num_prototypes, where the position corresponding to 
        prototype_idx is 1 and all others are 0.
        """
        return torch.eye(self.num_prototypes, device=self.device)[prototype_idx]
    
class LinearIntermediate(nn.Module):
    """
    A simple linear intermediate layer that maps count values to a higher-dimensional space.
    This avoids the discretization and potential gradient issues of one-hot encoding.
    """
    def __init__(self, num_prototypes, max_count, expansion_factor=None):
        """
        Args:
            num_prototypes: Number of prototypes in the model
            max_count: Maximum count value to consider
            expansion_factor: Factor to expand the feature dimension by (defaults to max_count)
        """
        super().__init__()
        self.num_prototypes = num_prototypes
        self.max_count = max_count
        self.expansion_factor = max_count if expansion_factor is None else expansion_factor
        
        # Create a simple linear layer that maps each prototype's count to expanded dimensions
        self.linear = nn.Linear(1, self.expansion_factor, bias=False)
        
        # Initialize to approximate one-hot behavior for easier comparison
        # Higher counts activate later dimensions
        with torch.no_grad():
            for i in range(self.expansion_factor):
                # Create a ramp that peaks at count i+1
                self.linear.weight[i, 0] = (i + 1) / self.max_count
        
    def forward(self, x):
        """
        Forward pass - maps count values to expanded feature space.
        
        Args:
            x: Input tensor of counts [batch_size, num_prototypes]
            
        Returns:
            Expanded tensor [batch_size, num_prototypes * expansion_factor]
        """
        batch_size = x.shape[0]
        
        # Reshape for linear layer processing
        x_reshaped = x.view(batch_size * self.num_prototypes, 1)
        
        # Apply linear transformation
        expanded = self.linear(x_reshaped)
        
        # Reshape back to batch format
        result = expanded.view(batch_size, self.num_prototypes * self.expansion_factor)
        
        return result

    def prototype_to_classifier_input_weights(self, prototype_idx):
        # Total length of the flattened output vector
        total_length = self.num_prototypes * self.expansion_factor

        # Create a zero tensor of the appropriate size
        sparse_vector = torch.zeros(total_length, device=self.linear.weight.device, dtype=self.linear.weight.dtype)
        
        # Get the non-zero weight values for a single prototype
        # Note: self.linear.weight has shape [expansion_factor, 1]
        prototype_weights = self.linear.weight[:, 0]
        
        # Compute the block indices corresponding to the given prototype
        start_idx = prototype_idx * self.expansion_factor
        end_idx = start_idx + self.expansion_factor
        
        # Place the prototype's weights into their block positions
        sparse_vector[start_idx:end_idx] = prototype_weights
        
        return sparse_vector