import torch
import unittest
from torch.autograd import gradcheck
import sys
import os

# Add project root directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Go up one level to PIPNet root
sys.path.append(project_root)

from pipnet.count_pipnet_utils import OneHotEncoder


class TestModifiedOneHotEncoder(unittest.TestCase):
    def setUp(self):
        # Set up the encoder with max_count=3
        self.encoder = OneHotEncoder(num_bins=3, use_ste=True)
        
        # Define a mock count tensor for testing
        self.counts = torch.tensor([
            [0.0, 1.0, 3.0],  # Row 0: zero, one, max
            [0.05, 2.0, 2.9], # Row 1: near-zero, two, near-max
            [1.0, 0.0, 0.2],  # Row 2: one, zero, near-zero
            [3.0, 2.0, 1.0]   # Row 3: max, two, one
        ], requires_grad=True)
        
        # Define fictional gradients for the encoded representation
        # Shape: [batch_size=4, num_prototypes=3, num_bins=3]
        self.grad_output = torch.zeros(4, 3, 3)
        
        # Gradients for position 0 (count=1): Positive (encourage count=1)
        self.grad_output[:, :, 0] = 2.0
        
        # Gradients for position 1 (count=2): Negative (discourage count=2)
        self.grad_output[:, :, 1] = -1.0
        
        # Gradients for position 2 (count=3): Very negative (strongly discourage count=3)
        self.grad_output[:, :, 2] = -3.0

    def test_forward_pass(self):
        """Test that zero counts map to all zeros in forward pass"""
        with torch.no_grad():
            encoded = self.encoder(self.counts)
        
        # Check shape
        self.assertEqual(encoded.shape, (4, 3, 3))
        
        # Check zero positions map to all zeros
        self.assertTrue(torch.all(encoded[0, 0, :] == 0))  # First zero count
        self.assertTrue(torch.all(encoded[2, 1, :] == 0))  # Another zero count
        
        # Check count=1 mapped to (1,0,0)
        self.assertTrue(torch.allclose(encoded[0, 1, :], torch.tensor([1.0, 0.0, 0.0])))
        
        # Check count=2 mapped to (0,1,0)
        self.assertTrue(torch.allclose(encoded[1, 1, :], torch.tensor([0.0, 1.0, 0.0])))
        
        # Check count=3 mapped to (0,0,1)
        self.assertTrue(torch.allclose(encoded[0, 2, :], torch.tensor([0.0, 0.0, 1.0])))

    def test_gradient_flow(self):
        """Test gradient flow from encoded representation back to counts"""
        # Forward pass
        encoded = self.encoder(self.counts)
        
        # Backward pass
        encoded.backward(self.grad_output)
        
        # Get gradients
        count_grads = self.counts.grad
        
        # Expected gradient behavior:
        # 1. Zero counts:
        #    - Should get positive gradient from position 0
        self.assertGreater(count_grads[0, 0].item(), 0)  # Zero count
        self.assertGreater(count_grads[2, 1].item(), 0)  # Another zero count
        
        # 2. Count=1:
        #    - Should consider: lose position 0 (2) and gain position 1 (-1)
        #    - Net effect: -1 - 2 = -3
        self.assertGreater(count_grads[0, 1].item(), 0)  # Count=1
        
        # 3. Count=2:
        #    - Should consider: lose position 1 (-(-1)=+1) and gain position 2 (-3)
        #    - Net effect: +1-(-3) = -2, discouraging increase to count=3
        #    - Should also consider: lose position 1 (-(-1)=+1) and gain position 0 (+2)
        #    - Net effect: -1*(+1-(+2)) = -1, discouraging decrease to count=1
        self.assertLess(count_grads[1, 1].item(), 0)  # Count=2
        
        # 4. Count=3 (max):
        #    - Should consider: lose position 2 (-(-3)=+3) and gain position 1 (-1)
        #    - Net effect: -1*(+3-(-1)) = -4, encouraging decrease to count=2
        self.assertLess(count_grads[0, 2].item(), 0)  # Count=3 (max)
        
        # 5. Specific value tests
        print(f"Zero count gradient: {count_grads[0, 0].item():.4f}")
        print(f"Count=1 gradient: {count_grads[0, 1].item():.4f}")
        print(f"Count=2 gradient: {count_grads[1, 1].item():.4f}")
        print(f"Count=3 gradient: {count_grads[0, 2].item():.4f}")
        
        # Check gradient stability
        self.assertFalse(torch.isnan(count_grads).any())
        self.assertFalse(torch.isinf(count_grads).any())

if __name__ == '__main__':
    unittest.main()