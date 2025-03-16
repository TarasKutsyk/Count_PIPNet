import unittest
import sys
import os
import torch
import torch.nn as nn
import argparse
from collections import namedtuple

# Add the parent directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the modules we want to test
from pipnet.count_pipnet import get_count_network, CountPIPNet

class TestCountPIPNet(unittest.TestCase):
    """
    Test suite for CountPIPNet with middle layers functionality
    """
    
    def setUp(self):
        """
        Set up test environment
        """
        # Create a mock arguments object
        self.args = argparse.Namespace(
            net='convnext_tiny_26',
            disable_pretrained=True,  # Use random weights for faster testing
            num_features=16,
            bias=False,
            use_mid_layers=False,
            num_stages=2
        )
        
        # Mock classes for simple testing
        self.num_classes = 10
        
    def create_model_with_args(self, **kwargs):
        """Helper function to create a model with custom args"""
        # Create a copy of the default args and update with kwargs
        args = argparse.Namespace(**vars(self.args))
        for key, value in kwargs.items():
            setattr(args, key, value)
        
        model, num_prototypes = get_count_network(
            self.num_classes, 
            args, 
            max_count=3, 
            use_ste=kwargs.get('use_ste', False)
        )
        return model, num_prototypes, args
    
    def test_basic_creation(self):
        """Test if the model can be created without errors"""
        model, num_prototypes, _ = self.create_model_with_args()
        
        # Basic checks
        self.assertIsInstance(model, CountPIPNet)
        self.assertEqual(model._num_classes, self.num_classes)
        self.assertEqual(num_prototypes, self.args.num_features)
        
        # Verify the model structure
        self.assertIsInstance(model._net, nn.Module)
        self.assertIsInstance(model._add_on, nn.Sequential)
        self.assertIsInstance(model._classification, nn.Module)
        
    def test_full_vs_mid_layer_models(self):
        """Test difference between full and middle-layer models"""
        # Create models with and without middle layers
        full_model, full_prototypes, _ = self.create_model_with_args(use_mid_layers=False)
        mid_model, mid_prototypes, _ = self.create_model_with_args(use_mid_layers=True, num_stages=2)
        
        # Both should have the same number of prototypes since we set num_features
        self.assertEqual(full_prototypes, mid_prototypes)
        
        # Check that the models are different
        full_params = sum(p.numel() for p in full_model._net.parameters())
        mid_params = sum(p.numel() for p in mid_model._net.parameters())
        
        # Mid-layer model should have fewer parameters
        self.assertLess(mid_params, full_params)
        
    def test_varying_num_stages(self):
        """Test with different numbers of stages"""
        # Create models with 1, 2, and 3 stages
        model1, _, _ = self.create_model_with_args(use_mid_layers=True, num_stages=1)
        model2, _, _ = self.create_model_with_args(use_mid_layers=True, num_stages=2)
        
        # Check that the models have different network sizes
        params1 = sum(p.numel() for p in model1._net.parameters())
        params2 = sum(p.numel() for p in model2._net.parameters())
        
        # Model with more stages should have more parameters
        self.assertLess(params1, params2)
        
    def test_forward_pass(self):
        """Test if forward pass works correctly"""
        # Create the model
        model, _, _ = self.create_model_with_args(use_mid_layers=True, num_stages=2)
        
        # Create a sample input
        x = torch.randn(2, 3, 224, 224)
        
        # Run forward pass
        proto_features, pooled, output = model(x)
        
        # Check shapes
        batch_size = x.size(0)
        self.assertEqual(output.shape, (batch_size, self.num_classes))
        self.assertEqual(pooled.shape[0], batch_size)
        
        # With max_count=3, the pooled should have dims [batch_size, num_prototypes * (max_count+1)]
        expected_pooled_dims = self.args.num_features * (3 + 1)  # max_count + 1
        self.assertEqual(pooled.shape[1], expected_pooled_dims)
        
    def test_ste_vs_non_ste(self):
        """Test difference between STE and non-STE models"""
        # Create models with and without STE
        ste_model, _, _ = self.create_model_with_args(use_ste=True)
        nonste_model, _, _ = self.create_model_with_args(use_ste=False)
        
        # Create a sample input
        x = torch.randn(2, 3, 224, 224)
        
        # Set models to eval mode to see difference in inference
        ste_model.eval()
        nonste_model.eval()
        
        # Run forward pass
        with torch.no_grad():
            _, ste_pooled, _ = ste_model(x, inference=True)
            _, nonste_pooled, _ = nonste_model(x, inference=True)
        
        # The outputs might be different, but should have the same shape
        self.assertEqual(ste_pooled.shape, nonste_pooled.shape)
            
    def test_dynamic_channel_detection(self):
        """Test if channel detection works correctly with different stage configurations"""
        # Test with different stage numbers
        for num_stages in [1, 2]:
            model, _, args = self.create_model_with_args(
                use_mid_layers=True, 
                num_stages=num_stages,
                num_features=0  # Let it use the detected channels
            )
            
            # Verify the model was created successfully
            self.assertIsInstance(model, CountPIPNet)
            
            # Create a sample input
            x = torch.randn(2, 3, 224, 224)
            
            # Verify forward pass works
            proto_features, pooled, output = model(x)
            
            # Output should match the expected shape
            self.assertEqual(output.shape, (2, self.num_classes))

if __name__ == '__main__':
    unittest.main()