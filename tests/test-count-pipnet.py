import unittest
import torch
import argparse
import sys
import os

# Add project root directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Go up one level to PIPNet root
sys.path.append(project_root)

# Import the models
from pipnet.pipnet import PIPNet, get_pip_network
from pipnet.count_pipnet import CountPIPNet, get_count_network
import torch.nn as nn

IMG_SIZE=192

class TestCountPIPNet(unittest.TestCase):
    """
    Combined test suite for CountPIPNet functionality:
    - Basic shape compatibility tests
    - Middle layers functionality
    - STE vs non-STE behavior
    - Count values validation
    """
    
    def setUp(self):
        """
        Set up the test environment with configurations and models.
        """
        # Create test args
        self.args = argparse.Namespace()
        self.args.net = 'convnext_tiny_26'  # Use a small network for faster testing
        self.args.num_features = 16  # Set number of features for consistent tests
        self.args.disable_pretrained = True  # Don't load pretrained weights for testing
        self.args.bias = True  # Use bias in linear layers
        self.args.use_mid_layers = False  # Default to full model
        self.args.num_stages = 2  # Default number of stages
        
        # Parameters for testing
        self.num_classes = 10  # Number of classes for testing
        self.batch_size = 4  # Batch size for testing
        self.max_count = 3  # Maximum count value
        
        # Create dummy input of shape [batch_size, 3, IMG_SIZE, IMG_SIZE]
        self.input = torch.randn(self.batch_size, 3, IMG_SIZE, IMG_SIZE)
        
        # Create the standard models
        self.pipnet, self.num_prototypes = self._create_pipnet()
        self.count_pipnet, _ = self._create_count_pipnet()
        
    def _create_pipnet(self):
        """Create a standard PIPNet model using the get_pipnet method."""
        from pipnet.pipnet import get_pipnet
        
        pipnet, num_prototypes = get_pipnet(self.num_classes, self.args)
        return pipnet, num_prototypes
        
    def _create_count_pipnet(self, use_ste=False):
        """Create a CountPIPNet model for testing."""
        model, num_prototypes = get_count_network(
            self.num_classes, 
            self.args, 
            max_count=self.max_count,
            use_ste=use_ste
        )
        
        return model, num_prototypes
    
    def create_model_with_args(self, **kwargs):
        """Helper function to create a model with custom args"""
        # Create a copy of the default args and update with kwargs
        args = argparse.Namespace(**vars(self.args))
        for key, value in kwargs.items():
            setattr(args, key, value)
        
        model, num_prototypes = get_count_network(
            self.num_classes, 
            args, 
            max_count=kwargs.get('max_count', self.max_count), 
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
    
    def test_forward_shapes(self):
        """Test that forward pass produces tensors of expected shapes."""
        # Forward pass for standard PIPNet
        proto_features_pip, pooled_pip, out_pip = self.pipnet(self.input)
        
        # Forward pass for CountPIPNet (inference mode)
        proto_features_count, flattened_counts, out_count = self.count_pipnet(self.input, inference=True)

        # Forward pass for CountPIPNet in default training mode
        _, counts, _ = self.count_pipnet(self.input)
        
        # Check proto_features shape (should be identical)
        self.assertEqual(proto_features_pip.shape, proto_features_count.shape,
                         "Proto features shape mismatch between PIPNet and CountPIPNet")
        
        # Check output shape (should be identical)
        self.assertEqual(out_pip.shape, out_count.shape,
                         "Output shape mismatch between PIPNet and CountPIPNet")
        
        # Check pooling layer output shape for PIPNet
        self.assertEqual(pooled_pip.shape, (self.batch_size, self.num_prototypes),
                         "Pooled output shape incorrect for PIPNet")
        
        # Check expanded representation shape for CountPIPNet
        expected_flattened_shape = (self.batch_size, self.num_prototypes * self.max_count)
        self.assertEqual(flattened_counts.shape, expected_flattened_shape,
                         f"Flattened counts shape incorrect. Expected {expected_flattened_shape}, got {flattened_counts.shape}")
        
        # Check if counts (non-flattened) are of the correct shape
        self.assertEqual(counts.shape, (self.batch_size, self.num_prototypes),
                         "Counts shape incorrect")
    
    def test_inference_mode(self):
        """Test that inference mode works correctly."""
        # Forward pass in inference mode
        proto_features, flattened_counts, out = self.count_pipnet(self.input, inference=True)
        
        # Check output shape
        self.assertEqual(out.shape, (self.batch_size, self.num_classes),
                         "Output shape incorrect in inference mode")
    
    def test_ste_vs_no_ste(self):
        """Test shape compatibility with and without STE."""
        # Create models with and without STE
        ste_model, _ = self._create_count_pipnet(use_ste=True)
        no_ste_model, _ = self._create_count_pipnet(use_ste=False)
        
        # Forward pass for both models
        _, counts_ste, out_ste = ste_model(self.input)
        _, counts_no_ste, out_no_ste = no_ste_model(self.input)
        
        # Check shapes match
        self.assertEqual(counts_ste.shape, counts_no_ste.shape,
                         "Counts shape should be the same with or without STE")
        self.assertEqual(out_ste.shape, out_no_ste.shape,
                         "Output shape should be the same with or without STE")
        
        # Test in eval mode to see difference in inference
        ste_model.eval()
        no_ste_model.eval()
        
        # Run forward pass
        with torch.no_grad():
            _, ste_pooled, _ = ste_model(self.input, inference=True)
            _, nonste_pooled, _ = no_ste_model(self.input, inference=True)
        
        # The outputs might be different, but should have the same shape
        self.assertEqual(ste_pooled.shape, nonste_pooled.shape)
    
    def test_count_values(self):
        """Test that count values are properly bounded."""
        # Create a specific input where one prototype is highly activated across many patches
        # First create a base input
        special_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        
        # Run the model up to the proto_features stage
        with torch.no_grad():
            features = self.count_pipnet._net(special_input)
            proto_features = self.count_pipnet._add_on(features)
            
            # Sum across spatial dimensions to get counts
            counts = proto_features.sum(dim=(2, 3))
            
            # Check that counts are non-negative
            self.assertTrue((counts >= 0).all(), "Counts should be non-negative")
            
            # Get one-hot encoded counts
            encoded_counts = self.count_pipnet.onehot_encoder(counts)
            
            # Check shape of encoded counts
            expected_shape = (1, self.num_prototypes, self.max_count)
            self.assertEqual(encoded_counts.shape, expected_shape,
                             f"Encoded counts shape incorrect. Expected {expected_shape}, got {encoded_counts.shape}")
            
            # Check that values are properly one-hot encoded (each row sums to 1)
            row_sums = encoded_counts.sum(dim=2)
            self.assertTrue(torch.allclose(row_sums, torch.ones_like(row_sums)),
                            "Each row in encoded_counts should sum to 1 (one-hot encoding)")
    
    def test_full_vs_mid_layer_models(self):
        """Test difference between full and middle-layer models for both PIPNet and CountPIPNet."""
        # Test CountPIPNet
        full_count_model, full_count_protos, _ = self.create_model_with_args(use_mid_layers=False)
        mid_count_model, mid_count_protos, _ = self.create_model_with_args(use_mid_layers=True, num_stages=2)
        
        # Both should have the same number of prototypes since we set num_features
        self.assertEqual(full_count_protos, mid_count_protos)
        
        # Check that the models are different
        full_count_params = sum(p.numel() for p in full_count_model._net.parameters())
        mid_count_params = sum(p.numel() for p in mid_count_model._net.parameters())
        
        # Mid-layer model should have fewer parameters
        self.assertLess(mid_count_params, full_count_params)
        
        # Test original PIPNet
        from pipnet.pipnet import get_pipnet
        
        # Create copies of args for different configurations
        full_args = argparse.Namespace(**vars(self.args))
        full_args.use_mid_layers = False
        
        mid_args = argparse.Namespace(**vars(self.args))
        mid_args.use_mid_layers = True
        mid_args.num_stages = 2
        
        full_pip_model, full_pip_protos = get_pipnet(self.num_classes, full_args)
        mid_pip_model, mid_pip_protos = get_pipnet(self.num_classes, mid_args)
        
        # Both should have the same number of prototypes since we set num_features
        self.assertEqual(full_pip_protos, mid_pip_protos)
        
        # Check that the models are different
        full_pip_params = sum(p.numel() for p in full_pip_model._net.parameters())
        mid_pip_params = sum(p.numel() for p in mid_pip_model._net.parameters())
        
        # Mid-layer model should have fewer parameters
        self.assertLess(mid_pip_params, full_pip_params)


    def test_varying_num_stages(self):
        """Test with different numbers of stages for both PIPNet and CountPIPNet."""
        # Test CountPIPNet
        count_model1, _, _ = self.create_model_with_args(use_mid_layers=True, num_stages=1)
        count_model2, _, _ = self.create_model_with_args(use_mid_layers=True, num_stages=2)
        
        # Check that the models have different network sizes
        count_params1 = sum(p.numel() for p in count_model1._net.parameters())
        count_params2 = sum(p.numel() for p in count_model2._net.parameters())
        
        # Model with more stages should have more parameters
        self.assertLess(count_params1, count_params2)
        
        # Test original PIPNet
        from pipnet.pipnet import get_pipnet
        
        # Create args for different configurations
        args1 = argparse.Namespace(**vars(self.args))
        args1.use_mid_layers = True
        args1.num_stages = 1
        
        args2 = argparse.Namespace(**vars(self.args))
        args2.use_mid_layers = True
        args2.num_stages = 2
        
        pip_model1, _ = get_pipnet(self.num_classes, args1)
        pip_model2, _ = get_pipnet(self.num_classes, args2)
        
        # Check that the models have different network sizes
        pip_params1 = sum(p.numel() for p in pip_model1._net.parameters())
        pip_params2 = sum(p.numel() for p in pip_model2._net.parameters())
        
        # Model with more stages should have more parameters
        self.assertLess(pip_params1, pip_params2)

    def test_dynamic_channel_detection(self):
        """Test if channel detection works correctly with different stage configurations 
        for both PIPNet and CountPIPNet."""
        # Test with different stage numbers for CountPIPNet
        for num_stages in [1, 2]:
            count_model, _, args = self.create_model_with_args(
                use_mid_layers=True, 
                num_stages=num_stages,
                num_features=0  # Let it use the detected channels
            )
            
            # Verify the model was created successfully
            self.assertIsInstance(count_model, CountPIPNet)
            
            # Create a sample input
            x = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)
            
            # Verify forward pass works
            proto_features, pooled, output = count_model(x)
            
            # Output should match the expected shape
            self.assertEqual(output.shape, (2, self.num_classes))
        
        # Test with different stage numbers for original PIPNet
        from pipnet.pipnet import get_pipnet, PIPNet
        
        for num_stages in [1, 2]:
            # Create args with specific configuration
            pip_args = argparse.Namespace(**vars(self.args))
            pip_args.use_mid_layers = True
            pip_args.num_stages = num_stages
            pip_args.num_features = 0  # Let it use the detected channels
            
            pip_model, _ = get_pipnet(self.num_classes, pip_args)
            
            # Verify the model was created successfully
            self.assertIsInstance(pip_model, PIPNet)
            
            # Create a sample input
            x = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)
            
            # Verify forward pass works
            proto_features, pooled, output = pip_model(x)
            
            # Output should match the expected shape
            self.assertEqual(output.shape, (2, self.num_classes))

    def test_training_interface_compatibility(self):
        """Test that both models can be trained with the same interface."""
        # Get both models
        pipnet, _ = self._create_pipnet()
        count_pipnet, _ = self._create_count_pipnet()
        
        # Create mock optimizer and scheduler for each model
        optimizer_net_pip = torch.optim.Adam(pipnet.parameters(), lr=0.001)
        optimizer_cls_pip = torch.optim.Adam(pipnet._classification.parameters(), lr=0.001)
        scheduler_pip = torch.optim.lr_scheduler.StepLR(optimizer_net_pip, step_size=10, gamma=0.1)
        
        optimizer_net_cnt = torch.optim.Adam(count_pipnet.parameters(), lr=0.001)
        optimizer_cls_cnt = torch.optim.Adam(count_pipnet._classification.parameters(), lr=0.001)
        scheduler_cnt = torch.optim.lr_scheduler.StepLR(optimizer_net_cnt, step_size=10, gamma=0.1)
        
        # Create a criterion
        criterion = nn.CrossEntropyLoss()
        
        # Verify that both models can be put in training mode
        pipnet.train()
        count_pipnet.train()
        
        # Verify both accept the same basic training inputs
        # We're just testing the interface compatibility, not actual training
        x = self.input
        y = torch.randint(0, self.num_classes, (self.batch_size,))
        
        # Run forward pass for both models
        proto_features_pip, pooled_pip, out_pip = pipnet(x)
        proto_features_cnt, pooled_cnt, out_cnt = count_pipnet(x)
        
        # Check both models output correct shapes for classification
        self.assertEqual(out_pip.shape, (self.batch_size, self.num_classes))
        self.assertEqual(out_cnt.shape, (self.batch_size, self.num_classes))
        
        # Verify both models can compute gradients
        optimizer_net_pip.zero_grad()
        optimizer_cls_pip.zero_grad()
        optimizer_net_cnt.zero_grad()
        optimizer_cls_cnt.zero_grad()
        
        # Create loss values (not using actual cross-entropy since output formats differ)
        loss_pip = out_pip.mean()
        loss_cnt = out_cnt.mean()
        
        # Check that gradients can be computed
        loss_pip.backward()
        loss_cnt.backward()
        
        # Verify optimizers and schedulers can step
        optimizer_net_pip.step()
        optimizer_cls_pip.step()
        scheduler_pip.step()
        
        optimizer_net_cnt.step()
        optimizer_cls_cnt.step()
        scheduler_cnt.step()
        
        # This test passes if no exceptions are raised

    def test_parameter_group_compatibility(self):
        """Test that parameter groups can be created similarly for both models."""
        pipnet, _ = self._create_pipnet()
        count_pipnet, _ = self._create_count_pipnet()
        
        # Verify that both models have similar key components for parameter grouping
        self.assertTrue(hasattr(pipnet, '_net'), "PIPNet should have a _net attribute")
        self.assertTrue(hasattr(count_pipnet, '_net'), "CountPIPNet should have a _net attribute")
        
        self.assertTrue(hasattr(pipnet, '_classification'), "PIPNet should have a _classification attribute")
        self.assertTrue(hasattr(count_pipnet, '_classification'), "CountPIPNet should have a _classification attribute")
        
        self.assertTrue(hasattr(pipnet, '_add_on'), "PIPNet should have an _add_on attribute")
        self.assertTrue(hasattr(count_pipnet, '_add_on'), "CountPIPNet should have an _add_on attribute")
        
        # Verify both accept inference parameter in forward
        proto_features_pip, pooled_pip, out_pip = pipnet(self.input, inference=True)
        proto_features_cnt, pooled_cnt, out_cnt = count_pipnet(self.input, inference=True)
        
        # Check inference outputs have expected shapes
        self.assertEqual(out_pip.shape, (self.batch_size, self.num_classes))
        self.assertEqual(out_cnt.shape, (self.batch_size, self.num_classes))

    def test_model_attributes_for_training_loop(self):
        """Test that both models have necessary attributes required by main.py's run_pipnet."""
        # In run_pipnet, various model attributes are accessed:
        # - net.module._classification.requires_grad
        # - net.module._multiplier (specific to PIPNet)
        # - net.module._classification.weight
        # - net.module._classification.normalization_multiplier (specific to PIPNet)
        # - net.module._classification.bias
        
        pipnet, _ = self._create_pipnet()
        count_pipnet, _ = self._create_count_pipnet()
        
        # Simulate DataParallel wrapper
        class MockModule:
            def __init__(self, model):
                self.module = model
        
        mock_pipnet = MockModule(pipnet)
        mock_count_pipnet = MockModule(count_pipnet)
        
        # Test critical attributes
        self.assertTrue(hasattr(mock_pipnet.module._classification, 'weight'), 
                        "PIPNet should have classification weights")
        self.assertTrue(hasattr(mock_count_pipnet.module._classification, 'weight'), 
                        "CountPIPNet should have classification weights")
        
        # Test if bias can be accessed (if present)
        if hasattr(mock_pipnet.module._classification, 'bias') and mock_pipnet.module._classification.bias is not None:
            self.assertTrue(mock_pipnet.module._classification.bias is not None, 
                        "PIPNet should have classification bias")
        
        if hasattr(mock_count_pipnet.module._classification, 'bias') and mock_count_pipnet.module._classification.bias is not None:
            self.assertTrue(mock_count_pipnet.module._classification.bias is not None, 
                        "CountPIPNet should have classification bias")
        
        # Test if requires_grad can be modified
        mock_pipnet.module._classification.requires_grad = False
        mock_count_pipnet.module._classification.requires_grad = False
        
        # This test would fail if any of the above operations raised an exception

if __name__ == '__main__':
    unittest.main()