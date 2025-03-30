import unittest
import torch
import argparse
import sys
import os
import numpy as np
import os
import pickle
import copy

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
        self.args.disable_pretrained = False  # Don't load pretrained weights for testing
        self.args.bias = False  # Use bias in linear layers
        self.args.use_mid_layers = True  # Default to full model
        self.args.num_stages = 3  # Default number of stages
        self.args.intermediate_layer = 'bilinear'
        
        # Parameters for testing
        self.num_classes = 9  # Number of classes for testing
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

        # Wrap it in DataParallel like the original
        model = nn.DataParallel(model)
        
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
    
    # def test_proto_features_shapes_and_counts(self):
    #     """Test the relationship between proto_features shape and counts"""
    #     # Create a CountPIPNet model
    #     model, num_prototypes = self._create_count_pipnet()
    #     model.eval()
        
    #     # Create dummy input
    #     x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        
    #     # Forward pass with hooks to capture intermediate tensors
    #     intermediate_outputs = {}
        
    #     def hook_fn(name):
    #         def hook(module, input, output):
    #             intermediate_outputs[name] = output
    #         return hook
        
    #     # Register hooks
    #     backbone_hook = model._net.register_forward_hook(hook_fn('backbone_output'))
    #     addon_hook = model._add_on.register_forward_hook(hook_fn('addon_output'))
        
    #     # Run model in inference mode for cleaner output
    #     with torch.no_grad():
    #         proto_features, counts, out = model(x)
        
    #     # Remove hooks
    #     backbone_hook.remove()
    #     addon_hook.remove()
        
    #     # Log dimensions of all intermediate tensors
    #     print(f"\n===== Shape Debugging =====")
    #     print(f"Input shape: {x.shape}")
    #     for name, tensor in intermediate_outputs.items():
    #         print(f"{name} shape: {tensor.shape}")
    #     print(f"proto_features shape: {proto_features.shape}")
    #     print(f"counts shape: {counts.shape}")
        
    #     # Check if the proto_features shape matches what we expect
    #     expected_shape = (1, num_prototypes, 28, 28)  # Expected shape based on convnext_tiny_26
    #     self.assertEqual(proto_features.shape[0], expected_shape[0], "Batch size mismatch")
    #     self.assertEqual(proto_features.shape[1], expected_shape[1], "Prototype count mismatch")
    #     print(f"Expected spatial dimensions: {expected_shape[2]}x{expected_shape[3]}")
    #     print(f"Actual spatial dimensions: {proto_features.shape[2]}x{proto_features.shape[3]}")
        
    #     # Test sum of proto_features vs counts for a specific prototype
    #     for p_idx in range(min(3, num_prototypes)):  # Test first 3 prototypes
    #         # Get the proto_features for this prototype
    #         p_features = proto_features[0, p_idx]
            
    #         # Calculate the sum
    #         p_sum = p_features.sum().item()
            
    #         # Get the corresponding count
    #         p_count = counts[0, p_idx].item()
            
    #         print(f"\nPrototype {p_idx} Analysis:")
    #         print(f"Sum of feature map: {p_sum:.4f}")
    #         print(f"Count value: {p_count:.4f}")
            
    #         # Print detailed statistics
    #         p_features_np = p_features.cpu().numpy()
    #         print(f"Max value: {p_features_np.max():.4f}")
    #         print(f"Mean value: {p_features_np.mean():.4f}")
    #         print(f"Values > 0.1: {(p_features_np > 0.1).sum()}")
    #         print(f"Values > 0.5: {(p_features_np > 0.5).sum()}")
            
    #         # Check locations of high activations
    #         high_activations = (p_features_np > 0.5).nonzero()
    #         if len(high_activations[0]) > 0:
    #             print("High activation locations (y, x):")
    #             for y, x in zip(high_activations[0], high_activations[1]):
    #                 print(f"  ({y}, {x}): {p_features_np[y, x]:.4f}")
            
    #         # Assert that the sum roughly equals the count
    #         self.assertAlmostEqual(p_sum, p_count, delta=0.1, 
    #                             msg=f"Sum of prototype {p_idx} features ({p_sum}) doesn't match count ({p_count})")
        
    #     # Detailed examination of proto_features for one prototype
    #     test_p_idx = 0  # Examine the first prototype
    #     print(f"\n===== Full Proto-Features Matrix for Prototype {test_p_idx} =====")
        
    #     p_features = proto_features[0, test_p_idx].cpu().numpy()
        
    #     # Instead of printing the full matrix which is too large,
    #     # print a summary and the non-zero values
    #     print(f"Matrix shape: {p_features.shape}")
    #     print(f"Sum: {p_features.sum():.4f}")
    #     print(f"Non-zero entries: {(p_features > 0.01).sum()}")
        
    #     # Print a heatmap visualizing all values
    #     import matplotlib.pyplot as plt
    #     plt.figure(figsize=(10, 8))
    #     plt.imshow(p_features, cmap='viridis')
    #     plt.colorbar(label='Activation')
    #     plt.title(f'Prototype {test_p_idx} Feature Map')
    #     plt.savefig(f'prototype_{test_p_idx}_feature_map_debug.png')
    #     plt.close()
        
    #     # Print the core implementation of the model's forward pass
    #     print("\n===== Model Forward Pass Implementation =====")
    #     import inspect
    #     print(inspect.getsource(model.forward))

    # def test_add_on_layer_impact(self):
    #     """Test how the add_on layers transform the feature dimensions"""
    #     model, num_prototypes = self._create_count_pipnet()
        
    #     # Create dummy features similar to what the backbone would output
    #     # Using a slightly larger batch size to test more thoroughly
    #     batch_size = 2
    #     backbone_features = torch.randn(batch_size, 192, 24, 24)
        
    #     # Apply the add_on layers directly
    #     with torch.no_grad():
    #         proto_features = model._add_on(backbone_features)
        
    #     print(f"\n===== Add-On Layer Testing =====")
    #     print(f"Backbone features shape: {backbone_features.shape}")
    #     print(f"Proto-features shape after add_on: {proto_features.shape}")
        
    #     # Test each component of the add_on layers to find which one changes the dimensions
    #     if isinstance(model._add_on, nn.Sequential):
    #         x = backbone_features
    #         for i, layer in enumerate(model._add_on):
    #             x = layer(x)
    #             print(f"After layer {i} ({type(layer).__name__}): {x.shape}")
        
    #     # Check if GumbelSoftmax or softmax preserves dimensions
    #     from pipnet.count_pipnet_utils import GumbelSoftmax
        
    #     # Test GumbelSoftmax directly
    #     if any(isinstance(m, GumbelSoftmax) for m in model._add_on.modules()):
    #         gumbel = next(m for m in model._add_on.modules() if isinstance(m, GumbelSoftmax))
    #         gumbel_out = gumbel(backbone_features)
    #         print(f"GumbelSoftmax output shape: {gumbel_out.shape}")
        
    #     # Test standard Softmax
    #     softmax = nn.Softmax(dim=1)
    #     softmax_out = softmax(backbone_features)
    #     print(f"Softmax output shape: {softmax_out.shape}")
        
    #     # Check if there's any Conv2d layer that might be changing the dimensions
    #     for name, module in model._add_on.named_modules():
    #         if isinstance(module, nn.Conv2d):
    #             print(f"Found Conv2d in add_on layers: {name}")
    #             print(f"  Kernel size: {module.kernel_size}")
    #             print(f"  Stride: {module.stride}")
    #             print(f"  Padding: {module.padding}")
                
    #             # Test if this Conv2d changes dimensions
    #             conv_out = module(backbone_features)
    #             print(f"  Output shape: {conv_out.shape}")

    # def test_counts_calculation(self):
    #     """Test how counts are calculated from proto_features"""
    #     model, num_prototypes = self._create_count_pipnet()
    #     model.eval()
        
    #     # Create a controlled proto_features tensor with known values
    #     # Create a 1x2x4x4 proto_features tensor (batch=1, prototypes=2, h=4, w=4)
    #     proto_features = torch.zeros(1, 2, 4, 4)
        
    #     # Set specific activations for the first prototype
    #     # Should sum to exactly 3.0
    #     proto_features[0, 0, 0, 0] = 1.0  # Full activation at (0,0)
    #     proto_features[0, 0, 1, 1] = 0.5  # Half activation at (1,1)
    #     proto_features[0, 0, 2, 2] = 1.0  # Full activation at (2,2)
    #     proto_features[0, 0, 3, 3] = 0.5  # Half activation at (3,3)
        
    #     # Set specific activations for the second prototype
    #     # Should sum to exactly 2.0
    #     proto_features[0, 1, 0, 1] = 1.0  # Full activation at (0,1)
    #     proto_features[0, 1, 1, 0] = 0.5  # Half activation at (1,0)
    #     proto_features[0, 1, 3, 3] = 0.5  # Half activation at (3,3)
        
    #     # Manually calculate the expected counts
    #     expected_counts = torch.tensor([[3.0, 2.0]])
        
    #     # Run the model's counting logic
    #     with torch.no_grad():
    #         # Calculate counts directly from proto_features
    #         counts = proto_features.sum(dim=(2, 3))
            
    #         # For comparison, try accessing the model's internal counting code
    #         # This will only work if you add a test method to CountPIPNet
    #         if hasattr(model, '_calculate_counts_for_testing'):
    #             model_counts = model._calculate_counts_for_testing(proto_features)
    #             print(f"Model's calculated counts: {model_counts}")
        
    #     print(f"\n===== Counts Calculation Testing =====")
    #     print(f"Proto-features tensor shape: {proto_features.shape}")
    #     print(f"Manual sum of prototype 0: {proto_features[0, 0].sum().item()}")
    #     print(f"Manual sum of prototype 1: {proto_features[0, 1].sum().item()}")
    #     print(f"Expected counts: {expected_counts}")
    #     print(f"Actual counts: {counts}")
        
    #     # Assert that the counts match what we expect
    #     torch.testing.assert_close(counts, expected_counts, 
    #                             msg=f"Calculated counts {counts} don't match expected {expected_counts}")
        
    # def test_basic_creation(self):
    #     """Test if the model can be created without errors"""
    #     model, num_prototypes, _ = self.create_model_with_args()
        
    #     # Basic checks
    #     self.assertIsInstance(model, CountPIPNet)
    #     self.assertEqual(model._num_classes, self.num_classes)
    #     self.assertEqual(num_prototypes, self.args.num_features)
        
    #     # Verify the model structure
    #     self.assertIsInstance(model._net, nn.Module)
    #     self.assertIsInstance(model._add_on, nn.Sequential)
    #     self.assertIsInstance(model._classification, nn.Module)
    
    # def test_forward_shapes(self):
    #     """Test that forward pass produces tensors of expected shapes."""
    #     # Forward pass for standard PIPNet
    #     proto_features_pip, pooled_pip, out_pip = self.pipnet(self.input)
        
    #     # Forward pass for CountPIPNet (inference mode)
    #     proto_features_count, flattened_counts, out_count = self.count_pipnet(self.input, inference=True)

    #     # Forward pass for CountPIPNet in default training mode
    #     _, counts, _ = self.count_pipnet(self.input)
        
    #     # Check proto_features shape (should be identical)
    #     self.assertEqual(proto_features_pip.shape, proto_features_count.shape,
    #                      "Proto features shape mismatch between PIPNet and CountPIPNet")
        
    #     # Check output shape (should be identical)
    #     self.assertEqual(out_pip.shape, out_count.shape,
    #                      "Output shape mismatch between PIPNet and CountPIPNet")
        
    #     # Check pooling layer output shape for PIPNet
    #     self.assertEqual(pooled_pip.shape, (self.batch_size, self.num_prototypes),
    #                      "Pooled output shape incorrect for PIPNet")
        
    #     # Check expanded representation shape for CountPIPNet
    #     expected_flattened_shape = (self.batch_size, self.num_prototypes * self.max_count)
    #     self.assertEqual(flattened_counts.shape, expected_flattened_shape,
    #                      f"Flattened counts shape incorrect. Expected {expected_flattened_shape}, got {flattened_counts.shape}")
        
    #     # Check if counts (non-flattened) are of the correct shape
    #     self.assertEqual(counts.shape, (self.batch_size, self.num_prototypes),
    #                      "Counts shape incorrect")
    
    # def test_inference_mode(self):
    #     """Test that inference mode works correctly."""
    #     # Forward pass in inference mode
    #     proto_features, flattened_counts, out = self.count_pipnet(self.input, inference=True)
        
    #     # Check output shape
    #     self.assertEqual(out.shape, (self.batch_size, self.num_classes),
    #                      "Output shape incorrect in inference mode")
    
    # def test_ste_vs_no_ste(self):
    #     """Test shape compatibility with and without STE."""
    #     # Create models with and without STE
    #     ste_model, _ = self._create_count_pipnet(use_ste=True)
    #     no_ste_model, _ = self._create_count_pipnet(use_ste=False)
        
    #     # Forward pass for both models
    #     _, counts_ste, out_ste = ste_model(self.input)
    #     _, counts_no_ste, out_no_ste = no_ste_model(self.input)
        
    #     # Check shapes match
    #     self.assertEqual(counts_ste.shape, counts_no_ste.shape,
    #                      "Counts shape should be the same with or without STE")
    #     self.assertEqual(out_ste.shape, out_no_ste.shape,
    #                      "Output shape should be the same with or without STE")
        
    #     # Test in eval mode to see difference in inference
    #     ste_model.eval()
    #     no_ste_model.eval()
        
    #     # Run forward pass
    #     with torch.no_grad():
    #         _, ste_pooled, _ = ste_model(self.input, inference=True)
    #         _, nonste_pooled, _ = no_ste_model(self.input, inference=True)
        
    #     # The outputs might be different, but should have the same shape
    #     self.assertEqual(ste_pooled.shape, nonste_pooled.shape)
    
    # def test_full_vs_mid_layer_models(self):
    #     """Test difference between full and middle-layer models for both PIPNet and CountPIPNet."""
    #     # Test CountPIPNet
    #     full_count_model, full_count_protos, _ = self.create_model_with_args(use_mid_layers=False)
    #     mid_count_model, mid_count_protos, _ = self.create_model_with_args(use_mid_layers=True, num_stages=3)
        
    #     # Both should have the same number of prototypes since we set num_features
    #     self.assertEqual(full_count_protos, mid_count_protos)
        
    #     # Check that the models are different
    #     full_count_params = sum(p.numel() for p in full_count_model._net.parameters())
    #     mid_count_params = sum(p.numel() for p in mid_count_model._net.parameters())
        
    #     # Mid-layer model should have fewer parameters
    #     self.assertLess(mid_count_params, full_count_params)
        
    #     # Test original PIPNet
    #     from pipnet.pipnet import get_pipnet
        
    #     # Create copies of args for different configurations
    #     full_args = argparse.Namespace(**vars(self.args))
    #     full_args.use_mid_layers = False
        
    #     mid_args = argparse.Namespace(**vars(self.args))
    #     mid_args.use_mid_layers = True
    #     mid_args.num_stages = 3
        
    #     full_pip_model, full_pip_protos = get_pipnet(self.num_classes, full_args)
    #     mid_pip_model, mid_pip_protos = get_pipnet(self.num_classes, mid_args)
        
    #     # Both should have the same number of prototypes since we set num_features
    #     self.assertEqual(full_pip_protos, mid_pip_protos)
        
    #     # Check that the models are different
    #     full_pip_params = sum(p.numel() for p in full_pip_model._net.parameters())
    #     mid_pip_params = sum(p.numel() for p in mid_pip_model._net.parameters())
        
    #     # Mid-layer model should have fewer parameters
    #     self.assertLess(mid_pip_params, full_pip_params)


    # def test_varying_num_stages(self):
    #     """Test with different numbers of stages for both PIPNet and CountPIPNet."""
    #     # Test CountPIPNet
    #     count_model1, _, _ = self.create_model_with_args(use_mid_layers=True, num_stages=3)
    #     count_model2, _, _ = self.create_model_with_args(use_mid_layers=True, num_stages=5)
        
    #     # Check that the models have different network sizes
    #     count_params1 = sum(p.numel() for p in count_model1._net.parameters())
    #     count_params2 = sum(p.numel() for p in count_model2._net.parameters())
        
    #     # Model with more stages should have more parameters
    #     self.assertLess(count_params1, count_params2)
        
    #     # Test original PIPNet
    #     from pipnet.pipnet import get_pipnet
        
    #     # Create args for different configurations
    #     args1 = argparse.Namespace(**vars(self.args))
    #     args1.use_mid_layers = True
    #     args1.num_stages = 3
        
    #     args2 = argparse.Namespace(**vars(self.args))
    #     args2.use_mid_layers = True
    #     args2.num_stages = 5
        
    #     pip_model1, _ = get_pipnet(self.num_classes, args1)
    #     pip_model2, _ = get_pipnet(self.num_classes, args2)
        
    #     # Check that the models have different network sizes
    #     pip_params1 = sum(p.numel() for p in pip_model1._net.parameters())
    #     pip_params2 = sum(p.numel() for p in pip_model2._net.parameters())
        
    #     # Model with more stages should have more parameters
    #     self.assertLess(pip_params1, pip_params2)

    # def test_dynamic_channel_detection(self):
    #     """Test if channel detection works correctly with different stage configurations 
    #     for both PIPNet and CountPIPNet."""
    #     # Test with different stage numbers for CountPIPNet
    #     for num_stages in [3, 5]:
    #         count_model, _, args = self.create_model_with_args(
    #             use_mid_layers=True, 
    #             num_stages=num_stages,
    #             num_features=0  # Let it use the detected channels
    #         )
            
    #         # Verify the model was created successfully
    #         self.assertIsInstance(count_model, CountPIPNet)
            
    #         # Create a sample input
    #         x = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)
            
    #         # Verify forward pass works
    #         proto_features, pooled, output = count_model(x)
            
    #         # Output should match the expected shape
    #         self.assertEqual(output.shape, (2, self.num_classes))
        
    #     # Test with different stage numbers for original PIPNet
    #     from pipnet.pipnet import get_pipnet, PIPNet
        
    #     for num_stages in [3, 5]:
    #         # Create args with specific configuration
    #         pip_args = argparse.Namespace(**vars(self.args))
    #         pip_args.use_mid_layers = True
    #         pip_args.num_stages = num_stages
    #         pip_args.num_features = 0  # Let it use the detected channels
            
    #         pip_model, _ = get_pipnet(self.num_classes, pip_args)
            
    #         # Verify the model was created successfully
    #         self.assertIsInstance(pip_model, PIPNet)
            
    #         # Create a sample input
    #         x = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)
            
    #         # Verify forward pass works
    #         proto_features, pooled, output = pip_model(x)
            
    #         # Output should match the expected shape
    #         self.assertEqual(output.shape, (2, self.num_classes))

    # def test_training_interface_compatibility(self):
    #     """Test that both models can be trained with the same interface."""
    #     # Get both models
    #     pipnet, _ = self._create_pipnet()
    #     count_pipnet, _ = self._create_count_pipnet()
        
    #     # Create mock optimizer and scheduler for each model
    #     optimizer_net_pip = torch.optim.Adam(pipnet.parameters(), lr=0.001)
    #     optimizer_cls_pip = torch.optim.Adam(pipnet._classification.parameters(), lr=0.001)
    #     scheduler_pip = torch.optim.lr_scheduler.StepLR(optimizer_net_pip, step_size=10, gamma=0.1)
        
    #     optimizer_net_cnt = torch.optim.Adam(count_pipnet.parameters(), lr=0.001)
    #     optimizer_cls_cnt = torch.optim.Adam(count_pipnet._classification.parameters(), lr=0.001)
    #     scheduler_cnt = torch.optim.lr_scheduler.StepLR(optimizer_net_cnt, step_size=10, gamma=0.1)
        
    #     # Create a criterion
    #     criterion = nn.CrossEntropyLoss()
        
    #     # Verify that both models can be put in training mode
    #     pipnet.train()
    #     count_pipnet.train()
        
    #     # Verify both accept the same basic training inputs
    #     # We're just testing the interface compatibility, not actual training
    #     x = self.input
    #     y = torch.randint(0, self.num_classes, (self.batch_size,))
        
    #     # Run forward pass for both models
    #     proto_features_pip, pooled_pip, out_pip = pipnet(x)
    #     proto_features_cnt, pooled_cnt, out_cnt = count_pipnet(x)
        
    #     # Check both models output correct shapes for classification
    #     self.assertEqual(out_pip.shape, (self.batch_size, self.num_classes))
    #     self.assertEqual(out_cnt.shape, (self.batch_size, self.num_classes))
        
    #     # Verify both models can compute gradients
    #     optimizer_net_pip.zero_grad()
    #     optimizer_cls_pip.zero_grad()
    #     optimizer_net_cnt.zero_grad()
    #     optimizer_cls_cnt.zero_grad()
        
    #     # Create loss values (not using actual cross-entropy since output formats differ)
    #     loss_pip = out_pip.mean()
    #     loss_cnt = out_cnt.mean()
        
    #     # Check that gradients can be computed
    #     loss_pip.backward()
    #     loss_cnt.backward()
        
    #     # Verify optimizers and schedulers can step
    #     optimizer_net_pip.step()
    #     optimizer_cls_pip.step()
    #     scheduler_pip.step()
        
    #     optimizer_net_cnt.step()
    #     optimizer_cls_cnt.step()
    #     scheduler_cnt.step()
        
    #     # This test passes if no exceptions are raised

    # def test_parameter_group_compatibility(self):
    #     """Test that parameter groups can be created similarly for both models."""
    #     pipnet, _ = self._create_pipnet()
    #     count_pipnet, _ = self._create_count_pipnet()
        
    #     # Verify that both models have similar key components for parameter grouping
    #     self.assertTrue(hasattr(pipnet, '_net'), "PIPNet should have a _net attribute")
    #     self.assertTrue(hasattr(count_pipnet, '_net'), "CountPIPNet should have a _net attribute")
        
    #     self.assertTrue(hasattr(pipnet, '_classification'), "PIPNet should have a _classification attribute")
    #     self.assertTrue(hasattr(count_pipnet, '_classification'), "CountPIPNet should have a _classification attribute")
        
    #     self.assertTrue(hasattr(pipnet, '_add_on'), "PIPNet should have an _add_on attribute")
    #     self.assertTrue(hasattr(count_pipnet, '_add_on'), "CountPIPNet should have an _add_on attribute")
        
    #     # Verify both accept inference parameter in forward
    #     proto_features_pip, pooled_pip, out_pip = pipnet(self.input, inference=True)
    #     proto_features_cnt, pooled_cnt, out_cnt = count_pipnet(self.input, inference=True)
        
    #     # Check inference outputs have expected shapes
    #     self.assertEqual(out_pip.shape, (self.batch_size, self.num_classes))
    #     self.assertEqual(out_cnt.shape, (self.batch_size, self.num_classes))

    # def test_model_attributes_for_training_loop(self):
    #     """Test that both models have necessary attributes required by main.py's run_pipnet."""
    #     # In run_pipnet, various model attributes are accessed:
    #     # - net.module._classification.requires_grad
    #     # - net.module._multiplier (specific to PIPNet)
    #     # - net.module._classification.weight
    #     # - net.module._classification.normalization_multiplier (specific to PIPNet)
    #     # - net.module._classification.bias
        
    #     pipnet, _ = self._create_pipnet()
    #     count_pipnet, _ = self._create_count_pipnet()
        
    #     # Simulate DataParallel wrapper
    #     class MockModule:
    #         def __init__(self, model):
    #             self.module = model
        
    #     mock_pipnet = MockModule(pipnet)
    #     mock_count_pipnet = MockModule(count_pipnet)
        
    #     # Test critical attributes
    #     self.assertTrue(hasattr(mock_pipnet.module._classification, 'weight'), 
    #                     "PIPNet should have classification weights")
    #     self.assertTrue(hasattr(mock_count_pipnet.module._classification, 'weight'), 
    #                     "CountPIPNet should have classification weights")
        
    #     # Test if bias can be accessed (if present)
    #     if hasattr(mock_pipnet.module._classification, 'bias') and mock_pipnet.module._classification.bias is not None:
    #         self.assertTrue(mock_pipnet.module._classification.bias is not None, 
    #                     "PIPNet should have classification bias")
        
    #     if hasattr(mock_count_pipnet.module._classification, 'bias') and mock_count_pipnet.module._classification.bias is not None:
    #         self.assertTrue(mock_count_pipnet.module._classification.bias is not None, 
    #                     "CountPIPNet should have classification bias")
        
    #     # Test if requires_grad can be modified
    #     mock_pipnet.module._classification.requires_grad = False
    #     mock_count_pipnet.module._classification.requires_grad = False
        
        # This test would fail if any of the above operations raised an exception

    def test_proto_features_with_loaded_checkpoint(self):
        """Test proto_features shapes and counts with a loaded checkpoint"""
        # Create a CountPIPNet model
        model, num_prototypes = self._create_count_pipnet()
        model.eval()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)  # Move model to device before wrapping
        model = nn.DataParallel(model)  # Wrap in DataParallel

        # After loading the checkpoint:
        model = model.to(device)  # Ensure model is on the correct device after loading
                
        # Hard-code a checkpoint path - adjust this to point to your actual checkpoint
        checkpoint_path = "./runs/12_shapes_bilinear_no_sparse_gaussian_noise/checkpoints/net_pretrained_d3f8da25c6"  # Adjust this path
        
        # Load the checkpoint
        try:
            print(f"\nLoading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)  # Use strict=False to handle potential mismatches
            print("Checkpoint loaded successfully")
        except Exception as e:
            self.fail(f"Failed to load checkpoint: {str(e)}")
        
        # Create dummy input
        x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
        
        # Forward pass
        with torch.no_grad():
            proto_features, counts, out = model(x)
        
        print(f"\n===== Loaded Model Test =====")
        print(f"Proto-features shape: {proto_features.shape}")
        print(f"Counts shape: {counts.shape}")
        
        # Check relationship between proto_features and counts
        for p_idx in range(min(3, num_prototypes)):
            # Get the proto_features for this prototype
            p_features = proto_features[0, p_idx]
            
            # Calculate various statistics
            p_features_np = p_features.cpu().numpy()
            p_sum = p_features_np.sum()
            p_max = p_features_np.max()
            p_mean = p_features_np.mean()
            p_count = counts[0, p_idx].item()
            
            print(f"\nPrototype {p_idx} Analysis with loaded weights:")
            print(f"Sum of feature map: {p_sum:.4f}")
            print(f"Count value: {p_count:.4f}")
            print(f"Max value: {p_max:.4f}")
            print(f"Mean value: {p_mean:.4f}")
            print(f"Non-zero values (>0.01): {(p_features_np > 0.01).sum()}")
            print(f"Values > 0.1: {(p_features_np > 0.1).sum()}")
            print(f"Values > 0.5: {(p_features_np > 0.5).sum()}")
            
            # Verify that sum of features approximately equals count
            delta = abs(p_sum - p_count)
            print(f"Difference between sum and count: {delta:.4f}")
            
            if delta > 0.1:
                print("WARNING: Large difference between feature map sum and count")
                
                # Let's look at what's inside counts more closely
                if hasattr(model, "_use_ste"):
                    print(f"Model uses STE: {model._use_ste}")
                
                # Analyze the distribution of values in both
                print(f"Feature map unique values: {np.unique(p_features_np)}")
                
                # Check if model is using Gumbel-Softmax which might discretize values
                from pipnet.count_pipnet_utils import GumbelSoftmax
                has_gumbel = any(isinstance(m, GumbelSoftmax) for m in model._add_on.modules())
                print(f"Model has GumbelSoftmax: {has_gumbel}")
                
                # Save the feature map as an image for inspection
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 8))
                plt.imshow(p_features_np, cmap='viridis')
                plt.colorbar(label='Activation')
                plt.title(f'Prototype {p_idx} Feature Map (Checkpoint loaded)')
                plt.savefig(f'checkpoint_prototype_{p_idx}_feature_map.png')
                plt.close()

    def test_feature_map_loading_from_input(self):
        """Test feature map generation with a real image input"""
        # Create a CountPIPNet model
        model, num_prototypes = self._create_count_pipnet()
        model.eval()

        # To this:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
 
        # Hard-code a checkpoint path
        checkpoint_path = "./runs/12_shapes_bilinear_no_sparse_gaussian_noise/checkpoints/net_pretrained_d3f8da25c6"  # Adjust this path
        
        # Load the checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            print("Checkpoint loaded successfully for image test")

        except Exception as e:
            self.fail(f"Failed to load checkpoint: {str(e)}")

        # After loading the checkpoint:
        model = model.to(device)  # Ensure model is on the correct device after loading
        
        # Try to load a real image for testing
        try:
            from PIL import Image
            import torchvision.transforms as transforms
            
            # Standard image transformations similar to what's used in the main code
            transform = transforms.Compose([
                transforms.Resize(IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Try to load an example image from your dataset
            image_path = "./data/geometric_shapes_no_noise/dataset/train/class_4/img_0000.png"  # Adjust path to a real image
            
            # Check if file exists
            import os
            if not os.path.exists(image_path):
                print(f"Image not found at {image_path}, using random noise instead")
                test_image = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
            else:
                # Load and transform the image
                img = Image.open(image_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
                test_image = img_tensor

            test_image = test_image.to(device)
            
            # Forward pass with the image
            with torch.no_grad():
                proto_features, counts, out = model(test_image)
            
            print(f"\n===== Real Image Test =====")
            print(f"Image shape: {test_image.shape}")
            print(f"Proto-features shape: {proto_features.shape}")
            print(f"Counts: {counts[0]}")  # First batch element
            print(f"Output scores: {out[0]}")  # First batch element
            
            # Check some high-activating prototypes
            top_prototypes = torch.argsort(counts[0], descending=True)[:3]
            
            for idx, p_idx in enumerate(top_prototypes):
                p_features = proto_features[0, p_idx]
                p_count = counts[0, p_idx].item()
                p_features_np = p_features.cpu().numpy()
                
                print(f"\nTop Prototype {idx+1}: #{p_idx.item()} (Count: {p_count:.4f})")
                print(f"Feature map sum: {p_features_np.sum():.4f}")
                print(f"Max activation: {p_features_np.max():.4f}")
                print(f"Number of activations > 0.5: {(p_features_np > 0.5).sum()}")
                
                # Save feature map visualization
                import matplotlib.pyplot as plt
                plt.figure(figsize=(12, 5))
                
                # Original image (if available)
                plt.subplot(1, 2, 1)
                if not os.path.exists(image_path):
                    plt.text(0.5, 0.5, "Random noise image", 
                            horizontalalignment='center', verticalalignment='center')
                else:
                    # Convert tensor to image for display
                    img_np = test_image[0].cpu().numpy().transpose(1, 2, 0)
                    # Denormalize
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img_np = std * img_np + mean
                    img_np = np.clip(img_np, 0, 1)
                    plt.imshow(img_np)
                plt.title("Input Image")
                plt.axis('off')
                
                # Feature map
                plt.subplot(1, 2, 2)
                plt.imshow(p_features_np, cmap='viridis')
                plt.colorbar(label='Activation')
                plt.title(f'Prototype {p_idx.item()} Feature Map')
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(f'real_image_prototype_{p_idx.item()}.png')
                plt.close()
                
        except Exception as e:
            print(f"Error during real image test: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    unittest.main()
    print(f'Project root: {project_root}')
