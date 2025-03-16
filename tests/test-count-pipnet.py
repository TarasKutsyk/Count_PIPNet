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
from pipnet.pipnet import PIPNet, get_network
from pipnet.count_pipnet import CountPIPNet, get_count_network


class TestCountPIPNet(unittest.TestCase):
    """
    Unit tests for the CountPIPNet architecture.
    Tests shape compatibility and basic functionality.
    """
    
    def setUp(self):
        """
        Set up the test environment with configurations and models.
        """
        # Create test args
        self.args = argparse.Namespace()
        self.args.net = 'convnext_tiny_26'  # Use a small network for faster testing
        self.args.num_features = 0  # Use default number of features
        self.args.disable_pretrained = True  # Don't load pretrained weights for testing
        self.args.bias = True  # Use bias in linear layers
        
        # Parameters for testing
        self.num_classes = 10  # Number of classes for testing
        self.batch_size = 4  # Batch size for testing
        self.max_count = 3  # Maximum count value
        
        # Create dummy input of shape [batch_size, 3, 224, 224]
        self.input = torch.randn(self.batch_size, 3, 224, 224)
        
        # Create the models
        self.pipnet, self.num_prototypes = self._create_pipnet()
        self.count_pipnet, _ = self._create_count_pipnet()
        
    def _create_pipnet(self):
        """Create a standard PIPNet model for comparison."""
        features, add_on_layers, pool_layer, classification_layer, num_prototypes = get_network(
            self.num_classes, self.args)
        
        pipnet = PIPNet(
            num_classes=self.num_classes,
            num_prototypes=num_prototypes,
            feature_net=features,
            args=self.args,
            add_on_layers=add_on_layers,
            pool_layer=pool_layer,
            classification_layer=classification_layer
        )
        
        return pipnet, num_prototypes
        
    def _create_count_pipnet(self, freeze_mode='none', use_ste=False):
        """Create a CountPIPNet model for testing."""
        model, num_prototypes = get_count_network(
            self.num_classes, 
            self.args, 
            max_count=self.max_count,
            freeze_mode=freeze_mode,
            use_ste=use_ste
        )
        
        return model, num_prototypes
    
    def test_forward_shapes(self):
        """Test that forward pass produces tensors of expected shapes."""
        # Forward pass for standard PIPNet
        proto_features_pip, pooled_pip, out_pip = self.pipnet(self.input)
        
        # Forward pass for CountPIPNet
        proto_features_count, flattened_counts, out_count = self.count_pipnet(self.input)
        
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
        expected_flattened_shape = (self.batch_size, self.num_prototypes * (self.max_count + 1))
        self.assertEqual(flattened_counts.shape, expected_flattened_shape,
                         f"Flattened counts shape incorrect. Expected {expected_flattened_shape}, got {flattened_counts.shape}")
    
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
        _, flattened_counts_ste, out_ste = ste_model(self.input)
        _, flattened_counts_no_ste, out_no_ste = no_ste_model(self.input)
        
        # Check shapes match
        self.assertEqual(flattened_counts_ste.shape, flattened_counts_no_ste.shape,
                         "Flattened counts shape should be the same with or without STE")
        self.assertEqual(out_ste.shape, out_no_ste.shape,
                         "Output shape should be the same with or without STE")
    
    def test_count_values(self):
        """Test that count values are properly bounded."""
        # Create a specific input where one prototype is highly activated across many patches
        # First create a base input
        special_input = torch.randn(1, 3, 224, 224)
        
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
            expected_shape = (1, self.num_prototypes, self.max_count + 1)
            self.assertEqual(encoded_counts.shape, expected_shape,
                             f"Encoded counts shape incorrect. Expected {expected_shape}, got {encoded_counts.shape}")
            
            # Check that values are properly one-hot encoded (each row sums to 1)
            row_sums = encoded_counts.sum(dim=2)
            self.assertTrue(torch.allclose(row_sums, torch.ones_like(row_sums)),
                            "Each row in encoded_counts should sum to 1 (one-hot encoding)")


if __name__ == '__main__':
    unittest.main()