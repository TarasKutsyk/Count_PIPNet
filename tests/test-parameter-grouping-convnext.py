import sys
import os
import unittest
import torch
import torch.nn as nn
import argparse
from collections import defaultdict

# Add project root to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.convnext_features import convnext_tiny_26_features
from pipnet.count_pipnet import CountPIPNet, GumbelSoftmax, NonNegLinear
from util.args import group_convnext_mid_layer_parameters

class TestConvNeXtParameterGrouping(unittest.TestCase):
    """
    Test suite for verifying ConvNeXt parameter grouping logic.
    
    These tests confirm that our new understanding of ConvNeXt's
    architecture is correctly reflected in the parameter grouping logic.
    """
    
    def setUp(self):
        """Set up common test resources."""
        # Fix random seed for reproducibility
        torch.manual_seed(42)
        
        # Create base model arguments
        self.args = argparse.Namespace(
            net='convnext_tiny_26',
            disable_pretrained=True,
            num_features=16,
            lr=0.0005,
            lr_net=0.0005,
            lr_block=0.0005,
            weight_decay=0.0,
            bias=False,
            seed=42,
            use_mid_layers=True,
            optimizer='Adam',
        )
    
    def create_model(self, num_stages):
        """Create a CountPIPNet model with specified number of stages."""
        # Set num_stages in args
        self.args.num_stages = num_stages
        
        # Create feature extractor
        feature_net = convnext_tiny_26_features(
            pretrained=False,
            use_mid_layers=True,
            num_stages=num_stages
        )
        
        # Create CountPIPNet components
        add_on_layers = nn.Sequential(
            nn.Conv2d(768, 16, kernel_size=1, stride=1, padding=0, bias=True),
            GumbelSoftmax(dim=1, tau=1.0)
        )
        
        expanded_dim = 16 * 4  # max_count + 1 = 3 + 1 = 4
        classification_layer = NonNegLinear(expanded_dim, 10, bias=False)
        
        # Create model
        model = CountPIPNet(
            num_classes=10,
            num_prototypes=16,
            feature_net=feature_net,
            args=self.args,
            add_on_layers=add_on_layers,
            classification_layer=classification_layer,
            max_count=3,
            freeze_mode='none',
            use_ste=False
        )
        
        return model
    
    def analyze_parameter_grouping(self, model, num_stages):
        """Apply parameter grouping logic and analyze the results."""
        # Apply parameter grouping
        params_to_train = []
        params_to_freeze = []
        params_backbone = []
        
        group_convnext_mid_layer_parameters(
            model._net, 
            params_to_train, 
            params_to_freeze, 
            params_backbone, 
            num_stages
        )
        
        # Analyze parameter grouping
        param_to_name = {}
        for name, param in model.named_parameters():
            if '_net.' in name:
                param_to_name[param] = name.split('_net.')[1]
        
        # Group parameters by stage and block
        stage_groups = {
            'train': defaultdict(list),
            'freeze': defaultdict(list),
            'backbone': defaultdict(list)
        }
        
        # Process each group
        for param in params_to_train:
            if param in param_to_name:
                name = param_to_name[param]
                parts = name.split('.')
                if len(parts) >= 2 and parts[0] == 'features' and parts[1].isdigit():
                    stage = int(parts[1])
                    stage_groups['train'][stage].append(name)
        
        for param in params_to_freeze:
            if param in param_to_name:
                name = param_to_name[param]
                parts = name.split('.')
                if len(parts) >= 2 and parts[0] == 'features' and parts[1].isdigit():
                    stage = int(parts[1])
                    stage_groups['freeze'][stage].append(name)
        
        for param in params_backbone:
            if param in param_to_name:
                name = param_to_name[param]
                parts = name.split('.')
                if len(parts) >= 2 and parts[0] == 'features' and parts[1].isdigit():
                    stage = int(parts[1])
                    stage_groups['backbone'][stage].append(name)
        
        return stage_groups, params_to_train, params_to_freeze, params_backbone
    
    def test_two_stage_model(self):
        """
        Test parameter grouping for a 2-stage model.
        
        Based on our analysis, for num_stages=2:
        - Stage 0: Stem → params_backbone
        - Stage 1: Content → params_to_freeze
        - Stage 2 (final): Transition → params_to_train
        """
        print("\n" + "="*80)
        print("Testing 2-stage model parameter grouping")
        print("="*80)
        
        # Create model and group parameters
        model = self.create_model(num_stages=2)
        stage_groups, params_to_train, params_to_freeze, params_backbone = self.analyze_parameter_grouping(model, 2)
        
        # Print parameter distribution
        print(f"Parameter counts: {len(params_to_train)} train, {len(params_to_freeze)} freeze, {len(params_backbone)} backbone")
        
        # Check that parameters are correctly grouped
        self.assertTrue(1 in stage_groups['freeze'], "Stage 1 should be in params_to_freeze")
        self.assertTrue(2 in stage_groups['train'], "Stage 2 should be in params_to_train")
        self.assertTrue(0 in stage_groups['backbone'], "Stage 0 should be in params_backbone")
        
        # Print stage distribution
        for group in ['train', 'freeze', 'backbone']:
            print(f"\n{group.upper()} stages: {sorted(stage_groups[group].keys())}")
        
        # Verify our understanding of the architecture
        # In a 2-stage model, we expect params_to_train to contain parameters from stage 2 (the final transition layer)
        self.assertGreater(len(stage_groups['train'][2]), 0, "params_to_train should contain stage 2 parameters")
        
    def test_four_stage_model(self):
        """
        Test parameter grouping for a 4-stage model.
        
        Based on our analysis, for num_stages=4:
        - Stage 0: Stem → params_backbone
        - Stage 1: Content → params_backbone
        - Stage 2: Transition → params_freeze
        - Stage 3: Content → params_freeze
        - Stage 4 (final): Transition → params_to_train
        """
        print("\n" + "="*80)
        print("Testing 4-stage model parameter grouping")
        print("="*80)
        
        # Create model and group parameters
        model = self.create_model(num_stages=4)
        stage_groups, params_to_train, params_to_freeze, params_backbone = self.analyze_parameter_grouping(model, 4)
        
        # Print parameter distribution
        print(f"Parameter counts: {len(params_to_train)} train, {len(params_to_freeze)} freeze, {len(params_backbone)} backbone")
        
        # Check that parameters are correctly grouped
        self.assertTrue(4 in stage_groups['train'], "Stage 4 should be in params_to_train")
        self.assertTrue(3 in stage_groups['freeze'], "Stage 3 should be in params_to_freeze")
        self.assertTrue(2 in stage_groups['backbone'], "Stage 2 should be in backbone")
        self.assertTrue(1 in stage_groups['backbone'], "Stage 1 should be in params_backbone")
        self.assertTrue(0 in stage_groups['backbone'], "Stage 0 should be in params_backbone")
        
        # Print stage distribution
        for group in ['train', 'freeze', 'backbone']:
            print(f"\n{group.upper()} stages: {sorted(stage_groups[group].keys())}")
        
        # Verify final stage is in params_to_train
        self.assertGreater(len(stage_groups['train'][4]), 0, "params_to_train should contain stage 4 parameters")
    
    def test_all_stage_configs(self):
        """Test parameter grouping for all valid stage configurations (2, 4, 6)."""
        valid_configs = [1, 3, 5, 7]  # Based on our analysis, these are the meaningful configurations
        
        for num_stages in valid_configs:
            print("\n" + "="*80)
            print(f"Testing {num_stages}-stage model parameter grouping")
            print("="*80)
            
            # Create model and group parameters
            model = self.create_model(num_stages=num_stages)
            stage_groups, params_to_train, params_to_freeze, params_backbone = self.analyze_parameter_grouping(model, num_stages)
            
            # Print parameter distribution
            print(f"Parameter counts: {len(params_to_train)} train, {len(params_to_freeze)} freeze, {len(params_backbone)} backbone")
            
            # Check final stage is in params_to_train
            output_stage = num_stages
            self.assertTrue(output_stage in stage_groups['train'], f"Stage {output_stage} should be in params_to_train")
            
            # Print stage distribution
            for group in ['train', 'freeze', 'backbone']:
                print(f"\n{group.upper()} stages: {sorted(stage_groups[group].keys())}")
            
            # Core verification: the stage that produces the output is in params_to_train
            self.assertGreater(len(stage_groups['train'][output_stage]), 0, 
                              f"params_to_train should contain stage {output_stage} parameters")

if __name__ == '__main__':
    unittest.main()