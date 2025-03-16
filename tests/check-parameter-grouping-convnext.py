#!/usr/bin/env python3
"""
Simple standalone script to test ConvNeXt parameter grouping
without running full unit tests.

Usage:
    python simple-param-grouping-test.py

This script will create ConvNeXt models with 1, 2, and 3 stages,
apply the parameter grouping logic, and display the results.
"""

import os
import sys
import torch
import torch.nn as nn
import argparse
from collections import defaultdict

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the necessary modules
from features.convnext_features import convnext_tiny_26_features
from util.args import get_optimizer_nn
from pipnet.count_pipnet import CountPIPNet, GumbelSoftmax, NonNegLinear

def create_model(num_stages=2):
    """Create a CountPIPNet model with ConvNeXt backbone"""
    args = argparse.Namespace(
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
        num_stages=num_stages,
        optimizer='Adam',
    )
    
    feature_net = convnext_tiny_26_features(
        pretrained=False,
        use_mid_layers=True,
        num_stages=num_stages
    )
    
    add_on_layers = nn.Sequential(
        nn.Conv2d(768, 16, kernel_size=1, stride=1, padding=0, bias=True),
        GumbelSoftmax(dim=1, tau=1.0)
    )
    
    expanded_dim = 16 * 4  # max_count + 1 = 3 + 1 = 4
    classification_layer = NonNegLinear(expanded_dim, 10, bias=False)
    
    model = CountPIPNet(
        num_classes=10,
        num_prototypes=16,
        feature_net=feature_net,
        args=args,
        add_on_layers=add_on_layers,
        classification_layer=classification_layer,
        max_count=3,
        freeze_mode='none',
        use_ste=False
    )
    
    model = nn.DataParallel(model)
    return model, args

def analyze_parameter_groups(model, args):
    """Analyze parameter grouping"""
    optimizer_net, _, params_to_freeze, params_to_train, params_backbone = get_optimizer_nn(model, args)
    
    # Group parameters by stage and block
    stage_blocks = defaultdict(lambda: defaultdict(list))
    
    # Get mappings from parameter to name
    param_to_name = {}
    for name, param in model.named_parameters():
        if '_net.' in name:
            param_to_name[param] = name.split('_net.')[1]
    
    # Categorize parameters
    for param in params_to_train:
        if param in param_to_name:
            name = param_to_name[param]
            parts = name.split('.')
            if len(parts) >= 3 and parts[0] == 'features':
                try:
                    stage = int(parts[1])
                    block = parts[2] if len(parts) > 2 else 'N/A'
                    stage_blocks['train'][(stage, block)].append(name)
                except ValueError:
                    stage_blocks['train'][('unknown', 'unknown')].append(name)
    
    for param in params_to_freeze:
        if param in param_to_name:
            name = param_to_name[param]
            parts = name.split('.')
            if len(parts) >= 3 and parts[0] == 'features':
                try:
                    stage = int(parts[1])
                    block = parts[2] if len(parts) > 2 else 'N/A'
                    stage_blocks['freeze'][(stage, block)].append(name)
                except ValueError:
                    stage_blocks['freeze'][('unknown', 'unknown')].append(name)
    
    for param in params_backbone:
        if param in param_to_name:
            name = param_to_name[param]
            parts = name.split('.')
            if len(parts) >= 3 and parts[0] == 'features':
                try:
                    stage = int(parts[1])
                    block = parts[2] if len(parts) > 2 else 'N/A'
                    stage_blocks['backbone'][(stage, block)].append(name)
                except ValueError:
                    stage_blocks['backbone'][('unknown', 'unknown')].append(name)
    
    return stage_blocks, param_to_name

def main():
    """Test parameter grouping with configurable stage configurations"""
    torch.manual_seed(42)
    
    for num_stages in list(range(1, 7 + 1)):
        model, args = create_model(num_stages)
        stage_blocks, param_to_name = analyze_parameter_groups(model, args)
        
        print(f"\n=== {num_stages}-STAGE MODEL PARAMETER GROUPING ===")
        
        # Count parameters in each group
        train_count = sum(len(params) for params in stage_blocks['train'].values())
        freeze_count = sum(len(params) for params in stage_blocks['freeze'].values())
        backbone_count = sum(len(params) for params in stage_blocks['backbone'].values())
        
        print(f"Parameter counts: {train_count} train, {freeze_count} freeze, {backbone_count} backbone")
        
        # Print parameter distribution by stage
        groups = ['train', 'freeze', 'backbone']
        for group in groups:
            print(f"\n{group.upper()} Parameters:")
            
            if not stage_blocks[group]:
                print("  No parameters in this group")
                continue
            
            # Group by stage for cleaner display
            by_stage = defaultdict(list)
            for (stage, block), params in stage_blocks[group].items():
                if isinstance(stage, int):
                    by_stage[stage].append((block, len(params)))
            
            # Print summary by stage
            for stage in sorted(by_stage.keys()):
                blocks = by_stage[stage]
                blocks_str = ", ".join([f"Block {block}: {count} params" for block, count in blocks])
                print(f"  Stage {stage}: {blocks_str}")
        
        # Validate if the model groups parameters according to our expectations
        last_stage = num_stages
        last_stage_in_train = any(stage == last_stage for (stage, _) in stage_blocks['train'].keys())
        
        # Check if early stages are in backbone
        early_stages_in_backbone = all(stage < num_stages - 1
                                    for (stage, _) in stage_blocks['backbone'].keys() 
                                    if isinstance(stage, int))
        
        if last_stage_in_train:
            print(f"✅ Last stage (Stage {last_stage}) parameters are in params_to_train!")
        else:
            print(f"❌ Last stage (Stage {last_stage}) parameters not in params_to_train!")
            
        if early_stages_in_backbone:
            print(f"✅ Early stages are in params_backbone as expected!")
        else:
            print(f"❌ Some early stages are not in params_backbone!")

if __name__ == '__main__':
    main()