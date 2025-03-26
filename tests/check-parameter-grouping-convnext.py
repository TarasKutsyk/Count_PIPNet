#!/usr/bin/env python3
"""
Simple standalone script to test ConvNeXt parameter grouping
for both PIPNet and CountPIPNet without running full unit tests.

Usage:
    python simple-param-grouping-test.py

This script will create PIPNet and CountPIPNet models with 1, 2, and 3 stages,
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
from pipnet.pipnet import PIPNet, get_pip_network, NonNegLinear as PIPNetNonNegLinear

def create_count_pipnet(num_stages=2):
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
    
    expanded_dim = 16 * 3
    classification_layer = NonNegLinear(expanded_dim, 10, bias=False)
    
    model = CountPIPNet(
        num_classes=10,
        num_prototypes=16,
        feature_net=feature_net,
        args=args,
        add_on_layers=add_on_layers,
        classification_layer=classification_layer,
        max_count=3,
        use_ste=False
    )
    
    model = nn.DataParallel(model)
    return model, args

def create_pipnet(num_stages=2):
    """Create a regular PIPNet model with ConvNeXt backbone"""
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
    
    # Use get_pip_network to create model components
    feature_net, add_on_layers, pool_layer, classification_layer, num_prototypes = get_pip_network(10, args)
    
    # Create the model
    model = PIPNet(
        num_classes=10,
        num_prototypes=num_prototypes,
        feature_net=feature_net,
        args=args,
        add_on_layers=add_on_layers,
        pool_layer=pool_layer,
        classification_layer=classification_layer
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

def analyze_model(model, args, model_type):
    """Analyze parameter grouping and print results for a model"""
    stage_blocks, param_to_name = analyze_parameter_groups(model, args)
    
    print(f"\n=== {model_type}: {args.num_stages}-STAGE MODEL PARAMETER GROUPING ===")
    
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
    last_stage = args.num_stages
    last_stage_in_train = any(stage == last_stage for (stage, _) in stage_blocks['train'].keys())
    
    # Check if early stages are in backbone
    early_stages_in_backbone = all(stage < args.num_stages - 1
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
    
    return stage_blocks

def compare_models(pipnet_blocks, countpipnet_blocks, num_stages):
    """Compare parameter grouping between PIPNet and CountPIPNet"""
    print(f"\n=== COMPARISON BETWEEN PIPNET AND COUNTPIPNET (STAGES: {num_stages}) ===")
    
    # Compare each parameter group
    for group in ['train', 'freeze', 'backbone']:
        # Get stages for each model
        pipnet_stages = set(stage for (stage, _) in pipnet_blocks[group].keys() if isinstance(stage, int))
        countpipnet_stages = set(stage for (stage, _) in countpipnet_blocks[group].keys() if isinstance(stage, int))
        
        # Check if the stages match
        if pipnet_stages == countpipnet_stages:
            print(f"✅ {group.upper()} group: Both models use the same stages: {sorted(pipnet_stages)}")
        else:
            print(f"❌ {group.upper()} group: Models use different stages:")
            print(f"   - PIPNet: {sorted(pipnet_stages)}")
            print(f"   - CountPIPNet: {sorted(countpipnet_stages)}")

def main():
    """Test parameter grouping with configurable stage configurations for both models"""
    torch.manual_seed(42)
    
    # Test for different numbers of stages
    for num_stages in list(range(1, 4)):  # Testing with 1, 2, and 3 stages
        # Create both types of models
        pipnet_model, pipnet_args = create_pipnet(num_stages)
        countpipnet_model, countpipnet_args = create_count_pipnet(num_stages)
        
        # Analyze both models
        pipnet_blocks = analyze_model(pipnet_model, pipnet_args, "PIPNET")
        countpipnet_blocks = analyze_model(countpipnet_model, countpipnet_args, "COUNTPIPNET")
        
        # Compare the models
        compare_models(pipnet_blocks, countpipnet_blocks, num_stages)
        
        print("\n" + "="*80)  # Separator between different stage configurations

if __name__ == '__main__':
    main()