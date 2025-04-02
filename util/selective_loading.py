"""
Utility functions for selectively loading parameters from checkpoints.
This is particularly useful for loading just the backbone parameters
while initializing the rest of the model (intermediate layers, classification head)
from scratch when testing different model configurations.
"""

import os
import torch
import torch.nn as nn
import re
from typing import Dict, Any, List, Set

def identify_backbone_parameters(model: nn.Module) -> Set[str]:
    """
    Identify which parameters belong to the backbone network.
    For CountPIPNet, these are parameters in the _net module.
    
    Args:
        model: The model to analyze
        
    Returns:
        Set of parameter names that are part of the backbone
    """
    backbone_params = set()
    
    for name, _ in model.named_parameters():
        if '_net.' in name:
            backbone_params.add(name)
        elif '_add_on.' in name:
            backbone_params.add(name)
            
    return backbone_params

def load_backbone_only(model: nn.Module, 
                       checkpoint_path: str, 
                       verbose: bool = True) -> Dict[str, Any]:
    """
    Load only the backbone parameters from a checkpoint file.
    Handles different prefix patterns between checkpoint and model.
    
    Args:
        model: The model to load parameters into
        checkpoint_path: Path to the checkpoint file
        verbose: Whether to print verbose loading information
        
    Returns:
        Dictionary containing metadata about the loading process
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    
    # Get model state dict from checkpoint
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict'] 
    else:
        state_dict = checkpoint
    
    # Identify backbone parameters
    backbone_params = identify_backbone_parameters(model)
    
    # Create a parameter name mapping to handle module prefix mismatches
    model_to_checkpoint_mapping = {}
    
    # Check if checkpoint keys have 'module.' prefix
    checkpoint_has_module_prefix = any(key.startswith('module.') for key in state_dict.keys())
    
    # Check if model parameter names have 'module.' prefix
    model_has_module_prefix = any(name.startswith('module.') for name in backbone_params)
    
    if verbose:
        print(f"Checkpoint has 'module.' prefix: {checkpoint_has_module_prefix}")
        print(f"Model has 'module.' prefix: {model_has_module_prefix}")
    
    # Create mappings based on prefix patterns
    for model_param_name in backbone_params:
        # Remove module prefix from model param name if present
        clean_name = model_param_name.replace('module.', '')
        
        # Determine the corresponding name in the checkpoint
        if checkpoint_has_module_prefix and not model_has_module_prefix:
            checkpoint_name = f"module.{model_param_name}"
        elif not checkpoint_has_module_prefix and model_has_module_prefix:
            checkpoint_name = clean_name
        else:
            # Same prefix pattern in both
            checkpoint_name = model_param_name
            
        model_to_checkpoint_mapping[model_param_name] = checkpoint_name
    
    if verbose:
        print(f"Checkpoint keys (first 10): {list(state_dict.keys())[:10]}")
        print(f"Looking for backbone params (first 10): {list(backbone_params)[:10]}")
        print(f"Mapped parameters (first 5 pairs):")
        for i, (model_name, checkpoint_name) in enumerate(list(model_to_checkpoint_mapping.items())[:5]):
            print(f"  {i+1}. Model: {model_name} -> Checkpoint: {checkpoint_name}")
    
    # Create a filtered state dict with only backbone parameters
    filtered_state_dict = {}
    mismatch_shapes = []
    missing_keys = []
    
    # For each model parameter name that should be loaded
    for model_name, checkpoint_name in model_to_checkpoint_mapping.items():
        if checkpoint_name in state_dict:
            # Get the parameter from the model
            param = None
            for n, p in model.named_parameters():
                if n == model_name:
                    param = p
                    break
            
            if param is None:
                missing_keys.append(model_name)
                continue
                
            # Check if shapes match
            if state_dict[checkpoint_name].shape == param.shape:
                filtered_state_dict[model_name] = state_dict[checkpoint_name]
            else:
                mismatch_shapes.append((model_name, checkpoint_name))
        else:
            missing_keys.append(model_name)
    
    # Statistics for logging
    total_backbone_params = len(backbone_params)
    loaded_params = len(filtered_state_dict)
    
    # Print loading information if verbose
    if verbose:
        print(f"\nLoading backbone parameters from: {checkpoint_path}")
        print(f"Total backbone parameters: {total_backbone_params}")
        print(f"Successfully loaded parameters: {loaded_params}")
        
        if missing_keys:
            print(f"Missing keys: {len(missing_keys)}")
            if len(missing_keys) < 10:
                print(f"  {missing_keys}")
                
        if mismatch_shapes:
            print(f"Shape mismatches: {len(mismatch_shapes)}")
            if len(mismatch_shapes) < 10:
                print(f"  {mismatch_shapes}")
    
    # Only load the backbone parameters if we found any
    if loaded_params > 0:
        model_dict = model.state_dict()
        model_dict.update(filtered_state_dict)
        model.load_state_dict(model_dict)
    
    return {
        "success": loaded_params > 0,
        "total_backbone_params": total_backbone_params,
        "loaded_params": loaded_params,
        "missing_keys": missing_keys,
        "mismatch_shapes": mismatch_shapes
    }

def load_shared_backbone(model: nn.Module, 
                         shared_pretrained_dir: str, 
                         verbose: bool = True) -> Dict[str, Any]:
    """
    Load a shared backbone model from a directory.
    Tries to load from both standard checkpoint locations.
    
    Args:
        model: The model to load backbone parameters into
        shared_pretrained_dir: Directory containing the checkpoint
        verbose: Whether to print verbose loading information
        
    Returns:
        Dictionary containing metadata about the loading process
    """
    # Try all possible checkpoint file locations
    possible_paths = [
        os.path.join(shared_pretrained_dir, "net_pretrained"),  # Standard path
        shared_pretrained_dir,  # In case the full path was provided
        os.path.join(shared_pretrained_dir, "checkpoint_best.pth"),  # Best checkpoint
        os.path.join(shared_pretrained_dir, "checkpoint_last.pth")   # Last checkpoint
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                return load_backbone_only(model, path, verbose)
            except Exception as e:
                if verbose:
                    print(f"Failed to load from {path}: {e}")
                continue
    
    # If we reach here, we couldn't load from any path
    if verbose:
        print(f"Could not load backbone from any checkpoint in {shared_pretrained_dir}")
    
    return {"success": False}