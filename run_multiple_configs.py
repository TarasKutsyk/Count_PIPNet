#!/usr/bin/env python3
"""
Script for running multiple PIPNet or CountPIPNet training sessions
with different configuration files sequentially, using a shared
pre-trained backbone model.
"""

import os
import sys
import argparse
import yaml
import time
from datetime import datetime
from copy import deepcopy
import json
import torch
import hashlib
import yaml

# Import the run_pipnet function directly, but NOT get_args
from main import run_pipnet

def parse_command_line_args():
    """Parse command line arguments for the multi-configuration runner."""
    parser = argparse.ArgumentParser(description='Run multiple PIPNet configurations')
    parser.add_argument('--config_list', 
                        type=str, 
                        required=True,
                        help='Path to a JSON file containing a list of YAML config file paths')
    parser.add_argument('--base_log_dir', 
                        type=str, 
                        default='./runs/multi_config',
                        help='Base directory for all run logs')
    parser.add_argument('--sequential', 
                        action='store_true',
                        help='Run configurations sequentially (default)')
    parser.add_argument('--continue_on_error', 
                        action='store_true',
                        help='Continue with next configuration if current one fails')
    parser.add_argument('--gpu_ids',
                        type=str,
                        default='',
                        help='GPU IDs to use (comma-separated)')
    parser.add_argument('--shared_pretrained_dir',
                        type=str,
                        default='',
                        help='Directory containing shared pre-trained model checkpoint')
    parser.add_argument('--pretraining_only_first_run',
                        action='store_true',
                        help='Only do pretraining in first run, then share the model')
    parser.add_argument('--pretraining_config',
                        type=str,
                        default='',
                        help='Path to YAML config file for pretraining (if different from regular configs)')
    parser.add_argument('--fresh_pretraining',
                        action='store_true',
                        help='Do fresh pretraining for each configuration (default: False)')
    parser.add_argument('--individual_pretraining_epochs',
                        type=int,
                        default=None,
                        help='Override pretraining epochs for individual runs when using fresh_pretraining')
    
    return parser.parse_args()

def load_config_list(config_list_path):
    """Load a list of configuration file paths from a JSON file."""
    try:
        with open(config_list_path, 'r') as f:
            config_list = json.load(f)
            
        if not isinstance(config_list, list):
            print(f"Error: {config_list_path} must contain a JSON array of configuration file paths")
            sys.exit(1)
            
        # Validate that all files exist
        for config_path in config_list:
            if not os.path.exists(config_path):
                print(f"Error: Configuration file {config_path} does not exist")
                sys.exit(1)
                
        return config_list
        
    except Exception as e:
        print(f"Error loading config list: {e}")
        sys.exit(1)

def verify_compatible_pretraining_params(configs):
    """Verify that all configs have compatible pretraining parameters."""
    
    if not configs:
        return True
    
    # Extract pretraining parameters from each config
    pretraining_params_list = []
    for config_path in configs:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
            # Create a dictionary of pretraining-relevant parameters
            pretraining_params = {
                'net': config.get('net', 'convnext_tiny_26'),
                'num_features': config.get('num_features', 0),
                'activation': config.get('activation', 'gumbel_softmax'),
                'use_mid_layers': config.get('use_mid_layers', False),
                'num_stages': config.get('num_stages', 2),
                'dataset': config.get('dataset', 'CUB-200-2011')
            }
            pretraining_params_list.append(pretraining_params)
    
    # Compare each config's pretraining params with the first one
    reference = pretraining_params_list[0]
    for i, params in enumerate(pretraining_params_list[1:], 1):
        for key in reference:
            if params[key] != reference[key]:
                print(f"Warning: Config {i+1} has different pretraining parameter '{key}' "
                      f"({params[key]} vs {reference[key]})")
                return False
                
    return True

def create_namespace_from_config(yaml_config_path, run_index, base_log_dir=None, gpu_ids=None, shared_pretrained_dir=None):
    """
    Create a namespace object with configuration parameters from a YAML file.
    This is an alternative to using argparse and get_args().
    
    Args:
        yaml_config_path: Path to the YAML configuration file
        run_index: Index of the current run
        base_log_dir: Base directory for logs
        gpu_ids: GPU IDs to use
        shared_pretrained_dir: Directory with shared pretrained model
        
    Returns:
        An argparse.Namespace object with configuration parameters
    """
    # Load configuration from YAML file
    with open(yaml_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create a namespace object
    args = argparse.Namespace()
    
    # Fill namespace with configuration parameters
    for key, value in config.items():
        setattr(args, key, value)
    
    # Set default values for required parameters if not present
    if not hasattr(args, 'log_dir'):
        args.log_dir = './runs/pipnet'
    if not hasattr(args, 'dataset'):
        args.dataset = 'CUB-200-2011'
    if not hasattr(args, 'batch_size'):
        args.batch_size = 64
    if not hasattr(args, 'net'):
        args.net = 'convnext_tiny_26'
    if not hasattr(args, 'state_dict_dir_net'):
        args.state_dict_dir_net = ''
    
    # Update log directory if base_log_dir is provided
    if base_log_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_name = os.path.splitext(os.path.basename(yaml_config_path))[0]
        args.log_dir = os.path.join(base_log_dir, f"{timestamp}_{run_index}_{config_name}")
    
    # Override GPU IDs if specified
    if gpu_ids:
        args.gpu_ids = gpu_ids
    
    # Set shared pretrained model directory if provided
    if shared_pretrained_dir:
        args.shared_pretrained_dir = shared_pretrained_dir
        # Skip pretraining since we'll use the shared pretrained model
        args.epochs_pretrain = 0
    
    # Keep track of the source config file
    args.source_config = yaml_config_path
    args.config = yaml_config_path
    
    return args

def run_all_configs(cmd_args):
    """Run all configurations sequentially."""
    config_list = load_config_list(cmd_args.config_list)
    
    print(f"Found {len(config_list)} configurations to run")
    
    # Verify that pretraining parameters are compatible if using shared pretraining
    if (cmd_args.shared_pretrained_dir or cmd_args.pretraining_only_first_run) and not cmd_args.fresh_pretraining:
        compatible = verify_compatible_pretraining_params(config_list)
        if not compatible:
            print("Warning: Configurations have incompatible pretraining parameters.")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                sys.exit(1)
    
    # Create the base log directory if it doesn't exist
    if not os.path.exists(cmd_args.base_log_dir):
        os.makedirs(cmd_args.base_log_dir)
    
    # Save a copy of the config list for reference
    with open(os.path.join(cmd_args.base_log_dir, 'config_list.json'), 'w') as f:
        json.dump(config_list, f, indent=2)
    
    # Track results for summary
    results = []
    
    # Dictionary to store paths to already completed pretraining checkpoints
    # Key: tuple(seed, num_stages, num_features), Value: path to 'net_pretrained'
    pretrained_checkpoints = {}
    
    # Run each full training configuration
    for i, config_path in enumerate(config_list):
        print(f"\n{'='*80}")
        print(f"Starting run {i+1}/{len(config_list)}: {config_path}")
        print(f"{'='*80}\n")

        run_start_time = time.time()
        current_run_shared_pretrained_dir = None # Reset for each run
        perform_pretraining_this_run = False # Flag to track if this run does the pretraining

        try:
            # Load the specific config for this run to extract pretraining keys
            with open(config_path, 'r') as f:
                current_config_data = yaml.safe_load(f)

            # --- NEW: Extract pretraining key parameters ---
            seed = current_config_data.get('seed') # Default seed if not specified
            num_stages = current_config_data.get('num_stages') # Default num_stages
            num_features = current_config_data.get('num_features') # Default num_features (means using backbone output channels)
            use_mid_layers = current_config_data.get('use_mid_layers', True)

            # Ensure num_stages is only relevant if use_mid_layers is True
            if not use_mid_layers:
                num_stages = -1 # Use a placeholder if mid_layers are not used

            pretrain_key = (seed, num_stages, num_features)

            # Create namespace for this run (as before)
            run_args = create_namespace_from_config(
                config_path,
                run_index=i+1,
                base_log_dir=cmd_args.base_log_dir,
                gpu_ids=cmd_args.gpu_ids
            )

            # Priority 1: Explicitly provided shared directory
            if cmd_args.shared_pretrained_dir:
                run_args.shared_pretrained_dir = cmd_args.shared_pretrained_dir
                run_args.epochs_pretrain = 0  # Don't pretrain if loading explicit path
                print(f"INFO: Using explicitly provided shared pretrain model: {run_args.shared_pretrained_dir}")
                current_run_shared_pretrained_dir = run_args.shared_pretrained_dir
                perform_pretraining_this_run = False

            # Priority 2: Fresh pretraining requested (only if no explicit path was given)
            elif cmd_args.fresh_pretraining:
                run_args.shared_pretrained_dir = '' # Ensure no loading
                # Override pretraining epochs if specified (optional)
                if cmd_args.individual_pretraining_epochs is not None:
                    run_args.epochs_pretrain = cmd_args.individual_pretraining_epochs
                print(f"INFO: Fresh pretraining requested. Performing pretraining (if epochs > 0) for this run.")
                perform_pretraining_this_run = run_args.epochs_pretrain > 0

            # Priority 3: Key-based shared pretraining (only if no explicit path and no fresh pretraining)
            elif pretrain_key in pretrained_checkpoints:
                # Found an existing checkpoint for this key
                run_args.shared_pretrained_dir = pretrained_checkpoints[pretrain_key]
                run_args.epochs_pretrain = 0 # Don't pretrain again
                print(f"INFO: Found shared pretrain checkpoint for key {pretrain_key}. Loading from: {run_args.shared_pretrained_dir}")
                current_run_shared_pretrained_dir = run_args.shared_pretrained_dir
                perform_pretraining_this_run = False

            # Priority 4: Perform pretraining for this key (no explicit path, no fresh pretraining, no existing key)
            else:
                run_args.shared_pretrained_dir = '' # Ensure it doesn't load anything accidentally
                print(f"INFO: No shared pretrain checkpoint found for key {pretrain_key}. Performing pretraining (if epochs > 0) in run: {run_args.log_dir}")
                perform_pretraining_this_run = run_args.epochs_pretrain > 0

            # Log the actual configuration being used (as before)
            print(f"Running with configuration:")
            for arg in vars(run_args):
                val = getattr(run_args, arg)
                if isinstance(val, str) and len(val) > 100:
                    # Truncate long strings
                    print(f"  {arg}: {val[:50]}...{val[-50:]}")
                else:
                    print(f"  {arg}: {val}")

            # Run PIPNet with these arguments (as before)
            run_pipnet(run_args)

            run_status = "completed"

            # Store checkpoint path if pretraining was done successfully ---
            if perform_pretraining_this_run and pretrain_key not in pretrained_checkpoints:
                 # Check if the pretraining checkpoint was actually created
                expected_checkpoint_path = os.path.join(run_args.log_dir, "checkpoints", "net_pretrained")
                if os.path.exists(expected_checkpoint_path):
                    pretrained_checkpoints[pretrain_key] = expected_checkpoint_path
                    print(f"INFO: Stored pretrained checkpoint for key {pretrain_key} at: {expected_checkpoint_path}")
                else:
                     print(f"WARNING: Pretraining was expected for key {pretrain_key} in {run_args.log_dir}, but checkpoint file not found at {expected_checkpoint_path}. It will be re-run if encountered again.")

        except Exception as e:
            run_status = f"failed: {str(e)}"
            print(f"Error during run {i+1}: {e}")
            import traceback
            traceback.print_exc()

            if not cmd_args.continue_on_error:
                print("Aborting remaining runs due to error")
                sys.exit(1)

        run_end_time = time.time()
        run_duration = run_end_time - run_start_time

        # Record the results
        results.append({
            "run_index": i+1,
            "config_path": config_path,
            "status": run_status,
            "duration": run_duration,
            "log_dir": getattr(run_args, 'log_dir', None), # Use getattr for safety
            "pretrain_key": pretrain_key,
            "pretraining_run": perform_pretraining_this_run,
            "loaded_checkpoint": current_run_shared_pretrained_dir
        })

        # Print summary after each run
        print(f"\n{'='*80}")
        print(f"Run {i+1}/{len(config_list)} {run_status}")
        print(f"Duration: {run_duration:.2f} seconds ({run_duration/60:.2f} minutes)")
        print(f"{'='*80}\n")
    
    # Print final summary
    print("\nAll runs completed. Summary:")
    for result in results:
        print(f"Run {result['run_index']}: {result['status']} in {result['duration']/60:.2f} minutes")
        if result['log_dir']:
            print(f"  Log directory: {result['log_dir']}")
    
    # Save summary to file
    summary_path = os.path.join(cmd_args.base_log_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Summary saved to {summary_path}")

if __name__ == "__main__":
    cmd_args = parse_command_line_args()
    run_all_configs(cmd_args)