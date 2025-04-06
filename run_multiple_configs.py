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
    
    # Handle shared pretraining if requested and not using fresh pretraining
    shared_pretrained_dir = cmd_args.shared_pretrained_dir
    
    if cmd_args.pretraining_only_first_run and not cmd_args.fresh_pretraining and not shared_pretrained_dir:
        print("\n=== Performing shared pretraining in first run ===")
        
        # Use pretraining config if provided, otherwise use first config
        pretraining_config_path = cmd_args.pretraining_config if cmd_args.pretraining_config else config_list[0]
        
        # Create a dedicated pretraining directory
        pretraining_dir = os.path.join(cmd_args.base_log_dir, "pretraining")
        if not os.path.exists(pretraining_dir):
            os.makedirs(pretraining_dir)
        
        # Create namespace for pretraining
        pretraining_args = create_namespace_from_config(
            pretraining_config_path, 
            run_index="pretrain",
            base_log_dir=pretraining_dir, 
            gpu_ids=cmd_args.gpu_ids
        )
        
        # Override log directory
        pretraining_args.log_dir = pretraining_dir
        
        # Print pretraining configuration
        print("Pretraining with configuration:")
        for key, value in vars(pretraining_args).items():
            print(f"  {key}: {value}")
        
        # Run pretraining
        run_pipnet(pretraining_args)
        
        # Now set shared_pretrained_dir to the pretraining checkpoint
        shared_pretrained_dir = os.path.join(pretraining_dir, "checkpoints", "net_pretrained")
        print(f"Using shared pretrained model from: {shared_pretrained_dir}")
    
    # Run each full training configuration
    for i, config_path in enumerate(config_list):
        print(f"\n{'='*80}")
        print(f"Starting run {i+1}/{len(config_list)}: {config_path}")
        print(f"{'='*80}\n")
        
        run_start_time = time.time()
        
        try:
            # Create namespace for this run
            run_args = create_namespace_from_config(
                config_path, 
                run_index=i+1,
                base_log_dir=cmd_args.base_log_dir, 
                gpu_ids=cmd_args.gpu_ids
            )
            
            # Handle pretraining options
            if cmd_args.fresh_pretraining:
                # Force fresh pretraining for this run
                run_args.shared_pretrained_dir = ''
                # Override pretraining epochs if specified
                if cmd_args.individual_pretraining_epochs is not None:
                    run_args.epochs_pretrain = cmd_args.individual_pretraining_epochs
                print(f"Using fresh pretraining with {run_args.epochs_pretrain} epochs")
            elif shared_pretrained_dir:
                # Use shared pretraining
                run_args.shared_pretrained_dir = shared_pretrained_dir
                run_args.epochs_pretrain = 0
                print(f"Using shared pretrained model from: {shared_pretrained_dir}")
            
            # Log the actual configuration being used
            print(f"Running with configuration:")
            for arg in vars(run_args):
                val = getattr(run_args, arg)
                if isinstance(val, str) and len(val) > 100:
                    # Truncate long strings
                    print(f"  {arg}: {val[:50]}...{val[-50:]}")
                else:
                    print(f"  {arg}: {val}")
            
            # Run PIPNet with these arguments
            run_pipnet(run_args)
            
            run_status = "completed"
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
            "log_dir": run_args.log_dir if 'run_args' in locals() else None
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