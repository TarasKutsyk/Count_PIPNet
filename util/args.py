import os
import argparse
import pickle
import numpy as np
import random
import torch
import torch.optim
import yaml

"""
    Utility functions for handling parsed arguments

"""
def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser('Train a PIP-Net')
    parser.add_argument('--config',
                    type=str,
                    default='',
                    help='Path to YAML config file')

    parser.add_argument('--dataset',
                        type=str,
                        default='CUB-200-2011',
                        help='Data set on PIP-Net should be trained')
    parser.add_argument('--validation_size',
                        type=float,
                        default=0.,
                        help='Split between training and validation set. Can be zero when there is a separate test or validation directory. Should be between 0 and 1. Used for partimagenet (e.g. 0.2)')
    parser.add_argument('--net',
                        type=str,
                        default='convnext_tiny_26',
                        help='Base network used as backbone of PIP-Net. Default is convnext_tiny_26 with adapted strides to output 26x26 latent representations. Other option is convnext_tiny_13 that outputs 13x13 (smaller and faster to train, less fine-grained). Pretrained network on iNaturalist is only available for resnet50_inat. Options are: resnet18, resnet34, resnet50, resnet50_inat, resnet101, resnet152, convnext_tiny_26 and convnext_tiny_13.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='Batch size when training the model using minibatch gradient descent. Batch size is multiplied with number of available GPUs')
    parser.add_argument('--batch_size_pretrain',
                        type=int,
                        default=128,
                        help='Batch size when pretraining the prototypes (first training stage)')
    parser.add_argument('--epochs',
                        type=int,
                        default=60,
                        help='The number of epochs PIP-Net should be trained (second training stage)')
    parser.add_argument('--epochs_pretrain',
                        type=int,
                        default = 10,
                        help='Number of epochs to pre-train the prototypes (first training stage). Recommended to train at least until the align loss < 1'
                        )
    parser.add_argument('--optimizer',
                        type=str,
                        default='Adam',
                        help='The optimizer that should be used when training PIP-Net')
    parser.add_argument('--lr',
                        type=float,
                        default=0.05, 
                        help='The optimizer learning rate for training the weights from prototypes to classes')
    parser.add_argument('--lr_block',
                        type=float,
                        default=0.0005, 
                        help='The optimizer learning rate for training the last conv layers of the backbone')
    parser.add_argument('--lr_net',
                        type=float,
                        default=0.0005, 
                        help='The optimizer learning rate for the backbone. Usually similar as lr_block.') 
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0.0,
                        help='Weight decay used in the optimizer')
    parser.add_argument('--disable_cuda',
                        action='store_true',
                        help='Flag that disables GPU usage if set')
    parser.add_argument('--log_dir',
                        type=str,
                        default='./runs/run_pipnet',
                        help='The directory in which train progress should be logged')
    parser.add_argument('--num_features',
                        type=int,
                        default = 0,
                        help='Number of prototypes. When zero (default) the number of prototypes is the number of output channels of backbone. If this value is set, then a 1x1 conv layer will be added. Recommended to keep 0, but can be increased when number of classes > num output channels in backbone.')
    parser.add_argument('--image_size',
                        type=int,
                        default=224,
                        help='Input images will be resized to --image_size x --image_size (square). Code only tested with 224x224, so no guarantees that it works for different sizes.')
    parser.add_argument('--state_dict_dir_net',
                        type=str,
                        default='',
                        help='The directory containing a state dict with a pretrained PIP-Net. E.g., ./runs/run_pipnet/checkpoints/net_pretrained')
    parser.add_argument('--freeze_epochs',
                        type=int,
                        default = 10,
                        help='Number of epochs where pretrained features_net will be frozen while training classification layer (and last layer(s) of backbone)'
                        )
    parser.add_argument('--dir_for_saving_images',
                        type=str,
                        default='visualization_results',
                        help='Directoy for saving the prototypes and explanations')
    parser.add_argument('--disable_pretrained',
                        action='store_true',
                        help='When set, the backbone network is initialized with random weights instead of being pretrained on another dataset).'
                        )
    parser.add_argument('--weighted_loss',
                        action='store_true',
                        help='Flag that weights the loss based on the class balance of the dataset. Recommended to use when data is imbalanced. ')
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        help='Random seed. Note that there will still be differences between runs due to nondeterminism. See https://pytorch.org/docs/stable/notes/randomness.html')
    parser.add_argument('--gpu_ids',
                        type=str,
                        default='',
                        help='ID of gpu. Can be separated with comma')
    parser.add_argument('--num_workers',
                        type=int,
                        default=8,
                        help='Num workers in dataloaders.')
    parser.add_argument('--bias',
                        action='store_true',
                        help='Flag that indicates whether to include a trainable bias in the linear classification layer.'
                        )
    parser.add_argument('--extra_test_image_folder',
                        type=str,
                        default='./experiments',
                        help='Folder with images that PIP-Net will predict and explain, that are not in the training or test set. E.g. images with 2 objects or OOD image. Images should be in subfolder. E.g. images in ./experiments/images/, and argument --./experiments')
    
    parser.add_argument('--pretrained_checkpoints_dir',
                    type=str,
                    default='',
                    help='Directory to search for pretrained checkpoints before using the current log_dir')
    parser.add_argument('--resume_training',
                    action='store_true',
                    help='Resume training from the last checkpoint')

    # New arguments for CountPIPNet  
    parser.add_argument('--model',
                        type=str,
                        default='pipnet',
                        help='Model type to use: "pipnet" (original) or "count_pipnet"')
    parser.add_argument('--use_mid_layers', action='store_true', 
                        help='Use only middle layers of the backbone')
    parser.add_argument('--num_stages', type=int, default=3,
                        help='Number of stages to use when using middle layers')
    parser.add_argument('--max_count',
                        type=int,
                        default=3,
                        help='Maximum count value to consider for CountPIPNet (counts >= max_count get mapped to max_count)')
    parser.add_argument('--use_ste',
                        type=eval,
                        choices=[True, False],
                        default=False,
                        help='Whether to use Straight-Through Estimator for full training of CountPiPNet')
    parser.add_argument(f'--activation',
                        type=str,
                        default='gumbel_softmax',
                        help='Prototype-enforcing function to apply to the backbone feature maps. Can be either softmax or gumbel_softmax (default)')
    
    # Parse known args first to get the config file path
    known_args, _ = parser.parse_known_args()
    
    # If config file is provided, load it and update defaults
    if known_args.config and os.path.exists(known_args.config):
        with open(known_args.config, 'r') as config_file:
            config = yaml.safe_load(config_file)
            print("Using the config parameters as default. The provided command-line arguments will still have precedence if provided.")
            
            # Create a map from arg name to option string
            arg_to_option = {}
            for action in parser._actions:
                if action.dest != 'help':  # Skip the help action
                    arg_to_option[action.dest] = action.dest
            
            # Update parser defaults with values from config file
            config_updates = {}
            for key, value in config.items():
                if key in arg_to_option:
                    config_updates[key] = value
                else:
                    print(f"Warning: Config contains unknown parameter '{key}'")
            
            if config_updates:
                parser.set_defaults(**config_updates)
    
    args = parser.parse_args()
    if len(args.log_dir.split('/'))>2:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)

    return args


def save_args(args: argparse.Namespace, directory_path: str) -> None:
    """
    Save the arguments in the specified directory as
        - a text file called 'args.txt'
        - a pickle file called 'args.pickle'
    :param args: The arguments to be saved
    :param directory_path: The path to the directory where the arguments should be saved
    """
    # If the specified directory does not exists, create it
    if not os.path.isdir(directory_path):
        os.mkdir(directory_path)
    # Save the args in a text file
    with open(directory_path + '/args.txt', 'w') as f:
        for arg in vars(args):
            val = getattr(args, arg)
            if isinstance(val, str):  # Add quotation marks to indicate that the argument is of string type
                val = f"'{val}'"
            f.write('{}: {}\n'.format(arg, val))
    # Pickle the args for possible reuse
    with open(directory_path + '/args.pickle', 'wb') as f:
        pickle.dump(args, f)                                                                               
    
def get_optimizer_nn(net, args: argparse.Namespace):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Create parameter groups
    params_to_freeze = []
    params_to_train = []
    params_backbone = []
    
    # Handle different network architectures
    if 'convnext' in args.net:
        if hasattr(args, 'use_mid_layers') and args.use_mid_layers:
            group_convnext_mid_layer_parameters(
                net.module._net,
                params_to_train,
                params_to_freeze,
                params_backbone,
                args.num_stages if hasattr(args, 'num_stages') else 2
            )
        else:
            # Standard ConvNeXt without mid-layers
            for name, param in net.module._net.named_parameters():
                if 'features.7.2' in name: 
                    params_to_train.append(param)
                elif 'features.7' in name or 'features.6' in name:
                    params_to_freeze.append(param)
                else:
                    params_backbone.append(param)
    elif 'resnet50' in args.net:
        # ResNet parameter grouping (unchanged)
        for name, param in net.module._net.named_parameters():
            if 'layer4.2' in name:
                params_to_train.append(param)
            elif 'layer4' in name or 'layer3' in name:
                params_to_freeze.append(param)
            elif 'layer2' in name:
                params_backbone.append(param)
            else:
                param.requires_grad = False
    else:
        print("Network is not ResNet or ConvNext.", flush=True)     
    
    # Classification layer parameters (unchanged)
    classification_weight = []
    classification_bias = []
    for name, param in net.module._classification.named_parameters():
        if 'weight' in name:
            classification_weight.append(param)
        elif 'multiplier' in name:
            param.requires_grad = False
        else:
            if args.bias:
                classification_bias.append(param)
    
    # Define parameter groups with weight decay only for classification weights
    paramlist_net = [
        {"params": params_backbone, "lr": args.lr_net, "weight_decay": 0.0},
        {"params": params_to_freeze, "lr": args.lr_block, "weight_decay": 0.0},
        {"params": params_to_train, "lr": args.lr_block, "weight_decay": 0.0},
        {"params": net.module._add_on.parameters(), "lr": args.lr_block*10., "weight_decay": 0.0}
    ]
                
    # Apply weight decay only to classification weights, not to bias
    paramlist_classifier = [
        {"params": classification_weight, "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": classification_bias, "lr": args.lr, "weight_decay": 0.0},
    ]
          
    # Create optimizers
    if args.optimizer == 'Adam':
        optimizer_net = torch.optim.AdamW(paramlist_net, lr=args.lr, weight_decay=0.0)
        optimizer_classifier = torch.optim.AdamW(paramlist_classifier, lr=args.lr, weight_decay=0.0)
        return optimizer_net, optimizer_classifier, params_to_freeze, params_to_train, params_backbone
    else:
        raise ValueError("this optimizer type is not implemented")

def group_convnext_mid_layer_parameters(model, params_to_train, params_to_freeze, params_backbone, num_stages=2):
    """
    Group ConvNeXt parameters when using mid-layers based on the observed architecture:
    
    ConvNeXt has an alternating stage pattern:
    - Even stages (0,2,4,6): Transform dimensions (stem, transitions)
    - Odd stages (1,3,5,7): Process content (maintain dimensions)
    
    Parameters are grouped by training priority:
    1. params_to_train: The final stage (highest training priority)
    2. params_to_freeze: Intermediate stages (medium priority)
    3. params_backbone: Early stages (lowest priority)
    """
    # Track parameter counts for verification
    counts = {'train': 0, 'freeze': 0, 'backbone': 0}
    stage_assignments = {}
    
    # Get the output stage number
    output_stage = num_stages
    
    # Group parameters based on stage number
    for name, param in model.named_parameters():
        # Extract stage number from parameter name
        if not name.startswith('features.'):
            continue
            
        parts = name.split('.')
        if len(parts) < 2 or not parts[1].isdigit():
            # Handle parameters without clear stage numbering
            params_backbone.append(param)
            counts['backbone'] += 1
            continue
            
        stage_num = int(parts[1])
        
        # Record which group this stage is assigned to
        if stage_num not in stage_assignments:
            stage_assignments[stage_num] = set()
        
        # Assign parameters to groups based on stage number
        if stage_num == output_stage:
            # The stage producing the final output gets highest priority
            params_to_train.append(param)
            counts['train'] += 1
            stage_assignments[stage_num].add('train')

            # print(f"Assigning {name} to train group", flush=True)
        elif stage_num == output_stage - 1:
            # The immediate previous stages get medium priority
            params_to_freeze.append(param)
            counts['freeze'] += 1
            stage_assignments[stage_num].add('freeze')
            # print(f"Assigning {name} to freeze group", flush=True)
        else:
            # Earlier stages get lowest priority
            params_backbone.append(param)
            counts['backbone'] += 1
            stage_assignments[stage_num].add('backbone')
            # print(f"Assigning {name} to backbone group", flush=True)
    
    # Print summary information
    print(f"\nParameter grouping for ConvNeXt with {num_stages} stages:", flush=True)
    print(f"Total parameters: {counts['train']} trainable, {counts['freeze']} freezable, {counts['backbone']} backbone", flush=True)
    
    # Print stage assignments for clarity
    print("\nStage assignments:", flush=True)
    for stage in sorted(stage_assignments.keys()):
        groups = ", ".join(stage_assignments[stage])
        print(f"  Stage {stage}: {groups}", flush=True)
    
    return params_to_train, params_to_freeze, params_backbone