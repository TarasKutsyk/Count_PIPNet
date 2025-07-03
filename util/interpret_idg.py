import os
import torch
import torch.nn as nn
import argparse
import pickle
import random
import numpy as np
from pathlib import Path
from PIL import Image
import plotly.express as px
import json
import matplotlib.pyplot as plt
from captum.attr import visualization as viz
import matplotlib.cm as cm

# Add base path to system path
# This is needed to import modules from the project root
def get_base_folder(project_root="PIPNet"):
    current_dir = os.getcwd()
    while True:
        if os.path.basename(current_dir) == project_root:
            break
        parent = os.path.dirname(current_dir)
        if parent == current_dir:
            raise RuntimeError(f"Project root {project_root} not found.")
        current_dir = parent
    return Path(current_dir)

base_path = get_base_folder()
import sys
sys.path.append(str(base_path))

from util.data import get_dataloaders, get_data
from util.saliencyMethods import IDG, IG
from pipnet.pipnet import get_pipnet
from pipnet.count_pipnet import get_count_network

# --- Configuration ---
USE_GLOBAL_CFG = True

GLOBAL_CFG = {
    'run_dir': str(base_path / 'runs' / 'pipnet' / '20250407_021157_15_pipnet_s21_stage7_p16'),
    'checkpoint_name': 'net_trained_best',
    'saliency_map_mode': 'prototype', # 'logit' or 'prototype'

    # Dataset params
    'dataset': 'geometric_shapes_no_noise',
    'image_path': 'class_6/img_0000.png',  # Set to a specific image path or leave as None for random sampling
    'sample_from_classes': [ # List of class names to sample from (for random sampling per class)
        'class_1',
        # 'class_2',
        # 'class_3',
        # 'class_4',
        # 'class_5',
        # 'class_6',
        # 'class_7',
        # 'class_8',
        # 'class_9',
    ], 
    'target_class': None, # Set to a specific class index or leave as None

    # Attribution method params
    'attr_method': 'ig', # 'ig', 'lig' or 'idg'
    'idg_steps': 200, # Number of interpolated samples
    'idg_baseline': 0.0, # Baseline value for interpolation
    'idg_batch_size': 10, # Number of images (x_i interpolated samples) to process in parallel

    # Visualization params
    'show_plots': True, # Whether to display the plots directly
    'prototype_activation_threshold': 0.01, # Min activation for a prototype to be considered for saliancy map
    'percentile': 98, # Percentile for clipping the attribution map
    'plot_individual_prototypes': True, # Whether to plot each prototype's attribution map separately
    'use_captum_viz': True, # If True, use Captum's vizualizer for individual prototype maps
    'alpha_overlay': 0.75, # Alpha for the blended heatmap in Captum visualization

    'random_seed': 42, # Seed for random sampling,
    'gpu_id': 0,
    'output_dir': str(base_path / 'visualizations' / 'idg_interpretations'),
}

def ATTR_FUNC(attr_method):
    if attr_method == 'ig':
        return lambda input, model, steps, batch_size, baseline, device, target_class: IG(input, model, steps, batch_size, 1.0, baseline, device, target_class)
    elif attr_method == 'lig':
        return lambda input, model, steps, batch_size, baseline, device, target_class: IG(input, model, steps, batch_size, 0.9, baseline, device, target_class)
    elif attr_method == 'idg':
        return lambda input, model, steps, batch_size, baseline, device, target_class: IDG(input, model, steps, batch_size, baseline, device, target_class)
    else:
        raise ValueError(f"Unknown attribution method: {attr_method}")

# --- PIP-Net Wrapper for IDG ---
class PIPNetWrapper(nn.Module):
    def __init__(self, model):
        super(PIPNetWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        _, _, out = self.model(x)
        return out

class PIPNetPrototypeWrapper(nn.Module):
    def __init__(self, model):
        super(PIPNetPrototypeWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        _, pooled, _ = self.model(x)
        return pooled

# --- Model and Data Loading ---
def load_model_for_interpretation(run_dir, checkpoint_name, gpu_id):
    metadata_dir = os.path.join(run_dir, 'metadata')
    args_path = os.path.join(metadata_dir, 'args.pickle')

    with open(args_path, 'rb') as f:
        args = pickle.load(f)

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')

    args.log_dir = run_dir
    trainloader, trainloader_pretraining, trainloader_normal, \
	trainloader_normal_augment, projectloader, testloader, test_projectloader, classes = get_dataloaders(args, device, base_path)

    if hasattr(args, 'model') and args.model == 'count_pipnet':
        net, num_prototypes = get_count_network(
			num_classes=len(classes), 
			args=args,
			device=device,
			max_count=getattr(args, 'max_count', 3),
			use_ste=getattr(args, 'use_ste', False))
    else:
        net, _ = get_pipnet(len(classes), args)

    net = net.to(device)

    with torch.no_grad():
        xs1, _, _ = next(iter(trainloader))
        xs1 = xs1.to(device)
        features = net._net(xs1)
        proto_features = net._add_on(features)
        args.wshape = proto_features.shape[-1]

    net = nn.DataParallel(net, device_ids=[gpu_id])
    checkpoint_path = os.path.join(run_dir, 'checkpoints', checkpoint_name)
    net.load_state_dict(torch.load(checkpoint_path, map_location=device)['model_state_dict'])
    net.eval()

    return net, testloader, classes, device, args

# --- Helper Functions ---
def normalize_attribution_map(attr_map_2d, percentile, proto_idx_for_log):
    """
    Normalizes a 2D attribution map robustly.
    First, it tries to normalize based on a percentile. If the dynamic range is too small,
    it falls back to normalizing by the global maximum to avoid zeroing out the map.
    """
    vmax_percentile = np.percentile(attr_map_2d, percentile)
    vmin = np.min(attr_map_2d)
    denominator = vmax_percentile - vmin

    if denominator > 1e-3:
        clipped_map = np.clip((attr_map_2d - vmin) / denominator, 0, 1)
    else:
        print(f'WARNING: Percentile-based denominator ({denominator:.2e}) is too small for prototype {proto_idx_for_log}.' +
              f'Falling back to global max normalization.')
        vmax_global = np.max(attr_map_2d)
        denominator_global = vmax_global - vmin
        if denominator_global > 1e-5:
            clipped_map = np.clip((attr_map_2d - vmin) / denominator_global, 0, 1)
        else:
            clipped_map = np.zeros_like(attr_map_2d)
    return clipped_map

# --- Main Interpretation Logic ---
def interpret(config):
    os.makedirs(config['output_dir'], exist_ok=True)

    # Load model and data
    net, testloader, classes, device, args = load_model_for_interpretation(
        config['run_dir'], config['checkpoint_name'], config['gpu_id']
    )

    # --- Interpretation Mode ---
    if config['saliency_map_mode'] == 'prototype':
        if not config['image_path']:
            raise ValueError("Image path must be provided for 'prototype' saliency map mode.")
        interpret_prototypes(net, testloader, classes, device, config)
    elif config['saliency_map_mode'] == 'logit':
        interpret_logits_for_dataset(net, testloader, classes, device, config)
    else:
        raise ValueError(f"Unknown saliency_map_mode: {config['saliency_map_mode']}")

def interpret_prototypes(net, testloader, classes, device, config):
    """Interpret prototype activations for one or more images.

    If config contains ``sample_from_classes`` as a list of class names, the function
    randomly samples (with a reproducible seed) **one** image from **each** requested
    class and interprets them sequentially. In that case the function calls itself
    recursively for each sampled image with ``sample_from_classes`` cleared so that
    the normal single-image logic is executed.  When the key is absent or ``None``,
    the original single-image path in ``config['image_path']`` is used.
    """

    # ------------------------------------------------------------------
    # Optional: handle random sampling of a single image **per class**
    # ------------------------------------------------------------------
    sample_classes = config.get('sample_from_classes')
    if sample_classes:
        # Ensure deterministic selection when a seed is provided
        seed = config.get('random_seed', 42)
        random.seed(seed)

        dataset_test_root = base_path / 'data' / config['dataset'] / 'dataset' / 'test'

        for cls_name in sample_classes:
            if cls_name not in classes:
                print(f"Warning: requested class '{cls_name}' not in class list – skipping.")
                continue

            class_dir = dataset_test_root / cls_name
            if not class_dir.exists():
                print(f"Warning: class directory '{class_dir}' does not exist – skipping.")
                continue

            # Collect files in class directory
            img_paths = [p for p in class_dir.iterdir() if p.is_file()]
            if not img_paths:
                print(f"Warning: no images found for class '{cls_name}' – skipping.")
                continue

            chosen_path = random.choice(img_paths)

            # Create a shallow copy of config with updated image path and sampling disabled
            new_config = dict(config)
            # Store relative path so downstream naming logic (parent + stem) works unchanged
            new_config['image_path'] = str(chosen_path.relative_to(dataset_test_root))
            new_config['sample_from_classes'] = None  # prevent nested sampling

            print(f"Selected random image '{chosen_path.name}' from class '{cls_name}'.")
            interpret_prototypes(net, testloader, classes, device, new_config)

        # After processing all classes we're done.
        return

    print("--- Interpreting Prototype Activations for a Single Image ---")
    attr_func = ATTR_FUNC(config['attr_method'])
    print("Using attribution method: " + config['attr_method'])

    # Load and transform the image
    try:
        # Construct full path.
        image_path = base_path / 'data' / config['dataset'] / 'dataset' / 'test' / config['image_path']
        image = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return

    # Infer class from path
    try:
        class_name_from_path = image_path.parent.name
        target_class_idx = classes.index(class_name_from_path)
        print(f"Inferred class '{class_name_from_path}' (index: {target_class_idx}) from path.")
    except (ValueError, IndexError):
        print(f"Warning: Could not infer class from path '{class_name_from_path}'. Using 0.")
        target_class_idx = 0

    transform = testloader.dataset.transform
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Get prototype activations for the image
    _, pooled, _ = net(image_tensor)
    pooled = pooled.squeeze(0)

    # Get classification weights
    classification_weights = net.module._classification.weight # shape [num_classes, num_prototypes]

    # Calculate weighted activations for the target class
    if target_class_idx is None:
        # Fallback to simple activation filtering if target_class_idx is not specified
        print("Warning: target_class_idx not specified. Filtering prototypes by activation only.")
        weighted_activations = pooled
    else:
        # Multiply prototype activations by their classification weights for the target class
        target_class_weights = classification_weights[target_class_idx]
        weighted_activations = pooled * target_class_weights

    # Filter for prototypes whose weighted activation is above the threshold
    active_protos_indices = torch.where(weighted_activations > config['prototype_activation_threshold'])[0]
    if len(active_protos_indices) == 0:
        print("No prototypes were active above the threshold. Nothing to interpret.")
        return

    print(f"Found {len(active_protos_indices)} active prototypes.")

    # Wrap model for prototype interpretation
    wrapped_model = PIPNetPrototypeWrapper(net)

    # Get color scale
    colors = px.colors.qualitative.Plotly
    if len(active_protos_indices) > len(colors):
        print(f"Warning: More active prototypes than available colors. Colors will be reused.")

    # Store prototype index and color for legend
    prototype_color_map = []
    # Store individual attribution maps and their weighted activations if plotting them separately
    individual_attribution_maps = []
    individual_weighted_activations = {}

    # --- Generate and Overlay Attribution Maps using Additive Blending ---
    final_rgb_overlay = np.zeros((image_tensor.shape[2], image_tensor.shape[3], 3))
    final_alpha_overlay = np.zeros((image_tensor.shape[2], image_tensor.shape[3], 1))

    for i, proto_idx in enumerate(active_protos_indices):
        print(f"  Generating map for prototype {proto_idx.item()}...")
        attribution_map = attr_func(
            input=image_tensor,
            model=wrapped_model,
            steps=config['idg_steps'],
            batch_size=config.get('idg_batch_size', 10),
            baseline=config['idg_baseline'],
            device=device,
            target_class=proto_idx.item()
        )

        # Process map
        attr_map_np = attribution_map.squeeze(0).detach().cpu().numpy()
        attr_map_np = np.transpose(attr_map_np, (1, 2, 0))
        image_2d = np.sum(np.abs(attr_map_np), axis=2)

        clipped_map = normalize_attribution_map(image_2d, config['percentile'], proto_idx.item())

        # --- Additive Blending ---
        color_hex = colors[i % len(colors)]
        src_rgb = np.array(tuple(int(color_hex.lstrip('#')[j:j+2], 16) / 255.0 for j in (0, 2, 4)))
        
        # Add the weighted color of the current prototype to the RGB overlay
        final_rgb_overlay += src_rgb * clipped_map[..., None]
        
        # The final alpha is the maximum alpha seen at each pixel
        final_alpha_overlay = np.maximum(final_alpha_overlay, clipped_map[..., None])

        # Store for legend
        prototype_color_map.append({'prototype_idx': proto_idx.item(), 'color': color_hex})

        # Store individual map and its weighted activation if needed
        if config.get('plot_individual_prototypes', False):
            # Store the raw numpy attribution map for individual plotting later
            individual_attribution_maps.append({'prototype_idx': proto_idx.item(), 'map': attr_map_np, 'color': color_hex})
            if target_class_idx is not None:
                individual_weighted_activations[proto_idx.item()] = weighted_activations[proto_idx].item()


    # --- Save and Plot --- 
    # Un-normalize original image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
    original_image_unnormalized = image_tensor.squeeze(0) * std + mean
    original_image_unnormalized = original_image_unnormalized.permute(1, 2, 0).detach().cpu().numpy()
    original_image_unnormalized = np.clip(original_image_unnormalized, 0, 1)

    # Save
    # Construct a more descriptive image name that includes the parent folder (e.g. "class_6_img_0000")
    img_path_obj = Path(config['image_path'])
    img_parent = img_path_obj.parent.name if img_path_obj.parent is not None else ''
    img_name = f"{img_parent}_{img_path_obj.stem}" if img_parent else img_path_obj.stem
    # Normalize the RGB overlay to ensure all values are in the [0, 1] range.
    final_rgb_overlay = np.clip(final_rgb_overlay, 0, 1)
    
    # Combine the normalized RGB and the alpha channel to create the final RGBA image.
    final_overlay = np.concatenate((final_rgb_overlay, final_alpha_overlay), axis=-1)

    # --- Save Combined Original and Composite Plot ---
    combined_save_path = os.path.join(config['output_dir'], f"combined_prototypes_{img_name}_{config['attr_method']}.png")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(original_image_unnormalized)
    ax1.set_title(f'Original Image: {img_name}')
    ax1.axis('off')

    ax2.imshow(final_overlay)
    ax2.set_title(f'Composite Prototype Attributions ({len(active_protos_indices)} maps)')
    ax2.axis('off')

    # Create custom legend for the composite plot
    legend_handles = []
    for item in prototype_color_map:
        color_rgb = tuple(int(item['color'].lstrip('#')[j:j+2], 16) / 255.0 for j in (0, 2, 4))
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                         markerfacecolor=color_rgb, markersize=10,
                                         label=f'P{item['prototype_idx']}'))
    ax2.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1, 1), title='Prototypes')
    
    plt.tight_layout()
    fig.savefig(combined_save_path, bbox_inches='tight')
    print(f"Saved combined original and composite prototype attribution map to {combined_save_path}")

    if config.get('show_plots', False):
        plt.show()
    
    plt.close(fig)

    # --- Save Figure Containing All Individual Prototype Maps (if enabled) ---
    if config.get('plot_individual_prototypes', False) and len(individual_attribution_maps) > 0:
        individual_output_dir = os.path.join(config['output_dir'], 'individual_prototypes')
        os.makedirs(individual_output_dir, exist_ok=True)

        num_maps = len(individual_attribution_maps)
        num_cols = 2
        num_rows = (num_maps + num_cols - 1) // num_cols  # Ceiling division

        fig_ind, axes_ind = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 5), squeeze=False)
        axes_ind = axes_ind.flatten()

        # Helper to plot one prototype map on a given axis (supports captum & manual)
        def _plot_single_prototype(ax, proto_item):
            proto_idx = proto_item['prototype_idx']
            attr_map_np = proto_item['map']

            # Edge Case: near-constant attribution map
            if np.abs(np.max(attr_map_np) - np.min(attr_map_np)) < 1e-5:
                ax.imshow(original_image_unnormalized)
                contribution_str = ''
                if proto_idx in individual_weighted_activations:
                    contribution = individual_weighted_activations[proto_idx]
                    contribution_str = f" (Contrib: {contribution:.2f})"

                ax.set_title(f'Prototype {proto_idx} Attribution{contribution_str}\n' + 
                             f'Constant Attribution of {np.mean(attr_map_np):.2f}')
                ax.axis('off')
                return

            if config.get('use_captum_viz', False):
                contribution_str = ""
                if proto_idx in individual_weighted_activations:
                    contribution = individual_weighted_activations[proto_idx]
                    contribution_str = f" (Contrib: {contribution:.2f})"
                
                title = f'P{proto_idx} Attribution{contribution_str}'
                viz.visualize_image_attr(
                    attr=attr_map_np,
                    original_image=original_image_unnormalized,
                    method='blended_heat_map',
                    sign='absolute_value',
                    outlier_perc=100 - config['percentile'],
                    plt_fig_axis=(fig_ind, ax),
                    show_colorbar=True,
                    title=title,
                    alpha_overlay=config['alpha_overlay'],
                    use_pyplot=False
                )
            else:
                image_2d = np.sum(np.abs(attr_map_np), axis=2)
                clipped_map = normalize_attribution_map(image_2d, config['percentile'], proto_idx)

                ax.imshow(original_image_unnormalized)
                overlay = np.zeros((*original_image_unnormalized.shape[:2], 4))

                color_rgb = np.array(tuple(int(proto_item['color'].lstrip('#')[j:j+2], 16) / 255.0 for j in (0, 2, 4)))
                overlay[..., :3] = color_rgb
                overlay[..., 3] = clipped_map
                ax.imshow(overlay)

                contribution_str = ""
                if proto_idx in individual_weighted_activations:
                    contribution = individual_weighted_activations[proto_idx]
                    contribution_str = f" (Contrib: {contribution:.2f})"
                ax.set_title(f'Prototype {proto_idx} Attribution{contribution_str}')
                ax.axis('off')
        # end _plot_single_prototype

        # Populate sub-plots
        for idx, proto_item in enumerate(individual_attribution_maps):
            _plot_single_prototype(axes_ind[idx], proto_item)

        # Remove any unused axes
        for idx in range(num_maps, len(axes_ind)):
            fig_ind.delaxes(axes_ind[idx])

        plt.tight_layout()
        plt.suptitle('Individual Prototype Attribution Maps', y=1.02)

        # Save the complete figure (instead of separate pngs per prototype)
        fig_ind_path = os.path.join(individual_output_dir, f"individual_prototypes_{img_name}_{config['attr_method']}.png")
        fig_ind.savefig(fig_ind_path, bbox_inches='tight')
        print(f"Saved figure with all individual prototype maps to {fig_ind_path}")

    # --- Display Plots (if enabled) ---
    if config.get('show_plots', False):
        # Just display the figure we already created above
        plt.show()
        plt.close(fig_ind)

        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        # ax1.imshow(original_image_unnormalized)
        # ax1.set_title(f'Original Image: {img_name}')
        # ax1.axis('off')

        # ax2.imshow(final_overlay)
        # ax2.set_title(f'Composite Prototype Attributions ({len(active_protos_indices)} maps)')
        # ax2.axis('off')

        # # Create custom legend
        # legend_handles = []
        # for item in prototype_color_map:
        #     color_rgb = tuple(int(item['color'].lstrip('#')[j:j+2], 16) / 255.0 for j in (0, 2, 4))
        #     legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', 
        #                                      markerfacecolor=color_rgb, markersize=10,
        #                                      label=f'P{item['prototype_idx']}'))
        # ax2.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1, 1), title='Prototypes')
        
        # plt.tight_layout()
        # plt.show()

        # if config.get('plot_individual_prototypes', False) and len(individual_attribution_maps) > 0:
        #     num_maps = len(individual_attribution_maps)
        #     num_cols = 2
        #     num_rows = (num_maps + num_cols - 1) // num_cols  # Ceiling division

        #     fig_ind, axes_ind = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 5), squeeze=False)
        #     axes_ind = axes_ind.flatten()

        #     if config.get('use_captum_viz', False):
        #         # --- Captum Visualization Path for Display ---
        #         print("--- Displaying Captum individual prototype visualization ---")
        #         for idx, item in enumerate(individual_attribution_maps):
        #             ax = axes_ind[idx]
        #             attr_map_np = item['map']
                    
        #             # Edge Case: Handle near-zero attribution maps for display
        #             if np.abs(np.max(attr_map_np) - np.min(attr_map_np)) < 1e-5:
        #                 ax.imshow(original_image_unnormalized)
        #                 ax.set_title(f'Prototype {item["prototype_idx"]} (Constant Attribution of {np.mean(attr_map_np):.2f})')
        #                 ax.axis('off')
        #                 continue

        #             contribution_str = ""
        #             if item['prototype_idx'] in individual_weighted_activations:
        #                 contribution = individual_weighted_activations[item['prototype_idx']]
        #                 contribution_str = f" (Contrib: {contribution:.2f})"
                    
        #             title = f'P{item["prototype_idx"]} Attribution{contribution_str}'

        #             _ = viz.visualize_image_attr(
        #                 attr=attr_map_np,
        #                 original_image=original_image_unnormalized,
        #                 method='blended_heat_map',
        #                 sign='absolute_value',
        #                 outlier_perc=100 - config['percentile'],
        #                 plt_fig_axis=(fig_ind, ax),
        #                 show_colorbar=True,
        #                 title=title,
        #                 alpha_overlay=config['alpha_overlay'],
        #                 use_pyplot=False # Important to avoid showing the plot prematurely
        #             )
        #     else:
        #         # --- Original Manual Visualization Path for Display ---
        #         for idx, item in enumerate(individual_attribution_maps):
        #             ax = axes_ind[idx]
        #             attr_map_np = item['map']

        #             # Re-calculate and normalize map for manual visualization
        #             image_2d = np.sum(np.abs(attr_map_np), axis=2)
        #             clipped_map = normalize_attribution_map(image_2d, config['percentile'], item['prototype_idx'])

        #             ax.imshow(original_image_unnormalized)
        #             overlay = np.zeros((*original_image_unnormalized.shape[:2], 4))
        #             color_rgb = np.array(tuple(int(item['color'].lstrip('#')[j:j+2], 16) / 255.0 for j in (0, 2, 4)))
        #             overlay[..., :3] = color_rgb
        #             overlay[..., 3] = clipped_map
        #             ax.imshow(overlay)

        #             contribution_str = ""
        #             if item['prototype_idx'] in individual_weighted_activations:
        #                 contribution = individual_weighted_activations[item['prototype_idx']]
        #                 contribution_str = f" (Contrib: {contribution:.2f})"

        #             ax.set_title(f'Prototype {item["prototype_idx"]} Attribution{contribution_str}')
        #             ax.axis('off')

        #     # Common cleanup for both paths
        #     for idx in range(num_maps, len(axes_ind)):
        #         fig_ind.delaxes(axes_ind[idx])

        #     plt.tight_layout()
        #     plt.suptitle('Individual Prototype Attribution Maps', y=1.02)
        #     plt.show()

def interpret_logits_for_dataset(net, testloader, classes, device, config):
    print("--- Interpreting Logits for a Sample from Each Class ---")
    # Wrap model for logit interpretation
    wrapped_model = PIPNetWrapper(net)

    # Sample one image from each class
    images_by_class = {i: [] for i in range(len(classes))}
    for images, labels in testloader:
        for i in range(len(images)):
            images_by_class[labels[i].item()].append(images[i])

    for class_idx, image_list in images_by_class.items():
        if not image_list:
            continue
        
        image_to_interpret = random.choice(image_list).unsqueeze(0).to(device)
        target_class = class_idx

        print(f"Interpreting a random image from class: {classes[target_class]}")

        # Run IDG
        attribution_map = IDG(
            input=image_to_interpret,
            model=wrapped_model,
            steps=config['idg_steps'],
            batch_size=config.get('idg_batch_size', 10),
            baseline=config['idg_baseline'],
            device=device,
            target_class=target_class
        )

        # --- Visualization ---
        # Un-normalize the original image for saving/plotting
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
        original_image_unnormalized = image_to_interpret.squeeze(0) * std + mean
        original_image_unnormalized = original_image_unnormalized.permute(1, 2, 0).detach().cpu().numpy()
        original_image_unnormalized = np.clip(original_image_unnormalized, 0, 1)

        # Process attribution map for visualization
        attr_map_np = attribution_map.squeeze(0).detach().cpu().numpy()
        attr_map_np = np.transpose(attr_map_np, (1, 2, 0))
        image_2d = np.sum(np.abs(attr_map_np), axis=2)

        # Apply clipping
        percentile = config.get('percentile', 99)
        vmax = np.percentile(image_2d, percentile)
        vmin = np.min(image_2d)
        clipped_map = np.clip((image_2d - vmin) / (vmax - vmin), 0, 1)

        # Save images
        original_save_path = os.path.join(config['output_dir'], f'original_{classes[target_class]}.png')
        attribution_save_path = os.path.join(config['output_dir'], f'idg_{classes[target_class]}.png')
        plt.imsave(original_save_path, original_image_unnormalized)
        plt.imsave(attribution_save_path, clipped_map, cmap='Blues')
        print(f"Saved original image to {original_save_path}")
        print(f"Saved attribution map to {attribution_save_path}")

        # Plot images if requested
        if config.get('show_plots', False):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            ax1.imshow(original_image_unnormalized)
            ax1.set_title(f'Original: {classes[target_class]}')
            ax1.axis('off')

            ax2.imshow(clipped_map, cmap='Blues')
            ax2.set_title('Attribution Map (IDG)')
            ax2.axis('off')
            
            plt.suptitle(f"Interpretation for class '{classes[target_class]}'")
            plt.tight_layout()
            plt.show()
    else:
        # Sample one image from each class
        images_by_class = {i: [] for i in range(len(classes))}
        for images, labels in testloader:
            for i in range(len(images)):
                images_by_class[labels[i].item()].append(images[i])

        for class_idx, image_list in images_by_class.items():
            if not image_list:
                continue
            
            image_to_interpret = random.choice(image_list).unsqueeze(0).to(device)
            target_class = class_idx

            print(f"Interpreting a random image from class: {classes[target_class]}")

            # Run IDG
            attribution_map = IDG(
                input=image_to_interpret,
                model=wrapped_model,
                steps=config['idg_steps'],
                batch_size=config.get('idg_batch_size', 10),
                baseline=config['idg_baseline'],
                device=device,
                target_class=target_class
            )

            # --- Visualization ---
            # Un-normalize the original image for saving/plotting
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
            original_image_unnormalized = image_to_interpret.squeeze(0) * std + mean
            original_image_unnormalized = original_image_unnormalized.permute(1, 2, 0).detach().cpu().numpy()
            original_image_unnormalized = np.clip(original_image_unnormalized, 0, 1)

            # Process attribution map for visualization
            attr_map_np = attribution_map.squeeze(0).detach().cpu().numpy()
            attr_map_np = np.transpose(attr_map_np, (1, 2, 0))
            image_2d = np.sum(np.abs(attr_map_np), axis=2)

            # Apply clipping
            percentile = config.get('percentile', 99)
            vmax = np.percentile(image_2d, percentile)
            vmin = np.min(image_2d)
            clipped_map = np.clip((image_2d - vmin) / (vmax - vmin), 0, 1)

            # Save images
            original_save_path = os.path.join(config['output_dir'], f'original_{classes[target_class]}.png')
            attribution_save_path = os.path.join(config['output_dir'], f'idg_{classes[target_class]}.png')
            plt.imsave(original_save_path, original_image_unnormalized)
            plt.imsave(attribution_save_path, clipped_map, cmap='Blues')
            print(f"Saved original image to {original_save_path}")
            print(f"Saved attribution map to {attribution_save_path}")

            # Plot images if requested
            if config.get('show_plots', False):
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
                ax1.imshow(original_image_unnormalized)
                ax1.set_title(f'Original: {classes[target_class]}')
                ax1.axis('off')

                ax2.imshow(clipped_map, cmap='Blues')
                ax2.set_title('Attribution Map (IDG)')
                ax2.axis('off')
                
                plt.suptitle(f"Interpretation for class '{classes[target_class]}'")
                plt.tight_layout()
                plt.show()

if __name__ == '__main__':
    if USE_GLOBAL_CFG:
        config = GLOBAL_CFG
    else:
        parser = argparse.ArgumentParser(description='Interpret PIP-Net with IDG')
        parser.add_argument('--run_dir', type=str, required=True, help='Directory of the trained model run')
        parser.add_argument('--checkpoint_name', type=str, default='net_trained_best', help='Name of the checkpoint file')
        parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset')
        parser.add_argument('--image_path', type=str, default=None, help='Path to a specific image to interpret')
        parser.add_argument('--target_class', type=int, default=None, help='Target class index for interpretation')
        parser.add_argument('--idg_steps', type=int, default=200, help='Number of steps for IDG')
        parser.add_argument('--idg_baseline', type=float, default=0.0, help='Baseline for IDG')
        parser.add_argument('--output_dir', type=str, default='./idg_interpretations', help='Directory to save outputs')
        parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use')
        parser.add_argument('--show_plots', action='store_true', help='Display plots directly')
        parser.add_argument('--percentile', type=int, default=99, help='Percentile for clipping the attribution map')
        parser.add_argument('--saliency_map_mode', type=str, default='logit', choices=['logit', 'prototype'], help='Saliency map generation mode')
        parser.add_argument('--prototype_activation_threshold', type=float, default=0.1, help='Minimum activation for a prototype to be interpreted')
        args = parser.parse_args()
        config = vars(args)

    interpret(config)
