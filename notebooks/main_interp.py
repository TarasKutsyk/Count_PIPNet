#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Enable auto-reloading of imports when they have been modified
from IPython import get_ipython
ipython = get_ipython(); assert ipython is not None
ipython.run_line_magic("load_ext", "autoreload")
ipython.run_line_magic("autoreload", "2")

import os
import json
import torch
import torch.nn as nn

# Disable gradient computation - this notebook will only perform forward passes
torch.set_grad_enabled(False)

from pathlib import Path
import sys
import os

# Add the base (root) directory to the path so we can import the util modules
def get_base_folder(project_root = "Count_PIPNet"):
	# Find the project root dynamically
	current_dir = os.getcwd()
	while True:
		if os.path.basename(current_dir) == project_root:  # Adjust to match your project root folder name
			break
		parent = os.path.dirname(current_dir)
		if parent == current_dir:  # Stop if we reach the system root (failsafe)
			raise RuntimeError(f"Project root {project_root} not found. Check your folder structure.")
		current_dir = parent

	return Path(current_dir)

base_path = get_base_folder("PIPNet") #"PIPNet"
print(f"Base path: {base_path}")
sys.path.append(str(base_path))


# In[2]:


from util.vis_pipnet import visualize_topk
from pipnet.count_pipnet import get_count_network
from util.checkpoint_manager import CheckpointManager
from util.data import get_dataloaders
from util.args import get_args
from util.vis_pipnet import visualize_topk


# In[3]:


# Device setup
GPU_TO_USE = 0

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = f"cuda:{GPU_TO_USE}" if torch.cuda.is_available() else "cpu"

print(f'>>> Using {device} device <<<')


# In[4]:


use_multi_experiment_dir = True

multi_experiment_dir = base_path / 'runs/pipnet'
multi_experiment_dir


# In[5]:


visualization_dir = base_path / 'visualizations'
os.makedirs(visualization_dir, exist_ok=True)


# In[6]:


summary_path = os.path.join(multi_experiment_dir, 'summary.json')

try:
	# Load the summary file to get all run directories
	with open(summary_path, 'r') as f:
		summary = json.load(f)

	print(f"Found {len(summary)} trained models")
except FileNotFoundError:
	print(f"Summary file NOT found at {summary_path}. Please ensure the training was completed and the summary file was generated.")


# # Loading the model

# In[7]:


def load_model(run_dir, checkpoint_name='net_trained_best', base_path=base_path, gpu_id=3):
	"""
	Load a model from a checkpoint directory for visualization purposes.

	Args:
		run_dir: Directory containing the run results
		checkpoint_name: Name of checkpoint to load (default: 'net_trained_best')
		base_path: Base path for dataset directories (default: None)
		gpu_id: GPU ID to use (default: 0)
		
	Returns:
		Tuple of (net, projectloader, classes, args, is_count_pipnet)
	"""
	# Step 1: Load the configuration used for this run
	metadata_dir = os.path.join(run_dir, 'metadata')
	args_path = os.path.join(metadata_dir, 'args.pickle')

	import pickle
	with open(args_path, 'rb') as f:
		args = pickle.load(f)
	print(f"Loaded configuration from {args_path}")

	# Explicitly set GPU ID to ensure device consistency
	if torch.cuda.is_available():
		args.gpu_ids = str(gpu_id)
		device = torch.device(f'cuda:{gpu_id}')
		torch.cuda.set_device(device)
	else:
		device = torch.device('cpu')

	print(f"Using device: {device}")

	# Step 2: Create dataloaders (needed for projectloader)
	args.log_dir = run_dir  # Use the run directory as log_dir
	trainloader, trainloader_pretraining, trainloader_normal, \
	trainloader_normal_augment, projectloader, testloader, test_projectloader, classes = get_dataloaders(args, device, base_path)

	# Step 3: Create a model with the same architecture
	if hasattr(args, 'model') and args.model == 'count_pipnet':
		is_count_pipnet = True
		net, num_prototypes = get_count_network(
			num_classes=len(classes), 
			args=args,
			device=device,
			max_count=getattr(args, 'max_count', 3),
			use_ste=getattr(args, 'use_ste', False))
	else:
		from pipnet.pipnet import get_pipnet
		is_count_pipnet = False
		net, num_prototypes = get_pipnet(len(classes), args)

	# Step 4: Move model to device (don't use DataParallel yet)
	net = net.to(device)

	# Step 5: Forward one batch through the backbone to get the latent output size
	# This needs to happen BEFORE loading the checkpoint
	with torch.no_grad():
		# Use a small batch to determine output shape
		xs1, _, _ = next(iter(trainloader))
		xs1 = xs1.to(device)

		# Single-forward pass without DataParallel
		features = net._net(xs1)
		proto_features = net._add_on(features)

		wshape = proto_features.shape[-1]
		args.wshape = wshape  # needed for calculating image patch size
		print(f"Output shape: {proto_features.shape}, setting wshape={wshape}")
            
	# Step 6: Now wrap with DataParallel
	device_ids = [gpu_id]
	print(f"Using device_ids: {device_ids}")
	net = nn.DataParallel(net, device_ids=device_ids)

	# Step 7: Direct checkpoint loading
	checkpoint_path = os.path.join(run_dir, 'checkpoints', checkpoint_name)
	if not os.path.exists(checkpoint_path):
		print(f"Checkpoint not found at {checkpoint_path}, trying alternative paths...")
		# Try with full path as fallback
		if os.path.exists(checkpoint_name):
			checkpoint_path = checkpoint_name
		else:
			# Try other common checkpoint names
			alternatives = [
				os.path.join(run_dir, 'checkpoints', 'net_trained_last'),
				os.path.join(run_dir, 'checkpoints', 'net_trained'),
				checkpoint_name # in case the direct path was passed
			]
			for alt_path in alternatives:
				if os.path.exists(alt_path):
					checkpoint_path = alt_path
					print(f"Found alternative checkpoint at {checkpoint_path}")
					break
			else:
				print("No checkpoint found")
				return None, None, None, None, None

	try:
		# Load just the model state dict, ignore optimizer states
		checkpoint = torch.load(checkpoint_path, map_location=device)
		
		if 'model_state_dict' in checkpoint:
			net.load_state_dict(checkpoint['model_state_dict'], strict=True)
			print(f"Successfully loaded model state from {checkpoint_path}")
			
			# Display additional information if available
			if 'epoch' in checkpoint:
				print(f"Checkpoint from epoch {checkpoint['epoch']}")
			if 'accuracy' in checkpoint:
				print(f"Model accuracy: {checkpoint['accuracy']:.4f}")
			
			return net, projectloader, testloader, classes, args
		else:
			print(f"Checkpoint doesn't contain model_state_dict")
			return None, None, None, None, None
			
	except Exception as e:
		print(f"Error loading checkpoint: {str(e)}")
		import traceback
		traceback.print_exc()
		return None, None, None, None, None


# # Pre-trained prototypes visualization

# In[8]:


RUN_PRETRAINED_VIZ = False


# In[9]:


# Assuming multi_experiment_dir is already defined as a Path
folders = [p for p in multi_experiment_dir.iterdir() if p.is_dir()]
name_filter = 'linear_full_train' 

# Print the folders
for folder in folders:
    if name_filter in folder.name:
        print(folder.name)  # or just print(folder) if you want full paths


# In[10]:


use_multi_experiment_dir = True

if use_multi_experiment_dir:
	checkpoint_to_load = '20250409_052530_8_final_s21_stage3_p16_linear_full_train'
	checkpoint_to_load_pretrain_dir = multi_experiment_dir / checkpoint_to_load
else:
	checkpoint_to_load = 'runs/45_shapesGN_linear'
	checkpoint_to_load_pretrain_dir = base_path / checkpoint_to_load

checkpoint_name = 'net_pretrained'

print(f'Loading a checkpoint {checkpoint_name} from {checkpoint_to_load_pretrain_dir}')


# In[11]:


if RUN_PRETRAINED_VIZ:
    net, projectloader, testloader, classes, args = load_model(checkpoint_to_load_pretrain_dir, gpu_id=GPU_TO_USE,
                                                               checkpoint_name=checkpoint_name)


# In[12]:


run_vis_dir = visualization_dir / checkpoint_to_load.split('/')[-1] / 'pre-trained'

print(f'Saving viz to {run_vis_dir}')

os.makedirs(run_vis_dir, exist_ok=True)


# In[13]:


if RUN_PRETRAINED_VIZ:
    topks = visualize_topk(net, projectloader, len(classes), device, run_vis_dir, args, k=10,
    					   plot_histograms=True, visualize_prototype_maps=True, are_pretraining_prototypes=True)
    print(f"Visualization saved to {run_vis_dir}")


# # Best trained model visualization

# In[14]:


RUN_BEST_TRAINED_VIZ = True


# In[15]:


checkpoint_to_load = '20250407_021157_15_pipnet_s21_stage7_p16' # 20250407_172248_20_pipnet_s21_stage7_p0
path_to_load = multi_experiment_dir / checkpoint_to_load

# path_to_load = base_path / 'runs/stage_3' / '20250401_064200_0_linear'

print(f'Loading a checkpoint from {path_to_load}...')


# In[16]:


net, projectloader, testloader, classes, args = load_model(path_to_load, gpu_id=GPU_TO_USE)


# In[17]:


run_vis_dir = visualization_dir /checkpoint_to_load / 'trained_best'

print(f'Saving viz to {run_vis_dir}')

os.makedirs(run_vis_dir, exist_ok=True)


# In[18]:


if RUN_BEST_TRAINED_VIZ:
    topks, non_zero_counts_per_prototype_and_class = visualize_topk(net, projectloader, len(classes), device, run_vis_dir, args, k=10,
                                            					    plot_histograms=True, visualize_prototype_maps=False, plot_always_histograms=True,
                                                                    normalize_frequencies=False, prototype_labels=checkpoint_to_load)
    print(f"Visualization saved to {run_vis_dir}")


# # Global explanation

# In[20]:


from util.histograms import class_idx_to_name

def calculate_global_explanation(net, classes):
    """
    Calculate the importance of each prototype for each class in the network.
    
    Args:
        net: The trained CountPIPNet model
        classes: List of class names
        
    Returns:
        Dictionary mapping class indices to tensors of prototype importances
    """
    # Detect if using CountPIPNet by checking for _max_count attribute
    is_count_pipnet = hasattr(net.module, '_max_count')
    num_prototypes = net.module._num_prototypes
    num_classes = net.module._num_classes

    if not is_count_pipnet:
        classification_weights = net.module._classification.weight # shape [num_classes, num_prototypes] 
        global_explanation = {c: classification_weights[c, :] for c in range(num_classes)}

        return global_explanation

    # Dictionary to store the importance of each prototype for each class
    class_prototype_importances = {}

    # Iterate through all prototypes
    for i in range(num_prototypes):
        # Get importance of this prototype for each class
        prototype_importance_per_class = net.module.get_prototype_importance_per_class(i)
        
        # Distribute the importance values to their respective classes
        for class_idx, class_importance in enumerate(prototype_importance_per_class):
            if class_idx not in class_prototype_importances.keys():
                class_prototype_importances[class_idx] = torch.zeros([num_prototypes], device=class_importance.device)

            class_prototype_importances[class_idx][i] += class_importance
    
    return class_prototype_importances


# In[29]:


from plotly.io import write_image

def show_global_explanation(net, classes, global_explanation=None, top_k_prototypes=None, output_path=None):
    """
    Visualize the global explanation as a heatmap using Plotly.
    
    Args:
        net: The trained CountPIPNet model
        classes: List of class names
        global_explanation: Pre-computed global explanation (optional)
        top_k_prototypes: Number of top prototypes to display per class (optional)
        output_path: Path to save the visualization (optional)
        
    Returns:
        Plotly figure object
    """
    import plotly.graph_objects as go
    import numpy as np
    
    # Calculate global explanation if not provided
    if global_explanation is None:
        global_explanation = calculate_global_explanation(net, classes)
    
    # Convert dictionary to numpy array for heatmap
    num_classes = len(global_explanation)
    num_prototypes = net.module._num_prototypes
    
    # Initialize the data matrix
    data_matrix = np.zeros((num_classes, num_prototypes))
    
    # Fill in the data matrix with importance values
    for class_idx, prototype_importances in global_explanation.items():
        data_matrix[class_idx] = prototype_importances.cpu().numpy()
    
    # Optionally filter to show only the top-k most important prototypes per class
    if top_k_prototypes is not None:
        # Create a mask of top-k prototypes per class
        top_k_mask = np.zeros_like(data_matrix, dtype=bool)
        for class_idx in range(num_classes):
            # Get indices of top-k prototypes for this class
            top_indices = np.argsort(data_matrix[class_idx])[-top_k_prototypes:]
            top_k_mask[class_idx, top_indices] = True
        
        # Apply mask (set non-top-k values to NaN for better visibility)
        filtered_data = np.where(top_k_mask, data_matrix, np.nan)
    else:
        filtered_data = data_matrix
    
    # Create x and y labels
    class_labels = [class_idx_to_name(i) for i in range(num_classes)]
    prototype_labels = [f"Prototype {i}" for i in range(num_prototypes)]

    # Define desired order for classes (if the dataset matches this organization)
    desired_order = [
        "1 Circle", "2 Circles", "3 Circles",
        "1 Triangle", "2 Triangles", "3 Triangles",
        "1 Square", "2 Squares", "3 Squares",
        "1 Hexagon", "2 Hexagons", "3 Hexagons"
    ]
    
    # Reorder data matrix and labels according to desired order
    reordered_data = filtered_data.copy()
    reordered_labels = class_labels.copy()
    
    # Create mapping from current order to desired order
    order_mapping = {}
    for i, label in enumerate(class_labels):
        if label in desired_order:
            new_idx = desired_order.index(label)
            order_mapping[i] = new_idx
    
    # If we have a valid mapping (matches our desired order scheme)
    if order_mapping:
        # Create new arrays with desired ordering
        sorted_indices = sorted(range(len(class_labels)), 
                               key=lambda i: order_mapping.get(i, len(desired_order) + i))
        reordered_data = filtered_data[sorted_indices]
        reordered_labels = [class_labels[i] for i in sorted_indices]
    
    # Keep original orientation (classes on y-axis, prototypes on x-axis)
    fig = go.Figure(data=go.Heatmap(
        z=reordered_data,
        x=prototype_labels,  # Prototypes on x-axis
        y=reordered_labels,  # Classes on y-axis
        colorscale='Plasma',
        hoverongaps=False,
        colorbar=dict(
            title="Importance",
            titleside="right"
        ),
        # Add hover template for better readability
        hovertemplate='Class: %{y}<br>%{x}<br>Importance: %{z:.3f}<extra></extra>'
    ))
    
    # Update layout for better readability
    fig.update_layout(
        # title="Global Explanation: Prototype Importance per Class",
        xaxis=dict(
            title="Prototypes",
            tickangle=-45,  # Angled prototype labels for better readability
        ),
        yaxis=dict(
            title="Classes",
        ),
        width=max(800, num_prototypes * 25),  # Adjust width based on number of prototypes
        height=max(600, num_classes * 30),    # Adjust height based on number of classes
        margin=dict(l=150, r=50, t=100, b=150)
    )
    
    # Save the figure if output path is provided
    if output_path:
        fig.write_html(output_path / 'global_explanation.html')
        write_image(fig, output_path / 'global_explanation.pdf', engine='orca')
        write_image(fig, output_path / 'global_explanation.png', engine='orca', scale=3)

        print(f"Saved global explanation visualization to {output_path}")
    
    return fig  # Return the figure object


# In[30]:


show_global_explanation(net, classes, output_path=run_vis_dir)


# In[21]:


weight_based_importance_scores = calculate_global_explanation(net, classes)
weight_based_importance_scores


# In[22]:


activation_based_importance_scores = non_zero_counts_per_prototype_and_class
activation_based_importance_scores


# ## Model specific explanation: `20250407_021157_15_pipnet_s21_stage7_p16`

# In[31]:


def get_prototype_groups(num_prototypes_total=16):
    # Define prototype groups
    shape_only_group = [4]
    count_only_group = [0, 3, 11, 14]
    mixed_group = [2, 6, 9, 12, 13]
    unique_group = [1, 7, 8, 10]

    # --- Labels for each prototype ---
    prototype_labels = [
        {"prototype": 0, "label":  "Count-1"},
        {"prototype": 1, "label":  "Circ(3)"},
        {"prototype": 2, "label":  "Tri(2, 3)"},
        {"prototype": 3, "label":  "Count-1"},
        {"prototype": 4, "label":  "Circ(:)"},
        {"prototype": 5, "label":  "Dead"},
        {"prototype": 6, "label":  "Tri(1, 3)"},
        {"prototype": 7, "label":  "Hex(3)"},
        {"prototype": 8, "label":  "Circ(2)"},
        {"prototype": 9, "label":  "Hex(2, 3)"},
        {"prototype": 10, "label": "Hex(1)"},
        {"prototype": 11, "label": "Count-2"},
        {"prototype": 12, "label": "Circ(1, 2)"},
        {"prototype": 13, "label": "Hex(1, 2)"},
        {"prototype": 14, "label": "Count-1"},
        {"prototype": 15, "label":  "Dead"}
    ]

    # Define group → color mapping (closest Plotly colors to LaTeX ones)
    group_to_color = {
        "count": "red",
        "shape": "deepskyblue",
        "mixed": "rgb(0, 100, 0)",
        "unique": "rgb(255, 207, 0)",
        "dead": "gray"
    }

    # Define group → order priority (lower = higher priority)
    group_to_priority = {
        "count": 2,
        "shape": 1,
        "mixed": 3,
        "unique": 4,
        "dead": 5
    }

    # Step 1: Concatenate
    all_grouped = shape_only_group + count_only_group + mixed_group + unique_group

    # Step 2: Check for overlaps
    seen = set()
    duplicates = set()
    for idx in all_grouped:
        if idx in seen:
            duplicates.add(idx)
        else:
            seen.add(idx)

    if duplicates:
        print("⚠️ Warning: Duplicate prototype indices found:", sorted(duplicates))
    else:
        print("✅ No overlaps found in group assignments.")

    # Step 3: Check for dead prototypes
    dead_group = []
    for i in range(num_prototypes_total):
        if i not in seen:
            print(f"⚠️ Prototype {i} not assigned to any group — marked as dead.")
            dead_group.append(i)

    if not dead_group:
        print("✅ No dead prototypes.")
    else:
        print("Dead prototypes:", dead_group)

    # Step 4: Reverse mapping from index to group
    index_to_group = {}
    for i in shape_only_group:
        index_to_group[i] = "shape"
    for i in count_only_group:
        index_to_group[i] = "count"
    for i in mixed_group:
        index_to_group[i] = "mixed"
    for i in unique_group:
        index_to_group[i] = "unique"

    # Step 5: Build label map
    label_map = {d["prototype"]: d["label"] for d in prototype_labels}

    # Step 6: Construct final list
    prototype_group_definitions = []
    for i in range(num_prototypes_total):
        group = index_to_group.get(i, "dead")
        entry = {
            "prototype": [i],  # wrapped in list
            "group_name": group,
            "color": group_to_color[group],
            "label": label_map.get(i, "Dead"),
            "order_priority": group_to_priority[group]
        }
        prototype_group_definitions.append(entry)

    # Optional: print the result
    for entry in prototype_group_definitions:
        print(entry)

    return prototype_group_definitions


# In[32]:


from plotly.io import write_image
import plotly.graph_objects as go
import numpy as np

def show_global_explanation(net, classes, global_explanation=None, top_k_prototypes=None, output_path=None, 
                            order_prototypes_by_group=True): # New parameter
    """
    Visualize the global explanation as a heatmap using Plotly.
    Optionally orders prototypes on x-axis by group and adds counts to legend.
    """
        
    num_classes = len(classes)
    num_prototypes_total = net.module._num_prototypes
    
    prototype_group_definitions = get_prototype_groups(num_prototypes_total)

    if global_explanation is None:
        global_explanation = calculate_global_explanation(net, classes)

    # --- PREPARE DATA AND LABELS (ORDERING LOGIC MOVED HERE) ---
    original_data_matrix = np.zeros((num_classes, num_prototypes_total))
    for class_idx in range(num_classes):
        if class_idx in global_explanation:
            importances_tensor = global_explanation[class_idx].cpu()
            if len(importances_tensor) == num_prototypes_total:
                original_data_matrix[class_idx] = importances_tensor.numpy()
            else:
                current_len = len(importances_tensor)
                if current_len < num_prototypes_total:
                    padded = np.pad(importances_tensor.numpy(), (0, num_prototypes_total - current_len), 'constant')
                    original_data_matrix[class_idx] = padded
                else:
                    original_data_matrix[class_idx] = importances_tensor.numpy()[:num_prototypes_total]

    x_tick_labels = [f"P{i}" for i in range(num_prototypes_total)]
    x_tick_colors = ["black"] * num_prototypes_total
    x_values_for_heatmap = list(range(num_prototypes_total))
    heatmap_data_to_plot = original_data_matrix.copy() # Start with original order

    # Map original prototype index to its group info for consistent coloring/labeling
    prototype_original_idx_to_group_info = {}
    for group_def in prototype_group_definitions:
        for p_idx in group_def["prototype"]:
            if 0 <= p_idx < num_prototypes_total:
                prototype_original_idx_to_group_info[p_idx] = group_def
    
    # Apply default "Other" to any unassigned prototypes for consistent x-tick coloring
    other_group_info_default = next((g for g in prototype_group_definitions if g["group_name"] == "Other"), 
                                    {"label": "Other", "color": "grey", "group_name": "Other"})

    for i in range(num_prototypes_total):
        if i in prototype_original_idx_to_group_info:
            group_info = prototype_original_idx_to_group_info[i]
            # Use the specific label from its definition if ordering is off, or for hover
            # If ordering is on, x_tick_labels will be overwritten by ordered labels
            x_tick_labels[i] = group_info["label"] if not order_prototypes_by_group else f"P{i}" # Default if ordered
            x_tick_colors[i] = group_info["color"]
        else: # Prototype not in any definition
            x_tick_labels[i] = f"P{i}"
            x_tick_colors[i] = other_group_info_default["color"]


    ordered_prototype_display_info_list = [] # For x-axis if ordered, and for hover
    if order_prototypes_by_group:
        prototypes_in_defined_groups = set()
        # Sort group definitions by order_priority, then by first prototype index
        sorted_group_definitions_for_ordering = sorted(prototype_group_definitions, key=lambda x: (x["order_priority"], x["prototype"][0] if x["prototype"] else float('inf')))

        for group_def in sorted_group_definitions_for_ordering:
            sorted_prototypes_in_group = sorted(group_def["prototype"])
            for p_original_idx in sorted_prototypes_in_group:
                if 0 <= p_original_idx < num_prototypes_total:
                    ordered_prototype_display_info_list.append({
                        "original_idx": p_original_idx,
                        "display_label": group_def["label"], # This will be the x-tick label
                        "color": group_def["color"],
                        "group_name": group_def["group_name"]
                    })
                    prototypes_in_defined_groups.add(p_original_idx)
        
        for p_original_idx in range(num_prototypes_total):
            if p_original_idx not in prototypes_in_defined_groups:
                ordered_prototype_display_info_list.append({
                    "original_idx": p_original_idx,
                    "display_label": f"P{p_original_idx}",
                    "color": other_group_info_default["color"],
                    "group_name": other_group_info_default["group_name"]
                })
        
        # Reorder heatmap_data_to_plot based on ordered_prototype_display_info_list
        temp_heatmap_data = np.zeros((num_classes, len(ordered_prototype_display_info_list)))
        for new_idx, info in enumerate(ordered_prototype_display_info_list):
            temp_heatmap_data[:, new_idx] = original_data_matrix[:, info["original_idx"]]
        heatmap_data_to_plot = temp_heatmap_data
        
        # Update x-axis values and labels for the ordered plot
        x_values_for_heatmap = list(range(len(ordered_prototype_display_info_list)))
        x_tick_labels = [info["display_label"] for info in ordered_prototype_display_info_list]
        x_tick_colors = [info["color"] for info in ordered_prototype_display_info_list]
    else:
        # If not ordering, create display_info_list matching original order for hover
        for i in range(num_prototypes_total):
            group_info = prototype_original_idx_to_group_info.get(i, other_group_info_default)
            ordered_prototype_display_info_list.append({
                "original_idx": i,
                "display_label": group_info.get("label", f"P{i}"), # Use group label or P_idx
                "color": group_info.get("color", "grey"),
                "group_name": group_info.get("group_name", "Other")
            })


    if top_k_prototypes is not None:
        top_k_mask = np.zeros_like(heatmap_data_to_plot, dtype=bool)
        for class_idx in range(num_classes):
            top_indices = np.argsort(heatmap_data_to_plot[class_idx])[-top_k_prototypes:]
            top_k_mask[class_idx, top_indices] = True
        filtered_heatmap_data = np.where(top_k_mask, heatmap_data_to_plot, np.nan)
    else:
        filtered_heatmap_data = heatmap_data_to_plot
    
    # Y-axis (class) reordering
    class_labels_y_axis = [class_idx_to_name(i) for i in range(num_classes)]
    desired_y_order = ["1 Circle", "2 Circles", "3 Circles", "1 Triangle", "2 Triangles", "3 Triangles", "1 Hexagon", "2 Hexagons", "3 Hexagons"]
    
    final_heatmap_data_y_reordered = np.zeros_like(filtered_heatmap_data)
    final_y_labels = [""] * num_classes
    
    temp_data_list = []
    temp_y_labels_list = []

    for desired_label in desired_y_order:
        if desired_label in class_labels_y_axis: # Use original class_labels_y_axis for indexing
            original_data_row_idx = class_labels_y_axis.index(desired_label)
            temp_data_list.append(filtered_heatmap_data[original_data_row_idx, :])
            temp_y_labels_list.append(class_labels_y_axis[original_data_row_idx])
    
    for i, label in enumerate(class_labels_y_axis):
        if label not in temp_y_labels_list:
            temp_data_list.append(filtered_heatmap_data[i, :])
            temp_y_labels_list.append(label)

    if temp_data_list:
        final_heatmap_data_y_reordered = np.array(temp_data_list)
        final_y_labels = temp_y_labels_list
    else:
        final_heatmap_data_y_reordered = filtered_heatmap_data
        final_y_labels = class_labels_y_axis

    # Prepare customdata for hover, using original index and the display label
    hover_custom_data_list = [[info["display_label"], info["original_idx"]] for info in ordered_prototype_display_info_list]


    fig = go.Figure(data=go.Heatmap(
        z=final_heatmap_data_y_reordered,
        x=x_values_for_heatmap, 
        y=final_y_labels,
        colorscale='Plasma',
        hoverongaps=False,
        colorbar=dict(title="Importance", titleside="right"),
        hovertemplate='Class: %{y}<br>Prototype: %{customdata[0]} (Original P%{customdata[1]})<br>Importance: %{z:.3f}<extra></extra>',
        customdata=np.array(hover_custom_data_list).T if hover_custom_data_list else None
    ))
    
    # --- COLORED BARS AT THE TOP (NO LABELS ON THEM) ---
    annotation_y_base_for_bars = 1.02
    annotation_bar_height = 0.03
    shapes_to_add = []

    if ordered_prototype_display_info_list: # Ensure list is not empty
        segment_start_new_idx = 0 # The display index (0 to N-1) of the first prototype in the current segment
        
        for current_display_idx, p_info_current_prototype in enumerate(ordered_prototype_display_info_list):
            # Properties of the prototype at the start of the current potential segment
            current_segment_color = ordered_prototype_display_info_list[segment_start_new_idx]["color"]
            current_segment_group_name = ordered_prototype_display_info_list[segment_start_new_idx]["group_name"]

            is_last_prototype_in_list = (current_display_idx == len(ordered_prototype_display_info_list) - 1)
            
            end_segment = False
            if is_last_prototype_in_list:
                end_segment = True
            else:
                p_info_next_prototype = ordered_prototype_display_info_list[current_display_idx + 1]
                # A segment ends if the *next* prototype has a different group_name or color
                # than the prototype that *started* the current segment.
                if (current_segment_group_name != p_info_next_prototype["group_name"] or \
                    current_segment_color != p_info_next_prototype["color"]):
                    end_segment = True
            
            if end_segment:
                # The segment to draw is from segment_start_new_idx to current_display_idx
                x0_val = segment_start_new_idx - 0.5
                x1_val = current_display_idx + 0.5 # Segment includes the current_display_idx prototype
                
                # --- DEBUG PRINT ---
                if current_display_idx >= len(ordered_prototype_display_info_list) - 2 : # Print for last few segments
                    print(f"Drawing bar for segment: StartIdx={segment_start_new_idx}, EndIdx={current_display_idx}, Color='{current_segment_color}', Group='{current_segment_group_name}'")
                # --- END DEBUG PRINT ---

                shapes_to_add.append(dict(
                    type="rect", xref="x", yref="paper",
                    x0=x0_val, y0=annotation_y_base_for_bars,
                    x1=x1_val, y1=annotation_y_base_for_bars + annotation_bar_height,
                    fillcolor=current_segment_color, # Use the color determined at the start of the segment
                    line_width=0.5, line_color="black", opacity=0.7,
                ))
                
                # Prepare for the next segment
                if not is_last_prototype_in_list:
                    segment_start_new_idx = current_display_idx + 1
                    
    fig.update_layout(shapes=shapes_to_add)

    # --- LEGEND WITH PROTOTYPE COUNTS PER GROUP ---
    legend_items = []
    added_group_names_to_legend = set()
    
    # Calculate counts for each group_name based on ordered_prototype_display_info_list
    group_name_counts = {}
    for info in ordered_prototype_display_info_list:
        group_name_counts[info["group_name"]] = group_name_counts.get(info["group_name"], 0) + 1
        
    # Use sorted_group_definitions to control legend order and get correct colors per group_name
    unique_legend_entries = {} # To store {group_name: color}
    for group_def in prototype_group_definitions: # Iterate original defs to get intended colors
        if group_def["group_name"] not in unique_legend_entries:
            unique_legend_entries[group_def["group_name"]] = group_def["color"]

    # Sort unique legend entries by group_name for consistent legend order
    # Or, could sort by order_priority of the first group_def that introduced the group_name
    sorted_legend_group_names = sorted(unique_legend_entries.keys())


    for group_name_for_legend in sorted_legend_group_names:
        color_for_legend = unique_legend_entries[group_name_for_legend]
        count_in_group = group_name_counts.get(group_name_for_legend, 0)
        legend_name_with_count = f"{group_name_for_legend}" # add ({count_in_group}) if you want to show the count
        
        legend_items.append(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(color=color_for_legend, size=10, symbol='square'),
            name=legend_name_with_count,
            legendgroup=group_name_for_legend 
        ))
    
    for item in legend_items:
        fig.add_trace(item)

    # --- UPDATE LAYOUT ---
    top_bar_height_paper_coords = annotation_bar_height
    legend_height_paper_coords = 0.05 
    total_top_elements_height_paper_coords = top_bar_height_paper_coords + legend_height_paper_coords + 0.02 

    plot_height_pixels_approx = max(650, num_classes * 35 + 200) 
    top_margin_pixels = int(total_top_elements_height_paper_coords * plot_height_pixels_approx)

    fig.update_layout(
        xaxis=dict(
            title="Prototypes (Grouped)" if order_prototypes_by_group else "prototype",
            tickmode='array',
            tickvals=x_values_for_heatmap,
            ticktext=[f'<span style="color:{color};">{label}</span>' for label, color in zip(x_tick_labels, x_tick_colors)],
            tickangle=-60, 
            automargin=True 
        ),
        yaxis=dict(
            title="Classes",
            automargin=True
        ),
        width=max(900, len(x_values_for_heatmap) * 45 + 150), 
        height=plot_height_pixels_approx, 
        margin=dict(l=150, r=50, t=top_margin_pixels + 20, b=180),
        legend_title_text=f'Prototype Categories ({num_prototypes_total} Prototypes)', # Updated legend title
        legend=dict(
            traceorder='normal',
            orientation="h",
            yanchor="bottom", 
            y=annotation_y_base_for_bars + annotation_bar_height + 0.01,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(255,255,255,0.7)',
            bordercolor='rgba(0,0,0,0.5)',
            borderwidth=1
        ),
    )
    if output_path:
        import pathlib # Ensure pathlib is imported
        output_path = pathlib.Path(output_path) # Convert to Path object if it's a string
        output_path.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        
        fig.write_html(output_path / 'global_explanation_annotated.html')
        try:
            write_image(fig, output_path / 'global_explanation_annotated.pdf', engine='orca') 
            write_image(fig, output_path / 'global_explanation_annotated.png', engine='orca', scale=3)
            print(f"Saved annotated global explanation visualization to {output_path}")
        except Exception as e:
            print(f"Could not save static images. Error: {e}")
            print("Please ensure 'kaleido' is installed (pip install -U kaleido) for static image export.")
            
    return fig


# In[33]:


show_global_explanation(net, classes, output_path=run_vis_dir, order_prototypes_by_group=False)


# In[34]:


show_global_explanation(net, classes, output_path=run_vis_dir, order_prototypes_by_group=True)


# In[ ]:




