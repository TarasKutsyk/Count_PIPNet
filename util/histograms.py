import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from tqdm import tqdm
from scipy import stats

from tqdm import tqdm
import argparse
import torch
import torch.nn.functional as F
import torch.utils.data
import os
# Required imports for visualization
from PIL import Image, ImageDraw as D
import torchvision.transforms as transforms
import torchvision
from util.func import get_patch_size
import random

def class_idx_to_name(idx):
    match = {
        0: "1 Cir",
        1: "1 Tri",
        2: "1 Hex",
        3: "2 Cir",
        4: "2 Tri",
        5: "2 Hex",
        6: "3 Cir",
        7: "3 Tri",
        8: "3 Hex",
    }

    return match.get(idx, f"Class {idx}")  # Fallback to generic if not found

def plot_prototype_activations_by_class(net, dataloader, device, output_dir, 
                                        only_important_prototypes=True, 
                                        prototype_importance=None,
                                        importance_threshold=1e-1,
                                        is_count_pipnet=None, 
                                        max_images=10000,
                                        class_idx_to_name=class_idx_to_name):
    """
    Plot class-conditional density plots of prototype activations.
    
    This function visualizes how different classes activate each prototype by creating
    density plots that show activation distributions per class. For CountPIPNet models,
    it adapts to handle discrete count values appropriately.
    
    Args:
        net: The model (PIPNet or CountPIPNet)
        dataloader: DataLoader for the dataset
        device: Device to run the model on
        output_dir: Directory to save the plots
        only_important_prototypes: If True, only plot prototypes relevant to classification
        prototype_importance: 1-D tensor of shape [num_prototypes] containing prototype importance scores
        importance_threshold: Threshold for considering a prototype important
        is_count_pipnet: Whether the model is CountPIPNet (auto-detected if None)
        max_images: Maximum number of images to process (to prevent memory issues)
        class_idx_to_name: Function that maps class indices to class names
    
    Returns:
        List of prototype indices that were visualized
    """
    import plotly.graph_objects as go
    import os
    import numpy as np
    from tqdm import tqdm
    import torch
    from scipy import stats
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Auto-detect if the model is CountPIPNet
    if is_count_pipnet is None:
        is_count_pipnet = hasattr(net.module, '_max_count')
    
    # Get max_count for CountPIPNet
    max_count = net.module._max_count if is_count_pipnet else None
    
    # Determine important prototypes based on classification weights
    num_prototypes = net.module._num_prototypes
    
    # Default class name function if not provided
    if class_idx_to_name is None:
        class_idx_to_name = lambda idx: f"Class {idx}"
    
    # Decide which prototypes to plot
    if only_important_prototypes and prototype_importance is not None:
        prototypes_to_plot = torch.where(prototype_importance > importance_threshold)[0].tolist()
        if not prototypes_to_plot:
            print("No prototypes with importance > threshold. Plotting all prototypes.")
            prototypes_to_plot = list(range(num_prototypes))
    else:
        prototypes_to_plot = list(range(num_prototypes))
    
    # Collect activation values for each prototype along with class labels
    net.eval()
    
    # Lists to store activations and corresponding class labels
    all_activations = []
    all_class_labels = []
    
    with torch.no_grad():
        for idx, (xs, ys) in enumerate(tqdm(dataloader, desc="Collecting activations")):
            if idx >= max_images:
                break
                
            xs = xs.to(device)
            
            # Get pooled activations
            _, pooled, _ = net_forward(xs, net, is_count_pipnet)
            
            # Store activations and class labels
            all_activations.append(pooled.cpu())
            all_class_labels.append(ys.cpu())
    
    # Concatenate all collected data
    all_activations = torch.cat(all_activations, dim=0)
    all_class_labels = torch.cat(all_class_labels, dim=0)
    
    # Get unique classes
    unique_classes = torch.unique(all_class_labels).tolist()
    
    # Define a color palette for classes with higher saturation and contrast
    bright_colors = [
        "#FF4500", "#00CED1", "#FFD700", "#32CD32", "#BA55D3",
        "#FF6347", "#4169E1", "#2E8B57", "#FF1493", "#1E90FF",
        "#FF8C00", "#00FA9A", "#9932CC", "#00BFFF", "#FF69B4"
    ]
    class_colors = {cls: bright_colors[i % len(bright_colors)] for i, cls in enumerate(unique_classes)}
    
    # Process each prototype
    for p in prototypes_to_plot:
        # Get activations for this prototype
        proto_activations = all_activations[:, p].numpy()
        
        # Create figure for this prototype
        fig = go.Figure()
        
        # Set up x-axis range with appropriate padding
        if is_count_pipnet:
            actual_max = np.max(proto_activations)
            x_max = max(max_count + 1, actual_max * 1.1)
            x_range = [0, x_max]
        else:
            non_zero_activations = proto_activations[proto_activations > 0.01]
            x_max = np.max(non_zero_activations) * 1.1 if len(non_zero_activations) > 0 else 1.0
            x_range = [0, x_max]
        
        # Process zero activations separately to avoid dominating the plot
        zero_mask = proto_activations < 0.01
        zero_counts = {cls: np.sum(zero_mask & (all_class_labels == cls).numpy()) for cls in unique_classes}
        zero_pct = np.sum(zero_mask) / len(proto_activations) * 100
        
        # Determine class activity for sorting - how frequently each class activates this prototype
        class_activity = {}
        for cls in unique_classes:
            class_mask = all_class_labels == cls
            class_activations = proto_activations[class_mask.numpy()]
            non_zero_activations = class_activations[class_activations > 0.01]
            
            class_activity[cls] = len(non_zero_activations) / len(class_activations) if len(class_activations) > 0 else 0
        
        # Sort classes by activity level (most active first)
        sorted_classes = sorted(class_activity.items(), key=lambda x: x[1], reverse=True)
        
        # Gather all class data
        all_class_data = {}
        for cls, _ in sorted_classes:
            class_mask = all_class_labels == cls
            class_activations = proto_activations[class_mask.numpy()]
            non_zero_activations = class_activations[class_activations > 0.01]
            
            if len(non_zero_activations) > 0:
                all_class_data[cls] = non_zero_activations
        
        # Scale factor to make bars a reasonable height
        scaling_factor = 0.3
        
        # Create visualization for each class
        class_traces = []
        
        for cls, activity in sorted_classes:
            class_mask = all_class_labels == cls
            class_activations = proto_activations[class_mask.numpy()]
            non_zero_activations = class_activations[class_activations > 0.01]
            
            if len(non_zero_activations) == 0:
                continue
                
            # Thicker lines for important classes (top 3)
            line_width = 4 if cls in [c for c, _ in sorted_classes[:3]] else 2
            class_name = class_idx_to_name(cls)
            
            # Determine if histogram or KDE approach is needed
            use_histogram = (
                len(non_zero_activations) <= 10 or 
                np.std(non_zero_activations) < 1e-6 or 
                len(np.unique(non_zero_activations)) < 3
            )
            
            if not use_histogram:
                try:
                    # Create KDE
                    kde = stats.gaussian_kde(non_zero_activations, bw_method='scott')
                    x_grid = np.linspace(0.01, x_range[1], 500)
                    y_density = kde(x_grid)
                    
                    # Scale density to reasonable height
                    max_y = np.max(y_density)
                    scaled_density = y_density / max_y * scaling_factor
                    
                    # Add density curve
                    class_traces.append(go.Scatter(
                        x=x_grid,
                        y=scaled_density,
                        mode='lines',
                        fill='tozeroy',
                        name=class_name,
                        line=dict(color=class_colors[cls], width=line_width),
                        opacity=0.7,
                    ))
                    
                    # Add annotation for the peak
                    peak_idx = np.argmax(y_density)
                    if y_density[peak_idx] > 0.1:
                        fig.add_annotation(
                            x=x_grid[peak_idx],
                            y=scaled_density[peak_idx] + 0.02,
                            text=f"{y_density[peak_idx]:.2f}",
                            showarrow=False,
                            font=dict(color=class_colors[cls], size=10),
                        )
                except np.linalg.LinAlgError:
                    use_histogram = True
            
            if use_histogram:
                # Calculate histogram counts
                values, counts = np.unique(non_zero_activations, return_counts=True)
                normalized_counts = counts / len(non_zero_activations)
                
                # Scale for consistent visualization
                max_val = np.max(normalized_counts)
                scaled_counts = normalized_counts / max_val * scaling_factor
                
                # Determine visualization mode
                mode = 'lines+markers' if cls in [c for c, _ in sorted_classes[:3]] else 'lines'
                marker = dict(size=8, symbol='circle') if mode == 'lines+markers' else dict()
                
                class_traces.append(go.Scatter(
                    x=values,
                    y=scaled_counts,
                    mode=mode,
                    name=class_name,
                    line=dict(color=class_colors[cls], width=line_width),
                    marker=marker,
                    opacity=0.9 if mode == 'lines+markers' else 0.7,
                ))
                
                # Add annotations for significant peaks
                peak_idx = np.argmax(normalized_counts)
                if normalized_counts[peak_idx] > 0.1:
                    fig.add_annotation(
                        x=values[peak_idx],
                        y=scaled_counts[peak_idx] + 0.02,
                        text=f"{normalized_counts[peak_idx]:.2f}",
                        showarrow=False,
                        font=dict(color=class_colors[cls], size=10),
                    )
        
        # Add traces to figure in reverse order (so most active classes are on top)
        for trace in reversed(class_traces):
            fig.add_trace(trace)
        
        # Add an annotation about zero values
        zero_text = f"Zero/near-zero activations: {zero_pct:.1f}% overall<br>"
        for cls in unique_classes:
            class_name = class_idx_to_name(cls)
            zero_text += f"{class_name}: {zero_counts[cls]} samples<br>"
            
        fig.add_annotation(
            x=0.02,
            y=0.95,
            xref="paper",
            yref="paper",
            text=zero_text,
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1,
        )
        
        # Add vertical lines for key thresholds
        if is_count_pipnet and max_count is not None:
            # Add lines at integer count values
            for count in range(1, min(max_count + 1, 4)):
                fig.add_vline(
                    x=count, 
                    line=dict(color="gray", width=1, dash="dash"),
                    annotation_text=str(count),
                    annotation_position="top"
                )
        else:
            # Add 0.1 threshold line for PIPNet
            fig.add_vline(
                x=0.1, 
                line=dict(color="black", width=1, dash="dash"),
                annotation_text="0.1 Threshold",
                annotation_position="top right"
            )
        
        # Build title
        title = f"Prototype {p} Activation by Class"
        if is_count_pipnet and max_count is not None:
            title += " (Count Values)"
        if prototype_importance is not None:
            title += f" (Importance: {prototype_importance[p].item():.4f})"
        
        # Add annotation for native class (with highest activity)
        if sorted_classes:
            native_class, native_activity = sorted_classes[0]
            if native_activity > 0.1:
                native_class_name = class_idx_to_name(native_class)
                fig.add_annotation(
                    x=0.5,
                    y=1.05,
                    xref="paper",
                    yref="paper",
                    text=f"Native Class: {native_class_name} (Activity: {native_activity:.1%})",
                    showarrow=False,
                    font=dict(size=14, color=class_colors[native_class]),
                )
        
        # Configure layout
        fig.update_layout(
            title=title,
            width=900,
            height=600,
            template="plotly_white",
            xaxis_title="Activation Value" if not is_count_pipnet else "Count Value",
            yaxis_title="Density",
            title_x=0.5,
            xaxis=dict(range=x_range),
            # Set y-axis range to accommodate scaled data and annotations
            yaxis=dict(range=[0, 0.5]),
            legend_title="Class",
            barmode='overlay',
            legend=dict(
                itemsizing='constant',
                font=dict(size=12)
            )
        )
        
        # Save figure
        fig.write_html(os.path.join(output_dir, f"prototype_{p}_class_distribution.html"))
        fig.write_image(os.path.join(output_dir, f"prototype_{p}_class_distribution.png"))
    
    # Create a summary heatmap showing average activation by class for each prototype
    z_data = np.zeros((len(unique_classes), len(prototypes_to_plot)))
    
    # Populate heatmap data with class-prototype activation values
    for i, cls in enumerate(unique_classes):
        for j, p_idx in enumerate(prototypes_to_plot):
            proto_activations = all_activations[:, p_idx].numpy()
            class_mask = all_class_labels == cls
            class_activations = proto_activations[class_mask.numpy()]
            non_zero_class_activations = class_activations[class_activations > 0.01]
            
            # Calculate average activation for non-zero values
            z_data[i, j] = np.mean(non_zero_class_activations) if len(non_zero_class_activations) > 0 else 0.0
    
    # Get class names for y-axis
    class_labels = [class_idx_to_name(cls) for cls in unique_classes]
    
    # Create heatmap visualization
    summary_fig = go.Figure(data=go.Heatmap(
        z=z_data,
        y=class_labels,
        x=[f'Proto {p}' for p in prototypes_to_plot],
        colorscale='Viridis',
        colorbar=dict(title="Avg Activation"),
    ))
    
    summary_fig.update_layout(
        title="Average Non-Zero Activation by Class and Prototype",
        width=1000,
        height=800,
        template="plotly_white",
        xaxis_title="Prototype",
        yaxis_title="Class",
        title_x=0.5,
    )
    
    # Save summary figure
    summary_fig.write_html(os.path.join(output_dir, "prototype_class_activation_summary.html"))
    summary_fig.write_image(os.path.join(output_dir, "prototype_class_activation_summary.png"))
    
    return prototypes_to_plot

def plot_prototype_activations_histograms(net, dataloader, device, output_dir, 
                                         only_important_prototypes=True, 
                                         prototype_importance=None,
                                         importance_threshold=1e-1,
                                         is_count_pipnet=None, 
                                         num_bins=100,  # Increased for granularity
                                         max_images=1000):
    """
    Plot histograms of prototype activations with colored regions for count ranges.
    
    Args:
        net: The model
        dataloader: DataLoader for the dataset
        device: Device to run the model on
        output_dir: Directory to save the plots
        only_important_prototypes: If True, only plot prototypes relevant to classification
        prototype_importance: 1-D tensor of shape [num_prototypes] containing prototypes importance scores
        importance_threshold: Threshold for considering a prototype important
        is_count_pipnet: Whether the model is CountPIPNet (auto-detected if None)
        num_bins: Number of bins for the histograms
        max_images: Maximum number of images to process (to prevent memory issues)
    """
    import plotly.graph_objects as go
    import einops
    import os
    import numpy as np
    from tqdm import tqdm
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Auto-detect if the model is CountPIPNet
    if is_count_pipnet is None:
        is_count_pipnet = hasattr(net.module, '_max_count')
    
    # Get max_count for CountPIPNet
    max_count = None
    if is_count_pipnet:
        max_count = net.module._max_count
    
    # Determine important prototypes based on classification weights
    num_prototypes = net.module._num_prototypes
    
    # Decide which prototypes to plot
    if only_important_prototypes:
        assert prototype_importance is not None, "prototype_importance tensor must be provided when only_important_prototypes = True"

        prototypes_to_plot = torch.where(prototype_importance > importance_threshold)[0].tolist()
        if not prototypes_to_plot:
            print("No prototypes with importance > threshold. Plotting all prototypes.")
            prototypes_to_plot = list(range(num_prototypes))
    else:
        prototypes_to_plot = list(range(num_prototypes))
    
    # Collect activation values for each prototype
    # Initialize tensor to store all activations
    all_activations = torch.zeros((min(max_images, len(dataloader)), num_prototypes))
    
    net.eval()
    with torch.no_grad():
        for idx, (xs, _) in enumerate(tqdm(dataloader, desc="Collecting activations")):
            if idx >= max_images:
                break
                
            xs = xs.to(device)
            
            # Get pooled activations
            _, pooled, _ = net_forward(xs, net, is_count_pipnet)
            all_activations[idx] = pooled.squeeze(0).cpu()
    
    # Define colors for different count ranges
    count_colors = [
        'rgba(200, 200, 200, 0.3)',  # 0 (gray)
        'rgba(152, 223, 138, 0.3)',  # 1 (light green)
        'rgba(44, 160, 44, 0.3)',    # 2 (green)
        'rgba(31, 119, 180, 0.3)',   # 3 (blue)
        'rgba(214, 39, 40, 0.3)',    # 4 (red)
        'rgba(197, 176, 213, 0.3)',  # 5 (purple)
        'rgba(255, 152, 150, 0.3)',  # 6 (pink)
        'rgba(255, 187, 120, 0.3)',  # 7 (orange)
        'rgba(227, 119, 194, 0.3)',  # 8 (violet)
    ]
    
    # Create histogram for each prototype
    for p in prototypes_to_plot:
        # Get activations for this prototype
        activations = all_activations[:, p].numpy()
        max_activation = np.max(activations)
        
        # Create a detailed histogram with many bins
        fig = go.Figure()
        
        # Add the main histogram
        fig.add_trace(go.Histogram(
            x=activations,
            nbinsx=num_bins,
            name="Activations",
            marker_color='rgba(44, 160, 44, 0.5)',
        ))
        
        # Set x-axis limit based on model type
        if is_count_pipnet and max_count is not None:
            x_max = max(max_count + 1.5, max_activation)
            
            # Add colored regions for each count value
            for count in range(max_count + 2):  # +2 to include max_count+
                x0 = count - 0.5 if count > 0 else 0  # Lower bound (but never below 0)
                
                if count <= max_count:
                    x1 = count + 0.5  # Upper bound for specific counts
                else:
                    x1 = x_max  # Upper bound for max_count+
                
                # Calculate percentage of samples in this range
                if count == 0:
                    in_range = (activations >= 0) & (activations < 0.5)
                elif count <= max_count:
                    in_range = (activations >= count - 0.5) & (activations < count + 0.5)
                else:
                    in_range = activations >= max_count + 0.5
                
                percentage = 100 * np.sum(in_range) / len(activations)
                
                # Choose color (cycle through color list if needed)
                color = count_colors[count % len(count_colors)]
                
                # Add shaded area
                fig.add_shape(
                    type="rect",
                    x0=x0,
                    x1=x1,
                    y0=0,
                    y1=1,
                    yref="paper",
                    fillcolor=color,
                    line=dict(width=0),
                    layer="below"
                )
                
                # Add annotation for percentage
                label = f"{count}+" if count > max_count else str(count)
                fig.add_annotation(
                    x=(x0 + x1) / 2,
                    y=0.9,
                    yref="paper",
                    text=f"{label}: {percentage:.1f}%",
                    showarrow=False,
                    font=dict(size=12, color="black")
                )
        else:
            # For PIPNet, use simpler approach without count regions
            x_max = max(1.5, max_activation)  # Ensure at least 0-1.5 range
            
            # Add reference line for the 0.1 threshold commonly used in visualization
            fig.add_vline(
                x=0.1, 
                line=dict(color="red", width=1, dash="dash"),
                annotation_text="0.1 Threshold",
                annotation_position="top right"
            )
            
            # Calculate percentage below and above threshold
            below_threshold = np.sum(activations < 0.1) / len(activations) * 100
            above_threshold = 100 - below_threshold
            
            # Add annotations for percentages
            fig.add_annotation(
                x=0.05,
                y=0.9,
                yref="paper",
                text=f"<0.1: {below_threshold:.1f}%",
                showarrow=False,
                font=dict(size=12, color="black")
            )
            
            fig.add_annotation(
                x=0.3,
                y=0.9,
                yref="paper",
                text=f"≥0.1: {above_threshold:.1f}%",
                showarrow=False,
                font=dict(size=12, color="black")
            )
        
        # Update layout
        title = f"Prototype {p} Activation Distribution"
        if is_count_pipnet and max_count is not None:
            title += " with Count Regions"
        title += f" (Importance: {prototype_importance[p].item():.4f})"
        
        fig.update_layout(
            title=title,
            width=900,
            height=500,
            template="plotly_white",
            bargap=0.1,
            xaxis_title="Activation Value" if not is_count_pipnet else "Count Value",
            yaxis_title="Number of Images",
            title_x=0.5,
            xaxis=dict(
                range=[0, x_max]
            ),
            showlegend=False
        )
        
        # Save figure
        fig.write_html(os.path.join(output_dir, f"prototype_{p}_histogram.html"))
        fig.write_image(os.path.join(output_dir, f"prototype_{p}_histogram.png"))
    
    # Create summary plot with all prototypes
    fig = go.Figure()
    
    # Calculate histogram data for each prototype 
    for p in prototypes_to_plot:
        activations = all_activations[:, p].numpy()
        
        # Add histogram trace for this prototype
        fig.add_trace(go.Histogram(
            x=activations,
            name=f"Proto {p}",
            opacity=0.7,
            nbinsx=num_bins,
        ))
    
    # Overlay histograms
    max_overall = torch.max(all_activations).item()
    
    # Set x-axis limit based on model type
    if is_count_pipnet and max_count is not None:
        x_max_overall = max(max_count + 1.5, max_overall)
    else:
        x_max_overall = max(1.5, max_overall)
    
    fig.update_layout(
        barmode='overlay',
        title="Activation Distributions Across Prototypes",
        width=1000,
        height=600,
        template="plotly_white",
        bargap=0.1,
        xaxis_title="Activation Value",
        yaxis_title="Number of Images",
        title_x=0.5,
        xaxis=dict(range=[0, x_max_overall])
    )
    
    # Add reference lines based on model type
    if is_count_pipnet and max_count is not None:
        # Add vertical lines for count thresholds
        for count in range(max_count + 2):
            if count > 0:  # Skip the 0-0.5 boundary
                fig.add_vline(
                    x=count - 0.5, 
                    line=dict(color="black", width=1, dash="dash"),
                    annotation_text=f"{count-1}/{count}" if count <= max_count else f"{max_count}+",
                    annotation_position="top"
                )
    else:
        # Add 0.1 threshold line for PIPNet
        fig.add_vline(
            x=0.1, 
            line=dict(color="red", width=1, dash="dash"),
            annotation_text="0.1 Threshold",
            annotation_position="top right"
        )
    
    # Save summary figure
    fig.write_html(os.path.join(output_dir, "all_prototypes_histograms.html"))
    fig.write_image(os.path.join(output_dir, "all_prototypes_histograms.png"))
    
    return prototypes_to_plot


def net_forward(xs, net, is_count_pipnet=True):
    """
    Performs a forward pass with suitable params depending on which network (pipnet or count_pipnet) is used.
    """
    run_in_inference = not is_count_pipnet # for count-pip-net run the model in training mode, for pipnet otherwise
    with torch.no_grad():
        pfs, pooled, out = net(xs, inference=run_in_inference)

    return pfs, pooled, out