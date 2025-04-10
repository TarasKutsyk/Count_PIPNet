# histograms.py

import numpy as np
import os
import plotly.graph_objects as go
import torch
import torch.utils.data
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Callable, Union

# --- Default Helper Functions (Can be replaced by user's implementations) ---

def class_idx_to_name(idx: int) -> str:
    """
    Default implementation to map a class index (integer) to a
    human-readable class name (string).
    """
    # Example mapping, replace with actual dataset class names if needed
    match = {
        0: "1 Circle", 1: "1 Triangle", 2: "1 Hexagon",
        3: "2 Circles", 4: "2 Triangles", 5: "2 Hexagons",
        6: "3 Circles", 7: "3 Triangles", 8: "3 Hexagons",
    }
    return match.get(idx, f"Class {idx}") # Fallback for unknown indices


def net_forward(
    xs: torch.Tensor,
    net: torch.nn.Module,
    is_count_pipnet: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Performs a forward pass through the network, handling the specific
    inference mode required by different PIPNet variants.

    Args:
        xs: Input image tensor batch.
        net: The trained PIPNet or CountPIPNet model.
        is_count_pipnet: Boolean flag indicating the model type.

    Returns:
        A tuple containing:
        - proto_features: Raw feature maps before pooling.
        - pooled: Pooled activation values (raw counts for CountPIPNet in non-inference).
        - out: Final classification layer output logits.
    """
    # CountPIPNet requires non-inference mode during analysis to get raw counts
    run_in_inference_mode = not is_count_pipnet

    with torch.no_grad():
        # Assuming the network's forward method signature is consistent
        proto_features, pooled, out = net(xs, inference=run_in_inference_mode)

    return proto_features, pooled, out

# --- Data Collection Helper ---

def _collect_activations(
    net: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    is_count_pipnet: bool,
    max_images: int,
    num_prototypes: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Collects pooled prototype activations and corresponding ground-truth class labels
    from the provided dataloader, essential for subsequent analysis.

    Args:
        net: The trained model.
        dataloader: DataLoader providing image batches and labels.
        device: Computation device ('cuda' or 'cpu').
        is_count_pipnet: Flag indicating the model type.
        max_images: Maximum number of images to process.
        num_prototypes: Expected number of prototypes from the model.

    Returns:
        Tuple (activations, labels) as numpy arrays, or (None, None) on failure.
    """
    net.eval() # Ensure model is in evaluation mode
    all_activations_list = []
    all_class_labels_list = []
    processed_images = 0

    print(f"Collecting activations from up to {max_images} images...")

    # Use tqdm for progress visualization
    pbar_collect = tqdm(dataloader, desc="Collecting activations", ncols=100, leave=False)
    with torch.no_grad():
        for xs, ys in pbar_collect:
            # Stop if the image limit has been reached
            if processed_images >= max_images:
                break

            xs = xs.to(device)
            current_batch_size = xs.size(0)

            # Determine how many images from this batch fit within the limit
            remaining_slots = max_images - processed_images
            actual_batch_size = min(current_batch_size, remaining_slots)

            # If no more images can be processed, skip the rest of the batch loop
            if actual_batch_size <= 0:
                continue

            # Slice the batch if processing fewer images than available
            if actual_batch_size < current_batch_size:
                xs = xs[:actual_batch_size]
                ys = ys[:actual_batch_size]

            # Perform the forward pass to get activations
            try:
                 _, pooled, _ = net_forward(xs, net, is_count_pipnet)
            except Exception as e:
                 print(f"\nError during forward pass: {e}. Skipping batch.")
                 continue


            # Validate the shape of the pooled activations
            if pooled.ndim < 2 or pooled.shape[1] != num_prototypes:
                 print(f"\nWarning: Unexpected pooled activation shape {pooled.shape}. Expected (batch, {num_prototypes}). Skipping batch.")
                 continue

            # Store the results from the current batch
            all_activations_list.append(pooled.cpu())
            all_class_labels_list.append(ys.cpu())
            processed_images += actual_batch_size

            # Update progress bar description
            pbar_collect.set_postfix_str(f"Images: {processed_images}/{max_images}")

    # Cleanly close the progress bar
    pbar_collect.close()

    # Provide final status on image collection
    if processed_images < max_images and processed_images > 0:
         print(f"\nFinished collecting activations (processed {processed_images} images).")
    elif processed_images >= max_images:
         print(f"\nReached max_images limit ({max_images}). Stopped data collection.")

    # Consolidate collected data into numpy arrays
    if not all_activations_list:
        print("Error: No activation data was collected (list is empty).")
        return None, None

    try:
        all_activations = torch.cat(all_activations_list, dim=0).numpy()
        all_class_labels = torch.cat(all_class_labels_list, dim=0).numpy()
    except Exception as e:
        print(f"Error concatenating collected data tensors: {e}")
        return None, None

    print(f"Successfully collected activations for {all_activations.shape[0]} images. Activation matrix shape: {all_activations.shape}")
    return all_activations, all_class_labels

# --- Report Generation Helper ---

def _generate_zero_report(
    output_dir: str,
    prototypes_analyzed_indices: List[int],
    initial_prototypes_indices: List[int],
    all_activations: np.ndarray,
    near_zero_threshold: float
) -> None:
    """
    Generates and saves a text report summarizing the frequency of near-zero
    activations for the analyzed prototypes.

    Args:
        output_dir: Directory to save the report.
        prototypes_analyzed_indices: Indices of prototypes included in the main analysis.
        initial_prototypes_indices: Indices initially selected based on importance.
        all_activations: Numpy array of collected activations.
        near_zero_threshold: Activation threshold defining 'near-zero'.
    """
    print("Generating near-zero activation report...")
    prototype_zero_stats: Dict[int, Dict] = {}
    num_images_analyzed = all_activations.shape[0]

    # Calculate near-zero statistics for each relevant prototype
    for p_idx in prototypes_analyzed_indices:
        # Ensure prototype index is valid for the collected data
        if p_idx >= all_activations.shape[1]:
            print(f"Warning (Report): Skipping prototype index {p_idx} - out of bounds for activations.")
            continue

        proto_activations = all_activations[:, p_idx]
        num_samples = len(proto_activations)

        if num_samples == 0:
            continue # Should not happen if index check passed, but safe

        # Calculate near-zero metrics
        zero_mask = proto_activations < near_zero_threshold
        num_zero = int(np.sum(zero_mask))
        zero_pct = (num_zero / num_samples * 100.0) if num_samples > 0 else 0.0
        prototype_zero_stats[p_idx] = {'num_zero': num_zero, 'total': num_samples, 'pct_zero': zero_pct}

    # Aggregate statistics across the analyzed prototypes
    all_zero_pcts = [stat_dict['pct_zero'] for stat_dict in prototype_zero_stats.values() if stat_dict['total'] > 0]
    if all_zero_pcts:
        avg_zero_pct = float(np.mean(all_zero_pcts))
        median_zero_pct = float(np.median(all_zero_pcts))
        std_zero_pct = float(np.std(all_zero_pcts))
        # Count prototypes that are almost always near-zero
        num_mostly_zero = sum(1 for pct in all_zero_pcts if pct > 95.0)
        # Count prototypes that are essentially always near-zero (allowing for float inaccuracy)
        num_always_zero = sum(1 for pct in all_zero_pcts if pct >= (100.0 - 1e-4))
    else:
        # Default values if no valid prototypes were analyzed
        avg_zero_pct = median_zero_pct = std_zero_pct = 0.0
        num_mostly_zero = num_always_zero = 0

    # --- Format Report Content ---
    report_lines = [
        "="*60,
        "      Zero/Near-Zero Activation Summary Report",
        "="*60,
        f"Threshold for near-zero activation: {near_zero_threshold}",
        f"Total images analyzed: {num_images_analyzed}",
        f"Prototypes initially selected (based on importance): {len(initial_prototypes_indices)}",
        f"Prototypes analyzed in detail (after outlier filtering): {len(prototypes_analyzed_indices)}\n",
        "Aggregate Statistics (based on analyzed prototypes):",
        f"  Average % near-zero activations: {avg_zero_pct:.2f}%",
        f"  Median % near-zero activations: {median_zero_pct:.2f}%",
        f"  Std Dev of % near-zero activations: {std_zero_pct:.2f}%",
        f"  Number of prototypes with >95% near-zero activations: {num_mostly_zero}",
        f"  Number of prototypes with ~100% near-zero activations: {num_always_zero}\n",
        "Per-Prototype Details (% near-zero for analyzed prototypes):",
        # Display details sorted by prototype index for consistency
        *(f"  Proto {p_idx: <4}: {stats['pct_zero']: >6.2f}% ({stats['num_zero']: >4}/{stats['total']: <4})"
          for p_idx, stats in sorted(prototype_zero_stats.items()))
    ]
    report_lines.append("="*60)

    # --- Save Report ---
    report_path = os.path.join(output_dir, "zero_activation_summary_report.txt")
    try:
        with open(report_path, 'w') as f:
            f.write("\n".join(report_lines))
        print(f"Saved zero activation report to: {report_path}")
    except IOError as e:
        print(f"Error saving zero activation report to {report_path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while saving the zero report: {e}")

# --- Heatmap Generation Helper ---

def _generate_summary_heatmap(
    output_dir: str,
    prototypes_for_heatmap: List[int],
    all_activations: np.ndarray,
    all_class_labels: np.ndarray,
    unique_classes: List[int],
    class_idx_to_name_func: Callable[[int], str],
    near_zero_threshold: float
) -> None:
    """
    Generates a heatmap summarizing the average non-zero activation for each
    prototype-class pair and saves it.

    Args:
        output_dir: Directory to save the heatmap.
        prototypes_for_heatmap: Indices of prototypes to include.
        all_activations: Numpy array of collected activations.
        all_class_labels: Numpy array of collected class labels.
        unique_classes: Sorted list of unique class indices.
        class_idx_to_name_func: Function mapping class indices to names.
        near_zero_threshold: Threshold defining non-zero activation.
    """
    print("\nGenerating summary heatmap...")

    # Check if there are any prototypes left to plot
    if not prototypes_for_heatmap:
         print("Warning: No prototypes provided for heatmap generation. Skipping heatmap.")
         return

    num_heatmap_protos = len(prototypes_for_heatmap)
    num_heatmap_classes = len(unique_classes)

    # Initialize matrix to store heatmap data (average non-zero activation)
    z_data = np.zeros((num_heatmap_classes, num_heatmap_protos))

# Inside _generate_summary_heatmap function:

    # Populate the heatmap data matrix
    for i, cls in enumerate(unique_classes):
        # Create a boolean mask for samples belonging to the current class
        class_mask = (all_class_labels == cls)

        # Skip if this class has no samples in the collected data
        if not np.any(class_mask):
            # Ensure the row corresponding to this class is filled with zeros
            # (or another appropriate default like np.nan if preferred)
            z_data[i, :] = 0.0
            continue

        for j, p_idx in enumerate(prototypes_for_heatmap):
            # Ensure prototype index is valid
            if p_idx >= all_activations.shape[1]:
                 print(f"Warning (Heatmap): Skipping prototype index {p_idx} - out of bounds.")
                 # Assign a default value if skipping
                 z_data[i, j] = 0.0 # Or np.nan
                 continue

            # Select activations for the current class and prototype
            class_proto_activations = all_activations[class_mask, p_idx]

            # *** Calculate average using ALL activations for this class-proto pair ***
            # Check if there are any activations for this class before averaging
            if len(class_proto_activations) > 0:
                z_data[i, j] = float(np.mean(class_proto_activations))
            else:
                # Assign 0 if somehow the class mask was non-empty but slicing failed
                # or if num_class_samples was 0 (already handled by outer check)
                z_data[i, j] = 0.0

    # --- Prepare Labels for Axes ---
    class_labels_names = [class_idx_to_name_func(cls) for cls in unique_classes]
    prototype_labels_names = [f'P{p}' for p in prototypes_for_heatmap] # Short label 'P{index}'

    # --- Create Heatmap Figure with Plotly ---
    summary_fig = go.Figure(data=go.Heatmap(
        z=z_data,                 # The calculated average activations
        y=class_labels_names,     # Class names on the y-axis
        x=prototype_labels_names, # Prototype names on the x-axis
        colorscale='Viridis',     # Choose a visually distinct colormap
        colorbar=dict(
            title="Avg <br>Activation", # Label for the color scale bar
            len=0.75,                           # Adjust colorbar length
            thickness=15                        # Adjust colorbar thickness
            ),
        zmin=0 # Force the color scale to start at 0
    ))

    # --- Configure Heatmap Layout ---
    summary_fig.update_layout(
        title="Average Activation by Class and Prototype (Filtered)",
        # Dynamically adjust plot size based on number of items
        width=max(800, num_heatmap_protos * 15 + 250),
        height=max(500, num_heatmap_classes * 15 + 200),
        template="plotly_white", # Use a clean layout theme
        xaxis_title="Prototype",
        yaxis_title="Class",
        title_x=0.5, # Center the main title
        xaxis=dict(
            tickangle=-60,    # Angle x-axis labels if many prototypes
            automargin=True   # Automatically adjust margin for labels
            ),
        yaxis=dict(
            tickmode='array', # Ensure all class labels are displayed
            tickvals=list(range(num_heatmap_classes)),
            ticktext=class_labels_names,
            automargin=True   # Automatically adjust margin for labels
            ),
        margin=dict(l=150, b=100, t=80, r=50) # Define plot margins
    )

    # --- Save Heatmap ---
    try:
        heatmap_path_html = os.path.join(output_dir, "prototype_class_activation_summary_filtered.html")
        heatmap_path_png = os.path.join(output_dir, "prototype_class_activation_summary_filtered.png")
        summary_fig.write_html(heatmap_path_html)
        summary_fig.write_image(heatmap_path_png, scale=2) # Higher resolution PNG
        print(f"Saved filtered summary heatmap to {heatmap_path_html} and .png")
    except IOError as e:
        print(f"Error saving summary heatmap files: {e}")
    except Exception as e:
        # Catch other potential Plotly/system errors
        print(f"An unexpected error occurred while saving the summary heatmap: {e}")


# --- Main Visualization Function ---

def plot_prototype_activations_by_class(
    net: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: str,
    only_important_prototypes: bool = True,
    prototype_importance: Optional[Union[np.ndarray, torch.Tensor]] = None,
    importance_threshold: float = 1e-1,
    is_count_pipnet: Optional[bool] = None,
    max_images: int = 10000,
    class_idx_to_name_func: Callable[[int], str] = class_idx_to_name,
    plot_outlier_threshold: float = 100.0,
    near_zero_threshold: float = 0.01,
    plot_always_histograms: bool = False,
    num_bins_continuous: int = 50
) -> List[int]:
    """
    Plots class-conditional histograms of prototype activations, showing normalized frequencies.

    Generates histogram plots showing the distribution of activation values (or counts)
    for selected prototypes, broken down by the ground-truth class of the input image.
    The y-axis represents the normalized frequency of activations *within that class's
    non-zero activations* for the specific prototype. It also produces a summary heatmap
    and a report on near-zero activations. Prototypes with extremely high average
    activations can be optionally excluded from plotting.

    Args:
        net: The trained model (PIPNet or CountPIPNet).
        dataloader: DataLoader for the dataset to analyze (e.g., projectloader).
        device: Device to run the model on ('cuda' or 'cpu').
        output_dir: Directory path to save the generated plots and reports.
        only_important_prototypes: If True, filter prototypes based on importance scores.
        prototype_importance: 1-D array/tensor of importance scores for each prototype.
                              Required if `only_important_prototypes` is True.
        importance_threshold: Minimum importance score for a prototype to be plotted when
                              `only_important_prototypes` is True.
        is_count_pipnet: Specifies if the model is CountPIPNet. Auto-detected if None.
        max_images: Maximum number of images to process from the dataloader.
        class_idx_to_name_func: Function mapping class indices to readable names.
        plot_outlier_threshold: Prototypes with avg non-zero activation above this value
                                will be skipped during plot generation and excluded from
                                the heatmap. Use `float('inf')` to disable.
        near_zero_threshold: Activations below this are considered 'near-zero' for reporting.

    Returns:
        A list of prototype indices initially selected based on importance criteria,
        before any outlier filtering was applied.
    """
    print("\n--- Starting Class-Conditional Activation Histogram Generation ---")

    # --- 1. Initialization and Setup ---
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating output directory '{output_dir}': {e}")
        return [] # Cannot proceed

    # Auto-detect model type if not specified
    if is_count_pipnet is None:
        is_count_pipnet = hasattr(net.module, '_max_count')
    model_type_str = 'CountPIPNet' if is_count_pipnet else 'PIPNet'
    print(f"Model Type: {model_type_str}")

    # Get model configuration
    try:
        max_count = net.module._max_count if is_count_pipnet else None
        num_prototypes = net.module._num_prototypes
    except AttributeError as e:
        print(f"Error accessing required model attributes (_max_count or _num_prototypes): {e}")
        return []

    # --- 2. Initial Prototype Selection (Based on Importance) ---
    initial_prototypes_to_plot: List[int] = []
    if only_important_prototypes:
        # Ensure prototype importance scores are provided and valid
        if prototype_importance is None:
            print("Error: `prototype_importance` is required when `only_important_prototypes=True`.")
            return []
        try:
            # Convert importance scores to a numpy array for consistent processing
            if isinstance(prototype_importance, torch.Tensor):
                prototype_importance_np = prototype_importance.detach().cpu().numpy()
            else:
                prototype_importance_np = np.array(prototype_importance)

            # Select prototype indices exceeding the importance threshold
            initial_prototypes_to_plot = np.where(prototype_importance_np > importance_threshold)[0].tolist()

            # If no prototypes meet the threshold, select all as a fallback
            if not initial_prototypes_to_plot:
                print(f"Warning: No prototypes found with importance > {importance_threshold}. Selecting all initially.")
                initial_prototypes_to_plot = list(range(num_prototypes))
        except Exception as e:
            print(f"Error processing prototype importance scores: {e}")
            return []
    else:
        # If not filtering by importance, select all prototypes
        initial_prototypes_to_plot = list(range(num_prototypes))

    # Exit if no prototypes are selected for any reason
    if not initial_prototypes_to_plot:
        print("Error: No prototypes selected for analysis.")
        return []
    print(f"Initially selected {len(initial_prototypes_to_plot)} prototypes based on importance criteria.")

    # --- 3. Data Collection ---
    # Collect activations and labels using the helper function
    all_activations, all_class_labels = _collect_activations(
        net, dataloader, device, is_count_pipnet, max_images, num_prototypes
    )

    # Exit if data collection failed
    if all_activations is None or all_class_labels is None:
        print("Aborting histogram generation due to data collection failure.")
        return initial_prototypes_to_plot # Return original list

    # --- 4. Calculate Overall Averages and Filter Outlier Prototypes ---
    print("\nCalculating overall average activations for outlier filtering...")
    overall_avg_activations: Dict[int, float] = {}
    for p_idx in initial_prototypes_to_plot:
        # Check index validity against collected data
        if p_idx >= all_activations.shape[1]:
             print(f"Warning (Filtering): Skipping prototype index {p_idx} - out of bounds.")
             continue

        proto_acts = all_activations[:, p_idx]
        # Calculate average based only on activations above the near-zero threshold
        non_zero_acts = proto_acts[proto_acts > near_zero_threshold]
        overall_avg_activations[p_idx] = float(np.mean(non_zero_acts)) if len(non_zero_acts) > 0 else 0.0

    # Create the final list of prototypes to plot, excluding outliers
    final_prototypes_to_plot = [
        p for p in initial_prototypes_to_plot
        if p in overall_avg_activations and overall_avg_activations[p] <= plot_outlier_threshold
    ]
    # Identify which prototypes were skipped
    skipped_outlier_prototypes = sorted(list(set(initial_prototypes_to_plot) - set(final_prototypes_to_plot)))

    if skipped_outlier_prototypes:
        print(f"Skipping plots for {len(skipped_outlier_prototypes)} prototypes with avg activation > {plot_outlier_threshold:.1f}: {skipped_outlier_prototypes}")
        for p in skipped_outlier_prototypes:
            print(f"  - Prototype {p}: Avg Activation = {overall_avg_activations[p]:.2f}")
    elif len(initial_prototypes_to_plot) > 0:
         print(f"No prototypes filtered based on outlier threshold (>{plot_outlier_threshold:.1f}).")

    # --- 5. Generate Near-Zero Activation Report ---
    # Report based on the final list of prototypes that will be plotted
    _generate_zero_report(
        output_dir, final_prototypes_to_plot, initial_prototypes_to_plot,
        all_activations, near_zero_threshold
    )

    # If all prototypes were filtered out, skip plotting
    if not final_prototypes_to_plot:
         print("\nWarning: All initially selected prototypes were filtered out as outliers. No individual plots or heatmap will be generated.")
         return initial_prototypes_to_plot # Return the original list

    # --- 6. Generate Individual Prototype Plots ---
    # Get unique class labels present in the collected data
    unique_classes = sorted(np.unique(all_class_labels).tolist())
    if not unique_classes:
        print("Error: No unique classes found in collected data. Cannot generate plots.")
        return initial_prototypes_to_plot

    # Define a color palette for distinguishing classes in plots
    bright_colors = ["#FF4500","#00CED1","#FFD700","#32CD32","#BA55D3","#FF6347","#4169E1","#2E8B57","#FF1493","#1E90FF","#FF8C00","#00FA9A","#9932CC","#00BFFF","#FF69B4"]
    class_colors = {cls: bright_colors[i % len(bright_colors)] for i, cls in enumerate(unique_classes)}

    print(f"\nGenerating individual plots for {len(final_prototypes_to_plot)} prototypes...")
    # Loop over the FINAL filtered list of prototypes
    plot_pbar = tqdm(final_prototypes_to_plot, desc="Generating prototype plots", ncols=100, leave=False)
    for p in plot_pbar:
        plot_pbar.set_postfix_str(f"Prototype: {p}")

        # --- 6a. Prepare Data for Prototype p ---
        # Index validity check
        if p >= all_activations.shape[1]:
            print(f"Warning (Plotting): Skipping prototype index {p} - out of bounds.")
            continue

        proto_activations = all_activations[:, p]
        fig = go.Figure() # Initialize a new figure for each prototype

        # --- 6b. Define X-axis Range ---
        if is_count_pipnet:
            # For discrete counts, ensure the range covers observed max + buffer
            actual_max = np.max(proto_activations) if len(proto_activations) > 0 else 0.0
            upper_bound = max(max_count + 1.5 if max_count is not None else 1.5, actual_max + 1.0)
            x_range = [-0.5, upper_bound] # Start at -0.5 for centered bars
        else:
            # For continuous activations, focus on the positive range
            non_zero_acts_p = proto_activations[proto_activations > near_zero_threshold]
            upper_bound = np.max(non_zero_acts_p) * 1.1 if len(non_zero_acts_p) > 0 else 1.0
            # Ensure range includes at least 0.1 for PIPNet threshold visualization
            x_range = [0.0, max(upper_bound, 0.15)]

        # --- 6c. Calculate Activation Statistics for Annotation ---
        # Calculate overall non-zero percentage for this prototype
        non_zero_mask_p = proto_activations >= near_zero_threshold # Use >= for non-zero
        total_samples_p = len(proto_activations)
        overall_non_zero_pct_p = (np.sum(non_zero_mask_p) / total_samples_p * 100.0) if total_samples_p > 0 else 0.0

        # Calculate non-zero counts and total samples per class
        non_zero_counts_per_class: Dict[int, int] = {}
        total_samples_per_class: Dict[int, int] = {}
        for cls in unique_classes:
            class_mask = (all_class_labels == cls)
            num_class_samples = int(np.sum(class_mask))
            total_samples_per_class[cls] = num_class_samples
            if num_class_samples > 0:
                # Count activations >= threshold within this class
                non_zero_counts_per_class[cls] = int(np.sum(proto_activations[class_mask] >= near_zero_threshold))
            else:
                non_zero_counts_per_class[cls] = 0

        # --- 6d. Determine Class Activity (Frequency) and Sort ---
        # This determines the plotting order (most active classes on top)
        class_activity: Dict[int, float] = {}
        for cls in unique_classes:
            class_mask = (all_class_labels == cls)
            num_class_samples = int(np.sum(class_mask))
            if num_class_samples == 0:
                class_activity[cls] = 0.0
                continue
            # Proportion of non-zero activations within this class
            non_zero_count = np.sum(proto_activations[class_mask] > near_zero_threshold)
            class_activity[cls] = non_zero_count / num_class_samples
        # Sort classes by descending frequency of activation
        sorted_classes = sorted(class_activity.items(), key=lambda item: item[1], reverse=True)

        # --- 6e. Create Histogram Traces for Each Class ---
        # The y-axis will represent normalized frequency *within the non-zero activations of that class*
        class_traces = []
        for cls, activity in sorted_classes:
            class_mask = (all_class_labels == cls)
            # Skip if no samples for this class
            if not np.any(class_mask):
                continue

            class_activations = proto_activations[class_mask]
            # Isolate activations above the near-zero threshold for frequency calculation
            non_zero_activations = class_activations[class_activations > near_zero_threshold]

            # Skip if this class has no non-zero activations for this prototype
            if len(non_zero_activations) == 0:
                continue

            class_name = class_idx_to_name_func(cls)
            # Visually distinguish the top 3 most frequently activating classes
            is_top_class = cls in [c for c, _ in sorted_classes[:3]]

            if is_count_pipnet and not plot_always_histograms:
                # --- Logic for Discrete Counts (CountPIPNet) ---
                values, counts = np.unique(non_zero_activations, return_counts=True)
                # Normalize frequency within this class's non-zero activations
                normalized_freq = counts / len(non_zero_activations)
                # X-values are the unique count values converted to strings for categorical axis
                x_values_for_plot = [str(v) for v in values]
                # Bar width is less relevant for categorical, let Plotly handle
                bar_width = None
            else:
                # --- Logic for Continuous Activations (PIPNet) ---
                # Define histogram range slightly wider than [0, 1] to catch edges if needed
                # Clip max range to avoid issues if activations slightly exceed 1 due to float precision
                hist_max_val = max(1.0, np.max(non_zero_activations) if len(non_zero_activations) > 0 else 1.0)
                hist_range = (near_zero_threshold, hist_max_val * 1.01) # Add small buffer at max

                # Calculate histogram counts and bin edges
                counts, bin_edges = np.histogram(
                    non_zero_activations,
                    bins=num_bins_continuous,
                    range=hist_range
                )
                # Normalize counts to get frequency
                normalized_freq = counts / len(non_zero_activations)
                # Calculate bin centers for plotting bar positions
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
                # Calculate bin width for consistent bar appearance
                bar_width = bin_edges[1] - bin_edges[0]
                # X-values are the bin centers
                x_values_for_plot = bin_centers

            # Create the Plotly Bar trace using calculated x, y, and width
            class_traces.append(go.Bar(
                x=x_values_for_plot,        # Use calculated bin centers or unique counts
                y=normalized_freq,          # Use calculated normalized frequency
                name=f"{class_name}",
                marker=dict(color=class_colors.get(cls, '#cccccc'), line=dict(width=0)),
                opacity=0.9 if is_top_class else 0.7,
                width=bar_width             # Use calculated bin width for continuous case
            ))

        # --- 6f. Add Traces and Annotations to Figure ---
        # Add traces in reverse order (most active class plotted last/on top)
        for trace in reversed(class_traces):
            fig.add_trace(trace)

        # *** MODIFIED: Annotation showing Non-Zero counts per class ***
        non_zero_text_lines = [f"<b>Non-Zero (>={near_zero_threshold:.2f}) Activations:</b><br>  Overall: {overall_non_zero_pct_p:.1f}%"]

        # Sort classes by the number of non-zero counts for this prototype, descending
        # Filter out classes with 0 total samples to avoid division errors or meaningless entries
        valid_classes_for_sort = [cls for cls in unique_classes if total_samples_per_class.get(cls, 0) > 0]
        sorted_non_zero_counts = sorted(
            valid_classes_for_sort,
            key=lambda cls: non_zero_counts_per_class.get(cls, 0),
            reverse=True
        )

        # Display details for top classes contributing to non-zero counts
        for cls in sorted_non_zero_counts:
            non_zero_count = non_zero_counts_per_class.get(cls, 0)
            total_count = total_samples_per_class.get(cls, 0)
            # Show classes with at least one non-zero count
            if non_zero_count > 0:
                class_name_nz = class_idx_to_name_func(cls)
                # Report as: Non-Zero Count / Total Samples for Class
                non_zero_text_lines.append(f"- {class_name_nz}: {non_zero_count}/{total_count}")

        non_zero_text = "<br>".join(non_zero_text_lines) # Combine lines with HTML line breaks

        # Add the annotation box to the plot
        fig.add_annotation(
            x=0.02, y=0.98,           # Position near top-left corner
            xref="paper", yref="paper", # Use relative paper coordinates
            text=non_zero_text,       # Formatted text showing non-zero stats
            showarrow=False,          # No arrow pointing to data
            bgcolor="rgba(255,255,255,0.85)", # Semi-transparent white background
            bordercolor="black",      # Black border
            borderwidth=1,
            align='left',             # Align text to the left within the box
            valign='top',             # Align box to the top
            font=dict(size=10)        # Font size for annotation
        )

        # Add Vertical Reference Lines
        if is_count_pipnet and max_count is not None:
            # Add dotted lines *between* integer count values
            for count_val in range(1, int(np.ceil(x_range[1]))):
                 fig.add_vline(x=count_val - 0.5, line=dict(color="darkgrey", width=1, dash="dot"))
                 # Add text label *at* the integer count position (e.g., "1", "2")
                 fig.add_annotation(x=count_val, y=1.0, yref='paper', yshift=5, text=str(count_val),
                                    showarrow=False, font=dict(size=10, color="darkgrey"))
        else:
             # For standard PIPNet, add a reference line at a common threshold (e.g., 0.1)
             line_val = 0.1
             fig.add_vline(x=line_val, line=dict(color="black", width=1, dash="dash"),
                           annotation_text=f"{line_val:.1f}", annotation_position="top right")

        # --- 6g. Configure Plot Title ---
        title_parts = [f"Prototype {p} Activation Distribution"]
        if is_count_pipnet:
            title_parts.append("(Counts)")

        # Safely retrieve and format the importance score
        imp_val_str = "N/A"
        if 'prototype_importance_np' in locals() and prototype_importance_np is not None:
             # Check if index 'p' is valid for the importance array
             if p < len(prototype_importance_np):
                  imp_val_str = f"{prototype_importance_np[p]:.4f}"
        title_parts.append(f"(Importance: {imp_val_str})")
        plot_title = " ".join(title_parts)

        # Add annotation for the class that most frequently activates this prototype
        if sorted_classes:
            native_class, native_activity = sorted_classes[0]
            # Add this annotation only if the activation frequency is somewhat significant
            if native_activity > 0.01:
                native_class_name = class_idx_to_name_func(native_class)
                fig.add_annotation(
                    x=0.5, y=1.07, # Position slightly above the main plot title
                    xref="paper", yref="paper",
                    text=f"Most Frequent Class: <b>{native_class_name}</b> ({native_activity:.1%})", # Highlight name
                    showarrow=False,
                    font=dict(size=14, color=class_colors.get(native_class, 'black')), # Use class color
                )

        # --- 6h. Configure Plot Layout ---
        fig.update_layout(
            title=dict(
                text=plot_title, # Set the main title text
                x=0.5,           # Center the title
                y=0.95           # Position title slightly lower
            ),
            width=1100, height=650, # Define overall plot dimensions
            template="plotly_white", # Use a clean background theme

            xaxis_title="Activation Value" if not is_count_pipnet else "Count Value",
            yaxis_title="Normalized Frequency (within class, non-zero acts.)", # Describe y-axis content

            xaxis=dict(range=x_range), # Apply calculated x-axis range

            yaxis=dict(
                autorange=True,      # Automatically determine y-axis range based on data
                showticklabels=True, # Display frequency values on the y-axis
                title_standoff=10,   # Space between y-axis title and ticks
                tickformat=".2f"     # Format y-axis tick labels to 2 decimal places
            ),

            legend_title_text="Class", # Set the title for the legend box
            barmode='overlay',       # Overlay bars from different classes at the same x-value
            bargap=0.1,              # Space between bars for different x-values
            bargroupgap=0.0,         # No space between bars at the same x-value (for overlay)

            legend=dict(
                itemsizing='constant', # Keep legend marker size consistent
                font=dict(size=11),    # Adjust legend font size
                traceorder='reversed', # Match legend order to the plotting order
                bgcolor='rgba(255,255,255,0.7)' # Semi-transparent background for legend
            ),
            margin=dict(t=120, b=80, l=80, r=50) # Adjust plot margins (top, bottom, left, right)
        )

        # --- 6i. Save Plot to Files ---
        try:
            plot_path_html = os.path.join(output_dir, f"prototype_{p}_class_distribution.html")
            plot_path_png = os.path.join(output_dir, f"prototype_{p}_class_distribution.png")
            # Save interactive HTML version
            fig.write_html(plot_path_html)
            # Save static PNG version at higher resolution
            fig.write_image(plot_path_png, scale=2)
        except Exception as e:
            # Report any errors during file saving
            print(f"  Error saving plot for prototype {p}: {e}")

    # Close the progress bar for plotting
    plot_pbar.close()
    print(f"\nFinished generating individual plots for {len(final_prototypes_to_plot)} prototypes.")

    # --- 7. Generate Summary Heatmap ---
    # Uses the same filtered list `final_prototypes_to_plot`
    _generate_summary_heatmap(
        output_dir, final_prototypes_to_plot, all_activations, all_class_labels,
        unique_classes, class_idx_to_name_func, near_zero_threshold
    )

    print("\n--- Class-Conditional Activation Histogram Generation Complete ---")

    # Return the list of prototypes initially selected before outlier filtering,
    # reflecting the selection based purely on the importance criteria.
    return initial_prototypes_to_plot

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