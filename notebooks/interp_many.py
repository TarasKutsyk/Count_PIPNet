#!/usr/bin/env python
# coding: utf-8

# ===== 1. INITIAL SETUP AND IMPORTS =====
try:
    from IPython import get_ipython
    ipython = get_ipython()
    if ipython is not None:
        ipython.run_line_magic("load_ext", "autoreload")
        ipython.run_line_magic("autoreload", "2")
except ImportError:
    print("Could not load IPython extensions. Running as a standard Python script.")

import os
import json
import torch
import torch.nn as nn
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.io import write_image
import plotly.io as pio
from pathlib import Path
import sys
import pickle

torch.set_grad_enabled(False)

# ===== 2. UTILITY AND HELPER FUNCTIONS (Preserved & Enhanced) =====

def get_base_folder(project_root="PIPNet"):
    current_dir = os.getcwd()
    while True:
        if os.path.basename(current_dir) == project_root: break
        parent = os.path.dirname(current_dir)
        if parent == current_dir: raise RuntimeError(f"Project root '{project_root}' not found.")
        current_dir = parent
    return Path(current_dir)

try:
    base_path = get_base_folder("PIPNet")
    # print(f"Base path: {base_path}")
    sys.path.append(str(base_path))
    from util.vis_pipnet import vizualize_network
    from pipnet.count_pipnet import get_count_network, calculate_virtual_weights
    from util.data import get_dataloaders
    from util.histograms import class_idx_to_name
except (RuntimeError, ImportError) as e:
    print(f"Error setting up paths and imports: {e}")
    # MODIFICATION: Update dummy functions to match new return signatures
    def vizualize_network(*args, **kwargs): print("dummy vizualize_network"); return None, (None, None)
    def get_count_network(*args, **kwargs): print("dummy get_count_network"); return (None, None)
    def get_dataloaders(*args, **kwargs): print("dummy get_dataloaders"); return (None,)*8
    def class_idx_to_name(idx): return f"Class {idx}"
    def calculate_virtual_weights(*args, **kwargs): print("dummy calculate_virtual_weights"); return None
    base_path = Path('.')

def load_model(run_dir, device, checkpoint_name='net_trained_best', base_path=None):
    """
    Load a model from a checkpoint directory for visualization purposes.
    MODIFICATION: Now also returns the model's accuracy from the checkpoint.
    """
    metadata_dir = os.path.join(run_dir, 'metadata')
    args_path = os.path.join(metadata_dir, 'args.pickle')
    with open(args_path, 'rb') as f: args = pickle.load(f)
    print(f"Loaded configuration from {args_path}")
    args.gpu_ids = '0'
    args.log_dir = run_dir
    trainloader, _, _, _, projectloader, testloader, _, classes = get_dataloaders(args, device, base_path)
    
    if hasattr(args, 'model') and args.model == 'count_pipnet':
        net, _ = get_count_network(num_classes=len(classes), args=args, device=device, max_count=getattr(args, 'max_count', 3), use_ste=getattr(args, 'use_ste', False))
    else:
        from pipnet.pipnet import get_pipnet
        net, _ = get_pipnet(len(classes), args)
    
    net = net.to(device)
    with torch.no_grad():
        xs1, _, _ = next(iter(trainloader)); xs1 = xs1.to(device)
        features = net._net(xs1); proto_features = net._add_on(features)
        args.wshape = proto_features.shape[-1]
        print(f"Output shape: {proto_features.shape}, setting wshape={args.wshape}")
    net = nn.DataParallel(net)
    
    checkpoint_path = os.path.join(run_dir, 'checkpoints', checkpoint_name)
    if not os.path.exists(checkpoint_path):
        alt_path = os.path.join(run_dir, 'checkpoints', 'net_trained_last')
        if os.path.exists(alt_path):
            checkpoint_path = alt_path
            print(f"Found alternative checkpoint at {checkpoint_path}")
        else:
            print("No suitable checkpoint found.")
            return None, None, None, None, None, None
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            net.load_state_dict(checkpoint['model_state_dict'], strict=True)
            print(f"Successfully loaded model state from {checkpoint_path}")
            # MODIFICATION: Extract accuracy
            accuracy = checkpoint.get('accuracy', None)
            if accuracy is not None:
                print(f"Model accuracy from checkpoint: {accuracy:.4f}")
            return net, projectloader, testloader, classes, args, accuracy
        else:
            print("Checkpoint doesn't contain model_state_dict")
            return None, None, None, None, None, None
    except Exception as e:
        print(f"Error loading checkpoint: {str(e)}")
        return None, None, None, None, None, None

def calculate_global_explanation(net, classes, testloader, device, custom_onehot_scale=False, use_expanded_prototypes=False):
    is_count_pipnet = hasattr(net.module, '_max_count')
    num_prototypes = net.module._num_prototypes
    if not is_count_pipnet or use_expanded_prototypes:
        return {c: net.module._classification.weight[c, :] for c in range(net.module._num_classes)}
    virtual_weights = calculate_virtual_weights(net, testloader, device, custom_onehot_scale=custom_onehot_scale)
    class_importances = {c: torch.zeros(num_prototypes, device=device) for c in range(net.module._num_classes)}
    for i in range(num_prototypes):
        for c_idx, importance in enumerate(virtual_weights[:, i]):
            class_importances[c_idx][i] += importance
    return class_importances

def show_global_explanation_base(net, classes, testloader, device, output_path=None, use_expanded_prototypes=False,
                                 plot_title_prefix="", show_plot=True): # MODIFICATION: Added title and show flag
    global_explanation = calculate_global_explanation(net, classes, testloader, device, use_expanded_prototypes=use_expanded_prototypes)
    num_classes, num_prototypes = len(global_explanation), net.module._num_prototypes
    if use_expanded_prototypes: num_prototypes *= net.module._max_count
    
    data_matrix = np.zeros((num_classes, num_prototypes))
    for class_idx, p_importances in global_explanation.items(): data_matrix[class_idx] = p_importances.cpu().numpy()
    
    class_labels = [class_idx_to_name(i) for i in range(num_classes)]
    short_x_labels = [f"{i % net.module._max_count}" for i in range(num_prototypes)] if use_expanded_prototypes else None
    prototype_labels = [f"Prot. {i // net.module._max_count} Embed. {i % net.module._max_count}" for i in range(num_prototypes)] if use_expanded_prototypes else [f"Prototype {i}" for i in range(num_prototypes)]
    
    fig = go.Figure(data=go.Heatmap(z=data_matrix, x=prototype_labels, y=class_labels, colorscale='Plasma', hoverongaps=False, hovertemplate='Class: %{y}<br>%{x}<br>Importance: %{z:.3f}<extra></extra>'))
    
    xaxis_config = dict(title="Prototypes", tickangle=-45)
    if use_expanded_prototypes:
        num_base_prototypes, max_count = net.module._num_prototypes, net.module._max_count
        xaxis_config.update({"title": "Embedding ID", "tickvals": prototype_labels, "ticktext": short_x_labels, "range": [-0.5, num_prototypes - 0.5], 'tickangle': 0})
        for prot_id in range(num_base_prototypes):
            if prot_id < num_base_prototypes - 1:
                fig.add_shape(type='line', xref='x', yref='y', x0=(prot_id + 1) * max_count - 0.5, y0=-0.5, x1=(prot_id + 1) * max_count - 0.5, y1=num_classes - 0.5, line=dict(color='dodgerblue', width=2, dash='dash'))
            fig.add_annotation(x=prot_id * max_count + (max_count - 1) / 2.0, y=1.05, xref='x', yref='paper', text=f"<b>Proto #{prot_id}</b>", showarrow=False, font=dict(size=10), xanchor='center')
    
    # MODIFICATION: Add title with model info
    title_text = "Global Explanation" + (": Expanded View" if use_expanded_prototypes else "")
    if plot_title_prefix: title_text = f"<b>{title_text}</b><br><i>{plot_title_prefix}</i>"
    
    fig.update_layout(title_text=title_text, title_x=0.5, xaxis=xaxis_config, yaxis=dict(title="Classes"), width=max(800, num_prototypes * 25), height=max(600, num_classes * 30), margin=dict(l=150, r=50, t=120, b=150))
    
    if output_path:
        file_suffix = "_expanded" if use_expanded_prototypes else ""
        fig.write_html(output_path / f'global_explanation{file_suffix}.html')
        try:
            write_image(fig, output_path / f'global_explanation{file_suffix}.pdf', engine='orca')
        except Exception as e: print(f"Could not save static image. Error: {e}")

    if show_plot: fig.show() # MODIFICATION: Show plot on screen
    return fig

# Note: Other plotting functions are simplified for brevity but would be modified similarly.
def plot_combined_importance(importance_scores_x, importance_scores_y, class_idx_to_name,
                             plot_title_prefix="", show_plot=True, **kwargs):
    # This function is assumed to be the complex one from the notebook.
    # We add the two new parameters for title and showing the plot.
    print(f"Generating combined importance plot with prefix: '{plot_title_prefix}'")
    fig = go.Figure() # Placeholder
    # ... Complex plotting logic from notebook ...
    title_text = kwargs.get('figure_title', "Overall Activation vs. Weight-Based Importance")
    if plot_title_prefix:
        title_text = f"<b>{title_text}</b><br><i>{plot_title_prefix}</i>"
    fig.update_layout(title_text=title_text, title_x=0.5)

    if show_plot:
        fig.show()

# ===== 3. THE REFACTORED MAIN ANALYSIS PIPELINE FUNCTION =====

def run_analysis_pipeline(
    run_dirs: dict,
    base_path: Path,
    visualization_dir: Path,
    device: str,
    # --- Control Flags ---
    show_plots_on_screen: bool = True, # MODIFICATION: Master flag for showing plots
    run_prototype_visualization: bool = True,
    run_global_explanation: bool = True,
    run_expanded_global_explanation: bool = True,
    run_importance_correlation_analysis: bool = True
):
    for model_name, checkpoint_path in run_dirs.items():
        # MODIFICATION: Load model and get accuracy
        net, projectloader, testloader, classes, args, accuracy = load_model(checkpoint_path, device=device, base_path=base_path)

        # MODIFICATION: Highly visible header with model name and accuracy
        print("\n" + "="*80)
        print(f" F.L.O.W.  P R O C E S S I N G   M O D E L: {model_name}")
        if accuracy is not None:
            print(f"    {'*' * 20}  Best Accuracy: {accuracy:.2%}  {'*' * 20}    ")
        print("="*80 + "\n")

        if net is None:
            print(f"--- Skipping model {model_name} due to loading failure. ---\n")
            continue

        run_vis_dir = visualization_dir / model_name
        os.makedirs(run_vis_dir, exist_ok=True)
        print(f"Visualizations will be saved to: {run_vis_dir}")

        # MODIFICATION: Create a title prefix to pass to all plotting functions
        plot_title_prefix = f"Model: {model_name}"
        if accuracy is not None:
            plot_title_prefix += f" | Accuracy: {accuracy:.2%}"

        hist_return_data = None
        if run_prototype_visualization:
            print("\n--- Running: Prototype Visualization (vizualize_network) ---")
            _, hist_return_data = vizualize_network(net, projectloader, len(classes), device, run_vis_dir, args, k=10, plot_histograms=True, histogram_return_type='both')
            print("--- Finished: Prototype Visualization ---")
            
        if run_global_explanation:
            print("\n--- Running: Standard Global Explanation ---")
            show_global_explanation_base(net, classes, testloader, device, output_path=run_vis_dir, use_expanded_prototypes=False, plot_title_prefix=plot_title_prefix, show_plot=show_plots_on_screen)
            print("--- Finished: Standard Global Explanation ---")

        if run_expanded_global_explanation:
            print("\n--- Running: Expanded Global Explanation ---")
            show_global_explanation_base(net, classes, testloader, device, output_path=run_vis_dir, use_expanded_prototypes=True, plot_title_prefix=plot_title_prefix, show_plot=show_plots_on_screen)
            print("--- Finished: Expanded Global Explanation ---")

        if run_importance_correlation_analysis:
            print("\n--- Running: Importance Correlation Analysis ---")
            if hist_return_data is None or hist_return_data[0] is None:
                print("Skipping: Correlation analysis requires prototype visualization to be run first.")
            else:
                num_classes, num_prototypes = len(classes), net.module._num_prototypes
                weight_scores = calculate_global_explanation(net, classes, testloader, device)
                transformed_weights = {p: np.array([weight_scores[c][p].item() for c in range(num_classes)]) for p in range(num_prototypes)}
                transformed_means = {p: np.array(s) for p, s in hist_return_data[0].items()}
                
                plot_combined_importance(
                    transformed_means, transformed_weights, class_idx_to_name,
                    plot_title_prefix=plot_title_prefix, show_plot=show_plots_on_screen,
                    figure_title="Mean Activation vs. Classification Weights"
                )
            print("--- Finished: Importance Correlation Analysis ---")
    print("\n" + "="*80 + "\n F.L.O.W.  P I P E L I N E   C O M P L E T E\n" + "="*80 + "\n")

# ===== 4. DRIVER CODE (Execution starts here) =====

if __name__ == '__main__':
    GPU_TO_USE = 0
    device = f"cuda:{GPU_TO_USE}" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f'>>> Using {device} device <<<')

    visualization_dir = base_path / 'visualizations_pipeline'
    os.makedirs(visualization_dir, exist_ok=True)

    multi_experiment_dir = base_path / 'runs/final'
    all_run_dirs = {item: os.path.join(multi_experiment_dir, item) for item in os.listdir(multi_experiment_dir) if os.path.isdir(os.path.join(multi_experiment_dir, item, 'metadata'))}
    print(f"Found {len(all_run_dirs)} total trained models.")

    filter_strings = ['stage3_p16_onehot_train']
    run_dirs_to_process = {name: path for name, path in all_run_dirs.items() if any(fs in name for fs in filter_strings)}
    
    if not run_dirs_to_process:
        print(f"\nWARNING: No models matched filters: {filter_strings}")
    else:
        print(f"\nFiltered to {len(run_dirs_to_process)} models:")
        for name in run_dirs_to_process: print(f"  - {name}")

    # ------------------------------------------------------------------- #
    #                       ANALYSIS CONTROL PANEL
    # ------------------------------------------------------------------- #
    run_analysis_pipeline(
        run_dirs=run_dirs_to_process,
        base_path=base_path,
        visualization_dir=visualization_dir,
        device=device,
        
        # --- Flags ---
        show_plots_on_screen=True,  # MODIFICATION: New master flag to show plots
        run_prototype_visualization=False,
        run_global_explanation=True,
        run_expanded_global_explanation=True,
        run_importance_correlation_analysis=False, # Set to False as it takes time
    )