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

def net_forward(xs, net, is_count_pipnet=True):
    """
    Performs a forward pass with suitable params depending on which network (pipnet or count_pipnet) is used.
    """
    run_in_inference = not is_count_pipnet # for count-pip-net run the model in training mode, for pipnet otherwise
    with torch.no_grad():
        pfs, pooled, out = net(xs, inference=run_in_inference)

    return pfs, pooled, out

@torch.no_grad()
def visualize_topk(net, projectloader, num_classes, device, foldername, 
                  args: argparse.Namespace, k=10, verbose=True, 
                  are_pretraining_prototypes=False, plot_histograms=True,
                  visualize_prototype_maps=True, max_feature_maps_per_prototype=3):
    """
    Wrapper function for prototype visualization that delegates to the appropriate implementation
    based on the model type (CountPIPNet vs regular PIPNet).
    """
    # Detect if using CountPIPNet by checking for _max_count attribute
    is_count_pipnet = hasattr(net.module, '_max_count')
    
    if is_count_pipnet:
        return visualize_topk_count_pipnet(
            net, projectloader, num_classes, device, foldername, 
            args, k, verbose, are_pretraining_prototypes, plot_histograms,
            visualize_prototype_maps, max_feature_maps_per_prototype
        )
    else:
        return visualize_topk_pipnet(
            net, projectloader, num_classes, device, foldername, 
            args, k, verbose, are_pretraining_prototypes, plot_histograms,
            visualize_prototype_maps, max_feature_maps_per_prototype
        )

@torch.no_grad()
def visualize_topk_pipnet(net, projectloader, num_classes, device, foldername, 
                          args: argparse.Namespace, k=10, verbose=True, 
                          are_pretraining_prototypes=False, plot_histograms=True,
                          visualize_prototype_maps=True, max_feature_maps_per_prototype=3):
    """
    Original visualization function for standard PIPNet models.
    This version uses the single-pass approach to avoid inconsistencies.
    """
    # Required imports for visualization
    from PIL import Image, ImageDraw as D
    
    print("Visualizing prototypes for topk (PIPNet)...", flush=True)
    dir = os.path.join(args.log_dir, foldername)
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    # Setup tracking dictionaries
    num_prototypes = net.module._num_prototypes
    patchsize, skip = get_patch_size(args)
    imgs = projectloader.dataset.imgs
    
    # Make sure the model is in evaluation mode
    net.eval()
    classification_weights = net.module._classification.weight
    
    # Determine prototype importance
    prototype_importance = torch.max(classification_weights, dim=0)[0]

    # When visualizing pretraining prototypes, include all prototypes regardless of weights
    if are_pretraining_prototypes:
        # Override prototype_importance with ones so all prototypes pass the threshold check
        prototype_importance = torch.ones_like(prototype_importance)

    # Plot activation histograms if requested
    if plot_histograms:
        histogram_dir = os.path.join(dir, "activation_histograms")
        os.makedirs(histogram_dir, exist_ok=True)
        
        # Use same setting for prototype importance as in main function
        only_important_prototypes = not are_pretraining_prototypes
        
        plot_prototype_activations_histograms(
            net=net,
            dataloader=projectloader,
            device=device,
            output_dir=histogram_dir,
            only_important_prototypes=only_important_prototypes,
            importance_threshold=1e-3,
            is_count_pipnet=False,
        )

    if verbose:
        # Debug information
        with torch.no_grad():
            print(f"Visualizing {'pretraining' if are_pretraining_prototypes else 'trained'} prototypes")

            xs, ys = next(iter(projectloader))
            xs, ys = xs.to(device), ys.to(device)
            
            # Run the model
            pfs, pooled, out = net(xs, inference=True)

            pooled = pooled.squeeze(0)
            pfs = pfs.squeeze(0)
            
            # Print basic shape information
            print(f"Pooled shape: {pooled.shape}, min: {pooled.min().item()}, max: {pooled.max().item()}")
            print(f"Feature maps shape: {pfs.shape}")
            print(f"Non-zero pooled values: {(pooled > 0.1).sum().item()} out of {pooled.numel()}")
            
            # Show pooled values
            print("Pooled values:", [round(v, 4) for v in pooled.tolist()])
            
            # Handle classification weights
            print(f"Classification weights shape: {net.module._classification.weight.shape}")
            
            if not are_pretraining_prototypes:
                print("Max classification weight per prototype:", [round(v, 4) for v in prototype_importance.tolist()])
                print("Prototypes with weight > 1e-3:", (prototype_importance > 1e-3).sum().item())
                
                # Show which specific prototypes are being used
                used_protos = torch.where(prototype_importance > 1e-3)[0].tolist()
                print("Prototype indices with weight > 1e-1:", used_protos)
            else:
                print("Visualizing all prototypes (ignoring classification weights during pretraining)")

    # Initialize storage for top-k prototypes
    prototype_storage = {}
    for p in range(num_prototypes):
        if are_pretraining_prototypes or prototype_importance[p] > 1e-1:
            prototype_storage[p] = []
    
    # Create feature maps directory if needed
    if visualize_prototype_maps:
        feature_maps_dir = os.path.join(dir, "feature_maps")
        os.makedirs(feature_maps_dir, exist_ok=True)
    
    # Show progress on progress bar
    img_iter = tqdm(enumerate(projectloader),
                    total=len(projectloader),
                    desc='Processing dataset (single pass)',
                    ncols=0)
    
    abstained = 0
    
    # Process each image only once
    for i, (xs, ys) in img_iter:
        xs, ys = xs.to(device), ys.to(device)
        
        with torch.no_grad():
            # Get model outputs
            proto_features, pooled, out = net(xs, inference=True)
            
            # Check if model abstained
            outmax = torch.amax(out, dim=1)[0]
            if outmax.item() == 0.:
                abstained += 1
            
            # Get image path
            img_path = imgs[i]
            if isinstance(img_path, tuple) or isinstance(img_path, list):
                img_path = img_path[0]
            
            # Process each prototype
            for p in prototype_storage.keys():
                # Get the activation score for this prototype
                score = pooled.squeeze(0)[p].item()
                
                # Get the prototype feature map
                feature_map = proto_features.squeeze(0)[p].clone().cpu()
                
                # Find the location of maximum activation
                max_h, h_idx = torch.max(feature_map, dim=0)
                max_w, w_idx = torch.max(max_h, dim=0)
                h_idx = h_idx[w_idx].item()
                w_idx = w_idx.item()
                
                # Calculate patch coordinates
                h_coor_min, h_coor_max, w_coor_min, w_coor_max = get_img_coordinates(
                    args.image_size, proto_features.shape, patchsize, skip, h_idx, w_idx
                )
                
                # Skip loading the image until we know we need it
                # Only process this prototype-image pair if it's a candidate for top-k
                images = prototype_storage[p]
                if len(images) < k or score > images[-1]['score']:
                    # Now load and process the image
                    image = transforms.Resize(size=(args.image_size, args.image_size))(Image.open(img_path))
                    img_tensor = transforms.ToTensor()(image).unsqueeze_(0)
                    img_tensor_patch = img_tensor[0, :, h_coor_min:h_coor_max, w_coor_min:w_coor_max].clone().cpu()
                    
                    # Store all the data we need
                    image_data = {
                        'index': i,
                        'image_path': img_path,
                        'image_tensor': img_tensor.squeeze(0).clone().cpu(),
                        'patch_tensor': img_tensor_patch,
                        'coords': (h_coor_min, h_coor_max, w_coor_min, w_coor_max),
                        'feature_map': feature_map,
                        'score': score,
                        'h_idx': h_idx,
                        'w_idx': w_idx
                    }
                    
                    # Add to the prototype's images if it's in the top-k
                    if len(images) < k:
                        images.append(image_data)
                        images.sort(key=lambda x: x['score'], reverse=True)
                    else:
                        # Already have k images, replace the lowest score
                        images[-1] = image_data
                        images.sort(key=lambda x: x['score'], reverse=True)
    
    print("Abstained:", abstained, flush=True)
    
    # Find prototypes without significant activations
    prototypes_not_used = []
    for p, images in prototype_storage.items():
        found_significant = False
        for img_data in images:
            if img_data['score'] > 0.1:
                found_significant = True
                break
        
        if not found_significant:
            prototypes_not_used.append(p)
    
    print(len(prototypes_not_used), "prototypes do not have any similarity score > 0.1. Will be ignored in visualisation.")
    
    # Create prototype visualizations
    all_tensors = []
    tensors_per_prototype = {p: [] for p in prototype_storage.keys()}
    saved = {p: 0 for p in prototype_storage.keys()}
    
    for p, images in prototype_storage.items():
        if p in prototypes_not_used:
            continue
        
        # Create tensor patches
        for img_data in images:
            # Add to the tensor collection
            saved[p] += 1
            tensors_per_prototype[p].append(img_data['patch_tensor'])
        
        # Add text next to each topk-grid
        if len(images) > 0:
            img_data = images[0]  # Use first image to get patch size
            patch_shape = img_data['patch_tensor'].shape
            
            text = "P " + str(p)
            txtimage = Image.new("RGB", (patch_shape[1], patch_shape[2]), (0, 0, 0))
            draw = D.Draw(txtimage)
            draw.text((patch_shape[1]//2, patch_shape[2]//2), text, anchor='mm', fill="white")
            txttensor = transforms.ToTensor()(txtimage)
            tensors_per_prototype[p].append(txttensor)
            
            # Save top-k image patches in grid
            try:
                grid = torchvision.utils.make_grid(tensors_per_prototype[p], nrow=k+1, padding=1)
                torchvision.utils.save_image(grid, os.path.join(dir, f"grid_topk_{p}.png"))
                if saved[p] >= k:
                    all_tensors.extend(tensors_per_prototype[p])
            except Exception as e:
                print(f"Error creating grid for prototype {p}: {e}")
    
    # Create a grid with all prototypes
    if len(all_tensors) > 0:
        grid = torchvision.utils.make_grid(all_tensors, nrow=k+1, padding=1)
        torchvision.utils.save_image(grid, os.path.join(dir, "grid_topk_all.png"))
    else:
        print("Prototypes not visualized. Try to pretrain longer.", flush=True)
    
    # Create feature map visualizations if requested
    if visualize_prototype_maps:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import numpy as np
        from scipy.ndimage import zoom
        
        print("Creating prototype feature map visualizations...", flush=True)
        
        for p, images in prototype_storage.items():
            if p in prototypes_not_used or len(images) == 0:
                continue
            
            # Create a prototype-specific directory
            proto_feature_dir = os.path.join(feature_maps_dir, f"prototype_{p}")
            os.makedirs(proto_feature_dir, exist_ok=True)
            
            # Select diverse activations - highest, middle, and lowest that's still > 0.1
            selected_indices = []
            
            # Always include highest activation
            selected_indices.append(0)
            
            # If we have more than 2 samples, add middle sample
            if len(images) > 2:
                selected_indices.append(len(images) // 2)
            
            # If we have more than 1 sample, add lowest sample that still exceeds threshold
            if len(images) > 1:
                # Find the lowest index where score > 0.1, or use the last element
                lowest_idx = len(images) - 1
                while lowest_idx > 0 and images[lowest_idx]['score'] < 0.1:
                    lowest_idx -= 1
                
                if lowest_idx not in selected_indices:
                    selected_indices.append(lowest_idx)
            
            # Limit to max 3 feature maps per prototype
            selected_indices = selected_indices[:max_feature_maps_per_prototype]
            
            # Create visualizations for selected samples
            for i, idx in enumerate(selected_indices):
                img_data = images[idx]
                
                # Extract data
                img_path = img_data['image_path']
                img_tensor = img_data['image_tensor']
                coords = img_data['coords']
                feature_map = img_data['feature_map']
                score = img_data['score']
                h_idx = img_data['h_idx']
                w_idx = img_data['w_idx']
                
                feature_map_np = feature_map.numpy()
                feature_map_sum = feature_map_np.sum()
                
                # Create a filename based on the prototype, rank and score
                base_filename = f"proto_{p}_rank_{i+1}_of_{len(selected_indices)}_score_{score:.3f}"
                
                # Convert image tensor to numpy for plotting
                img_np = img_tensor.permute(1, 2, 0).numpy()
                
                # Save the original image with patch rectangle
                plt.figure(figsize=(10, 8))
                plt.imshow(img_np)
                h_min, h_max, w_min, w_max = coords
                rect = plt.Rectangle((w_min, h_min), w_max-w_min, h_max-h_min, 
                                    fill=False, edgecolor='yellow', linewidth=2)
                plt.gca().add_patch(rect)
                plt.axis('off')
                plt.title(f"Prototype {p} - Activation: {score:.3f} (Map Sum: {feature_map_sum:.3f})")
                plt.tight_layout()
                plt.savefig(os.path.join(proto_feature_dir, f"{base_filename}_original.png"), 
                            bbox_inches='tight', dpi=150)
                plt.close()
                
                # Create side-by-side visualization with original and heatmap
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
                
                # Show original image on the left
                ax1.imshow(img_np)
                rect = plt.Rectangle((w_min, h_min), w_max-w_min, h_max-h_min, 
                                    fill=False, edgecolor='yellow', linewidth=2)
                ax1.add_patch(rect)
                ax1.set_title("Original Image")
                ax1.axis('off')
                
                # Show feature map on the right
                heatmap = ax2.imshow(feature_map_np, cmap='viridis')
                ax2.scatter(w_idx, h_idx, marker='x', color='red', s=100)
                ax2.set_title("Feature Map Heatmap")
                ax2.axis('off')
                fig.colorbar(heatmap, ax=ax2, label="Activation")
                
                plt.suptitle(f"Prototype {p} - Activation: {score:.3f}")
                plt.tight_layout()
                plt.savefig(os.path.join(proto_feature_dir, f"{base_filename}_feature_map.png"), 
                            bbox_inches='tight', dpi=150)
                plt.close()
                
                # Create overlay visualization
                plt.figure(figsize=(10, 8))
                plt.imshow(img_np)
                
                # Resize feature map to match image dimensions
                zoom_y = img_np.shape[0] / feature_map_np.shape[0]
                zoom_x = img_np.shape[1] / feature_map_np.shape[1]
                resized_feature_map = zoom(feature_map_np, (zoom_y, zoom_x))
                
                # Create a masked version of the feature map for cleaner overlay
                mask = resized_feature_map > 0.1  # Only show activations above threshold
                overlay = np.zeros((*resized_feature_map.shape, 4))  # RGBA
                
                # Create colormap
                cmap = cm.get_cmap('viridis')
                
                # Fill overlay with colors from colormap based on activation values
                for y in range(resized_feature_map.shape[0]):
                    for x in range(resized_feature_map.shape[1]):
                        if mask[y, x]:
                            color = cmap(resized_feature_map[y, x])
                            overlay[y, x] = (*color[:3], 0.7)  # RGB with 0.7 alpha
                
                plt.imshow(overlay, alpha=0.7)
                rect = plt.Rectangle((w_min, h_min), w_max-w_min, h_max-h_min, 
                                    fill=False, edgecolor='yellow', linewidth=2)
                plt.gca().add_patch(rect)
                plt.title(f"Prototype {p} - Feature Map Overlay (Activation: {score:.3f})")
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(proto_feature_dir, f"{base_filename}_overlay.png"), 
                            bbox_inches='tight', dpi=150)
                plt.close()
    
    # Only return the indices and scores for backward compatibility
    topks = {}
    for p, images in prototype_storage.items():
        if p not in prototypes_not_used:
            topks[p] = [(img_data['index'], img_data['score']) for img_data in images]
    
    return topks

@torch.no_grad()
def visualize_topk_count_pipnet(net, projectloader, num_classes, device, foldername, 
                               args: argparse.Namespace, k=10, verbose=True, 
                               are_pretraining_prototypes=False, plot_histograms=True,
                               visualize_prototype_maps=True, max_feature_maps_per_prototype=3):
    """
    Visualization function specially designed for CountPIPNet models.
    This version uses class information to select examples with different counts
    and processes everything in a single pass for efficiency.
    """
    # Required imports for visualization
    from PIL import Image, ImageDraw as D

    # Class to count mapping configuration
    # Maps class ranges to their corresponding count values
    # This can be edited based on dataset organization
    class_to_count_mapping = {
        # Format: (start_class_inclusive, end_class_inclusive): count_value
        (1, 3): 1,  # Classes 1-3 correspond to count 1
        (4, 6): 2,  # Classes 4-6 correspond to count 2
        (7, 9): 3   # Classes 7-9 correspond to count 3
    }
    
    # Function to determine count from class label
    def get_count_from_class(class_label):
        for (start, end), count in class_to_count_mapping.items():
            if start <= class_label + 1 <= end: # class labels are numbered from zero, so add one
                return count

        raise ValueError(f"No class label found for label {class_label + 1}")
    
    # Get available count values from the mapping
    available_counts = sorted(set(class_to_count_mapping.values()))
    
    print(f"Visualizing prototypes for topk (CountPIPNet with class-based count selection)...", flush=True)
    print(f"Using class-to-count mapping: {class_to_count_mapping}")
    print(f"Available counts: {available_counts}")
    
    dir = os.path.join(args.log_dir, foldername)
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    # Setup tracking dictionaries
    num_prototypes = net.module._num_prototypes
    max_count = net.module._max_count
    patchsize, skip = get_patch_size(args)
    imgs = projectloader.dataset.imgs
    
    # Make sure the model is in evaluation mode
    net.eval()
    classification_weights = net.module._classification.weight
    
    # Determine prototype importance for CountPIPNet
    prototype_importance = torch.tensor(
        [net.module.get_prototype_importance(i) for i in range(num_prototypes)]
    )

    if verbose:
        # Debug information
        with torch.no_grad():
            print(f"Detected model type: CountPIPNet with max_count={max_count}")
            print(f"Visualizing {'pretraining' if are_pretraining_prototypes else 'trained'} prototypes")
            print(f"Examples per count group: {k}")

            xs, ys = next(iter(projectloader))
            xs, ys = xs.to(device), ys.to(device)
            
            # Run the model - inference should be False to get raw counts
            pfs, pooled, out = net(xs, inference=False)

            pooled = pooled.squeeze(0)
            pfs = pfs.squeeze(0)
            
            # Print basic shape information
            print(f"Pooled shape: {pooled.shape}, min: {pooled.min().item()}, max: {pooled.max().item()}")
            print(f"Feature maps shape: {pfs.shape}")
            print(f"Non-zero pooled values: {(pooled > 0.1).sum().item()} out of {pooled.numel()}")
            
            # Show pooled values
            print("Pooled values (counts):", [round(v, 4) for v in pooled.tolist()])
            
            # Handle classification weights correctly for CountPIPNet
            if classification_weights.shape[1] > num_prototypes:
                print(f"Classification weights shape: {classification_weights.shape} (expanded)")
            else:
                print(f"Classification weights shape: {classification_weights.shape} (not expanded)")
            
            if not are_pretraining_prototypes:
                print("Max classification weight per prototype:", [round(v, 4) for v in prototype_importance.tolist()])
                print("Prototypes with weight > 1e-1:", (prototype_importance > 1e-1).sum().item())
                
                # Show which specific prototypes are being used
                used_protos = torch.where(prototype_importance > 1e-1)[0].tolist()
                print("Prototype indices with weight > 1e-1:", used_protos)
            else:
                print("Visualizing all prototypes (ignoring classification weights during pretraining)")

    # Plot activation histograms if requested
    if plot_histograms:
        histogram_dir = os.path.join(dir, "activation_histograms")
        os.makedirs(histogram_dir, exist_ok=True)
        
        plot_prototype_activations_histograms(
            net=net,
            dataloader=projectloader,
            device=device,
            output_dir=histogram_dir,
            only_important_prototypes=not are_pretraining_prototypes,
            prototype_importance=prototype_importance,
            importance_threshold=1e-1,
            is_count_pipnet=True,
        )

    # Create feature maps directory if needed
    if visualize_prototype_maps:
        feature_maps_dir = os.path.join(dir, "feature_maps")
        os.makedirs(feature_maps_dir, exist_ok=True)
    
    # Initialize top-k buffer for each prototype and count group
    # We'll maintain k examples with highest activation for each prototype and count
    topk_storage = {}
    for p in range(num_prototypes):
        if are_pretraining_prototypes or prototype_importance[p] > 0.1:
            topk_storage[p] = {}
            for count in available_counts:
                # Initialize with empty lists that can hold up to k items
                topk_storage[p][count] = []
    
    # Show progress on progress bar
    img_iter = tqdm(enumerate(projectloader),
                    total=len(projectloader),
                    desc='Single-pass processing for class-based count selection',
                    ncols=0)
    
    abstained = 0
    
    # Single pass through the dataset
    for i, (xs, ys) in img_iter:
        xs, ys = xs.to(device), ys.to(device)
        
        with torch.no_grad():
            # Get model outputs
            proto_features, pooled_counts, out = net(xs, inference=False)
            
            # Check if model abstained
            outmax = torch.amax(out, dim=1)[0]
            if outmax.item() == 0.:
                abstained += 1
            
            # Get image path
            img_path = imgs[i]
            if isinstance(img_path, tuple) or isinstance(img_path, list):
                img_path = img_path[0]
            
            # Get the class label and determine the count group
            class_label = ys.item()
            count_group = get_count_from_class(class_label)
            
            # Skip if not in our available counts
            if count_group not in available_counts:
                continue
                
            # Process each prototype
            for p in topk_storage.keys():
                # Get the count for this prototype (for reference, not used for grouping)
                model_count = pooled_counts.squeeze(0)[p].item()
                
                # Skip processing if activation is very low
                if model_count < 0.01:
                    continue
                
                # Get the prototype feature map
                feature_map = proto_features.squeeze(0)[p].clone().cpu()
                
                # Find the location of maximum activation
                max_h, h_idx = torch.max(feature_map, dim=0)
                max_w, w_idx = torch.max(max_h, dim=0)
                h_idx = h_idx[w_idx].item()
                w_idx = w_idx.item()
                
                # Calculate patch coordinates
                h_coor_min, h_coor_max, w_coor_min, w_coor_max = get_img_coordinates(
                    args.image_size, proto_features.shape, patchsize, skip, h_idx, w_idx
                )
                
                # Check if this image should be in the top-k for this prototype and count group
                topk_list = topk_storage[p][count_group]
                
                # If we haven't collected k examples yet, or this activation is higher than the lowest in our collection
                if len(topk_list) < k or model_count > topk_list[-1][0]:
                    # Only now we load and process the image since it's a candidate for top-k
                    image = transforms.Resize(size=(args.image_size, args.image_size))(Image.open(img_path))
                    img_tensor = transforms.ToTensor()(image).unsqueeze_(0)
                    img_tensor_patch = img_tensor[0, :, h_coor_min:h_coor_max, w_coor_min:w_coor_max].clone().cpu()
                    
                    # Full image data package to store
                    img_data = {
                        'index': i,
                        'image_path': img_path,
                        'class_label': class_label,
                        'count_group': count_group,
                        'model_count': model_count,
                        'coords': (h_coor_min, h_coor_max, w_coor_min, w_coor_max),
                        'feature_map': feature_map,
                        'h_idx': h_idx,
                        'w_idx': w_idx,
                        'image_tensor': img_tensor.squeeze(0).clone().cpu(),
                        'patch_tensor': img_tensor_patch
                    }
                    
                    # Update the top-k list
                    if len(topk_list) < k:
                        topk_list.append((model_count, img_data))
                        # Sort by model_count using a key function
                        topk_list.sort(key=lambda x: x[0], reverse=True)
                    else:
                        # Replace the lowest scoring example
                        topk_list[-1] = (model_count, img_data)
                        # Sort by model_count using a key function
                        topk_list.sort(key=lambda x: x[0], reverse=True)
    
    print("Abstained:", abstained, flush=True)
    
    # Extract just the image data from our sorted collections
    prototype_selected_images = {}
    for p in topk_storage.keys():
        prototype_selected_images[p] = []
        for count in available_counts:
            # Extract the img_data part, discarding the score used for sorting
            prototype_selected_images[p].extend([item[1] for item in topk_storage[p][count]])
    
    # Find prototypes without any examples
    prototypes_not_used = []
    for p, images in prototype_selected_images.items():
        if len(images) == 0:
            prototypes_not_used.append(p)
    
    print(f"{len(prototypes_not_used)} prototypes do not have any examples. Will be ignored in visualization.")
    
    # Create prototype visualizations uniformly sampled across count groups
    all_tensors = []
    tensors_per_prototype = {p: [] for p in prototype_selected_images.keys()}
    saved = {p: 0 for p in prototype_selected_images.keys()}

    for p, images in prototype_selected_images.items():
        if p in prototypes_not_used:
            continue
        
        # Group by count
        count_groups = {}
        for img_data in images:
            count = img_data['count_group']
            if count not in count_groups:
                count_groups[count] = []
            count_groups[count].append(img_data)
        
        # Sample uniformly across count groups
        uniform_samples = []
        if len(count_groups) > 0:
            # Calculate samples per count group
            samples_per_group = k // len(count_groups)
            extra_samples = k % len(count_groups)
            
            # Collect samples from each count group
            for count, count_images in sorted(count_groups.items()):
                # Sort by activation score
                sorted_images = sorted(count_images, key=lambda x: x['model_count'], reverse=True)
                # Take samples_per_group + 1 extra if needed
                samples_to_take = samples_per_group + (1 if extra_samples > 0 else 0)
                extra_samples -= 1 if extra_samples > 0 else 0
                
                # Add samples from this count group
                uniform_samples.extend(sorted_images[:min(samples_to_take, len(sorted_images))])
        
        # If we don't have enough samples, add more from any group
        if len(uniform_samples) < k:
            # Collect all unused images
            unused_images = []
            for count, count_images in count_groups.items():
                # Sort by activation score
                sorted_images = sorted(count_images, key=lambda x: x['model_count'], reverse=True)
                # Find the cutoff point for this group
                samples_to_take = samples_per_group + (1 if count in list(sorted(count_groups.keys()))[:extra_samples] else 0)
                # Add unused images
                if len(sorted_images) > samples_to_take:
                    unused_images.extend(sorted_images[samples_to_take:])
            
            # Sort unused images by activation score
            unused_images.sort(key=lambda x: x['model_count'], reverse=True)
            # Add as many as needed to reach k
            uniform_samples.extend(unused_images[:min(k - len(uniform_samples), len(unused_images))])
        
        # Ensure we have at most k samples
        uniform_samples = uniform_samples[:k]
        
        # Sort the final samples by activation score
        uniform_samples.sort(key=lambda x: x['model_count'], reverse=True)
        
        # Add to the tensor collection
        for img_data in uniform_samples:
            saved[p] += 1
            tensors_per_prototype[p].append(img_data['patch_tensor'])
        
        # Add text next to each topk-grid
        if len(uniform_samples) > 0:
            img_data = uniform_samples[0]  # Use first image to get patch size
            patch_shape = img_data['patch_tensor'].shape
            
            text = "P " + str(p)
            txtimage = Image.new("RGB", (patch_shape[1], patch_shape[2]), (0, 0, 0))
            draw = D.Draw(txtimage)
            draw.text((patch_shape[1]//2, patch_shape[2]//2), text, anchor='mm', fill="white")
            txttensor = transforms.ToTensor()(txtimage)
            tensors_per_prototype[p].append(txttensor)
            
            # Save top-k image patches in grid
            try:
                grid = torchvision.utils.make_grid(tensors_per_prototype[p], nrow=k+1, padding=1)
                torchvision.utils.save_image(grid, os.path.join(dir, f"grid_topk_{p}.png"))
                if saved[p] >= k:
                    all_tensors.extend(tensors_per_prototype[p])
            except Exception as e:
                print(f"Error creating grid for prototype {p}: {e}")

    # Create a grid with all prototypes
    if len(all_tensors) > 0:
        try:
            grid = torchvision.utils.make_grid(all_tensors, nrow=k+1, padding=1)
            torchvision.utils.save_image(grid, os.path.join(dir, "grid_topk_all.png"))
        except Exception as e:
            print(f"Error creating combined grid: {e}")
    else:
        print("Prototypes not visualized. Try to pretrain longer.", flush=True)
        
    # Create feature map visualizations with additional count information
    if visualize_prototype_maps:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import numpy as np
        from scipy.ndimage import zoom
        
        print("Creating prototype feature map visualizations with count information...", flush=True)
        
        for p, images in prototype_selected_images.items():
            if p in prototypes_not_used or len(images) == 0:
                continue
            
            # Create a prototype-specific directory
            proto_feature_dir = os.path.join(feature_maps_dir, f"prototype_{p}")
            os.makedirs(proto_feature_dir, exist_ok=True)
            
            # Group by count
            count_groups = {}
            for img_data in images:
                count = img_data['count_group']
                if count not in count_groups:
                    count_groups[count] = []
                count_groups[count].append(img_data)
            
            # Select best example for each count group
            selected_images = []
            for count, count_images in sorted(count_groups.items()):
                if len(count_images) > 0:
                    # Sort by model's count value to get the best example
                    sorted_images = sorted(count_images, key=lambda x: x['model_count'], reverse=True)
                    selected_images.append(sorted_images[0])
            
            # Limit to max_feature_maps_per_prototype
            selected_images = selected_images[:max_feature_maps_per_prototype]
            
            # Create visualizations for selected samples
            for i, img_data in enumerate(selected_images):
                # Extract data
                img_path = img_data['image_path']
                img_tensor = img_data['image_tensor']
                coords = img_data['coords']
                feature_map = img_data['feature_map']
                count_group = img_data['count_group']
                model_count = img_data['model_count']
                class_label = img_data['class_label']
                h_idx = img_data['h_idx']
                w_idx = img_data['w_idx']
                
                # Debug the feature map vs. count
                feature_map_np = feature_map.numpy()
                feature_map_sum = feature_map_np.sum()
                feature_map_max = feature_map_np.max()
                feature_map_mean = feature_map_np.mean()
                
                # Create a filename based on the prototype and count
                base_filename = f"proto_{p}_count_{count_group}_model_count_{model_count:.1f}_class_{class_label}"
                
                # Save debug information
                debug_info = f"""Prototype {p} Count Statistics:
Count Group (from class): {count_group}
Model Count Value: {model_count:.3f}
Class Label: {class_label}
Feature map shape: {feature_map_np.shape}
Feature map sum: {feature_map_sum:.3f} 
Feature map max value: {feature_map_max:.3f}
Feature map mean: {feature_map_mean:.3f}
"""
                with open(os.path.join(proto_feature_dir, f"{base_filename}_debug.txt"), 'w') as f:
                    f.write(debug_info)
                
                # Convert image tensor to numpy for plotting
                img_np = img_tensor.permute(1, 2, 0).numpy()
                
                # Save the original image with patch rectangle
                plt.figure(figsize=(10, 8))
                plt.imshow(img_np)
                h_min, h_max, w_min, w_max = coords
                rect = plt.Rectangle((w_min, h_min), w_max-w_min, h_max-h_min, 
                                    fill=False, edgecolor='yellow', linewidth=2)
                plt.gca().add_patch(rect)
                plt.axis('off')
                plt.title(f"Prototype {p} - Count: {count_group} (Model Count: {model_count:.1f}, Class: {class_label})")
                plt.tight_layout()
                plt.savefig(os.path.join(proto_feature_dir, f"{base_filename}_original.png"), 
                            bbox_inches='tight', dpi=150)
                plt.close()
                
                # Create side-by-side visualization with original and heatmap
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
                
                # Show original image on the left
                ax1.imshow(img_np)
                rect = plt.Rectangle((w_min, h_min), w_max-w_min, h_max-h_min, 
                                    fill=False, edgecolor='yellow', linewidth=2)
                ax1.add_patch(rect)
                ax1.set_title("Original Image")
                ax1.axis('off')
                
                # Show feature map on the right
                heatmap = ax2.imshow(feature_map_np, cmap='viridis')
                ax2.scatter(w_idx, h_idx, marker='x', color='red', s=100)
                ax2.set_title("Feature Map Heatmap")
                ax2.axis('off')
                fig.colorbar(heatmap, ax=ax2, label="Activation")
                
                plt.suptitle(f"Prototype {p} - Count: {count_group} (Model Count: {model_count:.1f})")
                plt.tight_layout()
                plt.savefig(os.path.join(proto_feature_dir, f"{base_filename}_feature_map.png"), 
                            bbox_inches='tight', dpi=150)
                plt.close()
                
                # Create overlay visualization
                plt.figure(figsize=(10, 8))
                plt.imshow(img_np)
                
                # Resize feature map to match image dimensions
                zoom_y = img_np.shape[0] / feature_map_np.shape[0]
                zoom_x = img_np.shape[1] / feature_map_np.shape[1]
                resized_feature_map = zoom(feature_map_np, (zoom_y, zoom_x))
                
                # Create a masked version of the feature map for cleaner overlay
                mask = resized_feature_map > 0.1  # Only show activations above threshold
                overlay = np.zeros((*resized_feature_map.shape, 4))  # RGBA
                
                # Create colormap
                cmap = cm.get_cmap('viridis')
                
                # Fill overlay with colors from colormap based on activation values
                for y in range(resized_feature_map.shape[0]):
                    for x in range(resized_feature_map.shape[1]):
                        if mask[y, x]:
                            color = cmap(resized_feature_map[y, x])
                            overlay[y, x] = (*color[:3], 0.7)  # RGB with 0.7 alpha
                
                plt.imshow(overlay, alpha=0.7)
                rect = plt.Rectangle((w_min, h_min), w_max-w_min, h_max-h_min, 
                                    fill=False, edgecolor='yellow', linewidth=2)
                plt.gca().add_patch(rect)
                plt.title(f"Prototype {p} - Count: {count_group} (Model Count: {model_count:.1f})")
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(proto_feature_dir, f"{base_filename}_overlay.png"), 
                            bbox_inches='tight', dpi=150)
                plt.close()
    
    # Only return the indices and scores for backward compatibility
    topks = {}
    for p, images in prototype_selected_images.items():
        if p not in prototypes_not_used:
            topks[p] = [(img_data['index'], img_data['model_count']) for img_data in images]
    
    return topks

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

def visualize(net, projectloader, num_classes, device, foldername, args: argparse.Namespace):
    print("Visualizing prototypes...", flush=True)
    dir = os.path.join(args.log_dir, foldername)
    if not os.path.exists(dir):
        os.makedirs(dir)

    is_count_pipnet = hasattr(net.module, '_max_count')

    near_imgs_dirs = dict()
    seen_max = dict()
    saved = dict()
    saved_ys = dict()
    tensors_per_prototype = dict()
    abstainedimgs = set()
    notabstainedimgs = set()
    
    for p in range(net.module._num_prototypes):
        near_imgs_dir = os.path.join(dir, str(p))
        near_imgs_dirs[p]=near_imgs_dir
        seen_max[p]=0.
        saved[p]=0
        saved_ys[p]=[]
        tensors_per_prototype[p]=[]
    
    patchsize, skip = get_patch_size(args)

    imgs = projectloader.dataset.imgs
    
    # skip some images for visualisation to speed up the process
    if len(imgs)/num_classes <10:
        skip_img=10
    elif len(imgs)/num_classes < 50:
        skip_img=5
    else:
        skip_img = 2
    print("Every", skip_img, "is skipped in order to speed up the visualisation process", flush=True)

    # Make sure the model is in evaluation mode
    net.eval()
    classification_weights = net.module._classification.weight
    # Show progress on progress bar
    img_iter = tqdm(enumerate(projectloader),
                    total=len(projectloader),
                    mininterval=100.,
                    desc='Visualizing',
                    ncols=0)

    # Iterate through the data
    images_seen_before = 0
    for i, (xs, ys) in img_iter: #shuffle is false so should lead to same order as in imgs
        if i % skip_img == 0:
            images_seen_before+=xs.shape[0]
            continue
        
        xs, ys = xs.to(device), ys.to(device)
        # Use the model to classify this batch of input data
        with torch.no_grad():
            softmaxes, pooled, out = net_forward(xs, net, is_count_pipnet)

        max_per_prototype, max_idx_per_prototype = torch.max(softmaxes, dim=0)
        # In PyTorch, images are represented as [channels, height, width]
        max_per_prototype_h, max_idx_per_prototype_h = torch.max(max_per_prototype, dim=1)
        max_per_prototype_w, max_idx_per_prototype_w = torch.max(max_per_prototype_h, dim=1)
        for p in range(0, net.module._num_prototypes):
            c_weight = torch.max(classification_weights[:,p]) #ignore prototypes that are not relevant to any class
            if c_weight>0:
                h_idx = max_idx_per_prototype_h[p, max_idx_per_prototype_w[p]]
                w_idx = max_idx_per_prototype_w[p]
                idx_to_select = max_idx_per_prototype[p,h_idx, w_idx].item()
                found_max = max_per_prototype[p,h_idx, w_idx].item()

                imgname = imgs[images_seen_before+idx_to_select]
                if out.max() < 1e-8:
                    abstainedimgs.add(imgname)
                else:
                    notabstainedimgs.add(imgname)
                
                if found_max > seen_max[p]:
                    seen_max[p]=found_max
               
                if found_max > 0.5:
                    img_to_open = imgs[images_seen_before+idx_to_select]
                    if isinstance(img_to_open, tuple) or isinstance(img_to_open, list): #dataset contains tuples of (img,label)
                        imglabel = img_to_open[1]
                        img_to_open = img_to_open[0]

                    image = transforms.Resize(size=(args.image_size, args.image_size))(Image.open(img_to_open).convert("RGB"))
                    img_tensor = transforms.ToTensor()(image).unsqueeze_(0) #shape (1, 3, h, w)
                    h_coor_min, h_coor_max, w_coor_min, w_coor_max = get_img_coordinates(args.image_size, softmaxes.shape, patchsize, skip, h_idx, w_idx)
                    img_tensor_patch = img_tensor[0, :, h_coor_min:h_coor_max, w_coor_min:w_coor_max]
                    saved[p]+=1
                    tensors_per_prototype[p].append((img_tensor_patch, found_max))
                    
                    save_path = os.path.join(dir, "prototype_%s")%str(p)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    draw = D.Draw(image)
                    draw.rectangle([(w_coor_min,h_coor_min), (w_coor_max, h_coor_max)], outline='yellow', width=2)
                    image.save(os.path.join(save_path, 'p%s_%s_%s_%s_rect.png'%(str(p),str(imglabel),str(round(found_max, 2)),str(img_to_open.split('/')[-1].split('.jpg')[0]))))
                    
        
        images_seen_before+=len(ys)

    print("num images abstained: ", len(abstainedimgs), flush=True)
    print("num images not abstained: ", len(notabstainedimgs), flush=True)
    for p in range(net.module._num_prototypes):
        if saved[p]>0:
            try:
                sorted_by_second = sorted(tensors_per_prototype[p], key=lambda tup: tup[1], reverse=True)
                sorted_ps = [i[0] for i in sorted_by_second]
                grid = torchvision.utils.make_grid(sorted_ps, nrow=16, padding=1)
                torchvision.utils.save_image(grid,os.path.join(dir,"grid_%s.png"%(str(p))))
            except RuntimeError:
                pass

# convert latent location to coordinates of image patch
def get_img_coordinates(img_size, softmaxes_shape, patchsize, skip, h_idx, w_idx):
    # in case latent output size is 26x26. For convnext with smaller strides. 
    if softmaxes_shape[1] == 26 and softmaxes_shape[2] == 26:
        #Since the outer latent patches have a smaller receptive field, skip size is set to 4 for the first and last patch. 8 for rest.
        h_coor_min = max(0,(h_idx-1)*skip+4)
        if h_idx < softmaxes_shape[-1]-1:
            h_coor_max = h_coor_min + patchsize
        else:
            h_coor_min -= 4
            h_coor_max = h_coor_min + patchsize
        w_coor_min = max(0,(w_idx-1)*skip+4)
        if w_idx < softmaxes_shape[-1]-1:
            w_coor_max = w_coor_min + patchsize
        else:
            w_coor_min -= 4
            w_coor_max = w_coor_min + patchsize
    else:
        h_coor_min = h_idx*skip
        h_coor_max = min(img_size, h_idx*skip+patchsize)
        w_coor_min = w_idx*skip
        w_coor_max = min(img_size, w_idx*skip+patchsize)                                    
    
    if h_idx == softmaxes_shape[1]-1:
        h_coor_max = img_size
    if w_idx == softmaxes_shape[2] -1:
        w_coor_max = img_size
    if h_coor_max == img_size:
        h_coor_min = img_size-patchsize
    if w_coor_max == img_size:
        w_coor_min = img_size-patchsize

    return h_coor_min, h_coor_max, w_coor_min, w_coor_max
    