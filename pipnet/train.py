from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import math

def train_pipnet(net, train_loader, optimizer_net, optimizer_classifier, 
                 scheduler_net, scheduler_classifier, criterion, epoch, nr_epochs, device, 
                 is_count_pipnet=False, pretrain=False, finetune=False, progress_prefix: str = 'Train Epoch',
                 apply_counting_loss=False):

    # Make sure the model is in train mode
    net.train()
    
    if pretrain:
        # Disable training of classification layer
        net.module._classification.requires_grad = False
        progress_prefix = 'Pretrain Epoch'
    else:
        # Enable training of classification layer (disabled in case of pretraining)
        net.module._classification.requires_grad = True
    
    # Store info about the procedure
    train_info = dict()
    total_loss = 0.
    total_acc = 0.

    # Store info about the procedure
    train_info = dict()
    total_loss = 0.
    total_acc = 0.
    
    # Add trackers for loss components
    align_loss_raw_total = 0.
    tanh_loss_raw_total = 0.
    class_loss_raw_total = 0.
    align_loss_weighted_total = 0.
    tanh_loss_weighted_total = 0.
    class_loss_weighted_total = 0.
    weight_entropy_raw_total = 0.
    weight_entropy_weighted_total = 0.

    iters = len(train_loader)
    # Show progress on progress bar. 
    train_iter = tqdm(enumerate(train_loader),
                    total=len(train_loader),
                    desc=progress_prefix+'%s'%epoch,
                    mininterval=2.,
                    ncols=0)
    
    count_param=0
    for name, param in net.named_parameters():
        if param.requires_grad:
            count_param+=1           
            
    print("Number of parameters that require gradient: ", count_param, flush=True)

    if pretrain:
        align_pf_weight = (epoch/nr_epochs)*1.
        unif_weight = 0.5 #ignored
        t_weight = 5.
        cl_weight = 0.
        # Start with no entropy loss during pretraining
        entropy_weight = 0.
    else:
        align_pf_weight = 2. 
        t_weight = 2.
        unif_weight = 0.
        cl_weight = 3.
        # Annealing entropy weight from low to high during training
        # Start with low weight and gradually increase it
        entropy_weight = 1 # * (epoch / nr_epochs)
    
    print("Align weight: ", align_pf_weight, ", U_tanh weight: ", t_weight, "Class weight:", cl_weight, flush=True)
    print("Pretrain?", pretrain, "Finetune?", finetune, flush=True)
    
    lrs_net = []
    lrs_class = []
    # Iterate through the data set to update leaves, prototypes and network
    for i, (xs1, xs2, ys) in train_iter:       
        
        xs1, xs2, ys = xs1.to(device), xs2.to(device), ys.to(device)
       
        # Reset the gradients
        optimizer_classifier.zero_grad(set_to_none=True)
        optimizer_net.zero_grad(set_to_none=True)
       
        # Perform a forward pass through the network
        proto_features, pooled, out = net(torch.cat([xs1, xs2]))
         # Extract individual loss components
        loss, acc, loss_components = calculate_loss(proto_features, pooled, out, ys, 
                                                   align_pf_weight, t_weight, unif_weight, cl_weight, 
                                                   net.module._classification.normalization_multiplier, 
                                                   pretrain, finetune, criterion, train_iter, 
                                                   is_count_pipnet=is_count_pipnet, print=True, EPS=1e-8,
                                                   apply_counting_loss=apply_counting_loss, net=net,
                                                   entropy_weight=entropy_weight)
        
        # Accumulate loss components
        align_loss_raw_total += loss_components['align']
        tanh_loss_raw_total += loss_components['tanh']
        class_loss_raw_total += loss_components['class']
        align_loss_weighted_total += loss_components['align_weighted']
        tanh_loss_weighted_total += loss_components['tanh_weighted']
        class_loss_weighted_total += loss_components['class_weighted']
        weight_entropy_raw_total += loss_components.get('weight_entropy', 0.0)
        weight_entropy_weighted_total += loss_components.get('weight_entropy_weighted', 0.0)
        
        # Compute the gradient
        loss.backward()
        
        if epoch == 1 and i==0:
            print("Gradient norms:")
            for name, param in net.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    if grad_norm > 0:
                        print(f"{name}: {grad_norm}")

        if not pretrain:
            optimizer_classifier.step()   
            scheduler_classifier.step(epoch - 1 + (i/iters))
            lrs_class.append(scheduler_classifier.get_last_lr()[0])
     
        if not finetune:
            optimizer_net.step()
            scheduler_net.step() 
            lrs_net.append(scheduler_net.get_last_lr()[0])
        else:
            lrs_net.append(0.)
            
        with torch.no_grad():
            total_acc+=acc
            total_loss+=loss.item()

        # if not pretrain:
        #     with torch.no_grad():
        #         net.module._classification.weight.copy_(torch.clamp(net.module._classification.weight.data - 1e-3, min=0.)) #set weights in classification layer < 1e-3 to zero
        #         net.module._classification.normalization_multiplier.copy_(torch.clamp(net.module._classification.normalization_multiplier.data, min=1.0)) 
        #         if net.module._classification.bias is not None:
        #             net.module._classification.bias.copy_(torch.clamp(net.module._classification.bias.data, min=0.))  

    # Store average loss components in train_info
    train_info['align_loss_raw'] = align_loss_raw_total/float(i+1)
    train_info['tanh_loss_raw'] = tanh_loss_raw_total/float(i+1)
    train_info['class_loss_raw'] = class_loss_raw_total/float(i+1)
    train_info['align_loss_weighted'] = align_loss_weighted_total/float(i+1)
    train_info['tanh_loss_weighted'] = tanh_loss_weighted_total/float(i+1)
    train_info['class_loss_weighted'] = class_loss_weighted_total/float(i+1)
    train_info['weight_entropy_raw'] = weight_entropy_raw_total/float(i+1)
    train_info['weight_entropy_weighted'] = weight_entropy_weighted_total/float(i+1)
    
    # Add a print statement at the end of the epoch to show loss breakdown
    print(f"\nEpoch {epoch} loss breakdown:")
    print(f"  Alignment loss: {train_info['align_loss_raw']:.4f} (raw), {train_info['align_loss_weighted']:.4f} (weighted)")
    print(f"  Tanh loss: {train_info['tanh_loss_raw']:.4f} (raw), {train_info['tanh_loss_weighted']:.4f} (weighted)")
    print(f"  Classification loss: {train_info['class_loss_raw']:.4f} (raw), {train_info['class_loss_weighted']:.4f} (weighted)")
    if is_count_pipnet:
        print(f"  Weight entropy loss: {train_info['weight_entropy_raw']:.4f} (raw), {train_info['weight_entropy_weighted']:.4f} (weighted)")

    train_info['train_accuracy'] = total_acc/float(i+1)
    train_info['loss'] = total_loss/float(i+1)
    train_info['lrs_net'] = lrs_net
    train_info['lrs_class'] = lrs_class
    
    return train_info

def calculate_loss(proto_features, pooled, out, ys1, align_pf_weight, t_weight, unif_weight, cl_weight, 
                   net_normalization_multiplier, pretrain, finetune, criterion, train_iter, 
                   is_count_pipnet=False, print=True, EPS=1e-10, apply_counting_loss=False,
                   net=None, entropy_weight=0.1):
    ys = torch.cat([ys1,ys1])
    pooled1, pooled2 = pooled.chunk(2)
    pf1, pf2 = proto_features.chunk(2)

    embv2 = pf2.flatten(start_dim=2).permute(0,2,1).flatten(end_dim=1)
    embv1 = pf1.flatten(start_dim=2).permute(0,2,1).flatten(end_dim=1)
    
    a_loss_pf = (align_loss(embv1, embv2.detach())+ align_loss(embv2, embv1.detach()))/2.
    
    C = 0.01
    if not is_count_pipnet:
        tanh_loss = -(torch.log(torch.tanh(C * torch.sum(pooled1,dim=0))+EPS).mean() + 
                      torch.log(torch.tanh(C * torch.sum(pooled2,dim=0))+EPS).mean())/2.
    else:
        tanh_loss = -(torch.log(torch.tanh(C * torch.sum(pooled1,dim=0))+EPS).mean() + 
                      torch.log(torch.tanh(C * torch.sum(pooled2,dim=0))+EPS).mean())/2.
        
    weight_entropy_loss = torch.tensor(0.0, device=pooled1.device)
    
    # Weight for count confidence loss
        
    loss_components = {
            'align': a_loss_pf.item(),
            'align_weighted': a_loss_pf.item() * align_pf_weight,
            'tanh': tanh_loss.item(),
            'tanh_weighted': tanh_loss.item() * t_weight,
            'class': 0.0,
            'class_weighted': 0.0,
            'weight_entropy': 0.0,
            'weight_entropy_weighted': 0.0
        }
    
    if not finetune:
        loss = align_pf_weight * a_loss_pf
        loss += t_weight * tanh_loss
    
    if not pretrain:
        softmax_inputs = torch.log1p(out**net_normalization_multiplier)
        class_loss = criterion(F.log_softmax((softmax_inputs),dim=1),ys)
        loss_components['class'] = class_loss.item()
        loss_components['class_weighted'] = class_loss.item() * cl_weight
        
        # Add entropy-based weight penalty for CountPIPNet
        if is_count_pipnet:
            weight_entropy_loss = calculate_weight_entropy_loss(net, is_count_pipnet=True)
            loss_components['weight_entropy'] = weight_entropy_loss.item()
            loss_components['weight_entropy_weighted'] = weight_entropy_loss.item() * entropy_weight
        
        if finetune:
            loss = cl_weight * class_loss
            # Add entropy loss during finetuning too
            if is_count_pipnet:
                loss += entropy_weight * weight_entropy_loss
        else:
            loss += cl_weight * class_loss
            # Add entropy loss during full training
            if is_count_pipnet:
                loss += entropy_weight * weight_entropy_loss

    acc=0.
    if not pretrain:
        ys_pred_max = torch.argmax(out, dim=1)
        correct = torch.sum(torch.eq(ys_pred_max, ys))
        acc = correct.item() / float(len(ys))
    if print: 
        with torch.no_grad():
            if pretrain:
                if is_count_pipnet:
                    train_iter.set_postfix_str(
                    f'L: {loss.item():.3f}, LA:{a_loss_pf.item():.2f}, LT:{tanh_loss.item():.3f}, num_scores>0.1:{torch.count_nonzero(torch.relu(pooled-0.1),dim=1).float().mean().item():.1f}',refresh=False)
                else:
                    train_iter.set_postfix_str(
                    f'L: {loss.item():.3f}, LA:{a_loss_pf.item():.2f}, LT:{tanh_loss.item():.3f}, num_scores>0.1:{torch.count_nonzero(torch.relu(pooled-0.1),dim=1).float().mean().item():.1f}',refresh=False)
            else:
                if finetune:
                    train_iter.set_postfix_str(
                    f'L:{loss.item():.3f},LC:{class_loss.item():.3f}, LA:{a_loss_pf.item():.2f}, LT:{tanh_loss.item():.3f}, LE:{weight_entropy_loss.item():.3f}, num_scores>0.1:{torch.count_nonzero(torch.relu(pooled-0.1),dim=1).float().mean().item():.1f}, Ac:{acc:.3f}',refresh=False)
                else:
                    if is_count_pipnet:
                        train_iter.set_postfix_str(
                        f'L:{loss.item():.3f},LC:{class_loss.item():.3f}, LA:{a_loss_pf.item():.2f}, LT:{tanh_loss.item():.3f}, LE:{weight_entropy_loss.item():.3f}, num_scores>0.1:{torch.count_nonzero(torch.relu(pooled-0.1),dim=1).float().mean().item():.1f}, Ac:{acc:.3f}',refresh=False)
                    else:
                        train_iter.set_postfix_str(
                        f'L:{loss.item():.3f},LC:{class_loss.item():.3f}, LA:{a_loss_pf.item():.2f}, LT:{tanh_loss.item():.3f}, num_scores>0.1:{torch.count_nonzero(torch.relu(pooled-0.1),dim=1).float().mean().item():.1f}, Ac:{acc:.3f}',refresh=False)            
    return loss, acc, loss_components

# Extra uniform loss from https://www.tongzhouwang.info/hypersphere/. Currently not used but you could try adding it if you want. 
def uniform_loss(x, t=2):
    # print("sum elements: ", torch.sum(torch.pow(x,2), dim=1).shape, torch.sum(torch.pow(x,2), dim=1)) #--> should be ones
    loss = (torch.pdist(x, p=2).pow(2).mul(-t).exp().mean() + 1e-10).log()
    return loss

# from https://gitlab.com/mipl/carl/-/blob/main/losses.py
def align_loss(inputs, targets, EPS=1e-12):
    assert inputs.shape == targets.shape
    assert targets.requires_grad == False
    
    loss = torch.einsum("nc,nc->n", [inputs, targets])
    loss = -torch.log(loss + EPS).mean()
    return loss

def calculate_weight_entropy_loss(net, is_count_pipnet=False):
    """
    Calculate entropy-based penalty for weights to encourage only one weight
    per prototype-count group to be active.
    
    Works with both flattened and structured weight representations.
    
    Args:
        net: The network model
        is_count_pipnet: Whether the model is CountPIPNet
        
    Returns:
        Entropy loss (higher entropy = higher loss)
    """
    if not is_count_pipnet:
        # Original PIPNet doesn't have count groups, so return zero
        return torch.tensor(0.0, device=net.module._classification.weight.device)
    
    # Get classification weights
    weights = net.module._classification.weight
    
    # Get model parameters
    max_count = net.module._max_count
    
    # Check if using structured classification
    is_structured = hasattr(net.module._classification, 'num_prototypes')
    
    if is_structured:
        # For StructuredCountClassifier, weights already have shape [num_classes, num_prototypes, max_count]
        # Just need to apply softmax over the count dimension for each prototype
        num_classes, num_prototypes, _ = weights.shape
        
        # Apply softmax over count dimension to get probability distribution
        # Apply small epsilon for numerical stability
        weights_prob = F.softmax(weights + 1e-10, dim=2)
        
        # Calculate entropy: -sum(p * log(p)) for each distribution
        # Higher entropy means more uniform distribution (bad)
        # Lower entropy means more concentrated distribution (good)
        entropy = -torch.sum(weights_prob * torch.log(weights_prob + 1e-10), dim=2)
        
        # Average entropy across classes and prototypes
        return entropy.mean()
    else:
        # For flattened NonNegLinear, weights have shape [num_classes, num_prototypes * max_count]
        num_classes, flattened_dim = weights.shape
        num_prototypes = flattened_dim // max_count
        
        # Reshape to [num_classes, num_prototypes, max_count]
        weights_reshaped = weights.view(num_classes, num_prototypes, max_count)
        
        # Apply softmax over count dimension to get probability distribution
        weights_prob = F.softmax(weights_reshaped + 1e-10, dim=2)
        
        # Calculate entropy: -sum(p * log(p)) for each distribution
        entropy = -torch.sum(weights_prob * torch.log(weights_prob + 1e-10), dim=2)
        
        # Average entropy across classes and prototypes
        return entropy.mean()