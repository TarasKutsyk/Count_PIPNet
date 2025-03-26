from tqdm import tqdm
import numpy as np
import torch
import torch.optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from util.log import Log
from util.func import topk_accuracy
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score, f1_score
import einops

@torch.no_grad()
def eval_pipnet(net,
        test_loader: DataLoader,
        epoch,
        device,
        log: Log = None,  
        progress_prefix: str = 'Eval Epoch'
        ) -> dict:
    
    net = net.to(device)
    # Make sure the model is in evaluation mode
    net.eval()

    is_count_pipnet = hasattr(net.module, '_max_count')  # Check if it's a CountPIPNet

    # Keep an info dict about the procedure
    info = dict()
    # Build a confusion matrix
    cm = np.zeros((net.module._num_classes, net.module._num_classes), dtype=int)

    global_top1acc = 0.
    global_top5acc = 0.
    global_sim_anz = 0.
    global_anz = 0.
    local_size_total = 0.
    y_trues = []
    y_preds = []
    y_preds_classes = []
    abstained = 0
    # Show progress on progress bar
    test_iter = tqdm(enumerate(test_loader),
                        total=len(test_loader),
                        desc=progress_prefix+' %s'%epoch,
                        mininterval=5.,
                        ncols=0)
    (xs, ys) = next(iter(test_loader))
    # Iterate through the test set
    for i, (xs, ys) in test_iter:
        xs, ys = xs.to(device), ys.to(device)
        
        with torch.no_grad():
            # net.module._classification.weight.copy_(torch.clamp(net.module._classification.weight.data - 1e-3, min=0.)) 
            # Use the model to classify this batch of input data
            _, pooled, out = net(xs, inference=True)
            max_out_score, ys_pred = torch.max(out, dim=1)
            ys_pred_scores = torch.amax(F.softmax((torch.log1p(out**net.module._classification.normalization_multiplier)),dim=1),dim=1)

            abstained += (max_out_score.shape[0] - torch.count_nonzero(max_out_score))

            if is_count_pipnet:
                # Get the maximum count parameter from the model
                max_count = net.module._max_count
                
                # Check if using StructuredCountClassifier
                using_structured_classifier = hasattr(net.module._classification, 'num_prototypes')
                
                if using_structured_classifier:
                    # With StructuredCountClassifier, weights already have shape [num_classes, num_prototypes, max_count]
                    num_raw_prototypes = net.module._num_prototypes
                    
                    # No need to reshape pooled if it already has the right structure [batch_size, num_prototypes, max_count]
                    if len(pooled.shape) == 3 and pooled.shape[2] == max_count:
                        reshaped_pooled = pooled
                    else:
                        # If pooled is flattened, reshape it to recover the structure
                        reshaped_pooled = einops.rearrange(
                            pooled, 
                            'b (p c) -> b p c', 
                            p=num_raw_prototypes, 
                            c=max_count
                        ) # [batch_size, num_raw_prototypes, max_count]
                    
                    # Sum across count dimension to get overall presence
                    count_presence = reshaped_pooled.sum(dim=2) # [batch_size, num_raw_prototypes]
                    
                    # Use weights directly from the structured classifier
                    nonzero_weights = net.module._classification.weight.relu() # [num_classes, num_prototypes, max_count]
                    nonzero_weights = nonzero_weights.sum(dim=2) # [num_classes, num_raw_prototypes]
                    
                    # Repeat for batch processing
                    repeated_weight = einops.repeat(
                        nonzero_weights,
                        'classes p -> classes b p',
                        b=count_presence.shape[0]
                    ) # [num_classes, batch_size, num_raw_prototypes]
                else:
                    # Original approach with flattened representation
                    # Calculate the number of actual prototypes (before one-hot encoding expanded them)
                    num_raw_prototypes = pooled.shape[1] // max_count
                    
                    # Reshape pooled to recover the original structure before flattening
                    reshaped_pooled = einops.rearrange(
                        pooled, 
                        'b (p c) -> b p c', 
                        p=num_raw_prototypes, 
                        c=max_count
                    ) # [batch_size, num_raw_prototypes, max_count]
                    
                    count_presence = reshaped_pooled.sum(dim=2) # [batch_size, num_raw_prototypes]
                    
                    # Reshape the classification weights to match the prototype-count structure
                    reshaped_weights = einops.rearrange(
                        net.module._classification.weight,
                        'classes (p c) -> classes p c',
                        p=num_raw_prototypes,
                        c=max_count
                    ) # [num_classes, num_raw_prototypes, max_count]
                    
                    # Extract weights for non-zero counts and sum them
                    nonzero_weights = reshaped_weights.sum(dim=2) # [num_classes, num_raw_prototypes]
                    
                    # Repeat for batch processing
                    repeated_weight = einops.repeat(
                        nonzero_weights,
                        'classes p -> classes b p',
                        b=count_presence.shape[0]
                    ) # [num_classes, batch_size, num_raw_prototypes]
                
                # Count significant prototype activations weighted by class importance
                sim_scores_anz = torch.count_nonzero(torch.gt(count_presence*repeated_weight, 1e-3).float(), dim=2).float()
                # [num_classes, batch_size] <- how many significant prototype activations exist for each class and image combination.
                
                # Count prototypes that contribute to each class decision
                weighted_contribution = einops.reduce(
                    torch.relu((count_presence*repeated_weight)-1e-3),
                    'classes b p -> classes b',
                    'sum'
                )
                local_size = torch.count_nonzero(torch.gt(weighted_contribution, 0.).float(), dim=1).float() 
                # [num_classes] <- how many images there are in the batch with non-zero evidence for each class 
                
                # Count activated prototypes per image (regardless of class)
                almost_nz = torch.count_nonzero(torch.gt(count_presence, 1e-3).float(), dim=1).float() 
                # [batch_size] - count of activated prototypes per image
            else: # Original PIPNet code with more comments
                # Repeat the classification weights for batch processing
                repeated_weight = net.module._classification.weight.unsqueeze(1).repeat(1,pooled.shape[0],1)
                # net.module._classification.weight: [num_classes, num_prototypes] - prototype-to-class weights
                # repeated_weight: [num_classes, batch_size, num_prototypes] - weights repeated for batch

                # Count non-zero prototype activations weighted by class importance
                sim_scores_anz = torch.count_nonzero(torch.gt(torch.abs(pooled*repeated_weight), 1e-3).float(),dim=2).float()
                # pooled*repeated_weight: [num_classes, batch_size, num_prototypes] - weighted prototype activations
                # torch.gt(...): [num_classes, batch_size, num_prototypes] - boolean mask of significant activations
                # sim_scores_anz: [num_classes, batch_size] - count of significant prototypes per class per image

                # Count prototypes that contribute to each class decision
                local_size = torch.count_nonzero(torch.gt(torch.relu((pooled*repeated_weight)-1e-3).sum(dim=1), 0.).float(),dim=1).float()
                # (pooled*repeated_weight).sum(dim=1): [num_classes, batch_size] - summed weighted activations per class
                # torch.gt(...): [num_classes, batch_size] - boolean mask of classes with significant evidence
                # local_size: [num_classes] - count of images where each class has significant evidence

                # Count of activated prototypes per image (regardless of class)
                almost_nz = torch.count_nonzero(torch.gt(torch.abs(pooled), 1e-3).float(),dim=1).float()
                # torch.count_nonzero(torch.relu(pooled-0.1),dim=1).float().mean().item()
                # torch.gt(torch.abs(pooled), 1e-3): [batch_size, num_prototypes] - boolean mask of activated prototypes
                # almost_nz: [batch_size] - count of activated prototypes per image

            # Extract information from the correct predicted class
            correct_class_sim_scores_anz = torch.diagonal(torch.index_select(sim_scores_anz, dim=0, index=ys_pred),0)
            # torch.index_select(sim_scores_anz, dim=0, index=ys_pred): [batch_size, batch_size] 
            #   - Selects rows from sim_scores_anz corresponding to each image's predicted class
            # torch.diagonal(..., 0): [batch_size]
            #   - For each image, gets the count of significant prototypes for its predicted class
            
            local_size_total += local_size.sum().item()

            global_sim_anz += correct_class_sim_scores_anz.sum().item()            
            global_anz += almost_nz.sum().item()
            
            # Update the confusion matrix
            cm_batch = np.zeros((net.module._num_classes, net.module._num_classes), dtype=int)
            for y_pred, y_true in zip(ys_pred, ys):
                cm[y_true][y_pred] += 1
                cm_batch[y_true][y_pred] += 1
            acc = acc_from_cm(cm_batch)
            test_iter.set_postfix_str(
                f'SimANZCC: {correct_class_sim_scores_anz.mean().item():.2f}, ANZ: {almost_nz.mean().item():.1f}, LocS: {local_size.mean().item():.1f}, Acc: {acc:.3f}', refresh=False
            )    

            (top1accs, top5accs) = topk_accuracy(out, ys, topk=[1,5])
            
            global_top1acc+=torch.sum(top1accs).item()
            global_top5acc+=torch.sum(top5accs).item()
            y_preds += ys_pred_scores.detach().tolist()
            y_trues += ys.detach().tolist()
            y_preds_classes += ys_pred.detach().tolist()
        
        del out
        del pooled
        del ys_pred
        
    print("PIP-Net abstained from a decision for", abstained.item(), "images", flush=True)            
    info['num non-zero prototypes'] = torch.gt(net.module._classification.weight,1e-3).any(dim=0).sum().item()
    print("sparsity ratio: ", (torch.numel(net.module._classification.weight)-torch.count_nonzero(torch.nn.functional.relu(net.module._classification.weight-1e-3)).item()) / torch.numel(net.module._classification.weight), flush=True)
    info['confusion_matrix'] = cm
    info['test_accuracy'] = acc_from_cm(cm)
    info['top1_accuracy'] = global_top1acc/len(test_loader.dataset)
    info['top5_accuracy'] = global_top5acc/len(test_loader.dataset)
    info['almost_sim_nonzeros'] = global_sim_anz/len(test_loader.dataset)
    info['local_size_all_classes'] = local_size_total / len(test_loader.dataset)
    info['almost_nonzeros'] = global_anz/len(test_loader.dataset)

    if net.module._num_classes == 2:
        tp = cm[0][0]
        fn = cm[0][1]
        fp = cm[1][0]
        tn = cm[1][1]
        print("TP: ", tp, "FN: ",fn, "FP:", fp, "TN:", tn, flush=True)
        sensitivity = tp/(tp+fn)
        specificity = tn/(tn+fp)
        print("\n Epoch",epoch, flush=True)
        print("Confusion matrix: ", cm, flush=True)
        try:
            for classname, classidx in test_loader.dataset.class_to_idx.items(): 
                if classidx == 0:
                    print("Accuracy positive class (", classname, classidx,") (TPR, Sensitivity):", tp/(tp+fn))
                elif classidx == 1:
                    print("Accuracy negative class (", classname, classidx,") (TNR, Specificity):", tn/(tn+fp))
        except ValueError:
            pass
        print("Balanced accuracy: ", balanced_accuracy_score(y_trues, y_preds_classes),flush=True)
        print("Sensitivity: ", sensitivity, "Specificity: ", specificity,flush=True)
        info['top5_accuracy'] = f1_score(y_trues, y_preds_classes)
        try:
            print("AUC macro: ", roc_auc_score(y_trues, y_preds, average='macro'), flush=True)
            print("AUC weighted: ", roc_auc_score(y_trues, y_preds, average='weighted'), flush=True)
        except ValueError:
            pass
    else:
        info['top5_accuracy'] = global_top5acc/len(test_loader.dataset) 

    return info

def acc_from_cm(cm: np.ndarray) -> float:
    """
    Compute the accuracy from the confusion matrix
    :param cm: confusion matrix
    :return: the accuracy score
    """
    assert len(cm.shape) == 2 and cm.shape[0] == cm.shape[1]

    correct = 0
    for i in range(len(cm)):
        correct += cm[i, i]

    total = np.sum(cm)
    if total == 0:
        return 1
    else:
        return correct / total