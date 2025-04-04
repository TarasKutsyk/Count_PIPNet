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
        progress_prefix: str = 'Eval Epoch',
        enforce_weight_sparsity=True,
        args=None, # TODO fix
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
    correct_class_local_size_mean = 0.
    all_classes_local_size_mean = 0.
    global_anz = 0.
    prototypes_per_class_total = 0.
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
            if enforce_weight_sparsity:
                # print('(TEST) Setting small weights to zero...')
                net.module._classification.weight.copy_(torch.clamp(net.module._classification.weight.data - 1e-3, min=0.)) 
            # Use the model to classify this batch of input data
            _, pooled, out = net(xs, inference=not is_count_pipnet)
            batch_size = xs.shape[0]
            num_classes = out.shape[1]

            max_out_score, ys_pred = torch.max(out, dim=1)
            ys_pred_scores = torch.amax(F.softmax((torch.log1p(out**net.module._classification.normalization_multiplier)),dim=1),dim=1)

            abstained += (max_out_score.shape[0] - torch.count_nonzero(max_out_score))

            if is_count_pipnet:
                # Calculate the number of actual prototypes
                num_raw_prototypes = net.module._num_prototypes
                assert num_raw_prototypes == pooled.shape[1]
                
                prototype_to_class_weigths = [net.module.get_prototype_importance_per_class(i) for i in range(num_raw_prototypes)]
                prototype_to_class_weigths = torch.stack(prototype_to_class_weigths, dim=0) # [num_raw_prototypes, num_classes]

                assert prototype_to_class_weigths.shape[0] == num_raw_prototypes and \
                       prototype_to_class_weigths.shape[1] == net.module._num_classes

                # Compute the weighted prototype activations for each class
                # First, rearrange to match the expected shape for broadcasting with pooled
                repeated_weight = einops.repeat(prototype_to_class_weigths, 
                                                'num_prototypes num_classes -> num_classes batch_size num_prototypes',
                                                batch_size=pooled.shape[0])
            else: # Original PIPNet approach
                # Repeat the classification weights for batch processing
                repeated_weight = net.module._classification.weight.unsqueeze(1).repeat(1,pooled.shape[0],1)
                # net.module._classification.weight: [num_classes, num_prototypes] - prototype-to-class weights
                # repeated_weight: [num_classes, batch_size, num_prototypes] - weights repeated for batch

            scores = pooled * repeated_weight
            # scores: [num_classes, batch_size, num_prototypes] - weighted prototype activations

            any_class_local_sizes, pred_class_local_sizes = compute_local_explanation_sizes(
                scores, ys_pred, threshold=1e-3
            )

            # Count prototypes that contribute to each class decision
            prototypes_per_class = torch.count_nonzero(
                torch.gt(torch.relu(scores-1e-3).mean(dim=1),
                            0.).float(),
                dim=1
            ).float()
            # scores.mean(dim=1): 
            #   - [num_classes, num_prototypes] - Mean of the thresholded prototype-class contributions over all images in the batch
            # torch.gt(...): 
            #   - [num_classes, num_prototypes] - Boolean mask indicating whether each prototype’s mean contribution exceeds zero
            # prototypes_per_class: [num_classes] - How many prototypes each class has with non-zero mean evidence across the batch

            # Count of activated prototypes per image (regardless of class)
            almost_nz = torch.count_nonzero(
                torch.gt(torch.abs(pooled), 1e-3).float(),
                dim=1
            ).float()
            # torch.gt(torch.abs(pooled), 1e-3): [batch_size, num_prototypes] - boolean mask of activated prototypes
            # almost_nz: [batch_size] - count of activated prototypes per image
            
            correct_class_local_size_mean += pred_class_local_sizes.mean(0).item()
            all_classes_local_size_mean += any_class_local_sizes.mean(0).item()
                        
            prototypes_per_class_total += prototypes_per_class.mean(0).item() # get an average number of relevant prototypes for each class
            global_anz += almost_nz.mean().item()
            
            # Update the confusion matrix
            cm_batch = np.zeros((net.module._num_classes, net.module._num_classes), dtype=int)
            for y_pred, y_true in zip(ys_pred, ys):
                cm[y_true][y_pred] += 1
                cm_batch[y_true][y_pred] += 1
            acc = acc_from_cm(cm_batch)
            test_iter.set_postfix_str(
                f'local_pred_class: {pred_class_local_sizes.mean().item():.2f}, ANZ: {almost_nz.mean().item():.1f}, Acc: {acc:.3f}', refresh=False
            )    

            (top1accs, top5accs) = topk_accuracy(out, ys, topk=[1,5])
            
            global_top1acc+=torch.mean(top1accs).item()
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
    info['top1_accuracy'] = global_top1acc/len(test_loader)
    info['local_size_for_true_class'] = correct_class_local_size_mean/len(test_loader)
    info['local_size_for_all_classes'] = all_classes_local_size_mean/len(test_loader)
    info['prototypes_per_class'] = prototypes_per_class_total / len(test_loader)
    info['almost_nonzeros'] = global_anz/len(test_loader)

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
        try:
            print("AUC macro: ", roc_auc_score(y_trues, y_preds, average='macro'), flush=True)
            print("AUC weighted: ", roc_auc_score(y_trues, y_preds, average='weighted'), flush=True)
        except ValueError:
            pass

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

def compute_local_explanation_sizes(
    scores: torch.Tensor,
    ys_pred: torch.Tensor,
    threshold: float = 1e-3
):
    """
    Compute two metrics for local explanation size:
      1) # of prototypes active for *any* class in each image
      2) # of prototypes active for that image's *predicted* class

    Args:
        scores: [num_classes, batch_size, num_prototypes] tensor of weighted prototype activations (pooled * repeated_weight)
        ys_pred: [batch_size] long tensor with predicted class indices in [0..num_classes-1].
        threshold: Activation threshold for deciding if a prototype is "present."
        
    Returns:
        any_class_sizes:  [batch_size]  (# of prototypes that passed threshold for any class)
        pred_class_sizes: [batch_size]  (# of prototypes that passed threshold for predicted class)
    """
    

    # Threshold scores to find which prototypes are "active"
    #    --> relevant: [num_classes, batch_size, num_prototypes], boolean
    relevant = torch.abs(scores) > threshold

    # -------------------------------------------------------------------------
    # (A) ANY-CLASS EXPLANATION SIZE
    #   For each image, a prototype is counted if it is active in *any* class.
    #   So we do an "any" reduction over num_classes, then a "sum" over prototypes.
    # -------------------------------------------------------------------------
    #    relevant_any_class: [batch_size, num_prototypes]
    relevant_any_class = relevant.any(dim=0)
    #    any_class_sizes: [batch_size]
    any_class_sizes = relevant_any_class.sum(dim=1)

    # -------------------------------------------------------------------------
    # (B) PREDICTED-CLASS EXPLANATION SIZE
    #   For each image i, we only look at the row = ys_pred[i] in relevant,
    #   then count active prototypes.
    # -------------------------------------------------------------------------
    #   Approach:
    #   1) Sum across prototypes -> shape [num_classes, batch_size]
    #   2) index_select the correct class for each image -> shape [batch_size, batch_size]
    #   3) take the diagonal -> shape [batch_size]
    #
    #   That yields "per-image, how many prototypes are active for the predicted class?"
    # -------------------------------------------------------------------------

    #   local_count_per_class: [num_classes, batch_size]
    local_count_per_class = relevant.sum(dim=2).float()
    #   pick the row for each predicted class
    #   => shape [batch_size, batch_size], then diagonal => [batch_size]
    selected_local_count = torch.index_select(local_count_per_class, dim=0, index=ys_pred)
    pred_class_sizes = torch.diagonal(selected_local_count, offset=0)

    return any_class_sizes.float(), pred_class_sizes.float()