import torch

def get_patch_size(args):
    if not args.use_mid_layers:
        # Default ConvNext/Resnet patch size from the paper
        patchsize = 32
    else:
        num_stages = args.num_stages
        if num_stages in [1, 2, 3]:
            patchsize = 16
        else:
            patchsize = 32
            
    skip = round((args.image_size - patchsize) / (args.wshape-1))
    return patchsize, skip

def init_weights_xavier(m):
    if type(m) == torch.nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain('sigmoid'))

# https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b?permalink_comment_id=3662215#gistcomment-3662215
def topk_accuracy(output, target, topk=[1,]):
    with torch.no_grad():
        num_classes = output.shape[1]
        # Cap k at number of classes instead of filtering
        adjusted_topk = [min(k, num_classes) for k in topk]
        
        maxk = max(adjusted_topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = (pred == target.unsqueeze(dim=0)).expand_as(pred)

        res = []
        for k, original_k in zip(adjusted_topk, topk):
            correct_k = correct[:k].reshape(-1).float()
            res.append(correct_k)
        return res