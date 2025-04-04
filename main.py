from util.log import Log
import torch.nn as nn
from util.args import get_args, save_args, get_optimizer_nn
from util.data import get_dataloaders
from util.func import init_weights_xavier
from util.checkpoint_manager import CheckpointManager
from pipnet.train import train_pipnet
from pipnet.test import eval_pipnet
import torch
from util.vis_pipnet import visualize, visualize_topk
from util.visualize_prediction import vis_pred, vis_pred_experiments
from util.selective_loading import load_shared_backbone
import sys, os
import random
import numpy as np
from shutil import copy
import matplotlib.pyplot as plt
from copy import deepcopy

from pipnet.pipnet import get_pipnet
from pipnet.count_pipnet import get_count_network

# Add this at the top of main.py
import hashlib
import json

def get_pretraining_config_hash(args):
    """Generate a unique identifier for pretraining configuration"""
    pretraining_params = {
        'max_count': getattr(args, 'max_count', 3),
        'use_mid_layers': getattr(args, 'use_mid_layers', False),
        'num_stages': getattr(args, 'num_stages', 2),
        'num_features': args.num_features,
        'activation': getattr(args, 'activation', 'gumbel_softmax'),
        'net': args.net,
        'dataset': args.dataset
    }
    param_str = json.dumps(pretraining_params, sort_keys=True)
    config_hash = hashlib.md5(param_str.encode()).hexdigest()[:10]
    return config_hash, pretraining_params

def run_pipnet(args=None):
    """
    Modified run_pipnet function to support shared backbone model loading.
    """
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    args = args or get_args()
    assert args.batch_size > 1

    # Create a logger
    log = Log(args.log_dir)
    print("Log dir: ", args.log_dir, flush=True)
    # Log the run arguments
    save_args(args, log.metadata_dir)
    
    gpu_list = args.gpu_ids.split(',')
    device_ids = []
    if args.gpu_ids!='':
        for m in range(len(gpu_list)):
            device_ids.append(int(gpu_list[m]))
    
    global device
    if not args.disable_cuda and torch.cuda.is_available():
        if len(device_ids)==1:
            device = torch.device('cuda:{}'.format(args.gpu_ids))
        elif len(device_ids)==0:
            device = torch.device('cuda')
            print("CUDA device set without id specification", flush=True)
            device_ids.append(torch.cuda.current_device())
        else:
            print("This code should work with multiple GPU's but we didn't test that, so we recommend to use only 1 GPU.", flush=True)
            device_str = ''
            for d in device_ids:
                device_str+=str(d)
                device_str+=","
            device = torch.device('cuda:'+str(device_ids[0]))
    else:
        device = torch.device('cpu')
     
    # Log which device was actually used
    print("Device used: ", device, "with id", device_ids, flush=True)
    
    # Obtain the dataset and dataloaders
    trainloader, trainloader_pretraining, trainloader_normal, trainloader_normal_augment, projectloader, testloader, test_projectloader, classes = get_dataloaders(args, device)
    if len(classes)<=20:
        if args.validation_size == 0.:
            print("Classes: ", testloader.dataset.class_to_idx, flush=True)
        else:
            print("Classes: ", str(classes), flush=True)

    # Create checkpoint manager
    checkpoint_manager = CheckpointManager(args, device)
    start_epoch = 1

    resume_training = False
    if hasattr(args, 'resume_training') and args.resume_training:
        resume_training = True

    use_gumbel_softmax = getattr(args, 'activation', 'gumbel_softmax') == 'gumbel_softmax'
    
    if hasattr(args, 'model') and args.model == 'count_pipnet':
        is_count_pipnet=True
        net, num_prototypes = get_count_network(
            num_classes=len(classes), 
            args=args,
            device=device,
            max_count=getattr(args, 'max_count', 3),
            use_ste=getattr(args, 'use_ste', False))
    else:
        is_count_pipnet=False
        net, num_prototypes = get_pipnet(len(classes), args)

    net = net.to(device=device)
    net = nn.DataParallel(net, device_ids=device_ids)
    
    optimizer_net, optimizer_classifier, params_to_freeze, params_to_train, params_backbone = get_optimizer_nn(net, args)   

    # Check for shared pretrained backbone
    shared_pretrained_loaded = False
    if hasattr(args, 'shared_pretrained_dir') and args.shared_pretrained_dir:
        try:
            shared_pretrained_result = load_shared_backbone(
                net, args.shared_pretrained_dir, verbose=True)
            shared_pretrained_loaded = shared_pretrained_result['success']
            
            if shared_pretrained_loaded:
                print(f"Successfully loaded shared pretrained backbone")
                print(f"Loaded {shared_pretrained_result['loaded_params']}/{shared_pretrained_result['total_backbone_params']} backbone parameters")
                
                # Skip pretraining since we've loaded the backbone
                args.epochs_pretrain = 0
            else:
                print(f"Failed to load shared pretrained backbone")
        except Exception as e:
            print(f"Error loading shared pretrained backbone: {e}")
            import traceback
            traceback.print_exc()

    # Only try regular checkpoints if shared pretrained wasn't loaded
    if not shared_pretrained_loaded:
        if resume_training:
            print("Attempting to resume training from last checkpoint", flush=True)
            resume_info = checkpoint_manager.load_trained_checkpoint(net, optimizer_net, optimizer_classifier)
            
            if resume_info['success']:
                # Skip pretraining if resuming
                args.epochs_pretrain = 0
                # Get the starting epoch if available
                if resume_info['epoch'] is not None:
                    start_epoch = resume_info['epoch'] + 1
                    print(f"Resuming training from epoch {start_epoch}", flush=True)
                else:
                    print("Resuming training from checkpoint without epoch information", flush=True)
        
        # If we're not resuming or resume failed, load pretrained checkpoint or initialize weights
        if not resume_training or not resume_info.get('success', False):
            with torch.no_grad():
                checkpoint_loaded = checkpoint_manager.load_pretrained_checkpoint(net, optimizer_net)
                
                if not checkpoint_loaded:
                    # Initialize weights
                    print("Initializing model weights", flush=True)
                    net.module._add_on.apply(init_weights_xavier)
                    torch.nn.init.normal_(net.module._classification.weight, mean=1.0, std=0.1)
                    if args.bias:
                        torch.nn.init.constant_(net.module._classification.bias, val=0.)
                    torch.nn.init.constant_(net.module._classification.normalization_multiplier, val=2.)
                    net.module._classification.normalization_multiplier.requires_grad = False
                    print("Classification layer initialized with mean", torch.mean(net.module._classification.weight).item(), flush=True)
                else:
                    print("Loaded pretrained checkpoint from standard location", flush=True)
                    args.epochs_pretrain = 0

    # If shared backbone was loaded or standard checkpoint failed, but we're using shared_pretrained_dir,
    # initialize the non-backbone parts of the model
    if shared_pretrained_loaded or (hasattr(args, 'shared_pretrained_dir') and args.shared_pretrained_dir):
        print("Initializing non-backbone components", flush=True)
        torch.nn.init.normal_(net.module._classification.weight, mean=1.0, std=0.1)
        if args.bias:
            torch.nn.init.constant_(net.module._classification.bias, val=0.)
        torch.nn.init.constant_(net.module._multiplier, val=2.)
        net.module._multiplier.requires_grad = False
        
    if hasattr(net.module, '_intermediate') and net.module._intermediate is not None:
        # Print information about the intermediate layer type
        intermediate_type = type(net.module._intermediate).__name__
        print(f"Found intermediate layer of type {intermediate_type}")
        
        # For OneHotEncoder, no initialization needed as it's parameter-free
        if intermediate_type == 'OneHotEncoder':
            print(f"OneHotEncoder doesn't require initialization", flush=True)
        # For LinearFull, LinearIntermediate, BilinearIntermediate, etc., 
        # the __init__ already contains custom initialization
        elif any(t in intermediate_type for t in ['Linear', 'Bilinear']):
            print(f"Using {intermediate_type}'s built-in custom initialization", flush=True)
        # For any other types, apply Xavier initialization as fallback
        elif hasattr(net.module._intermediate, 'apply'):
            print(f"Applying Xavier initialization to {intermediate_type}", flush=True)
            net.module._intermediate.apply(init_weights_xavier)

        print("Classification layer initialized with mean", torch.mean(net.module._classification.weight).item(), flush=True)
    
    # Define classification loss function and scheduler
    criterion = nn.NLLLoss(reduction='mean').to(device)
    scheduler_net = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_net, T_max=len(trainloader_pretraining)*args.epochs_pretrain, eta_min=args.lr_block/100., last_epoch=-1)

    # Forward one batch through the backbone to get the latent output size
    with torch.no_grad():
        xs1, _, _ = next(iter(trainloader))
        xs1 = xs1.to(device)
        proto_features, _, _ = net(xs1)
        wshape = proto_features.shape[-1]
        args.wshape = wshape #needed for calculating image patch size
        print("Output shape: ", proto_features.shape, flush=True)
    
    if net.module._num_classes == 2:
        # Create a csv log with additional loss component columns
        log.create_log('log_epoch_overview', 'epoch', 'test_top1_acc', 
                    'local_size_for_true_class', 'local_size_for_all_classes', 'prototypes_per_class', 'almost_nonzeros_pooled', 
                    'num_nonzero_prototypes', 'mean_train_acc', 'mean_train_loss_during_epoch',
                    'align_loss_raw', 'tanh_loss_raw', 'class_loss_raw',
                    'align_loss_weighted', 'tanh_loss_weighted', 'class_loss_weighted')
    else:
        # Create a csv log with additional loss component columns
        log.create_log('log_epoch_overview', 'epoch', 'test_top1_acc', 
                    'local_size_for_true_class', 'local_size_for_all_classes', 'prototypes_per_class', 'almost_nonzeros_pooled', 
                    'num_nonzero_prototypes', 'mean_train_acc', 'mean_train_loss_during_epoch',
                    'align_loss_raw', 'tanh_loss_raw', 'class_loss_raw',
                    'align_loss_weighted', 'tanh_loss_weighted', 'class_loss_weighted')
    
    
    lrs_pretrain_net = []
    # PRETRAINING PROTOTYPES PHASE
    for epoch in range(1, args.epochs_pretrain+1):
        for param in params_to_train:
            param.requires_grad = True
        for param in net.module._add_on.parameters():
            param.requires_grad = True
        for param in net.module._classification.parameters():
            param.requires_grad = False
        for param in net.module._intermediate.parameters():
            param.requires_grad = False
        
        for param in params_to_freeze:
            param.requires_grad = True # can be set to False when you want to freeze more layers
        for param in params_backbone:
            param.requires_grad = False #can be set to True when you want to train whole backbone (e.g. if dataset is very different from ImageNet)
        
        print("\nPretrain Epoch", epoch, "with batch size", trainloader_pretraining.batch_size, flush=True)
        
        # Pretrain prototypes
        train_info = train_pipnet(net, trainloader_pretraining, optimizer_net, optimizer_classifier, 
                                  scheduler_net, None, criterion, epoch, args.epochs_pretrain, device, 
                                  is_count_pipnet=is_count_pipnet, pretrain=True, finetune=False,
                                  enforce_weight_sparsity=args.enforce_weight_sparsity)
        
        # For CountPiPNet anneal the Gumbel-Softmax temperature
        if hasattr(args, 'model') and args.model == 'count_pipnet' and use_gumbel_softmax:
            # Configuration for temperature annealing
            start_temp = 1.0
            final_temp = 0.1 
            stabilization_epochs = int(args.epochs_pretrain * 0.25)  # Number of epochs to hold at final temperature
            
            # Calculate annealing period
            annealing_epochs = args.epochs_pretrain - stabilization_epochs
            
            # During pretraining
            if epoch <= args.epochs_pretrain:
                if epoch <= annealing_epochs:
                    # Linear decrease from start_temp to final_temp during annealing period
                    progress = epoch / annealing_epochs
                    temp = start_temp - (start_temp - final_temp) * progress
                else:
                    # Hold at final temperature during stabilization period
                    temp = final_temp
                    
                net.module.update_temperature(temp)
                print(f"Updated Gumbel-Softmax temperature to {temp:.3f} (Pretraining phase)", flush=True)

        lrs_pretrain_net+=train_info['lrs_net']
        plt.clf()
        plt.plot(lrs_pretrain_net)
        plt.savefig(os.path.join(args.log_dir,'lr_pretrain_net.png'))

        log.log_values('log_epoch_overview', epoch, "n.a.", "n.a.", "n.a.", "n.a.", "n.a.", "n.a.", "n.a.", 
                    train_info['loss'],
                    train_info['align_loss_raw'], train_info['tanh_loss_raw'], "n.a.",  # Classification loss is n.a.
                    train_info['align_loss_weighted'], train_info['tanh_loss_weighted'], "n.a.")
    
    if args.epochs_pretrain > 0 and not resume_training:
        net.eval()
        checkpoint_manager.save_pretrained_checkpoint(net, optimizer_net)

        with torch.no_grad():
            topks = visualize_topk(net, projectloader, len(classes), device, 'visualised_pretrained_prototypes_topk', args,
                                   k=10, are_pretraining_prototypes=True)
            
    # SECOND TRAINING PHASE
    # re-initialize optimizers and schedulers for second training phase
    if not resume_training or not resume_info['success']:
        optimizer_net, optimizer_classifier, params_to_freeze, params_to_train, params_backbone = get_optimizer_nn(net, args)
    scheduler_net = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_net, T_max=len(trainloader)*args.epochs, eta_min=args.lr_net/100.)
    # scheduler for the classification layer is with restarts, such that the model can re-active zeroed-out prototypes. Hence an intuitive choice. 
    if args.epochs<=30:
        scheduler_classifier = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_classifier, T_0=5, eta_min=0.001, T_mult=1, verbose=False)
    else:
        scheduler_classifier = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_classifier, T_0=10, eta_min=0.001, T_mult=1, verbose=False)
    for param in net.module.parameters():
        param.requires_grad = False
    for param in net.module._classification.parameters():
        param.requires_grad = True

    frozen = True
    lrs_net = []
    lrs_classifier = []

    # Initial finetuning phase
    epochs_to_finetune = args.epochs_finetune
    args.freeze_epochs += epochs_to_finetune
    
    for epoch in range(1, args.epochs + 1):
        # Special handling for CountPIPNet without STE
        count_pipnet_no_ste = (hasattr(args, 'model') and args.model == 'count_pipnet' and 
                                not getattr(args, 'use_ste', False))
        
        if epoch <= epochs_to_finetune:
            print(f'Finetuning...')
            # Freeze everything except classification and intermediate layer
            for param in net.module.parameters():
                param.requires_grad = False
            for param in net.module._classification.parameters():
                param.requires_grad = True
            for param in net.module._intermediate.parameters():
                param.requires_grad = True
            finetune = True
        else:
            finetune = False
            print(f'Finetuning ended.')
            
            # For CountPIPNet without STE, always keep everything frozen except classification layer
            if count_pipnet_no_ste:
                for param in net.module.parameters():
                    param.requires_grad = False
                for param in net.module._classification.parameters():
                    param.requires_grad = True
                print("\n Epoch", epoch, "CountPIPNet without STE: Training only classification layer", flush=True)
            # For original PIPNet or CountPIPNet with STE, follow regular unfreezing strategy
            else:
                if frozen:
                    # Unfreeze backbone after freeze_epochs
                    if epoch > args.freeze_epochs:
                        for param in net.module._add_on.parameters():
                            param.requires_grad = True
                        for param in net.module._intermediate.parameters():
                            param.requires_grad = True
                        for param in params_to_freeze:
                            param.requires_grad = True
                        for param in params_to_train:
                            param.requires_grad = True
                        for param in params_backbone:
                            param.requires_grad = True

                        frozen = False
                        print(f'Training everything...')
                    # Keep first layers of backbone frozen, train rest
                    else:
                        for param in params_to_freeze:
                            param.requires_grad = True
                        for param in net.module._add_on.parameters():
                            param.requires_grad = True
                        for param in net.module._intermediate.parameters():
                            param.requires_grad = True                            
                        for param in params_to_train:
                            param.requires_grad = True
                        for param in params_backbone:
                            param.requires_grad = False
                        print(f'Training train + freeze params...')
        
        print("\n Epoch", epoch, 
            "frozen:", frozen if not count_pipnet_no_ste else "N/A (CountPIPNet without STE)", 
            flush=True)    
        if args.enforce_weight_sparsity and (epoch==args.epochs or epoch%30==0) and args.epochs>1:
            # print('(MAIN) Setting small weights to zero')
            with torch.no_grad():
                torch.set_printoptions(profile="full")
                net.module._classification.weight.copy_(torch.clamp(net.module._classification.weight.data - 0.001, min=0.)) 
                print("Classifier weights: ", net.module._classification.weight[net.module._classification.weight.nonzero(as_tuple=True)], (net.module._classification.weight[net.module._classification.weight.nonzero(as_tuple=True)]).shape, flush=True)
                if args.bias:
                    print("Classifier bias: ", net.module._classification.bias, flush=True)
                torch.set_printoptions(profile="default")

        train_info = train_pipnet(net, trainloader, optimizer_net, optimizer_classifier, scheduler_net, 
                                  scheduler_classifier, criterion, epoch, args.epochs, device, 
                                  is_count_pipnet=is_count_pipnet, pretrain=False, finetune=finetune,
                                  enforce_weight_sparsity=args.enforce_weight_sparsity)

        lrs_net+=train_info['lrs_net']
        lrs_classifier+=train_info['lrs_class']
        # Evaluate model
        eval_info = eval_pipnet(net, testloader, epoch, device, log, enforce_weight_sparsity=args.enforce_weight_sparsity,
                                args=args)
        log.log_values('log_epoch_overview', epoch, eval_info['top1_accuracy'], 
            eval_info['local_size_for_true_class'], eval_info['local_size_for_all_classes'], eval_info['prototypes_per_class'], 
            eval_info['almost_nonzeros'], eval_info['num non-zero prototypes'], 
            train_info['train_accuracy'], train_info['loss'],
            train_info['align_loss_raw'], train_info['tanh_loss_raw'], train_info['class_loss_raw'],
            train_info['align_loss_weighted'], train_info['tanh_loss_weighted'], train_info['class_loss_weighted'])
            
        with torch.no_grad():
            # Save the checkpoint
            checkpoint_manager.save_trained_checkpoint(net, optimizer_net, optimizer_classifier, epoch)

            # Save the best checkpoint if this is the best model so far
            checkpoint_manager.save_best_checkpoint(net, optimizer_net, optimizer_classifier, 
                                                    epoch, eval_info['top1_accuracy'])
                    
            # Learning rate graphs (keep this part)
            plt.clf()
            plt.plot(lrs_net)
            plt.savefig(os.path.join(args.log_dir,'lr_net.png'))
            plt.clf()
            plt.plot(lrs_classifier)
            plt.savefig(os.path.join(args.log_dir,'lr_class.png'))
    
    net.eval()
    if args.epochs > 1:
        checkpoint_manager.save_trained_checkpoint(net, optimizer_net, optimizer_classifier, epoch="last")
        
    topks = visualize_topk(net, projectloader, len(classes), device, 'visualised_prototypes_topk', args)
    # visualize(net, projectloader, len(classes), device, 'visualised_prototypes', args)

    # Now load and visualize the best model's prototypes
    print("\nLoading best model for prototype visualization...", flush=True)
    best_model_info = checkpoint_manager.load_best_checkpoint(net, optimizer_net, optimizer_classifier)
    if best_model_info['success']:
        print(f"Loaded best model from epoch {best_model_info['epoch']} with accuracy {best_model_info['accuracy']:.4f} for visualization", flush=True)
        # Create a dedicated folder for the best model prototypes
        best_model_folder = f'visualised_prototypes_topk_best_model_epoch{best_model_info["epoch"]}'
        topks_best = visualize_topk(net, projectloader, len(classes), device, best_model_folder, args)
        print(f"Best model prototypes visualized in folder: {best_model_folder}", flush=True)
    else:
        print("Failed to load best model for prototype visualization", flush=True)

    # # set weights of prototypes that are never really found in projection set to 0
    # set_to_zero = []
    # if topks:
    #     for prot in topks.keys():
    #         found = False
    #         for (i_id, score) in topks[prot]:
    #             if score > 0.1:
    #                 found = True
    #         if not found:
    #             torch.nn.init.zeros_(net.module._classification.weight[:,prot])
    #             set_to_zero.append(prot)
    #     print("Weights of prototypes", set_to_zero, "are set to zero because it is never detected with similarity>0.1 in the training set", flush=True)
    #     eval_info = eval_pipnet(net, testloader, "notused"+str(args.epochs), device, log)
    #     log.log_values('log_epoch_overview', "notused"+str(args.epochs), eval_info['top1_accuracy'], eval_info['local_size_for_true_class'], eval_info['prototypes_per_class'], eval_info['almost_nonzeros'], eval_info['num non-zero prototypes'], "n.a.", "n.a.")

    # print("classifier weights: ", net.module._classification.weight, flush=True)
    # print("Classifier weights nonzero: ", net.module._classification.weight[net.module._classification.weight.nonzero(as_tuple=True)], (net.module._classification.weight[net.module._classification.weight.nonzero(as_tuple=True)]).shape, flush=True)
    # print("Classifier bias: ", net.module._classification.bias, flush=True)
    # # Print weights and relevant prototypes per class
    # for c in range(net.module._classification.weight.shape[0]):
    #     relevant_ps = []
    #     proto_weights = net.module._classification.weight[c,:]
    #     for p in range(net.module._classification.weight.shape[1]):
    #         if proto_weights[p]> 1e-3:
    #             relevant_ps.append((p, proto_weights[p].item()))
    #     if args.validation_size == 0.:
    #         print("Class", c, "(", list(testloader.dataset.class_to_idx.keys())[list(testloader.dataset.class_to_idx.values()).index(c)],"):","has", len(relevant_ps),"relevant prototypes: ", relevant_ps, flush=True)
        
    # visualize predictions 
    # testset_img0_path = test_projectloader.dataset.samples[0][0]
    # test_path = os.path.split(os.path.split(testset_img0_path)[0])[0]
    # vis_pred(net, test_path, classes, device, args) 
    # if args.extra_test_image_folder != '':
    #     if os.path.exists(args.extra_test_image_folder):   
    #         vis_pred_experiments(net, args.extra_test_image_folder, classes, device, args)

    print("Done!", flush=True)

if __name__ == '__main__':
    args = get_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create log directory if it doesn't exist
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)
    
    # Define paths for log files
    print_dir = os.path.join(args.log_dir, 'out.txt')
    tqdm_dir = os.path.join(args.log_dir, 'tqdm.txt')
    
    # Create a custom stream that writes to both console and file
    class Tee:
        def __init__(self, stdout, file):
            self.stdout = stdout
            self.file = file
            
        def write(self, message):
            self.stdout.write(message)
            self.file.write(message)
            
        def flush(self):
            self.stdout.flush()
            self.file.flush()
    
    # Save original stdout/stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # Open log files
    out_file = open(print_dir, 'w')
    tqdm_file = open(tqdm_dir, 'w')
    
    # Replace stdout/stderr with Tee objects
    sys.stdout = Tee(original_stdout, out_file)
    sys.stderr = Tee(original_stderr, tqdm_file)
    
    try:
        run_pipnet(args)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore original stdout/stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        
        # Close log files
        out_file.close()
        tqdm_file.close()
