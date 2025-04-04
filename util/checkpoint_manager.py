# checkpoint_manager.py

import os
import json
import hashlib
import pickle
import torch

class CheckpointManager:
    """Manages loading and saving of model checkpoints based on configuration hashes."""
    
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.config_hash, self.pretraining_params = self._get_pretraining_config_hash()
        
    def _get_pretraining_config_hash(self):
        """Generate a unique identifier for pretraining configuration"""
        pretraining_params = {
            'epochs_pretrain': self.args.epochs_pretrain,
            'max_count': getattr(self.args, 'max_count', 3),
            'use_ste': getattr(self.args, 'use_ste', False),
            'use_mid_layers': getattr(self.args, 'use_mid_layers', False),
            'num_stages': getattr(self.args, 'num_stages', 2),
            'num_features': self.args.num_features,
            'activation': getattr(self.args, 'activation', 'gumbel_softmax'),
            'net': self.args.net,
            'dataset': self.args.dataset,
            'batch_size_pretrain': self.args.batch_size_pretrain
        }
        param_str = json.dumps(pretraining_params, sort_keys=True)
        config_hash = hashlib.md5(param_str.encode()).hexdigest()[:10]
        return config_hash, pretraining_params
    
    def _get_search_directories(self):
        """Get list of directories to search for checkpoints in priority order"""
        search_dirs = []
        if hasattr(self.args, 'pretrained_checkpoints_dir') and self.args.pretrained_checkpoints_dir:
            search_dirs.append(os.path.join(self.args.pretrained_checkpoints_dir, 'checkpoints'))
        search_dirs.append(os.path.join(self.args.log_dir, 'checkpoints'))
        return search_dirs
    
    def _ensure_checkpoint_dir_exists(self):
        """Ensure checkpoint directory exists"""
        checkpoint_dir = os.path.join(self.args.log_dir, 'checkpoints')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        return checkpoint_dir
    
    def load_pretrained_checkpoint(self, net, optimizer_net):
        """
        Attempt to load a pretrained checkpoint matching the current configuration.
        
        Args:
            net: The network model to load weights into
            optimizer_net: The optimizer to load state into
            
        Returns:
            bool: True if a checkpoint was loaded, False otherwise
        """
        # If explicit checkpoint path is provided, use that
        if self.args.state_dict_dir_net:
            try:
                print(f"\nLoading specified checkpoint: {self.args.state_dict_dir_net}", flush=True)
                checkpoint = torch.load(self.args.state_dict_dir_net, map_location=self.device)
                net.load_state_dict(checkpoint['model_state_dict'], strict=True)
                optimizer_net.load_state_dict(checkpoint['optimizer_net_state_dict'])
                print("Specified checkpoint loaded successfully", flush=True)
                return True
            except Exception as e:
                print(f"Error loading specified checkpoint: {str(e)}", flush=True)
                return False
        
        # Otherwise, search for matching configuration checkpoint
        print(f"\nSearching for pretrained model with hash: {self.config_hash}", flush=True)
        
        for search_dir in self._get_search_directories():
            if not os.path.exists(search_dir):
                print(f"Directory {search_dir} does not exist, skipping", flush=True)
                continue
                
            checkpoint_path = os.path.join(search_dir, f'net_pretrained_{self.config_hash}')
            print(f"Checking for checkpoint at: {checkpoint_path}", flush=True)
            
            if os.path.exists(checkpoint_path):
                try:
                    checkpoint = torch.load(checkpoint_path, map_location=self.device)
                    net.load_state_dict(checkpoint['model_state_dict'], strict=True)
                    optimizer_net.load_state_dict(checkpoint['optimizer_net_state_dict'])
                    print(f"Auto-loaded pretrained model from {checkpoint_path}", flush=True)
                    return True
                except Exception as e:
                    print(f"Error loading checkpoint: {str(e)}", flush=True)
        
        print("No valid matching checkpoint found", flush=True)
        return False
    
    def save_pretrained_checkpoint(self, net, optimizer_net):
        """
        Save a pretrained checkpoint with the current configuration hash.
        
        Args:
            net: The network model to save
            optimizer_net: The optimizer to save
        """
        if self.args.state_dict_dir_net or self.args.epochs_pretrain <= 0:
            return
            
        net.eval()
        checkpoint_dir = self._ensure_checkpoint_dir_exists()
        
        # Paths for saving
        checkpoint_path = os.path.join(checkpoint_dir, f'net_pretrained_{self.config_hash}')
        params_path = os.path.join(checkpoint_dir, f'net_pretrained_{self.config_hash}_params.pkl')
        
        # Also save with standard name for backward compatibility
        standard_path = os.path.join(checkpoint_dir, 'net_pretrained')
        
        try:
            # Save with hash-based name
            checkpoint_data = {
                'model_state_dict': net.state_dict(),
                'optimizer_net_state_dict': optimizer_net.state_dict(),
                'config_hash': self.config_hash
            }
            
            torch.save(checkpoint_data, checkpoint_path)
            
            # Save parameter details
            with open(params_path, 'wb') as f:
                pickle.dump(self.pretraining_params, f)
                
            # Also save with standard name
            torch.save(checkpoint_data, standard_path)
            
            print(f"Saved pretrained model with hash: {self.config_hash}", flush=True)
            print(f"Saved to: {checkpoint_path}", flush=True)
        except Exception as e:
            print(f"Error saving checkpoint: {str(e)}", flush=True)
            
        net.train()
    
    def save_trained_checkpoint(self, net, optimizer_net, optimizer_classifier, epoch=None):
        """
        Save a checkpoint after training/fine-tuning.
        
        Args:
            net: The network model to save
            optimizer_net: The network optimizer to save
            optimizer_classifier: The classifier optimizer to save
            epoch: Current epoch number or "last" for final checkpoint
        """
        net.eval()
        checkpoint_dir = self._ensure_checkpoint_dir_exists()
        
        # Create the checkpoint data
        checkpoint_data = {
            'model_state_dict': net.state_dict(),
            'optimizer_net_state_dict': optimizer_net.state_dict(),
            'optimizer_classifier_state_dict': optimizer_classifier.state_dict()
        }
        
        # Add epoch information if available
        if epoch is not None and epoch != "last" and isinstance(epoch, int):
            checkpoint_data['epoch'] = epoch
        
        # Save the regular checkpoint (always updated)
        regular_path = os.path.join(checkpoint_dir, 'net_trained')
        torch.save(checkpoint_data, regular_path)
        
        # Save epoch-specific or final checkpoint if requested
        if epoch is not None:
            if epoch == "last":
                specific_path = os.path.join(checkpoint_dir, 'net_trained_last')
            # elif isinstance(epoch, int) and epoch % 30 == 0:
            #     specific_path = os.path.join(checkpoint_dir, f'net_trained_{epoch}')
            else:
                # Don't save for other epochs
                net.train()
                return
                
            torch.save(checkpoint_data, specific_path)
            print(f"Saved checkpoint to {specific_path}", flush=True)
        
        net.train()

    def load_trained_checkpoint(self, net, optimizer_net, optimizer_classifier, checkpoint_name='net_trained_last'):
        """
        Load a checkpoint from the main training phase.
        
        Args:
            net: The network model to load weights into
            optimizer_net: The network optimizer to load state into
            optimizer_classifier: The classifier optimizer to load state into
            checkpoint_name: Name of checkpoint file (default: 'net_trained_last')
            
        Returns:
            dict: Information about loaded checkpoint including success status and epoch if available
        """
        result = {'success': False, 'checkpoint_path': None, 'epoch': None}
        
        # Look in current log_dir first
        checkpoint_dir = os.path.join(self.args.log_dir, 'checkpoints')
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        
        # If specified path doesn't exist but a direct path was given, use that
        if not os.path.exists(checkpoint_path) and os.path.exists(checkpoint_name):
            checkpoint_path = checkpoint_name
            
        if not os.path.exists(checkpoint_path):
            print(f"Could not find training checkpoint at {checkpoint_path}", flush=True)
            return result
            
        try:
            print(f"Loading training checkpoint: {checkpoint_path}", flush=True)
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model state
            net.load_state_dict(checkpoint['model_state_dict'], strict=True)
            
            # Load optimizer states
            if 'optimizer_net_state_dict' in checkpoint:
                optimizer_net.load_state_dict(checkpoint['optimizer_net_state_dict'])
            
            if 'optimizer_classifier_state_dict' in checkpoint:
                optimizer_classifier.load_state_dict(checkpoint['optimizer_classifier_state_dict'])
                
            # Extract epoch if available
            if 'epoch' in checkpoint:
                result['epoch'] = checkpoint['epoch']

            if 'accuracy' in checkpoint:
                result['accuracy'] = checkpoint['accuracy']
                print(f"Loaded model with accuracy: {checkpoint['accuracy']:.4f}", flush=True)
                
            result['success'] = True
            result['checkpoint_path'] = checkpoint_path
            print(f"Successfully loaded training checkpoint from {checkpoint_path}", flush=True)
            
        except Exception as e:
            print(f"Error loading training checkpoint: {str(e)}", flush=True)
            
        return result

    def save_best_checkpoint(self, net, optimizer_net, optimizer_classifier, epoch, accuracy):
        """
        Save a checkpoint if it's the best performing model so far.
        
        Args:
            net: The network model to save
            optimizer_net: The network optimizer to save
            optimizer_classifier: The classifier optimizer to save
            epoch: Current epoch number
            accuracy: The test accuracy for this checkpoint
        """
        net.eval()
        checkpoint_dir = self._ensure_checkpoint_dir_exists()
        
        # Create the checkpoint data
        checkpoint_data = {
            'model_state_dict': net.state_dict(),
            'optimizer_net_state_dict': optimizer_net.state_dict(),
            'optimizer_classifier_state_dict': optimizer_classifier.state_dict(),
            'epoch': epoch,
            'accuracy': accuracy
        }
        
        # Path for the best model checkpoint
        best_path = os.path.join(checkpoint_dir, 'net_trained_best')
        
        # Check if we should save this as the best model
        should_save = True
        if os.path.exists(best_path):
            try:
                prev_checkpoint = torch.load(best_path, map_location=self.device)
                prev_acc = prev_checkpoint.get('accuracy', 0)
                if prev_acc >= accuracy:
                    # Previous model was better
                    should_save = False
            except Exception as e:
                print(f"Error checking previous best checkpoint: {str(e)}", flush=True)
        
        if should_save:
            torch.save(checkpoint_data, best_path)
            print(f"Saved new best model with accuracy {accuracy:.4f} at epoch {epoch}", flush=True)
        
        net.train()

    def load_best_checkpoint(self, net, optimizer_net, optimizer_classifier):
        """
        Load the best performing model checkpoint.
        
        Args:
            net: The network model to load weights into
            optimizer_net: The network optimizer to load state into
            optimizer_classifier: The classifier optimizer to load state into
            
        Returns:
            dict: Information about loaded checkpoint including success status, accuracy and epoch
        """
        return self.load_trained_checkpoint(net, optimizer_net, optimizer_classifier, 
                                            checkpoint_name='net_trained_best')