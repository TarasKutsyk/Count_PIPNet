import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from features.resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet50_features_inat, resnet101_features, resnet152_features
from features.convnext_features import convnext_tiny_26_features, convnext_tiny_13_features 
import torch
from torch import Tensor

class PIPNet(nn.Module):
    def __init__(self,
                 num_classes: int,
                 num_prototypes: int,
                 feature_net: nn.Module,
                 args: argparse.Namespace,
                 add_on_layers: nn.Module,
                 pool_layer: nn.Module,
                 classification_layer: nn.Module
                 ):
        super().__init__()
        assert num_classes > 0
        self._num_features = args.num_features
        self._num_classes = num_classes
        self._num_prototypes = num_prototypes
        self._net = feature_net
        self._add_on = add_on_layers
        self._pool = pool_layer
        self._classification = classification_layer
        self._multiplier = classification_layer.normalization_multiplier

    def forward(self, xs, inference=False):
        features = self._net(xs) 
        proto_features = self._add_on(features)
        pooled = self._pool(proto_features)
        if inference:
            clamped_pooled = torch.where(pooled < 0.1, 0., pooled)  #during inference, ignore all prototypes that have 0.1 similarity or lower
            out = self._classification(clamped_pooled) #shape (bs*2, num_classes)
            return proto_features, clamped_pooled, out
        else:
            out = self._classification(pooled) #shape (bs*2, num_classes) 
            return proto_features, pooled, out


base_architecture_to_features = {'resnet18': resnet18_features,
                                 'resnet34': resnet34_features,
                                 'resnet50': resnet50_features,
                                 'resnet50_inat': resnet50_features_inat,
                                 'resnet101': resnet101_features,
                                 'resnet152': resnet152_features,
                                 'convnext_tiny_26': convnext_tiny_26_features,
                                 'convnext_tiny_13': convnext_tiny_13_features}

# adapted from https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
class NonNegLinear(nn.Module):
    """Applies a linear transformation to the incoming data with non-negative weights`
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(NonNegLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.normalization_multiplier = nn.Parameter(torch.ones((1,),requires_grad=True))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input,torch.relu(self.weight), self.bias)


def get_pip_network(num_classes: int, args: argparse.Namespace): 
    if 'convnext' in args.net:
        # Get the feature extractor backbone (ConvNeXt only)
        use_mid_layers = getattr(args, 'use_mid_layers', False)
        num_stages = getattr(args, 'num_stages', 2)
        
        backbone_nn = base_architecture_to_features[args.net](
            pretrained=not args.disable_pretrained,
            use_mid_layers=use_mid_layers,
            num_stages=num_stages)
    elif 'res' in args.net:
        backbone_nn = base_architecture_to_features[args.net](pretrained=not args.disable_pretrained)
    else:
        raise Exception('other base architecture NOT implemented')
        
    first_add_on_layer_in_channels = \
        [i for i in backbone_nn.modules() if isinstance(i, nn.Conv2d)][-1].out_channels

    if args.num_features == 0:
        num_prototypes = first_add_on_layer_in_channels
        print("Number of prototypes: ", num_prototypes, flush=True)
        add_on_layers = nn.Sequential(
            nn.Softmax(dim=1), #softmax over every prototype for each patch, such that for every location in image, sum over prototypes is 1                
    )
    else:
        num_prototypes = args.num_features
        print("Number of prototypes set from", first_add_on_layer_in_channels, "to", num_prototypes,". Extra 1x1 conv layer added. Not recommended.", flush=True)
        add_on_layers = nn.Sequential(
            nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=num_prototypes, kernel_size=1, stride = 1, padding=0, bias=True), 
            nn.Softmax(dim=1), #softmax over every prototype for each patch, such that for every location in image, sum over prototypes is 1                
    )
    pool_layer = nn.Sequential(
                nn.AdaptiveMaxPool2d(output_size=(1,1)), #outputs (bs, ps,1,1)
                nn.Flatten() #outputs (bs, ps)
                ) 
    
    if args.bias:
        classification_layer = NonNegLinear(num_prototypes, num_classes, bias=True)
    else:
        classification_layer = NonNegLinear(num_prototypes, num_classes, bias=False)
        
    return backbone_nn, add_on_layers, pool_layer, classification_layer, num_prototypes

def get_pipnet(num_classes: int, args: argparse.Namespace):
    """
    Create a complete PIPNet model with the specified parameters.
    
    Args:
        num_classes: Number of output classes
        args: Command line arguments
        
    Returns:
        Tuple of (model, num_prototypes)
    """
    feature_net, add_on_layers, pool_layer, classification_layer, num_prototypes = get_pip_network(num_classes, args)
    
    model = PIPNet(
        num_classes=num_classes,
        num_prototypes=num_prototypes,
        feature_net=feature_net, 
        args=args,
        add_on_layers=add_on_layers,
        pool_layer=pool_layer,
        classification_layer=classification_layer
    )
    
    return model, num_prototypes