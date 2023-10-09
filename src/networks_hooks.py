"""Module containing the networks with hooks attached to the conv layers"""
import collections

import torch
import torch.nn as nn
import torchvision.models as models

import utils

module_names = {}
all_activations = collections.defaultdict(list)


def reset_global_data():
    """Reset the global data"""
    global all_activations
    all_activations = collections.defaultdict(list)
    

def get_act(self, _, outp):
    """Print activations"""
    avgpool_layer = nn.AvgPool2d(kernel_size=outp.shape[-1])
    pooled_output = avgpool_layer.forward(outp).squeeze()
    # print(mods[self], input[0].shape, output.shape, pooled_output.shape)
    all_activations[module_names[self]].append(pooled_output)


class ResNet18BackboneWithConvActivations(nn.Module):
    """ResNet18Backbone without the fully connected layer"""
    
    def __init__(self, pretrained=False, weights=None, tb_writer=None, track_gradients=False):
        super().__init__()
        assert not pretrained or weights is None, \
            "Resnet18 init asking for both ILSVRC init and pretrained_weights provided. Choose one..."
        self.pretrained = pretrained
        self.tb_writer = tb_writer
        self.track_gradients = track_gradients
        assert not self.track_gradients or self.tb_writer is not None, \
            "track_gradients is True, but tb_writer not defined..."
        if self.pretrained:
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.model = models.resnet18(weights=None)
            self.model.apply(utils.weights_init)
        self.fc_size = self.model.fc.in_features
        self.model.fc = nn.Identity()
        
        if weights is not None:
            self.model.load_state_dict(weights)

        for name, mod in self.model.named_modules():
            if "conv" in name:
                module_names[mod] = name
                mod.register_forward_hook(get_act)
    
    def forward(self, x, step=None):
        """Forward method"""
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        if self.track_gradients and x.requires_grad:
            x.register_hook(lambda grad: self.tb_writer.add_scalar('Grad/Layer0', grad.abs().mean(), global_step=step))
        
        x = self.model.layer1(x)
        if self.track_gradients and x.requires_grad:
            x.register_hook(lambda grad: self.tb_writer.add_scalar('Grad/Layer1', grad.abs().mean(), global_step=step))
        
        x = self.model.layer2(x)
        if self.track_gradients and x.requires_grad:
            x.register_hook(lambda grad: self.tb_writer.add_scalar('Grad/Layer2', grad.abs().mean(), global_step=step))
        
        x = self.model.layer3(x)
        if self.track_gradients and x.requires_grad:
            x.register_hook(lambda grad: self.tb_writer.add_scalar('Grad/Layer3', grad.abs().mean(), global_step=step))
        
        x = self.model.layer4(x)
        if self.track_gradients and x.requires_grad:
            x.register_hook(lambda grad: self.tb_writer.add_scalar('Grad/Layer4', grad.abs().mean(), global_step=step))
        
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        if self.track_gradients and x.requires_grad:
            x.register_hook(lambda grad: self.tb_writer.add_scalar('Grad/AvgPool', grad.abs().mean(), global_step=step))
        return x
    
    def set_train(self):
        """Set network in train mode"""
        self.model.train()
    
    def set_eval(self):
        """Set network in eval mode"""
        self.model.eval()
    
    def get_params(self) -> dict:
        """Return the parameters of the module"""
        return {'backbone_params': self.model.parameters()
                }



