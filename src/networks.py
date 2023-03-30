"""Module containing the network implementations for the algorithms"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18Backbone(nn.Module):
    """ResNet18Backbone without the fully connected layer"""
    def __init__(self, pretrained=False, backbone_weights=None):
        super().__init__()
        assert not pretrained or backbone_weights is None, \
            "Resnet18 init asking for both ILSVRC init and pretrained_weights provided. Choose one..."
        self.pretrained = pretrained
        if self.pretrained:
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.model = models.resnet18(weights=None)
        self.fc_size = self.model.fc.in_features
        self.model.fc = nn.Identity()
        
        if backbone_weights is not None:
            self.model.load_state_dict(backbone_weights)
        
    def forward(self, x):
        """Forward method"""
        return self.model.forward(x)
    
    def set_train(self):
        """Set network in train mode"""
        self.model.train()
        
    def set_eval(self):
        """Set network in eval mode"""
        self.model.eval()
        
    def parameters(self, recurse=True):
        """Return the parameters of the module"""
        return self.model.parameters()
    
    
    