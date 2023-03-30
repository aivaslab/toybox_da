"""Module containing the network implementations for the algorithms"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18Backbone(nn.Module):
    """ResNet18Backbone without the fully connected layer"""
    def __init__(self, pretrained=False, weights=None):
        super().__init__()
        assert not pretrained or weights is None, \
            "Resnet18 init asking for both ILSVRC init and pretrained_weights provided. Choose one..."
        self.pretrained = pretrained
        if self.pretrained:
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.model = models.resnet18(weights=None)
        self.fc_size = self.model.fc.in_features
        self.model.fc = nn.Identity()
        
        if weights is not None:
            self.model.load_state_dict(weights)
        
    def forward(self, x):
        """Forward method"""
        return self.model.forward(x)
    
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
    
    
class ResNet18MTL(nn.Module):
    """Definition for MTL network with ResNet18"""
    def __init__(self, pretrained=False, backbone_weights=None, num_classes=12,
                 ssl_weights=None, classifier_weights=None):
        super().__init__()
        self.backbone = ResNet18Backbone(pretrained=pretrained, weights=backbone_weights)
        self.backbone_fc_size = self.backbone.fc_size
        self.num_classes = num_classes
        
        self.classifier_head = nn.Linear(self.backbone_fc_size, self.num_classes)
        self.ssl_head = nn.Sequential(nn.Linear(self.backbone_fc_size, self.backbone_fc_size),
                                      nn.ReLU(),
                                      nn.Linear(self.backbone_fc_size, 128))
        
        if classifier_weights is not None:
            self.classifier_head.load_state_dict(classifier_weights)
        if ssl_weights is not None:
            self.ssl_head.load_state_dict(ssl_weights)
        
    def forward_all(self, x):
        """Forward method"""
        feats = self.backbone.forward(x)
        classifier_logits = self.classifier_head.forward(feats)
        ssl_logits = self.ssl_head.forward(feats)
        
        return feats, classifier_logits, ssl_logits
    
    def forward_class(self, x):
        """Forward using only classifier"""
        feats = self.backbone.forward(x)
        return self.classifier_head.forward(feats)

    def forward_ssl(self, x):
        """Forward using only classifier"""
        feats = self.backbone.forward(x)
        return self.ssl_head.forward(feats)
    
    def set_train(self):
        """Set network in train mode"""
        self.backbone.train()
        self.classifier_head.train()
        self.ssl_head.train()
        
    def set_eval(self):
        """Set network in eval mode"""
        self.backbone.eval()
        self.classifier_head.eval()
        self.ssl_head.eval()
        
    def get_params(self) -> dict:
        """Return a dictionary of the parameters of the model"""
        return {'backbone_params': self.backbone.parameters(),
                'classifier_params': self.classifier_head.parameters(),
                'ssl_head': self.ssl_head.parameters()
                }
    