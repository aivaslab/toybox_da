"""Module with networks for the mtl model"""
import torch.nn as nn

import networks
import utils


class ResNet18MTLWithBottleneck(nn.Module):
    """Definition for MTL network with ResNet18"""
    
    def __init__(self, pretrained=False, backbone_weights=None, num_classes=12, ssl_weights=None,
                 classifier_weights=None, bottleneck_weights=None):
        super().__init__()
        self.backbone = networks.ResNet18Backbone(pretrained=pretrained, weights=backbone_weights)
        self.backbone_fc_size = self.backbone.fc_size
        self.num_classes = num_classes
        self.bottleneck_dim = 256

        self.bottleneck = nn.Sequential(
            nn.Linear(self.backbone_fc_size, self.bottleneck_dim),
            nn.BatchNorm1d(self.bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        if bottleneck_weights is not None:
            self.bottleneck.load_state_dict(bottleneck_weights)
        else:
            self.bottleneck.apply(utils.weights_init)

        self.classifier_head = nn.Linear(self.bottleneck_dim, self.num_classes)
        
        if classifier_weights is not None:
            self.classifier_head.load_state_dict(classifier_weights)
        else:
            self.classifier_head.apply(utils.weights_init)
        
        self.ssl_head = nn.Sequential(nn.Linear(self.backbone_fc_size, self.backbone_fc_size),
                                      nn.ReLU(),
                                      nn.Linear(self.backbone_fc_size, 128))
    
        if ssl_weights is not None:
            self.ssl_head.load_state_dict(ssl_weights)
        else:
            self.ssl_head.apply(utils.weights_init)
    
    def forward_all(self, x):
        """Forward method"""
        feats = self.backbone.forward(x)
        bottl_feats = self.bottleneck.forward(feats)
        classifier_logits = self.classifier_head.forward(bottl_feats)
        ssl_logits = self.ssl_head.forward(feats)
        
        return bottl_feats, classifier_logits, ssl_logits
    
    def forward_class(self, x):
        """Forward using only classifier"""
        feats = self.backbone.forward(x)
        bottl_feats = self.bottleneck.forward(feats)
        return self.classifier_head.forward(bottl_feats)
    
    def forward_ssl(self, x):
        """Forward using only classifier"""
        feats = self.backbone.forward(x)
        return self.ssl_head.forward(feats)
    
    def set_train(self):
        """Set network in train mode"""
        self.backbone.train()
        self.bottleneck.train()
        self.classifier_head.train()
        self.ssl_head.train()
    
    def set_eval(self):
        """Set network in eval mode"""
        self.backbone.eval()
        self.bottleneck.eval()
        self.classifier_head.eval()
        self.ssl_head.eval()
    
    def get_params(self) -> dict:
        """Return a dictionary of the parameters of the model"""
        return {'backbone_params': self.backbone.parameters(),
                'bottleneck_params': self.bottleneck.parameters(),
                'classifier_params': self.classifier_head.parameters(),
                'ssl_head': self.ssl_head.parameters()
                }
