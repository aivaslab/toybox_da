"""Module for implementing the networks for the domain adaptation models"""
import torch
import torch.nn as nn

import networks
import utils


class ResNet18JAN(nn.Module):
    """Definition for JAN network with ResNet18"""
    
    def __init__(self, pretrained=False, backbone_weights=None, num_classes=12, bottleneck_weights=None,
                 classifier_weights=None, dropout=0, p=0.5):
        super().__init__()
        self.backbone = networks.ResNet18Backbone(pretrained=pretrained, weights=backbone_weights)
        self.backbone_fc_size = self.backbone.fc_size
        self.num_classes = num_classes
        self.bottleneck_dim = 256
        self.dropout = dropout
        self.p = p
        self.bottleneck = nn.Sequential(
            nn.Linear(self.backbone_fc_size, self.bottleneck_dim, bias=False),
            nn.BatchNorm1d(self.bottleneck_dim),
            nn.ReLU()
        )
        if bottleneck_weights is not None:
            self.bottleneck.load_state_dict(bottleneck_weights)
        else:
            nn.init.kaiming_normal_(self.bottleneck[0].weight.data, nonlinearity='relu')

        self.backbone_dropout = nn.Dropout(p=self.p) if self.dropout in [1, 3] else nn.Identity()
        self.bottleneck_dropout = nn.Dropout(p=self.p) if self.dropout in [2, 3] else nn.Identity()
        
        self.classifier_head = nn.Linear(self.bottleneck_dim, self.num_classes)
        if classifier_weights is not None:
            self.classifier_head.load_state_dict(classifier_weights)
        else:
            nn.init.kaiming_normal_(self.classifier_head.weight.data, nonlinearity='relu')
    
    def forward(self, x):
        """Forward method"""
        backbone_feats = self.backbone.forward(x)
        backbone_drop_feats = self.backbone_dropout.forward(backbone_feats)
        bottleneck_feats = self.bottleneck.forward(backbone_drop_feats)
        bottleneck_drop_feats = self.bottleneck_dropout.forward(bottleneck_feats)
        return backbone_drop_feats, bottleneck_drop_feats, self.classifier_head.forward(bottleneck_drop_feats)
    
    def set_train(self):
        """Set network in train mode"""
        self.backbone.train()
        self.backbone_dropout.train()
        self.bottleneck.train()
        self.bottleneck_dropout.train()
        self.classifier_head.train()
    
    def set_linear_eval(self):
        """Set backbone in eval and cl in train mode"""
        self.backbone.eval()
        self.backbone_dropout.train()
        self.bottleneck.train()
        self.bottleneck_dropout.train()
        self.classifier_head.train()
    
    def set_eval(self):
        """Set network in eval mode"""
        self.backbone_dropout.eval()
        self.backbone.eval()
        self.bottleneck.eval()
        self.bottleneck_dropout.eval()
        self.classifier_head.eval()
    
    def get_params(self) -> dict:
        """Return a dictionary of the parameters of the model"""
        return {'backbone_params': self.backbone.parameters(),
                'backbone_dropout': self.backbone_dropout.parameters(),
                'bottleneck_params': self.bottleneck.parameters(),
                'classifier_params': self.classifier_head.parameters(),
                }

    def save_model(self, fpath: str):
        """Save the model"""
        save_dict = {
            'type': self.__class__.__name__,
            'dropout': self.dropout,
            'backbone': self.backbone.model.state_dict(),
            'backbone_dropout': self.backbone_dropout.state_dict(),
            'bottleneck': self.bottleneck.state_dict(),
            'bottleneck_dropout': self.bottleneck_dropout.state_dict(),
            'classifier': self.classifier_head.state_dict(),
        }
        torch.save(save_dict, fpath)


class ResNet18SSLJAN(nn.Module):
    """Definition for JAN network with ResNet18"""

    def __init__(self, pretrained=False, backbone_weights=None, num_classes=12, bottleneck_weights=None,
                 ssl_weights=None, classifier_weights=None, dropout=0, p=0.5):
        super().__init__()
        self.backbone = networks.ResNet18SSL(pretrained=pretrained, backbone_weights=backbone_weights,
                                             ssl_weights=ssl_weights)
        self.backbone_fc_size = 128
        self.num_classes = num_classes
        self.bottleneck_dim = 256
        self.dropout = dropout
        self.p = p
        self.bottleneck = nn.Sequential(
            nn.Linear(self.backbone_fc_size, self.bottleneck_dim),
            nn.BatchNorm1d(self.bottleneck_dim),
            nn.ReLU()
        )
        if bottleneck_weights is not None:
            self.bottleneck.load_state_dict(bottleneck_weights)
        else:
            self.bottleneck.apply(utils.weights_init)

        self.backbone_dropout = nn.Dropout(p=self.p) if self.dropout in [1, 3] else nn.Identity()
        self.bottleneck_dropout = nn.Dropout(p=self.p) if self.dropout in [2, 3] else nn.Identity()

        self.classifier_head = nn.Linear(self.bottleneck_dim, self.num_classes)
        if classifier_weights is not None:
            self.classifier_head.load_state_dict(classifier_weights)
        else:
            self.classifier_head.apply(utils.weights_init)

    def forward(self, x):
        """Forward method"""
        backbone_feats = self.backbone.forward(x)
        backbone_drop_feats = self.backbone_dropout.forward(backbone_feats)
        bottleneck_feats = self.bottleneck.forward(backbone_drop_feats)
        bottleneck_drop_feats = self.bottleneck_dropout.forward(bottleneck_feats)
        return backbone_drop_feats, bottleneck_drop_feats, self.classifier_head.forward(bottleneck_drop_feats)

    def set_train(self):
        """Set network in train mode"""
        self.backbone.train()
        self.backbone_dropout.train()
        self.bottleneck.train()
        self.bottleneck_dropout.train()
        self.classifier_head.train()

    def set_linear_eval(self):
        """Set backbone in eval and cl in train mode"""
        self.backbone.eval()
        self.backbone_dropout.train()
        self.bottleneck.train()
        self.bottleneck_dropout.train()
        self.classifier_head.train()

    def set_eval(self):
        """Set network in eval mode"""
        self.backbone_dropout.eval()
        self.backbone.eval()
        self.bottleneck.eval()
        self.bottleneck_dropout.eval()
        self.classifier_head.eval()

    def get_params(self) -> dict:
        """Return a dictionary of the parameters of the model"""
        return {'backbone_params': self.backbone.parameters(),
                'backbone_dropout': self.backbone_dropout.parameters(),
                'bottleneck_params': self.bottleneck.parameters(),
                'classifier_params': self.classifier_head.parameters(),
                }

    def save_model(self, fpath: str):
        """Save the model"""
        save_dict = {
            'type': self.__class__.__name__,
            'dropout': self.dropout,
            'backbone': self.backbone.model.state_dict(),
            'backbone_dropout': self.backbone_dropout.state_dict(),
            'bottleneck': self.bottleneck.state_dict(),
            'bottleneck_dropout': self.bottleneck_dropout.state_dict(),
            'classifier': self.classifier_head.state_dict(),
        }
        torch.save(save_dict, fpath)


def test():
    """Method to test the models"""
    model = ResNet18JAN(pretrained=True, num_classes=12)
    x = torch.rand((3, 3, 224, 224))
    ou = model.backbone.forward(x)
    print(ou.shape)
    ou = model.bottleneck.forward(ou)
    print(ou.shape)
    ou = model.classifier_head.forward(ou)
    print(ou.shape)
    
    
if __name__ == "__main__":
    test()
