"""Module containing the network implementations for the algorithms"""

import torch
import torch.nn as nn
import torchvision.models as models

import utils


class ResNet18Backbone(nn.Module):
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
        
    def forward(self, x, step=None):
        """Forward method"""
        x = self.model.conv1.forward(x)
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


class ResNet18Sup(nn.Module):
    """Definition for Supervised network with ResNet18"""
    
    def __init__(self, pretrained=False, backbone_weights=None, num_classes=12, classifier_weights=None):
        super().__init__()
        self.backbone = ResNet18Backbone(pretrained=pretrained, weights=backbone_weights)
        self.backbone_fc_size = self.backbone.fc_size
        self.num_classes = num_classes
        
        self.classifier_head = nn.Linear(self.backbone_fc_size, self.num_classes)
        
        if classifier_weights is not None:
            self.classifier_head.load_state_dict(classifier_weights)
        else:
            self.classifier_head.apply(utils.weights_init)
    
    def forward(self, x):
        """Forward method"""
        feats = self.backbone.forward(x)
        return self.classifier_head.forward(feats)
    
    def set_train(self):
        """Set network in train mode"""
        self.backbone.train()
        self.classifier_head.train()
        
    def set_linear_eval(self):
        """Set backbone in eval and cl in train mode"""
        self.backbone.eval()
        self.classifier_head.train()
    
    def set_eval(self):
        """Set network in eval mode"""
        self.backbone.eval()
        self.classifier_head.eval()
    
    def get_params(self) -> dict:
        """Return a dictionary of the parameters of the model"""
        return {'backbone_params': self.backbone.parameters(),
                'classifier_params': self.classifier_head.parameters(),
                }


class ResNet18Sup24(nn.Module):
    """Definition for Supervised network with ResNet18 for the 24-class experiments"""
    
    def __init__(self, pretrained=False, backbone_weights=None, num_classes=24, classifier_weights=None):
        super().__init__()
        self.backbone = ResNet18Backbone(pretrained=pretrained, weights=backbone_weights)
        self.backbone_fc_size = self.backbone.fc_size
        self.num_classes = num_classes
        
        self.classifier_head = nn.Linear(self.backbone_fc_size, self.num_classes)
        
        if classifier_weights is not None:
            self.classifier_head.load_state_dict(classifier_weights)
        else:
            self.classifier_head.apply(utils.weights_init)
    
    def forward(self, x):
        """Forward method"""
        feats = self.backbone.forward(x)
        return self.classifier_head.forward(feats)
    
    def set_train(self):
        """Set network in train mode"""
        self.backbone.train()
        self.classifier_head.train()
    
    def set_linear_eval(self):
        """Set backbone in eval and cl in train mode"""
        self.backbone.eval()
        self.classifier_head.train()
    
    def set_eval(self):
        """Set network in eval mode"""
        self.backbone.eval()
        self.classifier_head.eval()
    
    def get_params(self) -> dict:
        """Return a dictionary of the parameters of the model"""
        return {'backbone_params': self.backbone.parameters(),
                'classifier_params': self.classifier_head.parameters(),
                }


class ResNet18SupWithDomain(nn.Module):
    """Network for experiments of combined learning of class and domain"""
    
    def __init__(self, pretrained=False, backbone_weights=None, num_classes=12, classifier_weights=None):
        super().__init__()
        self.backbone = ResNet18Backbone(pretrained=pretrained, weights=backbone_weights)
        self.backbone_fc_size = self.backbone.fc_size
        self.num_classes = num_classes
        
        self.classifier_head = nn.Linear(self.backbone_fc_size, self.num_classes)
        
        if classifier_weights is not None:
            self.classifier_head.load_state_dict(classifier_weights)
        else:
            self.classifier_head.apply(utils.weights_init)
            
        self.domain_head = nn.Sequential(nn.Linear(self.backbone_fc_size, 1000), nn.ReLU(),
                                         nn.Linear(1000, 1))
        self.domain_head.apply(utils.weights_init)
    
    def forward(self, x):
        """Forward method"""
        feats = self.backbone.forward(x)
        logits = self.classifier_head.forward(feats)
        dom_logits = self.domain_head.forward(feats)
        # print(dom_logits)
        return logits, dom_logits.squeeze()
    
    def set_train(self):
        """Set network in train mode"""
        self.backbone.train()
        self.classifier_head.train()
        self.domain_head.train()
    
    def set_linear_eval(self):
        """Set backbone in eval and cl in train mode"""
        self.backbone.eval()
        self.classifier_head.train()
        self.domain_head.eval()
    
    def set_eval(self):
        """Set network in eval mode"""
        self.backbone.eval()
        self.classifier_head.eval()
        self.domain_head.eval()
    
    def get_params(self) -> dict:
        """Return a dictionary of the parameters of the model"""
        return {'backbone_params': self.backbone.parameters(),
                'classifier_params': self.classifier_head.parameters(),
                'domain_head_params': self.domain_head.parameters()
                }


class ResNet18SSL(nn.Module):
    """Definition for SSL network with ResNet18"""
    
    def __init__(self, pretrained=False, backbone_weights=None, ssl_weights=None):
        super().__init__()
        self.backbone = ResNet18Backbone(pretrained=pretrained, weights=backbone_weights)
        self.backbone_fc_size = self.backbone.fc_size
        
        self.ssl_head = nn.Sequential(nn.Linear(self.backbone_fc_size, self.backbone_fc_size),
                                      nn.ReLU(),
                                      nn.Linear(self.backbone_fc_size, 128))
        
        if ssl_weights is not None:
            self.ssl_head.load_state_dict(ssl_weights)
        else:
            self.ssl_head.apply(utils.weights_init)
    
    def forward(self, x):
        """Forward using only classifier"""
        feats = self.backbone.forward(x)
        return self.ssl_head.forward(feats)
    
    def set_train(self):
        """Set network in train mode"""
        self.backbone.train()
        self.ssl_head.train()
    
    def set_eval(self):
        """Set network in eval mode"""
        self.backbone.eval()
        self.ssl_head.eval()
    
    def get_params(self) -> dict:
        """Return a dictionary of the parameters of the model"""
        return {'backbone_params': self.backbone.parameters(),
                'ssl_head': self.ssl_head.parameters()
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
        else:
            self.classifier_head.apply(utils.weights_init)
        
        if ssl_weights is not None:
            self.ssl_head.load_state_dict(ssl_weights)
        else:
            self.ssl_head.apply(utils.weights_init)
        
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
    