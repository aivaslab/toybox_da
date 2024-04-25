"""Module containing the network implementations for the algorithms"""

import torch
import torch.nn as nn
import torchvision.models as models

import utils


class ResNet18BackboneWithActivations(nn.Module):
    """ResNet18Backbone without the fully connected layer and additionally returns activations at the end of each
    residual layer"""

    MAX_POOLS = {
        'conv1': nn.MaxPool2d(kernel_size=14, stride=14, padding=0),
        'layer1': nn.MaxPool2d(kernel_size=7, stride=7, padding=0),
        'layer2': nn.MaxPool2d(kernel_size=5, stride=5, padding=2),
        'layer3': nn.MaxPool2d(kernel_size=4, stride=4, padding=1),
        'layer4': nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
        'avgpool': nn.Identity()
    }

    FC_SIZES = {
        'conv1': 4096,
        'layer1': 4096,
        'layer2': 4608,
        'layer3': 4096,
        'layer4': 4608,
        'avgpool': 512
    }
    
    def __init__(self, pretrained=False, weights=None):
        super().__init__()
        assert not pretrained or weights is None, \
            "Resnet18 init asking for both ILSVRC init and pretrained_weights provided. Choose one..."
        self.pretrained = pretrained
        if self.pretrained:
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.model = models.resnet18(weights=None)
            self.model.apply(utils.weights_init)
        self.fc_size = self.model.fc.in_features
        self.model.fc = nn.Identity()
        
        if weights is not None:
            self.model.load_state_dict(weights)
    
    def forward(self, x):
        """Forward method"""
        x = self.model.conv1.forward(x)
        x = self.model.bn1(x)
        
        layer0_out = self.model.relu(x)
        layer0_out_reduced = self.MAX_POOLS['conv1'](layer0_out)
        layer0_out_reduced = layer0_out_reduced.view(layer0_out_reduced.size(0), -1)

        layer0_out = self.model.maxpool(layer0_out)
        layer1_out = self.model.layer1(layer0_out)
        layer1_out_reduced = self.MAX_POOLS['layer1'](layer1_out)
        layer1_out_reduced = layer1_out_reduced.view(layer1_out_reduced.size(0), -1)
        
        layer2_out = self.model.layer2(layer1_out)
        layer2_out_reduced = self.MAX_POOLS['layer2'](layer2_out)
        layer2_out_reduced = layer2_out_reduced.view(layer2_out_reduced.size(0), -1)
        
        layer3_out = self.model.layer3(layer2_out)
        layer3_out_reduced = self.MAX_POOLS['layer3'](layer3_out)
        layer3_out_reduced = layer3_out_reduced.view(layer3_out_reduced.size(0), -1)
        
        layer4_out = self.model.layer4(layer3_out)
        layer4_out_reduced = self.MAX_POOLS['layer4'](layer4_out)
        layer4_out_reduced = layer4_out_reduced.view(layer4_out_reduced.size(0), -1)
        
        avgpool_out = self.model.avgpool(layer4_out)
        avgpool_out = torch.flatten(avgpool_out, 1)
        avgpool_out_reduced = self.MAX_POOLS['avgpool'](avgpool_out)
        return layer0_out_reduced, \
            layer1_out_reduced, layer2_out_reduced, layer3_out_reduced, layer4_out_reduced, \
            avgpool_out_reduced
    
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

    def count_trainable_parameters(self):
        """Count the number of trainable parameters"""
        num_params = sum(p.numel() for p in self.backbone.parameters())
        num_params_trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        num_params += sum(p.numel() for p in self.classifier_head.parameters())
        num_params_trainable += sum(p.numel() for p in self.classifier_head.parameters() if p.requires_grad)
        return num_params_trainable, num_params

    def forward(self, x):
        """Forward method"""
        feats = self.backbone.forward(x)
        return self.classifier_head.forward(feats)

    def freeze_train(self):
        """Unfreeze all weights for training"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        for param in self.classifier_head.parameters():
            param.requires_grad = True

    def freeze_eval(self):
        """Freeze all weights for evaluation"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.classifier_head.parameters():
            param.requires_grad = False

    def freeze_linear_eval(self):
        """Freeze all weights for linear eval training"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.classifier_head.parameters():
            param.requires_grad = True

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

    def save_model(self, fpath: str):
        """Save the model"""
        save_dict = {
            'type': self.__class__.__name__,
            'backbone': self.backbone.model.state_dict(),
            'classifier': self.classifier_head.state_dict(),
        }
        torch.save(save_dict, fpath)


class ResNet18SupLargeMargin(nn.Module):
    """Definition for Supervised network with ResNet18"""
    
    def __init__(self, pretrained=False, backbone_weights=None, num_classes=12, margin=3):
        super().__init__()
        self.backbone = ResNet18Backbone(pretrained=pretrained, weights=backbone_weights)
        self.backbone_fc_size = self.backbone.fc_size
        self.num_classes = num_classes
        self.margin=margin
        import lsoftmax
        self.classifier_head = lsoftmax.LSoftmaxLinear(input_features=self.backbone_fc_size,
                                                       output_features=num_classes, margin=margin,
                                                       device=torch.device("cuda"))
        self.classifier_head.reset_parameters()
    
    def forward(self, x, target=None):
        """Forward method"""
        feats = self.backbone.forward(x)
        logits = self.classifier_head.forward(feats, target)
        return logits
    
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


class ResNet18Sup12x2(nn.Module):
    """Definition for Supervised network with ResNet18"""
    
    def __init__(self, pretrained=False, backbone_weights=None, num_classes=12, classifier_weights=None):
        super().__init__()
        self.backbone = ResNet18Backbone(pretrained=pretrained, weights=backbone_weights)
        self.backbone_fc_size = self.backbone.fc_size
        self.num_classes = num_classes
        
        self.classifier_head_1 = nn.Linear(self.backbone_fc_size, self.num_classes)
        self.classifier_head_2 = nn.Linear(self.backbone_fc_size, self.num_classes)
        
        if classifier_weights is not None:
            self.classifier_head_1.load_state_dict(classifier_weights)
        else:
            self.classifier_head_1.apply(utils.weights_init)
        self.classifier_head_2.apply(utils.weights_init)
    
    def forward(self, x):
        """Forward method"""
        feats = self.backbone.forward(x)
        bsize = x.shape[0]
        src_size = bsize // 2
        src_feats, trgt_feats = feats[:src_size], feats[src_size:]
        return src_feats, trgt_feats, \
            self.classifier_head_1.forward(src_feats), self.classifier_head_2.forward(trgt_feats)
    
    def forward_1(self, x):
        """Forward method through classifier 1"""
        feats = self.backbone.forward(x)
        return feats, self.classifier_head_1.forward(feats)

    def forward_2(self, x):
        """Forward method through classifier 2"""
        feats = self.backbone.forward(x)
        return feats, self.classifier_head_2.forward(feats)

    def set_train(self):
        """Set network in train mode"""
        self.backbone.train()
        self.classifier_head_1.train()
        self.classifier_head_2.train()
    
    def set_linear_eval(self):
        """Set backbone in eval and cl in train mode"""
        self.backbone.eval()
        self.classifier_head_1.train()
        self.classifier_head_2.train()
    
    def set_eval(self):
        """Set network in eval mode"""
        self.backbone.eval()
        self.classifier_head_1.eval()
        self.classifier_head_2.eval()
    
    def get_params(self) -> dict:
        """Return a dictionary of the parameters of the model"""
        return {'backbone_params': self.backbone.parameters(),
                'classifier_1_params': self.classifier_head_1.parameters(),
                'classifier_2_params': self.classifier_head_2.parameters(),
                }


class ResNet18Sup12x2v2(nn.Module):
    """Definition for Supervised network with ResNet18"""
    
    def __init__(self, pretrained=False, backbone_weights=None, num_classes=12, classifier_weights=None):
        super().__init__()
        self.backbone = ResNet18Backbone(pretrained=pretrained, weights=backbone_weights)
        self.backbone_fc_size = self.backbone.fc_size
        self.num_classes = num_classes
        
        self.classifier_head_1 = nn.Linear(self.backbone_fc_size, self.num_classes)
        self.classifier_head_2 = nn.Linear(self.backbone_fc_size, self.num_classes)
        
        if classifier_weights is not None:
            self.classifier_head_1.load_state_dict(classifier_weights)
        else:
            self.classifier_head_1.apply(utils.weights_init)
        self.classifier_head_2.apply(utils.weights_init)
    
    def forward(self, x):
        """Forward method"""
        feats = self.backbone.forward(x)
        bsize = x.shape[0]
        src_size = bsize // 2
        src_feats, trgt_feats = feats[:src_size], feats[src_size:]
        return src_feats, trgt_feats, \
            self.classifier_head_1.forward(src_feats), self.classifier_head_1.forward(trgt_feats),\
            self.classifier_head_2.forward(src_feats), self.classifier_head_2.forward(trgt_feats)
    
    def forward_1(self, x):
        """Forward method through classifier 1"""
        feats = self.backbone.forward(x)
        return feats, self.classifier_head_1.forward(feats)
    
    def forward_2(self, x):
        """Forward method through classifier 2"""
        feats = self.backbone.forward(x)
        return feats, self.classifier_head_2.forward(feats)
    
    def set_train(self):
        """Set network in train mode"""
        self.backbone.train()
        self.classifier_head_1.train()
        self.classifier_head_2.train()
    
    def set_linear_eval(self):
        """Set backbone in eval and cl in train mode"""
        self.backbone.eval()
        self.classifier_head_1.train()
        self.classifier_head_2.train()
    
    def set_eval(self):
        """Set network in eval mode"""
        self.backbone.eval()
        self.classifier_head_1.eval()
        self.classifier_head_2.eval()
    
    def get_params(self) -> dict:
        """Return a dictionary of the parameters of the model"""
        return {'backbone_params': self.backbone.parameters(),
                'classifier_1_params': self.classifier_head_1.parameters(),
                'classifier_2_params': self.classifier_head_2.parameters(),
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


class ResNet18DualSupJAN(nn.Module):
    """Definition for JAN network with ResNet18"""
    
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
        backbone_feats = self.backbone.forward(x)
        return backbone_feats, self.classifier_head.forward(backbone_feats)
    
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

    def freeze_train(self):
        """Unfreeze all weights for training"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        for param in self.ssl_head.parameters():
            param.requires_grad = False

    def freeze_eval(self):
        """Freeze all weights for eval"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.ssl_head.parameters():
            param.requires_grad = False

    def count_trainable_parameters(self):
        """Count the number of trainable parameters"""
        num_params = sum(p.numel() for p in self.backbone.parameters())
        num_params_trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        num_params += sum(p.numel() for p in self.ssl_head.parameters())
        num_params_trainable += sum(p.numel() for p in self.ssl_head.parameters() if p.requires_grad)
        return num_params_trainable, num_params
    
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

    def save_model(self, fpath: str):
        """Save the model"""
        save_dict = {
            'type': self.__class__.__name__,
            'backbone': self.backbone.model.state_dict(),
            'ssl_head': self.ssl_head.state_dict()
        }
        torch.save(save_dict, fpath)
    
    
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
    