"""Module with networks for the mtl model"""
import torch.nn as nn

import networks
import utils


class ResNet18MTLWithBottleneck(nn.Module):
    """Definition for MTL network with ResNet18"""
    
    def __init__(self, pretrained=False, backbone_weights=None, num_classes=12, ssl_weights=None,
                 classifier_weights=None, bottleneck_weights=None, tb_writer=None, track_gradients=False):
        super().__init__()
        self.tb_writer = tb_writer
        self.track_gradients = track_gradients
        self.backbone = networks.ResNet18Backbone(pretrained=pretrained, weights=backbone_weights, tb_writer=tb_writer,
                                                  track_gradients=self.track_gradients)
        assert not self.track_gradients or self.tb_writer is not None, \
            "track_gradients is True, but tb_writer not defined..."
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
    
    def forward_all(self, x, step=None):
        """Forward method"""
        feats = self.backbone.forward(x, step=step)
        if self.track_gradients and feats.requires_grad:
            feats.register_hook(lambda grad: self.tb_writer.add_scalar('Grad/Feats', grad.abs().mean(),
                                                                       global_step=step))
        
        bottl_feats = self.bottleneck.forward(feats)
        if self.track_gradients and bottl_feats.requires_grad:
            bottl_feats.register_hook(lambda grad: self.tb_writer.add_scalar('Grad/Bottl_Feats', grad.abs().mean(),
                                                                             global_step=step))
        
        classifier_logits = self.classifier_head.forward(bottl_feats)
        if self.track_gradients and classifier_logits.requires_grad:
            classifier_logits.register_hook(lambda grad: self.tb_writer.add_scalar('Grad/Cl_logits', grad.abs().mean(),
                                                                                   global_step=step))
        
        ssl_logits = self.ssl_head.forward(feats)
        if self.track_gradients and ssl_logits.requires_grad:
            ssl_logits.register_hook(lambda grad: self.tb_writer.add_scalar('Grad/SSL_logits', grad.abs().mean(),
                                                                            global_step=step))
        
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
