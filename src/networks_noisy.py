"""Module with networks for the noisy labels models"""
import torch.nn as nn

import networks
import utils


class ResNet18Noisy1(nn.Module):
    """Definition for MTL network with ResNet18"""
    
    def __init__(self, pretrained=False, backbone_weights=None, num_classes=12, classifier_weights=None, tb_writer=None,
                 track_gradients=False):
        super().__init__()
        self.tb_writer = tb_writer
        self.track_gradients = track_gradients
        self.backbone = networks.ResNet18Backbone(pretrained=pretrained, weights=backbone_weights, tb_writer=tb_writer,
                                                  track_gradients=self.track_gradients)
        assert not self.track_gradients or self.tb_writer is not None, \
            "track_gradients is True, but tb_writer not defined..."
        self.backbone_fc_size = self.backbone.fc_size
        self.num_classes = num_classes
        
        self.classifier_head = nn.Linear(self.backbone_fc_size, self.num_classes)
        
        if classifier_weights is not None:
            self.classifier_head.load_state_dict(classifier_weights)
        else:
            self.classifier_head.apply(utils.weights_init)
    
    def forward(self, x, step=None):
        """Forward method"""
        feats = self.backbone.forward(x, step=step)
        if self.track_gradients and feats.requires_grad:
            feats.register_hook(lambda grad: self.tb_writer.add_scalar('Grad/Feats', grad.abs().mean(),
                                                                       global_step=step))
        
        classifier_logits = self.classifier_head.forward(feats)
        if self.track_gradients and classifier_logits.requires_grad:
            classifier_logits.register_hook(lambda grad: self.tb_writer.add_scalar('Grad/Cl_logits', grad.abs().mean(),
                                                                                   global_step=step))
        
        return feats, classifier_logits
    
    def freeze_network(self):
        """Freeze parameters of the network"""
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.classifier_head.parameters():
            p.requires_grad = False
            
    def unfreeze_network(self):
        """Unfreeze parameters of the network"""
        for p in self.backbone.parameters():
            p.requires_grad = True
        for p in self.classifier_head.parameters():
            p.requires_grad = True
    
    def set_train(self):
        """Set network in train mode"""
        self.backbone.train()
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
