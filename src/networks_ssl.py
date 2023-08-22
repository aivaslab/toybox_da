"""This file contains the network definitions for models trained with self-supervised learning"""
import torch.nn as nn

import numpy as np

import networks
import utils


class SimCLRResNet18(nn.Module):
    """Class definition for SimCLR with Resnet-18 as the underlying architecture"""
    def __init__(self):
        super().__init__()
        self.student_backbone = networks.ResNet18Backbone(pretrained=False, weights=None, tb_writer=None,
                                                          track_gradients=False)
        self.teacher_backbone = networks.ResNet18Backbone(pretrained=False, weights=None, tb_writer=None,
                                                          track_gradients=False)
        self.feat_num = self.student_backbone.fc_size
        self.student_projection = nn.Sequential(nn.Linear(self.feat_num, self.feat_num), nn.ReLU(inplace=True),
                                                nn.Linear(self.feat_num, 128))
        self.student_projection.apply(utils.weights_init)
        self.teacher_projection = nn.Sequential(nn.Linear(self.feat_num, self.feat_num), nn.ReLU(inplace=True),
                                                nn.Linear(self.feat_num, 128))
        
        self.teacher_backbone.load_state_dict(self.student_backbone.state_dict())
        self.teacher_projection.load_state_dict(self.student_projection.state_dict())
        self.start_beta = 0.996
        self.beta = self.start_beta
        
    def update_beta(self, step, total_steps):
        """Update the weight of the beta parameter based on the number of steps in training"""
        self.beta = 1 - (1 - self.beta) * (np.cos(np.pi * step / total_steps) + 1) / 2.0
        
    def update_teacher_weights(self):
        """Update the weights of the teacher using EMA"""
        for current_weights, ma_weights in zip(self.student_backbone.parameters(), self.teacher_backbone.parameters()):
            old_weight, up_weight = ma_weights.data, current_weights.data
            ma_weights.data = self.beta * old_weight + up_weight * (1 - self.beta)

        for current_weights, ma_weights in zip(self.student_projection.parameters(),
                                               self.teacher_projection.parameters()):
            old_weight, up_weight = ma_weights.data, current_weights.data
            ma_weights.data = self.beta * old_weight + up_weight * (1 - self.beta)
            
    def student_forward(self, x):
        """Forward method for student network"""
        feats = self.student_backbone.forward(x)
        feats = self.student_projection.forward(feats)
        return feats
    
    def teacher_forward(self, x):
        """Forward method for student network"""
        feats = self.teacher_backbone.forward(x)
        feats = self.teacher_projection.forward(feats)
        return feats
    
    def freeze_teacher(self):
        """Set requires_grad for teacher network to False"""
        for params in self.teacher_backbone.parameters():
            params.requires_grad = False
        for params in self.teacher_projection.parameters():
            params.requires_grad = False

    def freeze_student(self):
        """Set requires_grad for student network to False"""
        for params in self.student_backbone.parameters():
            params.requires_grad = False
        for params in self.student_projection.parameters():
            params.requires_grad = False
            
    def unfreeze_student(self):
        """Set requires_grad for student network to True"""
        for params in self.student_backbone.parameters():
            params.requires_grad = True
        for params in self.student_projection.parameters():
            params.requires_grad = True

        