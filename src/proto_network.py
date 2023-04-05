"""Module implementing the basic network that performs prototypical classification"""
import torch
import torch.nn as nn
import torch.nn.functional as func

import networks
import utils


class ResNet18Proto(nn.Module):
    """Class definition for the ResNet-18 backbone for the model"""
    
    def __init__(self, pretrained=False, backbone_weights=None, num_classes=12, bottleneck_weights=None,
                 prototypes=None, classifier_weights=None):
        super(ResNet18Proto, self).__init__()
        
        self.backbone = networks.ResNet18Backbone(pretrained=pretrained, weights=backbone_weights)
        self.backbone_fc_size = self.backbone.fc_size
        self.num_classes = num_classes
        
        self.backbone.model.avgpool = nn.Identity()
        self.resnet_out_channels = 512
        self.resnet_out_width = 7
        self.num_prototypes_per_class = 10
        self.prototype_shape = (self.num_prototypes_per_class * self.num_classes, 128, 1, 1)
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=self.resnet_out_channels, out_channels=2 * self.prototype_shape[1],
                      kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=2 * self.prototype_shape[1], out_channels=self.prototype_shape[1],
                      kernel_size=(1, 1)),
            nn.Sigmoid()
        )
        if bottleneck_weights is not None:
            self.bottleneck.load_state_dict(bottleneck_weights)
        else:
            self.bottleneck.apply(utils.weights_init)
        
        # TODO: add loading for prototypes
        self.prototypes = nn.Parameter(torch.rand(self.prototype_shape), requires_grad=True)
        if prototypes is not None:
            self.prototypes.data = prototypes

        self.prototype_class_identity = torch.zeros(self.prototype_shape[0], self.num_classes)

        num_prototypes_per_class = self.prototype_shape[0] // self.num_classes

        for j in range(self.prototype_shape[0]):
            self.prototype_class_identity[j, j // num_prototypes_per_class] = 1
        
        self.classifier = nn.Linear(self.prototype_shape[0], self.num_classes, bias=False)
        if classifier_weights is not None:
            self.classifier.load_state_dict(classifier_weights)
        else:
            self.set_classifier_incorrect_connection(incorrect_strength=-0.5)

    def get_weights_status(self):
        """Return a string with the training status of different parameters"""
        backbone_params_total = sum([p.numel() for p in self.backbone.parameters()])
        backbone_params_train = sum([p.numel() for p in self.backbone.parameters() if p.requires_grad])
        bottleneck_params_total = sum([p.numel() for p in self.bottleneck.parameters()])
        bottleneck_params_train = sum([p.numel() for p in self.bottleneck.parameters() if p.requires_grad])
        prototype_params_total = sum([p.numel() for p in self.prototypes])
        prototype_params_train = sum([p.numel() for p in self.prototypes if p.requires_grad])
        classifier_params_total = sum([p.numel() for p in self.classifier.parameters()])
        classifier_params_train = sum([p.numel() for p in self.classifier.parameters() if p.requires_grad])
        
        ret_str = "                  Backbone    Addl_layers     Proto      Last       All\n" \
                  "Train Params     {:9d} {:14d} {:9d} {:9d} {:9d}\n" \
                  "Total Params     {:9d} {:14d} {:9d} {:9d} {:9d}". \
            format(
                backbone_params_train, bottleneck_params_train, prototype_params_train, classifier_params_train,
                backbone_params_train + bottleneck_params_train + prototype_params_train + classifier_params_train,
                backbone_params_total, bottleneck_params_total, prototype_params_total, classifier_params_total,
                backbone_params_total + bottleneck_params_total + prototype_params_total + classifier_params_total
            )
        return ret_str

    def set_classifier_incorrect_connection(self, incorrect_strength):
        """the incorrect strength will be actual strength if -0.5 then input -0.5"""
    
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations
    
        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.classifier.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)

    def _l2_dist_conv(self, x):
        x2 = x ** 2
        x2 = torch.sum(x2, dim=1, keepdim=True).repeat(1, self.prototype_shape[0], 1, 1)
    
        p2 = self.prototypes ** 2
        p2 = torch.sum(p2, dim=(1, 2, 3)).view(-1, 1, 1)
    
        xp = func.conv2d(input=x, weight=self.prototypes)
        distances = func.relu(x2 - 2 * xp + p2)
    
        return distances

    def conv_forward(self, x):
        """Forward method for the resnet and 1d conv layers"""
        x = self.backbone.model.conv1.forward(x)
        x = self.backbone.model.bn1.forward(x)
        x = self.backbone.model.relu.forward(x)
        x = self.backbone.model.maxpool.forward(x)
        x = self.backbone.model.layer1.forward(x)
        x = self.backbone.model.layer2.forward(x)
        x = self.backbone.model.layer3.forward(x)
        x = self.backbone.model.layer4.forward(x)
        x = self.bottleneck.forward(x)
        return x

    def prototype_distances(self, x):
        """Get the distance to prototypes"""
        conv_features = self.conv_forward(x)
        distances = self._l2_dist_conv(conv_features)
        min_distances = -func.max_pool2d(input=-distances, kernel_size=(distances.size(2), distances.size(3)))
        min_distances = min_distances.view(-1, self.prototype_shape[0])
    
        return min_distances

    @staticmethod
    def _dist_to_similarity(distances):
        epsilon = 1e-4
        return torch.log((distances + 1) / (distances + epsilon))

    def push_forward(self, x):
        """this method is needed for the pushing operation"""
        conv_features = self.conv_forward(x)
        distances = self._l2_dist_conv(conv_features)
        return conv_features, distances

    def forward(self, x):
        """Combined forward method"""
        min_distances = self.prototype_distances(x)
    
        similarities = self._dist_to_similarity(min_distances)
        logits = self.classifier(similarities)
    
        return logits, min_distances

    def set_network_warmup(self):
        """Set the network in warmup mode"""
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.bottleneck.parameters():
            p.requires_grad = True
        self.prototypes.requires_grad = True
        for p in self.classifier.parameters():
            p.requires_grad = True

    def set_network_joint(self):
        """Set the network in warmup mode"""
        for p in self.backbone.parameters():
            p.requires_grad = True
        for p in self.bottleneck.parameters():
            p.requires_grad = True
        self.prototypes.requires_grad = True
        for p in self.classifier.parameters():
            p.requires_grad = True

    def set_network_last(self):
        """Set the network in warmup mode"""
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.bottleneck.parameters():
            p.requires_grad = False
        self.prototypes.requires_grad = False
        for p in self.classifier.parameters():
            p.requires_grad = True

    def set_network_eval(self):
        """Set the network in warmup mode"""
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.bottleneck.parameters():
            p.requires_grad = False
        self.prototypes.requires_grad = False
        for p in self.classifier.parameters():
            p.requires_grad = False
