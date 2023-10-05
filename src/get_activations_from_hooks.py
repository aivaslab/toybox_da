"""Module to run and extract activations using hooks"""
import torch
import torch.utils.data as torchdata
import torchvision.transforms as transforms

import argparse
import os
import numpy as np

import datasets
import networks_hooks


def get_parser():
    """Returns parser for the run"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir-path", type=str, required=True, help="Path for the saved model")
    return vars(parser.parse_args())


def get_activations_using_hook(args):
    """Method to retrieve activations for the specified model using hooks"""
    fpath = args['dir_path'] + "final_model.pt"
    assert os.path.isfile(fpath)
    network_data = torch.load(fpath)
    backbone_weights = network_data['backbone']
    network = networks_hooks.ResNet18BackboneWithConvActivations(weights=backbone_weights)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=datasets.TOYBOX_MEAN, std=datasets.TOYBOX_STD)
    ])
    rng = np.random.default_rng(0)
    for cl in datasets.TOYBOX_CLASSES:
        dataset = datasets.ToyboxDatasetClass(rng=rng, cl=cl, transform=transform, num_images_per_class=1500)
        dataloader = torchdata.DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4, drop_last=False)
        network.cuda()
        
        for idxs, images, labels in dataloader:
            images = images.cuda()
            with torch.no_grad():
                network(images)
                
        for k in networks_hooks.all_activations.keys():
            activations = networks_hooks.all_activations[k]
            cat_activations = torch.cat(activations, dim=0)
            print(k, cat_activations.shape)
    


def test_hooks():
    """Code to test modules"""
    net = networks_hooks.ResNet18BackboneWithConvActivations()
    x = torch.rand(size=(3, 3, 224, 224))
    y = net.forward(x)
    for k in networks_hooks.all_activations.keys():
        print(k, len(networks_hooks.all_activations[k]))
    
    x = torch.rand(size=(3, 3, 224, 224))
    y = net.forward(x)
    y = net.forward(x)
    y = net.forward(x)
    
    for k in networks_hooks.all_activations.keys():
        print(k, len(networks_hooks.all_activations[k]))
        
        
def main():
    """Main method"""
    exp_args = get_parser()
    get_activations_using_hook(args=exp_args)


if __name__ == "__main__":
    main()
    