"""Use provided model to get the activations from provided dataset"""
import numpy as np
import os

import torch
import torch.utils.data as torchdata
import torchvision.transforms as transforms

import datasets
import networks


def get_activations(net, fc_size, data):
    """Use the data to get activation"""
    len_train_data = len(data)
    data_loader = torchdata.DataLoader(data, batch_size=512, shuffle=False, num_workers=4)

    activations = torch.zeros(len_train_data, fc_size)

    for (_, idx), images, _ in data_loader:
        images = images.cuda()
        with torch.no_grad():
            feats = net.forward(images)
            feats = feats.cpu()
        activations[idx] = feats
    # all_activations = torch.cat([activations, mean_activations], dim=0)
    return activations.numpy()


def get_activations_sup(model_path, out_path):
    """Get the activations from a supervised model"""
    model_path = model_path
    os.makedirs(out_path, exist_ok=True)

    load_file = torch.load(model_path)
    net = networks.ResNet18Backbone(weights=load_file['backbone'], pretrained=False)
    fc_size = net.fc_size
    net.cuda()
    net.set_eval()

    transform_toybox = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),
                                           transforms.Resize((224, 224)),
                                           transforms.Normalize(mean=datasets.TOYBOX_MEAN, std=datasets.TOYBOX_STD)])

    toybox_train_data = datasets.ToyboxDatasetAll(train=True, transform=transform_toybox, hypertune=True)
    print(len(toybox_train_data))
    activations = get_activations(net=net, fc_size=fc_size, data=toybox_train_data)
    print(activations.shape)
    np.save(file=out_path+"toybox_train_activations.npy", arr=activations)

    toybox_test_data = datasets.ToyboxDatasetAll(train=False, transform=transform_toybox, hypertune=False)
    print(len(toybox_test_data))
    activations = get_activations(net=net, fc_size=fc_size, data=toybox_test_data)
    print(activations.shape)
    np.save(file=out_path+"toybox_test_activations.npy", arr=activations)

    transform_in12 = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Resize((224, 224)),
                                         transforms.Normalize(mean=datasets.IN12_MEAN, std=datasets.IN12_STD)])
    in12_train_data = datasets.DatasetIN12All(train=True, transform=transform_in12, hypertune=True)
    print(len(in12_train_data))
    activations = get_activations(net=net, fc_size=fc_size, data=in12_train_data)
    print(activations.shape)
    np.save(file=out_path+"in12_train_activations.npy", arr=activations)

    in12_test_data = datasets.DatasetIN12All(train=False, transform=transform_in12, hypertune=False)
    print(len(in12_test_data))
    activations = get_activations(net=net, fc_size=fc_size, data=in12_test_data)
    print(activations.shape)
    np.save(file=out_path+"in12_test_activations.npy", arr=activations)


if __name__ == "__main__":
    get_activations_sup(model_path="../temp/final_model_src_pre.pt", out_path="../temp/activations/")
    