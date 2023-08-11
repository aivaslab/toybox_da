"""Use provided model to calculate the extract all activations from resnet"""
import numpy as np
import os
import argparse

import torch
import torch.utils.data as torchdata
import torchvision.transforms as transforms

import datasets
import networks


def get_all_activations(net, data):
    """Use the data to get activations from net for all layers of resnet"""
    len_train_data = len(data)
    data_loader = torchdata.DataLoader(data, batch_size=512, shuffle=False, num_workers=4)
    fc_sizes = net.FC_SIZES
    
    layer0_activations = torch.zeros((len_train_data, fc_sizes['conv1']), dtype=torch.float)
    layer1_activations = torch.zeros((len_train_data, fc_sizes['layer1']), dtype=torch.float)
    layer2_activations = torch.zeros((len_train_data, fc_sizes['layer2']), dtype=torch.float)
    layer3_activations = torch.zeros((len_train_data, fc_sizes['layer3']), dtype=torch.float)
    layer4_activations = torch.zeros((len_train_data, fc_sizes['layer4']), dtype=torch.float)
    avgpool_activations = torch.zeros((len_train_data, fc_sizes['avgpool']), dtype=torch.float)
    
    indices = torch.zeros(len_train_data, dtype=torch.long)
    labels = torch.zeros(len_train_data, dtype=torch.long)
    for (idx, act_idx), images, lbls in data_loader:
        images = images.cuda()
        with torch.no_grad():
            l0_feats, l1_feats, l2_feats, l3_feats, l4_feats, avgpool_feats = net.forward(images)
            l0_feats, l1_feats, l2_feats, l3_feats, l4_feats, avgpool_feats = \
                l0_feats.cpu(), l1_feats.cpu(), l2_feats.cpu(), l3_feats.cpu(), l4_feats.cpu(), avgpool_feats.cpu()
        indices[idx] = act_idx
        labels[idx] = lbls
        layer0_activations[idx] = l0_feats
        layer1_activations[idx] = l1_feats
        layer2_activations[idx] = l2_feats
        layer3_activations[idx] = l3_feats
        layer4_activations[idx] = l4_feats
        avgpool_activations[idx] = avgpool_feats
    # all_activations = torch.cat([activations, mean_activations], dim=0)
    return indices.numpy(), labels.numpy(), layer0_activations.numpy(), layer1_activations.numpy(), \
        layer2_activations.numpy(), layer3_activations.numpy(), \
        layer4_activations.numpy(), avgpool_activations.numpy()


def get_all_activations_sup(model_path, out_path):
    """Get the activations from a supervised model"""
    os.makedirs(out_path, exist_ok=True)
    assert model_path is not None
    load_file = torch.load(model_path)
    net = networks.ResNet18BackboneWithActivations(weights=load_file['backbone'], pretrained=False)
    # net = networks.ResNet18BackboneWithActivations(pretrained=True)
    # fc_size = net.fc_size
    net.cuda()
    net.set_eval()
    
    transform_toybox = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),
                                           transforms.Resize((224, 224)),
                                           transforms.Normalize(mean=datasets.TOYBOX_MEAN, std=datasets.TOYBOX_STD)
                                           ])
    
    toybox_train_data = datasets.ToyboxDataset(rng=np.random.default_rng(0), train=True, transform=transform_toybox,
                                               hypertune=True, num_instances=-1, num_images_per_class=1500)
    print(len(toybox_train_data))
    tb_idxs, tb_lbls, tb_l0_f, tb_l1_f, \
        tb_l2_f, tb_l3_f, tb_l4_f, tb_avg_f = get_all_activations(net=net, data=toybox_train_data)
    print(tb_idxs.shape, tb_lbls.shape, tb_l0_f.shape, tb_l1_f.shape, tb_l2_f.shape, tb_l3_f.shape, tb_l4_f.shape,
          tb_avg_f.shape)
    
    np.save(file=out_path + "tb_tr_l0_f.npy", arr=tb_l0_f)
    np.save(file=out_path + "tb_tr_l1_f.npy", arr=tb_l1_f)
    np.save(file=out_path + "tb_tr_l2_f.npy", arr=tb_l2_f)
    np.save(file=out_path + "tb_tr_l3_f.npy", arr=tb_l3_f)
    np.save(file=out_path + "tb_tr_l4_f.npy", arr=tb_l4_f)
    np.save(file=out_path + "tb_tr_avg_f.npy", arr=tb_avg_f)
    np.save(file=out_path + "tb_tr_idxs.npy", arr=tb_idxs)
    np.save(file=out_path + "tb_tr_labels.npy", arr=tb_lbls)

    del tb_l0_f, tb_l1_f, tb_l2_f, tb_l3_f, tb_l4_f, tb_avg_f, toybox_train_data, tb_idxs, tb_lbls
    import time
    time.sleep(5)
    # toybox_test_data = datasets.ToyboxDataset(rng=np.random.default_rng(0), train=False, transform=transform_toybox,
    #                                           hypertune=False)
    # print(len(toybox_test_data))
    # indices, activations = get_activations(net=net, fc_size=fc_size, data=toybox_test_data)
    # print(indices.shape, activations.shape)
    # np.save(file=out_path + "toybox_test_activations.npy", arr=activations)
    # np.save(file=out_path + "toybox_test_indices.npy", arr=indices)
    #
    transform_in12 = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),
                                         transforms.Resize((224, 224)),
                                         transforms.Normalize(mean=datasets.IN12_MEAN, std=datasets.IN12_STD)])
    in12_train_data = datasets.DatasetIN12All(train=True, transform=transform_in12, hypertune=True)
    print(len(in12_train_data))
    in12_idxs, in12_lbls, in12_l0_f, in12_l1_f, \
        in12_l2_f, in12_l3_f, in12_l4_f, in12_avg_f = get_all_activations(net=net, data=in12_train_data)
    print(in12_idxs.shape, in12_lbls.shape, in12_l0_f.shape, in12_l1_f.shape, in12_l2_f.shape, in12_l3_f.shape,
          in12_l4_f.shape, in12_avg_f.shape)

    np.save(file=out_path + "in12_tr_l0_f.npy", arr=in12_l0_f)
    np.save(file=out_path + "in12_tr_l1_f.npy", arr=in12_l1_f)
    np.save(file=out_path + "in12_tr_l2_f.npy", arr=in12_l2_f)
    np.save(file=out_path + "in12_tr_l3_f.npy", arr=in12_l3_f)
    np.save(file=out_path + "in12_tr_l4_f.npy", arr=in12_l4_f)
    np.save(file=out_path + "in12_tr_avg_f.npy", arr=in12_avg_f)
    np.save(file=out_path + "in12_tr_idxs.npy", arr=in12_idxs)
    np.save(file=out_path + "in12_tr_labels.npy", arr=in12_lbls)

    del in12_l0_f, in12_l1_f, in12_l2_f, in12_l3_f, in12_l4_f, in12_avg_f, in12_train_data, in12_idxs, in12_lbls
    # indices, activations = get_activations(net=net, fc_size=fc_size, data=in12_train_data)
    # print(indices.shape, activations.shape)
    # np.save(file=out_path + "in12_train_activations.npy", arr=activations)
    # np.save(file=out_path + "in12_train_indices.npy", arr=indices)
    #
    # in12_test_data = datasets.DatasetIN12All(train=False, transform=transform_in12, hypertune=False)
    # print(len(in12_test_data))
    # indices, activations = get_activations(net=net, fc_size=fc_size, data=in12_test_data)
    # print(indices.shape, activations.shape)
    # np.save(file=out_path + "in12_test_indices.npy", arr=indices)
    # np.save(file=out_path + "in12_test_activations.npy", arr=activations)


def get_args():
    """Parser with arguments"""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dir-path", "-d", type=str, required=True, help="Path where model is stored.")
    return vars(parser.parse_args())


def test_network():
    """Code to test network output"""
    x = torch.rand(size=(64, 3, 224, 224))
    net = networks.ResNet18BackboneWithActivations(pretrained=True)
    with torch.no_grad():
        feats = net.forward(x)
    for f in feats:
        print(f.shape)


if __name__ == "__main__":
    # test_network()
    # get_activations_sup(model_path="", out_path="")
    args = get_args()
    dir_name = args['dir_path']
    assert os.path.isdir(dir_name)
    get_all_activations_sup(model_path=dir_name + "final_model.pt", out_path=dir_name + "all_activations/")
