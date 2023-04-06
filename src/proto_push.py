"""
Module implementing the push method for ProtoPNet
"""
import torch
import numpy as np
import time
import os


def push_prototypes(dataloader,  # pytorch dataloader (must be unnormalized in [0,1])
                    network,  # pytorch network with prototype_vectors
                    transform,
                    out_dir=None,
                    epochs=None
                    ):
    """Push method"""
    start_time = time.time()
    
    prototype_shape = network.prototype_shape
    n_prototypes = network.prototype_shape[0]
    # saves the closest distance seen so far
    
    global_min_proto_dist = np.full(n_prototypes, np.inf)
    # saves the patch representation that gives the current smallest distance
    global_min_fmap_patches = np.zeros([prototype_shape[0], prototype_shape[1], prototype_shape[2], prototype_shape[3]])
    
    num_classes = network.num_classes
    
    proto_rf_boxes = np.full(shape=[n_prototypes, 6], fill_value=-1)
    proto_bound_boxes = np.full(shape=[n_prototypes, 6], fill_value=-1)
    
    proto_out_dir = out_dir
    if out_dir is not None and epochs is not None:
        proto_out_dir = out_dir + 'epoch-' + str(epochs) + "/"
        os.makedirs(proto_out_dir, exist_ok=False)
    
    for push_iter, (idxs, search_batch_input, search_y) in enumerate(dataloader):
        '''
        start_index_of_search keeps track of the index of the image
        assigned to serve as prototype
        '''
        update_prototypes_on_batch(search_batch_input,
                                   transform,
                                   network,
                                   global_min_proto_dist,
                                   global_min_fmap_patches,
                                   search_y=search_y,
                                   num_classes=num_classes,
                                   out_dir=proto_out_dir)
    
    if proto_out_dir is not None:
        np.save(os.path.join(proto_out_dir, "bb" + '-receptive_field' + str(epochs) + '.npy'),
                proto_rf_boxes)
        np.save(os.path.join(proto_out_dir, "bb" + str(epochs) + '.npy'),
                proto_bound_boxes)
    
    prototype_update = np.reshape(global_min_fmap_patches, tuple(prototype_shape))
    network.prototypes.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).cuda())
    end_time = time.time()
    print("Time taken for pushing: {:2f}".format(end_time - start_time))


# update each prototype for current search batch
def update_prototypes_on_batch(search_batch_input,
                               transform,
                               network,
                               global_min_proto_dist,  # this will be updated
                               global_min_fmap_patches,  # this will be updated
                               search_y=None,  # required if class_specific == True
                               num_classes=None,
                               out_dir=None
                               ):
    """Update the prototypes using samples from current batch"""
    network.eval()
    search_batch = transform(search_batch_input)
    
    with torch.no_grad():
        search_batch = search_batch.cuda()
        # this computation currently is not parallelized
        protoL_input_torch, proto_dist_torch = network.push_forward(search_batch)
    
    protoL_input_ = np.copy(protoL_input_torch.detach().cpu().numpy())
    proto_dist_ = np.copy(proto_dist_torch.detach().cpu().numpy())
    
    del protoL_input_torch, proto_dist_torch
    
    class_to_img_index_dict = {key: [] for key in range(num_classes)}
    # img_y is the image's integer label
    for img_index, img_y in enumerate(search_y):
        img_label = img_y.item()
        class_to_img_index_dict[img_label].append(img_index)
    
    prototype_shape = network.prototype_shape
    n_prototypes = prototype_shape[0]
    proto_h = prototype_shape[2]
    proto_w = prototype_shape[3]
    
    for j in range(n_prototypes):
        # if n_prototypes_per_class != None:
        # target_class is the class of the class_specific prototype
        target_class = torch.argmax(network.prototype_class_identity[j]).item()
        # if there is no images of the target_class from this batch
        # we go on to the next prototype
        if len(class_to_img_index_dict[target_class]) == 0:
            continue
        proto_dist_j = proto_dist_[class_to_img_index_dict[target_class]][:, j, :, :]
        
        batch_min_proto_dist_j = np.amin(proto_dist_j)
        if batch_min_proto_dist_j < global_min_proto_dist[j]:
            batch_argmin_proto_dist_j = list(np.unravel_index(np.argmin(proto_dist_j, axis=None), proto_dist_j.shape))
            
            '''
            change the argmin index from the index among
            images of the target class to the index in the entire search
            batch
            '''
            batch_argmin_proto_dist_j[0] = class_to_img_index_dict[target_class][batch_argmin_proto_dist_j[0]]
            
            # retrieve the corresponding feature map patch
            img_index_in_batch = batch_argmin_proto_dist_j[0]
            fmap_height_start_index = batch_argmin_proto_dist_j[1]
            fmap_height_end_index = fmap_height_start_index + proto_h
            fmap_width_start_index = batch_argmin_proto_dist_j[2]
            fmap_width_end_index = fmap_width_start_index + proto_w
            
            batch_min_fmap_patch_j = protoL_input_[img_index_in_batch, :,
                                                   fmap_height_start_index:fmap_height_end_index,
                                                   fmap_width_start_index:fmap_width_end_index]
            
            global_min_proto_dist[j] = batch_min_proto_dist_j
            global_min_fmap_patches[j] = batch_min_fmap_patch_j
            
            if out_dir is None:
                continue
    
    del class_to_img_index_dict
