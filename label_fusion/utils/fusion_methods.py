import numpy as np
import torch
import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn

def certainty_aggregation(num_classes, num_teachers, label_list, certainty_list, threshold, gpu_id):
    ###
    # img_size: (B, H, W) -> (4, 1024, 2048) for GTA5 to Cityecapes
    # tensor_size: (B, H, W, C) -> (4, 1024, 2048, 19) for GTA5 to Cityecapes
    ###
    img_size = label_list[0].shape 
    tensor_size = certainty_list[0].shape 

    # Generate the pseudo labels
    pseudo_label = 255*torch.ones(img_size, dtype=torch.uint8).cuda(gpu_id)
    pseudo_label_mask = torch.zeros(img_size, dtype=torch.uint8).cuda(gpu_id)
    certainty_sum = torch.zeros(tensor_size).cuda(gpu_id)

    # Sum up the certainty tensors
    for l in range(num_teachers):
        certainty_sum = certainty_sum + certainty_list[l]
    certainty_sum = certainty_sum / num_teachers

    # mask those pixels with certainty < threshold
    pseudo_label_value, pseudo_label_idx = torch.max(certainty_sum, dim=3)
    pseudo_label_idx.type(torch.uint8)

    # Take the threshold
    pseudo_label_mask[pseudo_label_value >= threshold] = 1
    pseudo_label = pseudo_label_idx*pseudo_label_mask + pseudo_label*(1-pseudo_label_mask)

    del pseudo_label_mask, certainty_sum
    
    return pseudo_label

def majority_aggregation(num_classes, num_teachers, label_list, certainty_list, gpu_id):
    ###
    # img_size: (B, H, W) -> (4, 1024, 2048) for GTA5 to Cityecapes
    # tensor_size: (B, H, W, C) -> (4, 1024, 2048, 19) for GTA5 to Cityecapes
    ###
    img_size = label_list[0].shape 
    tensor_size = certainty_list[0].shape 
    
    pseudo_label_count = torch.zeros(tensor_size).cuda(gpu_id)
    for i in range(num_classes):
        for l in range(num_teachers):
            # Build the one-hot tensor 
            class_map = torch.zeros(img_size).cuda(gpu_id)
            class_map[label_list[l] == i] = 1
            # Sum up the votes
            pseudo_label_count[:,:,:,i] += class_map
    
    pseudo_label_value, pseudo_label_idx = torch.max(pseudo_label_count, dim=3)
    return pseudo_label_idx


def PLF_aggregation(num_classes, num_teachers, extraction_list, label_list, certainty_list, threshold, gpu_id, RF=(5, 5)):
    ###
    # img_size: (B, H, W) -> (4, 1024, 2048) for GTA5 to Cityecapes
    # tensor_size: (B, H, W, C) -> (4, 1024, 2048, 19) for GTA5 to Cityecapes
    ###
    img_size = label_list[0].shape 
    tensor_size = certainty_list[0].shape 

    # Generate the pseudo labels
    pseudo_label = torch.zeros(img_size, dtype=torch.uint8).cuda(gpu_id)
    pseudo_label_cubic = torch.zeros((img_size[0],num_classes,img_size[1],img_size[2])).cuda(gpu_id)
    pseudo_label_cubic_certainty = torch.zeros((img_size[0],num_classes,img_size[1],img_size[2])).cuda(gpu_id)

    count_map = torch.zeros(img_size, dtype=torch.uint8).cuda(gpu_id)
    
    # Extract the tensor
    for p in range(num_classes):
        # Extract pseudo label
        i, m = extraction_list[p][0], extraction_list[p][1]
        pseudo_label[label_list[m] == i] = i
        count_map[label_list[m] == i] += 1

        # Build the cubic
        class_map = torch.zeros(img_size).cuda(gpu_id)
        class_map[label_list[m] == i] = 1
        pseudo_label_cubic[:,i,:,:] = class_map

        # Build certainty cubic
        pseudo_label_cubic_certainty[:,i,:,:] = certainty_list[m][:,:,:,i]
    
    # Generate the mask M
    M = torch.zeros(img_size, dtype=torch.uint8).cuda(gpu_id)
    M[count_map == 1] = 1

    # Generate the tensors for disagreement
    avgPooling = nn.AvgPool2d(RF, stride=(1, 1), padding=(int(RF[0]/2), int(RF[1]/2))).cuda(gpu_id)
    # Pseudo label
    pseudo_label_cubic = avgPooling(pseudo_label_cubic)
    _, pseudo_label_disagree = torch.max(pseudo_label_cubic, dim=1)
    pseudo_label_disagree = pseudo_label_disagree.type(torch.ByteTensor).cuda(gpu_id)
    pseudo_label = pseudo_label*M + pseudo_label_disagree*(1-M)
    # Certainty tensor
    certainty_disagree = avgPooling(pseudo_label_cubic_certainty)
    print(certainty_disagree.shape)
    print(pseudo_label.unsqueeze(0).type(torch.LongTensor).cuda(gpu_id).shape)
    certainty_disagree = torch.gather(certainty_disagree, 1, pseudo_label.unsqueeze(1).type(torch.LongTensor).cuda(gpu_id)).squeeze(1)
    certainty_agree = torch.gather(pseudo_label_cubic_certainty, 1, pseudo_label.unsqueeze(1).type(torch.LongTensor).cuda(gpu_id)).squeeze(1)
    certainty_tensor = certainty_agree*M + certainty_disagree*(1-M)

    mask_threshold = torch.zeros(img_size, dtype=torch.uint8).cuda(gpu_id)
    mask_threshold[certainty_tensor >= threshold] = 1

    unlabeled = 255*torch.ones(img_size, dtype=torch.uint8).cuda(gpu_id)
    pseudo_label = pseudo_label*mask_threshold + unlabeled*(1-mask_threshold)

    return pseudo_label
