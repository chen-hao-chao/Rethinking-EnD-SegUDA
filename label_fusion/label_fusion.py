import argparse
import numpy as np
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image, ImageFile
from torch.utils import data, model_zoo
import yaml
import time
import os
from dataset.cityscapes_dataset_extract import cityscapesExtraction
from utils.fusion_methods import PLF_aggregation, certainty_aggregation, majority_aggregation

#### Model Settings  ####
SOURCE_DOMAIN = 'gta5'
TARGET_DOMAIN = 'cityscapes'
IGNORE_LABEL = 255
NUM_CLASSES = 19
BATCH_SIZE = 2
GPU = 3
INPUT_SIZE_TARGET = '2048,1024'
GT_SIZE = '2048,1024'
FUSION_MODE = 'PLF'
THRESHOLD = 90

####  Path Settings  ####
DATA_DIRECTORY_TARGET = '../../Warehouse/Cityscapes/data/'
DATA_LIST_PATH_TARGET = './dataset/cityscapes_list/train.txt'
RESULT_DIR = '../../Warehouse/Cityscapes/data/gtFine/PLF'

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)

def get_arguments():
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    #### Model Settings  ####
    parser.add_argument("--fusion-mode", type=str, default=FUSION_MODE, choices=['majority', 'certainty', 'PLF'],
                        help="available options : majority, certainty, PLF")
    parser.add_argument("--source-domain", type=str, default=SOURCE_DOMAIN, choices=['gta5', 'synthia'],
                        help="available options : gta5, synthia")
    parser.add_argument("--target-domain", type=str, default=TARGET_DOMAIN, choices=['cityscapes'],
                        help="available options : cityscapes")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--gpu-id", type=int, default=GPU,
                        help = 'choose gpus')
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="choose gpu device.")
    parser.add_argument("--threshold", type=int, default=THRESHOLD,
                        help="the threshold to filter some noises out.")
    parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--gt-size", type=str, default=GT_SIZE,
                        help="Comma-separated string with height and width of gt images.")
    
    ####  Path Settings  ####
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                        help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--result-dir", type=str, default=RESULT_DIR,
                        help="Path to the results. (prediction)")

    return parser.parse_args()

def main():
    print("Testing...")
    
    # args parsing
    args = get_arguments()
    w, h = map(int, args.input_size_target.split(','))
    args.input_size_target = (w, h)
    crop_size = (h, w)
    w, h = map(int, args.gt_size.split(','))
    args.gt_size = (w, h)
    
    # create result dir
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # Target loader
    total = 2975
    if args.source_domain == 'synthia':
        targetloader_MRKLD = data.DataLoader(
            cityscapesExtraction(args.data_dir_target, args.data_list_target, set='CRST_PL_SYNTHIA',
                    max_iters=total/args.batch_size),
        batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)

        targetloader_Mobile = data.DataLoader(
            cityscapesExtraction(args.data_dir_target, args.data_list_target, set='Mobile_PL_SYNTHIA', 
                    max_iters=total/args.batch_size),
        batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)

        targetloader_MRNet = data.DataLoader(
            cityscapesExtraction(args.data_dir_target, args.data_list_target, set='MRNet_PL_SYNTHIA', 
                    max_iters=total/args.batch_size),
        batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)
    
    elif args.source_domain == 'gta5':
        targetloader_Seg_Uncertainty = data.DataLoader(
            cityscapesExtraction(args.data_dir_target, args.data_list_target, set='Seg_Uncertainty',
                    max_iters=total/args.batch_size),
        batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)

        targetloader_DACS = data.DataLoader(
            cityscapesExtraction(args.data_dir_target, args.data_list_target, set='DACS', 
                    max_iters=total/args.batch_size),
        batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)

        targetloader_MRKLD = data.DataLoader(
            cityscapesExtraction(args.data_dir_target, args.data_list_target, set='MRKLD', 
                    max_iters=total/args.batch_size),
        batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)

        targetloader_CBST = data.DataLoader(
            cityscapesExtraction(args.data_dir_target, args.data_list_target, set='CBST', 
                    max_iters=total/args.batch_size),
        batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)
    
    sm = torch.nn.Softmax(dim = 1)
    log_sm = torch.nn.LogSoftmax(dim = 1)
    kl_distance = nn.KLDivLoss( reduction = 'none')
    interp = nn.Upsample(size=crop_size, mode='bilinear', align_corners=True)

    if args.source_domain == 'synthia':
        # evaluate
        for index, img_data in enumerate(zip(targetloader_MRKLD, targetloader_Mobile, targetloader_MRNet) ):
            print(str(index*args.batch_size)+" / 2975")

            batch_0, batch_1, batch_2 = img_data
            label_0, label_arr_0, name_0 = batch_0
            label_1, label_arr_1, name_1 = batch_1
            label_2, label_arr_2, name_2 = batch_2

            label_0 = label_0.cuda(args.gpu_id)
            label_1 = label_1.cuda(args.gpu_id)
            label_2 = label_2.cuda(args.gpu_id)
            
            label_arr_0 = label_arr_0.cuda(args.gpu_id)
            label_arr_1 = label_arr_1.cuda(args.gpu_id)
            label_arr_2 = label_arr_2.cuda(args.gpu_id)
            
            ### ----------------
            # aggregation
            # [index, method]
            # (0:MRKLD) (1:Mobile) (2:MRNet)
            extraction_list = [
                [4,0], 
                [3,0],
                [11,2],
                [14,2],
                [6,2],
                [7,0], 
                [5,2], 
                [13,2],
                [1,2],
                [15,2],
                [10,2],
                [8,2],
                [2,2],
                [9,1],
                [12,2],
                [0,2]
            ]
            label_list = [label_0, label_1, label_2]
            certainty_list = [label_arr_0, label_arr_1, label_arr_2]

            num_classes = 16
            num_teachers = 3
            with torch.no_grad():
                if args.fusion_mode == 'PLF':
                    pseudo_label = PLF_aggregation(num_classes, num_teachers, extraction_list, label_list, certainty_list, args.threshold, args.gpu_id)
                if args.fusion_mode == 'majority':
                    pseudo_label = majority_aggregation(num_classes, num_teachers, label_list, certainty_list, args.threshold, args.gpu_id)
                if args.fusion_mode == 'certainty':
                    pseudo_label = certainty_aggregation(num_classes, num_teachers, label_list, certainty_list, args.threshold, args.gpu_id)

                pseudo_label = pseudo_label.cpu().data.numpy()
            
            pseudo_label = np.asarray(pseudo_label, dtype=np.uint8)

            for i in range(args.batch_size):
                name_0[i] = '%s/%s' % (args.result_dir, name_0[i])
                
                output_col = colorize_mask(pseudo_label[i,:,:])
                output_col.save('%s_color.png' % (name_0[i]))

                output = Image.fromarray(pseudo_label[i,:,:])
                output.save('%s.png' % (name_0[i]))
                
            del pseudo_label, output_col, output

    elif args.source_domain == 'gta5':
        # evaluate
        for index, img_data in enumerate(zip(targetloader_Seg_Uncertainty, targetloader_DACS, targetloader_MRKLD, targetloader_CBST) ):
            print(str(index*args.batch_size)+" / 2975")

            batch_0, batch_1, batch_2, batch_3 = img_data
            label_0, label_arr_0, name_0 = batch_0
            label_1, label_arr_1, name_1 = batch_1
            label_2, label_arr_2, name_2 = batch_2
            label_3, label_arr_3, name_3 = batch_3

            label_0 = label_0.cuda(args.gpu_id)
            label_1 = label_1.cuda(args.gpu_id)
            label_2 = label_2.cuda(args.gpu_id)
            label_3 = label_3.cuda(args.gpu_id)
            
            label_arr_0 = label_arr_0.cuda(args.gpu_id)
            label_arr_1 = label_arr_1.cuda(args.gpu_id)
            label_arr_2 = label_arr_2.cuda(args.gpu_id)
            label_arr_3 = label_arr_3.cuda(args.gpu_id)
            
            ### ----------------
            # aggregation
            # [index, method]
            # (0:MRNet) (1:DACS) (2:MRKLD) (3:CBST)
            
            extraction_list = [
                [16,3],
                [3,0],
                [12,1],
                [4,1],
                [9,1],
                [5,2],
                [17,2],
                [6,0],
                [18,1],
                [7,1],
                [15,0],
                [14,1],
                [1,1],
                [11,1],
                [8,1],
                [2,1],
                [10,1],
                [13,1],
                [0,1]
            ]

            label_list = [label_0, label_1, label_2, label_3]
            certainty_list = [label_arr_0, label_arr_1, label_arr_2, label_arr_3]
            
            num_classes = 19
            num_teachers = 4
            
            with torch.no_grad():
                if args.fusion_mode == 'PLF':
                    pseudo_label = PLF_aggregation(num_classes, num_teachers, extraction_list, label_list, certainty_list, args.threshold, args.gpu_id)
                if args.fusion_mode == 'certainty':
                    pseudo_label = certainty_aggregation(num_classes, num_teachers, label_list, certainty_list, args.threshold, args.gpu_id)
                if args.fusion_mode == 'majority':
                    pseudo_label = majority_aggregation(num_classes, num_teachers, label_list, certainty_list, args.gpu_id)

                pseudo_label = pseudo_label.cpu().data.numpy()
            
            pseudo_label = np.asarray(pseudo_label, dtype=np.uint8)

            for i in range(args.batch_size):
                name_0[i] = '%s/%s' % (args.result_dir, name_0[i])
                
                output_col = colorize_mask(pseudo_label[i,:,:])
                output_col.save('%s_color.png' % (name_0[i]))

                output = Image.fromarray(pseudo_label[i,:,:])
                output.save('%s.png' % (name_0[i]))
                
            del pseudo_label, output_col, output

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

if __name__ == '__main__':
    main()