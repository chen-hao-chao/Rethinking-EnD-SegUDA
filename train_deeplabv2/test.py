import argparse
import numpy as np
import sys
import torch
import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn
import yaml
import time
import os

from modeling.deeplabv2 import Deeplabv2
from utils.evaluate import evaluate


#### Model Settings  ####
SOURCE_DOMAIN = 'gta5'
TARGET_DOMAIN = 'cityscapes'
MODEL = 'Deeplab'
BACKBONE = 'resnet'
IGNORE_LABEL = 255
NUM_CLASSES = 19
BATCH_SIZE = 20
RESTORE_FROM = './weights/model.pth'
GPU = 0
INPUT_SIZE = '1024,512'
GT_SIZE = '2048,1024'

####  Path Settings  ####
DATA_DIRECTORY = '../../Warehouse/Cityscapes/data'
DATA_LIST_PATH = './dataset/cityscapes_list/val.txt'
GT_DIR = '../../Warehouse/Cityscapes/data/gtFine/val'
GT_LIST_PATH = './dataset/cityscapes_list'
RESULT_DIR = './result/test'

def get_arguments():
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    #### Model Settings  ####
    parser.add_argument("--model", type=str, default=MODEL, choices=['Deeplab'],
                        help="Model Choice Deeplab.")
    parser.add_argument("--source-domain", type=str, default=SOURCE_DOMAIN, choices=['gta5', 'synthia'],
                        help="available options : gta5, synthia")
    parser.add_argument("--target-domain", type=str, default=TARGET_DOMAIN, choices=['cityscapes'],
                        help="available options : cityscapes")
    parser.add_argument("--backbone", type=str, default=BACKBONE, choices=['resnet', 'mobilenet', 'drn'],
                        help="available options : resnet, mobilenet, drn")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu-id", type=int, default=GPU,
                        help = 'choose gpus')
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="choose gpu device.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--gt-size", type=str, default=GT_SIZE,
                        help="Comma-separated string with height and width of gt images.")
    
    ####  Path Settings  ####
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--gt-dir", type=str, default=GT_DIR,
                        help="Path to the directory containing the target ground truth dataset.")
    parser.add_argument("--gt-list", type=str, default=GT_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--result-dir", type=str, default=RESULT_DIR,
                        help="Path to the results. (prediction)")

    return parser.parse_args()

def main():
    print("Testing...")
    
    # args parsing

    args = get_arguments()
    w, h = map(int, args.input_size.split(','))
    args.input_size = (w, h)
    w, h = map(int, args.gt_size.split(','))
    args.gt_size = (w, h)
    
    # create result dir
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # load model
    model = Deeplabv2(num_classes=args.num_classes, backbone=args.backbone)
    saved_state_dict = torch.load(args.restore_from)
    model.load_state_dict(saved_state_dict)
    model.cuda(args.gpu_id)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Number of parameters: " + str(pytorch_total_params))
    
    # evaluate
    tt = time.time()
    _, avg_time = evaluate(args, args.gt_dir, args.gt_list, args.result_dir, model)
    print('Time used: {} sec'.format(time.time()-tt))
    print(avg_time)

if __name__ == '__main__':
    main()