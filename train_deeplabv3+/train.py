import argparse
import torch
import torch.nn as nn
import numpy as np
import pickle
from torch.autograd import Variable
import torch.optim as optim
from torch.utils import data, model_zoo
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import sys
import os
import os.path as osp
import random
import time
import yaml
from tensorboardX import SummaryWriter

from trainer import AD_Trainer
from utils.tool import adjust_learning_rate, Timer 
from utils.evaluate import evaluate
from dataset.cityscapes_dataset_pl import cityscapesDataSet
from dataset.gta5_dataset import gta5DataSet
from dataset.synthia_dataset import synthiaDataSet

#### Hyperparameters ####
WEIGHT_DECAY = 0.0005
LEARNING_RATE = 1e-4
MOMENTUM = 0.9
MAX_VALUE = 7
POWER = 0.9
HARD_LABEL = 80

#### Model Settings  ####
SOURCE_DOMAIN = 'gta5'
TARGET_DOMAIN = 'cityscapes'
MODEL = 'DeepLab'
BACKBONE = 'drn'
BATCH_SIZE = 10
NUM_WORKERS = 1
IGNORE_LABEL = 255
NUM_CLASSES = 19
NUM_STEPS = 100001
SAVE_PRED_EVERY = 1000
WARM_UP = 0
INPUT_SIZE = '1024,512'
GT_SIZE = '2048,1024'
CROP_SIZE = '512,256'
SET = 'PLF_source'
NORM_STYLE = 'bn' # or in
AUTOAUG = False
AUTOAUG_TARGET = False

####  Path Settings  ####
DATA_DIRECTORY = '../../Warehouse/Cityscapes/data'
DATA_LIST_PATH = './dataset/cityscapes_list/train.txt'
SNAPSHOT_DIR = './snapshots'
LOG_DIR = './log'
RESTORE_FROM = '../../weights/weights/gta5/deeplabv3+/source/drn/model_34.80.pth'
GT_DIR = '../../Warehouse/Cityscapes/data/gtFine/val'
GT_LIST_PATH = './dataset/cityscapes_list'
RESULT_DIR = './result/gta5'

def get_arguments():
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    #### Hyperparameters ####
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--max-value", type=float, default=MAX_VALUE,
                        help="Max Value of Class Weight.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--only-hard-label",type=float, default=HARD_LABEL,  
                         help="class balance.")

    #### Model Settings  ####
    parser.add_argument("--source-domain", type=str, default=SOURCE_DOMAIN, choices=['gta5', 'synthia'],
                        help="available options : carla, gta5, synthia")
    parser.add_argument("--target-domain", type=str, default=TARGET_DOMAIN, choices=['cityscapes'],
                        help="available options : carla, cityscapes")
    parser.add_argument("--model", type=str, default=MODEL, choices=['Deeplab'],
                        help="available options : DeepLab")
    parser.add_argument("--backbone", type=str, default=BACKBONE, choices=['resnet', 'mobilenet', 'drn'],
                        help="available options : resnet, mobilenet, drn")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--warm-up", type=float, default=WARM_UP,
                        help = 'warm up iteration')
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--gt-size", type=str, default=GT_SIZE,
                        help="Comma-separated string with height and width of gt images.")
    parser.add_argument("--crop-size", type=str, default=CROP_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose adaptation set.")
    parser.add_argument("--norm-style", type=str, default=NORM_STYLE,
                        help="Norm Style in the final classifier.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--often-balance", action='store_true', help="balance the apperance times.")
    parser.add_argument("--class-balance", action='store_true', help="class balance.")
    parser.add_argument("--use-se", action='store_true', help="use se block.")
    parser.add_argument("--train_bn", action='store_true', help="train batch normalization.")
    parser.add_argument("--sync_bn", action='store_true', help="sync batch normalization.")
    parser.add_argument("--cpu", action='store_true', help="choose to use cpu device.")
    parser.add_argument("--gpu-ids", type=str, default='0', help = 'choose gpus')
    parser.add_argument("--tensorboard", action='store_false', help="choose whether to use tensorboard.")
    parser.add_argument("--autoaug_target", action='store_true', help="use augmentation or not" )
    parser.add_argument("--autoaug", action='store_true', help="use augmentation or not" )
    parser.add_argument("--fp16", action="store_true",
                        help="Use FP16.")

    ####  Path Settings  ####
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--gt-dir", type=str, default=GT_DIR,
                        help="Path to the folder containing the gt data.")
    parser.add_argument("--gt-list", type=str, default=GT_LIST_PATH,
                        help="Path to the folder includeing the list of gt.")
    parser.add_argument("--result-dir", type=str, default=RESULT_DIR,
                        help="Path to the results. (prediction)")
    parser.add_argument("--log-dir", type=str, default=LOG_DIR,
                        help="Path to the directory of log.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    
    return parser.parse_args()


args = get_arguments()

# save opts
if not os.path.exists(args.snapshot_dir):
    os.makedirs(args.snapshot_dir)

def main():
    """Create the model and start the training."""
    print("NUMBER OF CLASSES: ", str(args.num_classes))

    w, h = map(int, args.input_size.split(','))
    args.input_size = (w, h)

    w, h = map(int, args.crop_size.split(','))
    args.crop_size = (h, w)

    w, h = map(int, args.gt_size.split(','))
    args.gt_size = (w, h)

    cudnn.enabled = True
    cudnn.benchmark = True
    
    # create result dir
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    str_ids = args.gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        gid = int(str_id)
        if gid >=0:
            gpu_ids.append(gid)

    num_gpu = len(gpu_ids)
    args.multi_gpu = False
    if num_gpu>1:
        args.multi_gpu = True
        Trainer = AD_Trainer(args)
    else:
        Trainer = AD_Trainer(args)
        
    TARGET_IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
    
    trainloader = data.DataLoader(
        cityscapesDataSet(args.data_dir, args.data_list,
                max_iters=args.num_steps * args.batch_size,
                resize_size=args.input_size,
                crop_size=args.crop_size,
                set=args.set, scale=False, mirror=args.random_mirror, mean=TARGET_IMG_MEAN, autoaug = args.autoaug_target, source_domain=args.source_domain),
    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    trainloader_iter = enumerate(trainloader)

    # set up tensor board
    if args.tensorboard:
        args.log_dir += '/'+ os.path.basename(args.snapshot_dir)
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)

        writer = SummaryWriter(args.log_dir)
    
    # load mIOU (FOR EARLY STOPPING)
    best_mIoUs = 0

    for i_iter in range(args.num_steps):

        loss_seg_value = 0

        adjust_learning_rate(Trainer.gen_opt , i_iter, args)

        _, batch = trainloader_iter.__next__()
        images, labels, _, _ = batch
        images = images.cuda()
        labels = labels.long().cuda()

        with Timer("Elapsed time in update: %f"):
            loss_seg = Trainer.gen_update(images, labels, i_iter)
            loss_seg_value += loss_seg.item()

        if args.tensorboard:
            scalar_info = {
                'loss_seg': loss_seg_value
            }

            if i_iter % 100 == 0:
                for key, val in scalar_info.items():
                    writer.add_scalar(key, val, i_iter)

        print('\033[1m iter = %8d/%8d \033[0m loss_seg = %.3f' %(i_iter, args.num_steps, loss_seg_value))

        del loss_seg

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            mIoUs, _ = evaluate(args, args.gt_dir, args.gt_list, args.result_dir, Trainer.G)
            writer.add_scalar('mIOU', round(np.nanmean(mIoUs)*100, 2), int(i_iter/args.save_pred_every)) # (TB)
            if round(np.nanmean(mIoUs) * 100, 2) > best_mIoUs:
                print('save model ...')
                best_mIoUs = round(np.nanmean(mIoUs) * 100, 2)
                torch.save(Trainer.G.state_dict(), osp.join(args.snapshot_dir, 'model.pth'))

    if args.tensorboard:
        writer.close()


if __name__ == '__main__':
    main()
