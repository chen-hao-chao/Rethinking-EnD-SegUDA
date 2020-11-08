import numpy as np
import argparse
import json
from PIL import Image
from os.path import join
####
import scipy
from scipy import ndimage
import numpy as np
from multiprocessing import Pool
from PIL import Image, ImageFile
import torch
import torch.nn as nn #
from torch.utils import data, model_zoo
from os.path import join

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]

zero_pad = 256 * 3 - len(palette)

for i in range(zero_pad):
    palette.append(0)

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

####

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def label_mapping(input, mapping):
    output = np.copy(input)
    for ind in range(len(mapping)):
        output[input == mapping[ind][0]] = mapping[ind][1]
    return np.array(output, dtype=np.int64)


def compute_mIoU(gt_dir, pred_dir, set, devkit_dir='', source_domain='gta5'):
    """
    Compute IoU given the predicted colorized images and 
    """
    if source_domain == 'gta5':
        with open(join(devkit_dir, 'info_gta5_to_cityscapes.json'), 'r') as fp:
            info = json.load(fp)
    elif source_domain == 'synthia':
        with open(join(devkit_dir, 'info_synthia_to_cityscapes.json'), 'r') as fp:
            info = json.load(fp)

    num_classes = np.int(info['classes'])
    print(('Num classes', num_classes))
    name_classes = np.array(info['label'], dtype=np.str)
    mapping = np.array(info['label2train'], dtype=np.int)
    hist = np.zeros((num_classes, num_classes))

    #image_path_list = join(devkit_dir, set+'.txt')
    #label_path_list = join(devkit_dir, set+'_label.txt')
    
    image_path_list = join(devkit_dir, 'val.txt')
    label_path_list = join(devkit_dir, 'label.txt')
    
    gt_imgs = open(label_path_list, 'r').read().splitlines()
    gt_imgs = [join(gt_dir, x) for x in gt_imgs]
    pred_imgs = open(image_path_list, 'r').read().splitlines()
    pred_imgs = [join(pred_dir, x.split('/')[-1]) for x in pred_imgs]

    for ind in range(len(gt_imgs)):
        pred = np.array(Image.open(pred_imgs[ind]))
        label = np.array(Image.open(gt_imgs[ind]))
        label = label_mapping(label, mapping)
        #print(label.shape)
        #pred_s = Image.fromarray(label.astype(np.uint8)*50)
        #pred_s.save('cool.png')

        if len(label.shape) == 3 and label.shape[2]==4:
            label = label[:,:,0]
        if len(label.flatten()) != len(pred.flatten()):
            print(('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()), len(pred.flatten()), gt_imgs[ind], pred_imgs[ind])))
            continue
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
        if ind > 0 and ind % 10 == 0:
            print(('{:d} / {:d}: {:0.2f}'.format(ind, len(gt_imgs), 100*np.mean(per_class_iu(hist)))))
    
    mIoUs = per_class_iu(hist)
    for ind_class in range(num_classes):
        print(('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2))))
    print(('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2))))
    return mIoUs


def main(args):
   compute_mIoU(args.gt_dir, args.pred_dir, args.devkit_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('gt_dir', type=str, help='directory which stores CityScapes val gt images')
    parser.add_argument('pred_dir', type=str, help='directory which stores CityScapes val pred images')
    parser.add_argument('--devkit_dir', default='dataset/cityscapes_list', help='base directory of cityscapes')
    args = parser.parse_args()
    main(args)
