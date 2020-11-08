import os
import os.path as osp
import numpy as np
import random
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image, ImageFile
from dataset.autoaugment import ImageNetPolicy
#from autoaugment import ImageNetPolicy

ImageFile.LOAD_TRUNCATED_IMAGES = True

class cityscapesExtraction(data.Dataset):
    def __init__(self, root, list_path, set, max_iters=None, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.ignore_label = ignore_label
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        self.set = set
        
        for name in self.img_ids:
            img_file = osp.join(self.root, "leftImg8bit/train/%s" % (name))
            name = name.split('/')[1]
            name = name.split('.png')[0]
            label_file = osp.join(self.root, "gtFine/%s/%s" % (self.set, name))
            #label_file = osp.join(self.root, "%s" % (name))
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        #tt = time.time()
        datafiles = self.files[index]
        name = datafiles["name"]

        #image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"]+'.png')
        label_array = np.load(datafiles["label"]+'.npy')

        #image = np.asarray(image, np.float32)
        label = np.asarray(label, np.uint8)

        return label.copy(), label_array.copy(), name


if __name__ == '__main__':
    #x = np.load('../../result/Seg_Uncertainty/aachen_000000_000019_leftImg8bit.npy')
    #print(np.amax(x), np.amin(x))
    dst = cityscapesExtraction('../../result/Seg_Uncertainty', './cityscapes_list/train.txt', set='')
    trainloader = data.DataLoader(dst, batch_size=1, shuffle=False)
    for i, data in enumerate(trainloader):
        label, label_array, _ = data
        #label_array = np.asarray(label_array, np.uint8)
        print(torch.min(label_array))
        '''
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            img = Image.fromarray(np.uint8(img) )
            img.save('Cityscape_Demo.jpg')
            img = Image.fromarray(np.uint8(label_list[0][0]*255) )
            img.save('Cityscape_label.jpg')
        '''
