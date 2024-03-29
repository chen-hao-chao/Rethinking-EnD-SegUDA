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

ImageFile.LOAD_TRUNCATED_IMAGES = True

class cityscapesDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, resize_size=(1024, 512), crop_size=(512, 1024), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255, set='val', autoaug=False, source_domain='synthia'): #
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.resize_size = resize_size
        self.autoaug = autoaug
        self.h = crop_size[0]
        self.w = crop_size[1]
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        self.set = set
        
        '''
        if source_domain == 'gta5':
            self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                                19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                                26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
        elif source_domain == 'synthia':
            self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                                19: 6, 20: 7, 21: 8, 23: 9, 24: 10, 25: 11,
                                26: 12, 28: 13, 32: 14, 33: 15}
        '''

        for name in self.img_ids:
            img_file = osp.join(self.root, "leftImg8bit/train/%s" % (name))
            name = name.split('/')[1]
            label_file = osp.join(self.root, "gtFine/%s/%s" % (self.set, name))
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

        image, label = Image.open(datafiles["img"]).convert('RGB'), Image.open(datafiles["label"])
        # resize
        image, label = image.resize(self.resize_size, Image.BICUBIC), label.resize(self.resize_size, Image.NEAREST)
        if self.autoaug:
            policy = ImageNetPolicy()
            image = policy(image)

        image, label = np.asarray(image, np.float32), np.asarray(label, np.uint8)
        

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))
        x1 = random.randint(0, image.shape[1] - self.h)
        y1 = random.randint(0, image.shape[2] - self.w)
        image = image[:, x1:x1+self.h, y1:y1+self.w]
        label = label[x1:x1+self.h, y1:y1+self.w]

        if self.is_mirror and random.random() < 0.5:
            image = np.flip(image, axis = 2)
            label = np.flip(label, axis = 1)

        return image.copy(), label.copy(), np.array(size), name


if __name__ == '__main__':
    dst = cityscapesDataSet('./', './', mean=(0,0,0), set = 'train')
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, _, _, _ = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            img = Image.fromarray(np.uint8(img) )
            img.save('Cityscape_Demo.jpg')
        break
