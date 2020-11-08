import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image, ImageFile
from dataset.autoaugment import ImageNetPolicy
#from autoaugment import ImageNetPolicy
import cv2
ImageFile.LOAD_TRUNCATED_IMAGES = True


class synthiaDataSet(data.Dataset):
    def __init__(self, root, list_path, class_index, max_iters=None, resize_size=(1024, 512), crop_size=(512, 1024), mean=(128, 128, 128), scale=False, mirror=True, ignore_label=255, autoaug = False):
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
        self.class_index = class_index
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        '''
        self.id_to_trainid = {
                3: 0,
                4: 1,
                2: 2,
                21: 3,
                5: 4,
                7: 5,
                15: 6,
                9: 7,
                6: 8,
                1: 9,
                10: 10,
                17: 11,
                8: 12,
                19: 13,
                12: 14,
                11: 15,
            }
        '''
        for name in self.img_ids:
            name_ = name.split('.')
            img_file = osp.join(self.root, "images/%s" % name)
            label_file = osp.join(self.root, "labels/%s(3).%s" % (name_[0],name_[1]))
            depth_file = osp.join(self.root, "depth/%s" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "depth": depth_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        name = datafiles["name"]
        image = Image.open(datafiles["img"]).convert('RGB')
        #label = Image.open(datafiles["label"]).convert('RGB')
        #depth = Image.open(datafiles["depth"])
        label = cv2.imread(datafiles["label"], -1)[:, :, -1]
        

        # resize
        if self.scale:
            random_scale = 0.8 + random.random()*0.4 # 0.8 - 1.2
            image = image.resize( ( round(self.resize_size[0] * random_scale), round(self.resize_size[1] * random_scale)) , Image.BICUBIC)
            #label = label.resize( ( round(self.resize_size[0] * random_scale), round(self.resize_size[1] * random_scale)) , Image.NEAREST)
            #depth = depth.resize( ( round(self.resize_size[0] * random_scale), round(self.resize_size[1] * random_scale)) , Image.NEAREST)
            label = cv2.resize(label, (round(self.resize_size[0] * random_scale), round(self.resize_size[1] * random_scale)), interpolation=cv2.INTER_NEAREST)
        else:
            image = image.resize( ( self.resize_size[0], self.resize_size[1] ) , Image.BICUBIC)
            #label = label.resize( ( self.resize_size[0], self.resize_size[1] ) , Image.NEAREST)
            #depth = depth.resize( ( self.resize_size[0], self.resize_size[1] ) , Image.NEAREST)
            label = cv2.resize(label, ( self.resize_size[0], self.resize_size[1] ), interpolation=cv2.INTER_NEAREST)

        if self.autoaug:
            policy = ImageNetPolicy()
            image = policy(image)
        
        image = np.asarray(image, np.float32)
        #label = np.asarray(label, np.uint8)[:,:,2]
        #depth = np.asarray(depth, np.float32)[:,:,0]
        #depth = (65536.0 / (depth + 1.0)) # inverse depth
        #depth /= np.amax(depth)

        # re-assign labels to match the format of Cityscapes
        label_copy = np.ones(label.shape, dtype=np.uint8)
        for k, v in list(self.id_to_trainid.items()):
            label_copy[label == k] = v
        label_copy[label == self.class_index] = 0

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))

        for i in range(10): #find hard samples
            x1 = random.randint(0, image.shape[1] - self.h)
            y1 = random.randint(0, image.shape[2] - self.w)
            tmp_image = image[:, x1:x1+self.h, y1:y1+self.w]
            tmp_label_copy = label_copy[x1:x1+self.h, y1:y1+self.w]
            #tmp_depth = depth[x1:x1+self.h, y1:y1+self.w]
            u =  np.unique(tmp_label_copy)
            
            if len(u) > 10:
                break
            else:
                continue

        image = tmp_image
        label_copy = tmp_label_copy
        #depth = tmp_depth

        if self.is_mirror and random.random() < 0.5:
            image = np.flip(image, axis = 2)
            label_copy = np.flip(label_copy, axis = 1)
            #depth = np.flip(depth, axis = 1)

        return image.copy(), label_copy.copy(), np.array(size), name


if __name__ == '__main__':
    dst = SynthiaDataSet('../../../../work/lance5487/Warehouse/SYNTHIA', './synthia_list/train.txt', mean=(0,0,0), autoaug=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, label, depth, _, _ = data
        print(imgs.shape, label.shape, depth.shape)
        if i == 0:
            plt.imshow(label[0])
            plt.colorbar(label='Distance to Camera')
            plt.savefig('depth.png')
            '''
            img = torchvision.utils.make_grid(imgs[3]).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            img = Image.fromarray(np.uint8(img) )
            img.save('Synthia_Demo.jpg')
            '''
        break
