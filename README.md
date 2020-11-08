# Semantic Segmentation Based Unsupervised Domain Adaptation via Pseudo-Label Fusion 

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)

### File Structure
```
weights/
├── synthia/
├── gta5/
MMD/
├── train_deeplabv2/
├── train_deeplabv3+/
├── ...
...
Warehouse/
├── SYNTHIA/
│   ├── labels/
│   ├── images/
│   ├── depth/
|   |   ├── 0000000.png
|   |   ├── 0000001.png
|   |   ├── ...
├── GTA5/
│   ├── image/
│   ├── labels/
|   |   ├── 00000.png
|   |   ├── 00001.png
|   |   ├── ...
├── Cityscapes/
│   ├── data/
│   │   ├── gtFine/
│   │   ├── leftImg8bit/
│   │   │   ├── train/
│   |   |   ├── val/
│   |   |   ├── test/
│   │   |   |   ├── aachen
│   │   |   |   ├── ...
```
### Training
Quick start:
1. Down the pre-generated pseudo label here. (currently unavilable)
2. Place the pseudo label in the `Cityscapes/data/gtFine` folder and train with the following command:
```
cd train_deeplabv3+
python train.py 
```

The whole training procedure:
1. Train the teacher models
  - [DACS](https://github.com/vikolss/DACS)
  - [CRST](https://github.com/yzou2/CRST)
  - [CBST](https://github.com/yzou2/CBST)
  - [R-MRNet](https://github.com/layumi/Seg-Uncertainty)
2. Generate the pseudo labels and the output tensor
3. Fuse the pseudo labels
```
cd label_fusion
python3 label_fusion.py
```
4. Place the pseudo label in the `Cityscapes/data/gtFine` folder and train with the following command:
```
cd train_deeplabv3+
python train.py 
```


### Testing
```
================ GTA5 ================
{ Deeplabv2 }
cd train_deeplabv2
python test.py --restore-from ../../weights/weights/gta5/deeplabv2/resnet/certainty/model_50.13.pth
python test.py --restore-from ../../weights/weights/gta5/deeplabv2/resnet/priority/model_52.96.pth
python test.py --restore-from ../../weights/weights/gta5/deeplabv2/resnet/majority/model_52.76.pth
python test.py --backbone drn --restore-from ../../weights/weights/gta5/deeplabv2/drn/certainty/model_53.83.pth
python test.py --backbone drn --restore-from ../../weights/weights/gta5/deeplabv2/drn/priority/model_54.35.pth
python test.py --backbone drn --restore-from ../../weights/weights/gta5/deeplabv2/drn/majority/model_55.25.pth
python test.py --backbone mobilenet --restore-from ../../weights/weights/gta5/deeplabv2/mobilenet/certainty/model_48.23.pth
python test.py --backbone mobilenet --restore-from ../../weights/weights/gta5/deeplabv2/mobilenet/priority/model_51.35.pth
python test.py --backbone mobilenet --restore-from ../../weights/weights/gta5/deeplabv2/mobilenet/majority/model_50.98.pth

{ Deeplabv3+ }
cd train_deeplabv3+
python test.py --restore-from ../../weights/weights/gta5/deeplabv3+/resnet/certainty/model_51.97.pth
python test.py --restore-from ../../weights/weights/gta5/deeplabv3+/resnet/priority/model_55.12.pth
python test.py --restore-from ../../weights/weights/gta5/deeplabv3+/resnet/majority/model_54.75.pth
python test.py --backbone drn --restore-from ../../weights/weights/gta5/deeplabv3+/drn/certainty/model_54.85.pth
python test.py --backbone drn --restore-from ../../weights/weights/gta5/deeplabv3+/drn/priority/model_57.94.pth
python test.py --backbone drn --restore-from ../../weights/weights/gta5/deeplabv3+/drn/majority/model_57.65.pth
python test.py --backbone mobilenet --restore-from ../../weights/weights/gta5/deeplabv3+/mobilenet/certainty/model_51.64.pth
python test.py --backbone mobilenet --restore-from ../../weights/weights/gta5/deeplabv3+/mobilenet/priority/model_54.74.pth
python test.py --backbone mobilenet --restore-from ../../weights/weights/gta5/deeplabv3+/mobilenet/majority/model_54.95.pth

============== SYNTHIA ===============
{ Deeplabv2 }
cd train_deeplabv2
python test.py --num-classes 16 --source-domain synthia --restore-from ../../weights/weights/synthia/deeplabv2/resnet/model_47.93.pth

{ Deeplabv3+ }
cd train_deeplabv3+
python test.py --num-classes 16 --source-domain synthia --backbone drn --restore-from ../../weights/weights/synthia/deeplabv3+/drn/model_51.76.pth
python test.py --num-classes 16 --source-domain synthia --backbone mobilenet --restore-from ../../weights/weights/synthia/deeplabv3+/mobilenet/model_50.28.pth
```

### Pretrained Weights
You can download the pretrained model here. (currently unavilable)

### Prerequisites
- Python 3.6
- Pytorch 1.5.0

Download the dependencies:
```
pip install requirement.txt
```

### Acknowledgement
The code is heavily borrowed from the following works:
- MRNet: https://github.com/layumi/Seg-Uncertainty
- Deeplabv3+: https://github.com/jfzhang95/pytorch-deeplab-xception
