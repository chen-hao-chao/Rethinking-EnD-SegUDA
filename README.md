# Semantic Segmentation Based Unsupervised Domain Adaptation via Pseudo Label Fusion 

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)

### File Structure
```
weights/
├── synthia/
├── gta5/
PLF/
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
Quick Start:
1. Download the pre-generated pseudo labels [here](https://drive.google.com/drive/folders/1OwoHsM4pV1aQLrhm9cH7EV3286o4KXuN?usp=sharing).
2. Place the pseudo labels in `Cityscapes/data/gtFine` folder and train the model with the following commands:
```
cd train_deeplabv3+
python train.py --class-balance --often-balance --restore-from ../../weights/weights/gta5/source/resnet/model_30.32.pth
```

The whole training procedure:
1. Train the teacher models
  - [DACS](https://github.com/vikolss/DACS)
  - [CRST](https://github.com/yzou2/CRST)
  - [CBST](https://github.com/yzou2/CBST)
  - [R-MRNet](https://github.com/layumi/Seg-Uncertainty)
2. Generate the pseudo labels and the output tensors. (NOTE: it is recommended that the certainty tensors should be first mapped to 0~100 and stored using byte tensors for memory conservation.)

3. Fuse the pseudo labels
```
cd label_fusion
python3 label_fusion.py
```
4. Place the pseudo labels in `Cityscapes/data/gtFine` folder and follow the instructions in "Quick Start" to train the model.

### Testing
```
================ GTA5 ================
{ Deeplabv2 }
cd train_deeplabv2
python test.py --restore-from ../../weights/weights/gta5/deeplabv2/resnet/PLF/model_52.76.pth
{ Deeplabv3+ }
cd train_deeplabv3+
python test.py --backbone drn --restore-from ../../weights/weights/gta5/deeplabv3+/drn/PLF/model_57.65.pth
python test.py --backbone mobilenet --restore-from ../../weights/weights/gta5/deeplabv3+/mobilenet/PLF/model_54.95.pth

============== SYNTHIA ===============
{ Deeplabv2 }
cd train_deeplabv2
python test.py --num-classes 16 --source-domain synthia --restore-from ../../weights/weights/synthia/deeplabv2/resnet/PLF/model_47.93.pth
{ Deeplabv3+ }
cd train_deeplabv3+
python test.py --num-classes 16 --source-domain synthia --backbone drn --restore-from ../../weights/weights/synthia/deeplabv3+/drn/PLF/model_51.76.pth
python test.py --num-classes 16 --source-domain synthia --backbone mobilenet --restore-from ../../weights/weights/synthia/deeplabv3+/mobilenet/PLF/model_50.28.pth
```

### Pretrained Weights
You can download the pre-trained weights [here](https://drive.google.com/drive/folders/1OwoHsM4pV1aQLrhm9cH7EV3286o4KXuN?usp=sharing).

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
