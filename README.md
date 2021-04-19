# Rethinking Ensemble-Distillation for Semantic Segmentation Based Unsupervised Domain Adaption

### Conference
- CVPR Workshop (LLID) 2021
- NVIDIA GTC 2021

### File Structure
```
weights/
├── weights/
|   ├── synthia/
|   ├── gta5/
|   |   ├── gta5_ours_drn_57.98.pth
|   |   ├── ...
Rethinking_EnD_UDA/
├── label_fusion/
├── train_deeplabv2/
├── train_deeplabv3+/
├── ...
Warehouse/
├── SYNTHIA/
│   ├── labels/
│   ├── images/
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
1. Download the pre-generated pseudo labels [here](https://drive.google.com/drive/folders/1NjMDpjH6ESN9Nb9m9d48LLctvDsQn-uV?usp=sharing).
2. Place the pseudo labels in `Cityscapes/data/gtFine` folder and train the model with the following commands:
```
cd train_deeplabv3+
python train.py --class-balance --often-balance --backbone drn --restore-from ../../weights/weights/gta5/source/model_34.80.pth
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
{ Deeplabv3+ }
cd train_deeplabv3+
python test.py --backbone drn --restore-from ../../weights/weights/gta5/gta5_ours_drn_57.98.pth

============== SYNTHIA ===============
{ Deeplabv3+ }
cd train_deeplabv3+
python test.py --num-classes 16 --source-domain synthia --backbone drn --restore-from ../../weights/weights/synthia/synthia_ours_drn_59.95.pth
```

### Pretrained Weights
You can download the pre-trained weights [here](https://drive.google.com/drive/folders/18OFsUlhPvYdKyiSGoLbRIgbpzK4OOX2c?usp=sharing).

### Prerequisites
- Python 3.6
- Pytorch 1.5.0

Download the dependencies:
```
pip install requirement.txt
```


### Acknowledgement
The code is heavily borrowed from the following works:
- RMRNet: https://github.com/layumi/Seg-Uncertainty
- Deeplabv3+: https://github.com/jfzhang95/pytorch-deeplab-xception
