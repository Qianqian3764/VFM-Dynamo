# VFM-Dynamo
This is the official PyTorch implementation for VFM-Dynamo (IJCAI 2026).
## Requirements
1. Install the dependencies with:
```shell
pip install -r requirements.txt
```
2. The correlation layer is borrowed from [NVIDIA-flownet2-pytorch](https://github.com/NVIDIA/flownet2-pytorch)
```shell
cd correlation_package
python setup.py install
```
## Model zoo
You can download the pretrained model weights from the following [link](https://drive.google.com/drive/folders/1MVkAokl_dcJ-AGklU3nFSGP5sYUnoaU7?dmr=1&ec=wgc-drive-hero-goto).
## Ground Truth Data Preparation and Evaluation
To prepare the ground truth depth maps, run:
```shell
python export_gt_depth.py --data_path kitti_data --split eigen
```
To evaluate a model on KITTI, run:
```shell
CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py \
--load_weights_folder ./pretrained_model/ \
--width 640 \
--height 192 \
--data_path /data/KITTI/kitti_raw_data \
```
