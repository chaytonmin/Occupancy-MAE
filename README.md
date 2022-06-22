# Voxel-MAE: Masked Autoencoders for Pre-training Large-scale Point Clouds

Repository for our arxiv paper ["Voxel-MAE: Masked Autoencoders for Pre-training Large-scale Point Clouds"](https://arxiv.org/abs/2206.09900).

## Introduction
Mask-based pre-training has achieved great success for self-supervised learning in image, video and language, without manually annotated supervision. However,  it has not yet been studied in the field of 3D object detection. As the point clouds in 3D object detection is large-scale, it is impossible to reconstruct the input point clouds. In this paper, we propose a mask voxel classification network for large-scale point clouds pre-training, named Voxel-MAE. Our key idea is to divide the point clouds into voxel representations and classify whether the voxel contains point clouds. This simple strategy makes the network to be voxel-aware of the object shape, thus improving the performance of 3D object detection. Our Voxel-MAE will open  new research area of self-supervised learning about large-scale point clouds to ultimately enhance the perception ability of the autonomous vehicle. Extensive experiments show great effectiveness of our pre-trained model with 3D object detectors (SECOND, CenterPoint, and PV-RCNN) on three popular datasets (KITTI, Waymo, and nuScenes).

<p align="center">
<img src="docs/Voxel-MAE.png" width="100%"/>Flowchart of Voxel-MAE
</p>

## Installation

Please refer to [INSTALL.md](docs/INSTALL.md) for the installation of [OpenPCDet(v0.5)](https://github.com/open-mmlab/OpenPCDet).

## Getting Started

Please refer to [GETTING_STARTED.md](docs/GETTING_STARTED.md) .

## Usage

### First Pre-training Voxel-MAE

KITTI:

```
Train with multiple GPUs:
bash ./scripts/dist_train_voxel_mae.sh ${NUM_GPUS}  --cfg_file cfgs/kitti_models/voxel_mae_kitti.yaml --batch_size ${BATCH_SIZE}

Train with a single GPU:
python3 train_voxel_mae.py  --cfg_file cfgs/kitti_models/voxel_mae_kitti.yaml --batch_size 4
```

Waymo:

```
python3 train_voxel_mae.py  --cfg_file cfgs/kitti_models/voxel_mae_waymo.yaml --batch_size 4
```

nuScenes:

```
python3 train_voxel_mae.py  --cfg_file cfgs/kitti_models/voxel_mae_nuscenes.yaml --batch_size 4
```

### Then traing OpenPCDet

Same as [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) with pre-trained model from our Voxel-MAE.

## Performance

### KITTI 3D Object Detection

The results are the 3D detection performance of moderate difficulty on the *val* set of KITTI dataset. Results of OpenPCDet are from [here](https://github.com/open-mmlab/OpenPCDet) .

|                                                       | Car@R11 | Pedestrian@R11 | Cyclist@R11 | download |
| ----------------------------------------------------- | :-----: | :------------: | :---------: | :------: |
| [SECOND](tools/cfgs/kitti_models/second.yaml)         |  78.62  |     52.98      |    67.15    |          |
| Voxel-MAE+SECOND                                      |         |                |             |          |
| [SECOND-IoU](tools/cfgs/kitti_models/second_iou.yaml) |  79.09  |     55.74      |    71.31    |          |
| Voxel-MAE+SECOND-IoU                                  |         |                |             |          |
| [PV-RCNN](tools/cfgs/kitti_models/pv_rcnn.yaml)       |  83.61  |     57.90      |    70.47    |          |
| Voxel-MAE+PV-RCNN                                     |         |                |             |          |

### Waymo Open Dataset Baselines

Similar to  [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) , all models are trained with **a single frame** of **20% data (~32k frames)** of all the training samples , and the results of each cell here are mAP/mAPH calculated by the official Waymo evaluation metrics on the **whole** validation set (version 1.2).    

| Performance@(train with 20\% Data)                           |      Vec_L1 |   Vec_L2    |   Ped_L1    |   Ped_L2    |   Cyc_L1    |   Cyc_L2    |
| ------------------------------------------------------------ | ----------: | :---------: | :---------: | :---------: | :---------: | :---------: |
| [SECOND](tools/cfgs/waymo_models/second.yaml)                | 70.96/70.34 | 62.58/62.02 | 65.23/54.24 | 57.22/47.49 | 57.13/55.62 | 54.97/53.53 |
| Voxel-MAE+SECOND                                             |             |             |             |             |             |             |
| [CenterPoint](tools/cfgs/waymo_models/centerpoint_without_resnet.yaml) | 71.33/70.76 | 63.16/62.65 | 72.09/65.49 | 64.27/58.23 | 68.68/67.39 | 66.11/64.87 |
| Voxel-MAE+CenterPoint                                        |             |             |             |             |             |             |
| [PV-RCNN (AnchorHead)](tools/cfgs/waymo_models/pv_rcnn.yaml) | 75.41/74.74 | 67.44/66.80 | 71.98/61.24 | 63.70/53.95 | 65.88/64.25 | 63.39/61.82 |
| Voxel-MAE+PV-RCNN (AnchorHead                                |             |             |             |             |             |             |
| [PV-RCNN (CenterHead)](tools/cfgs/waymo_models/pv_rcnn_with_centerhead_rpn.yaml) | 75.95/75.43 | 68.02/67.54 | 75.94/69.40 | 67.66/61.62 | 70.18/68.98 | 67.73/66.57 |
| Voxel-MAE+PV-RCNN (CenterHead)                               |             |             |             |             |             |             |
| [PV-RCNN++](tools/cfgs/waymo_models/pv_rcnn_plusplus.yaml)   | 77.82/77.32 | 69.07/68.62 | 77.99/71.36 | 69.92/63.74 | 71.80/70.71 | 69.31/68.26 |
| Voxel-MAE+PV-RCNN++                                          |             |             |             |             |             |             |



### NuScenes 3D Object Detection Baselines

|                                                              |  mATE | mASE  | mAOE  | mAVE  | mAAE  |  mAP  |  NDS  |
| ------------------------------------------------------------ | ----: | :---: | :---: | :---: | :---: | :---: | :---: |
| [SECOND-MultiHead (CBGS)](tools/cfgs/nuscenes_models/cbgs_second_multihead.yaml) | 31.15 | 25.51 | 26.64 | 26.26 | 20.46 | 50.59 | 62.29 |
| Voxel-MAE+SECOND-MultiHead (CBGS)                            |       |       |       |       |       |       |       |
| [CenterPoint (voxel_size=0.1)](tools/cfgs/nuscenes_models/cbgs_voxel01_res3d_centerpoint.yaml) | 30.11 | 25.55 | 38.28 | 21.94 | 18.87 | 56.03 | 64.54 |
| Voxel-MAE+CenterPoint                                        |       |       |       |       |       |       |       |

##  License

Our code and dataset are released under the Apache 2.0 license.

## Acknowledgement

This repository is based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet).


## Citation 
If you find this project useful in your research, please consider cite:


```
@ARTICLE{Voxel-MAE,
    title={Voxel-MAE: Masked Autoencoders for Pre-training Large-scale Point Clouds},
    author={{Min}, Chen and {Zhao}, Dawei and {Xiao}, Liang and {Nie}, Yiming and {Dai}, Bin}},
    journal = {arXiv e-prints},
    year={2022}
}
```

