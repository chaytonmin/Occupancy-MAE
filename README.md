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

