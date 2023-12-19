from functools import partial
import random
import numpy as np
import torch
import torch.nn as nn

from ...utils.spconv_utils import replace_feature, spconv
from ...utils import common_utils
from .spconv_backbone import post_act_block


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, indice_key=None, norm_fn=None):
        super(SparseBasicBlock, self).__init__()
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x.features

        assert x.features.dim() == 2, 'x.features.dim()=%d' % x.features.dim()

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity)
        out = replace_feature(out, self.relu(out.features))

        return out


class Voxel_MAE_res(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        
        self.masked_ratio = model_cfg.MASKED_RATIO
        
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        
        self.num_point_features = 16                

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose3d(128, 32, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose3d(32, 8, 3, padding=1, output_padding=1, stride=(4,2,2), bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU()
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose3d(8, 1, 3, padding=1, output_padding=1, stride=(3,2,2), bias=False),
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.forward_re_dict = {}
        
    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        pred = self.forward_re_dict['pred']
        target = self.forward_re_dict['target']
        loss = self.criterion(pred, target)

        tb_dict = {
            'loss_rpn': loss.item()
        }

        return loss, tb_dict

    @staticmethod
    def fisher_yates_shuffle(tensor):
        size = tensor.size(0)
        random_idx = torch.randperm(size)
        shuffled_tensor = tensor[random_idx]
        return shuffled_tensor

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']

        select_ratio = 1 - self.masked_ratio # ratio for select voxel

        # voxel_coords_distance = (voxel_coords[:,2]**2 + voxel_coords[:,3]**2)**0.5
        voxel_size = torch.tensor(self.voxel_size[::-1]).to(voxel_coords.device)
        point_cloud_range = torch.tensor(list(self.point_cloud_range)[0:3][::-1]).to(voxel_coords.device)
        voxel_range = (voxel_coords[:, 1:] * voxel_size) + point_cloud_range + (voxel_size * 0.5)    # z, y, x
        voxel_coords_distance = (voxel_range[:,1]**2 + voxel_range[:,2]**2)**0.5
        
        select_30 = voxel_coords_distance[:]<=30
        select_30to50 = (voxel_coords_distance[:]>30) & (voxel_coords_distance[:]<=50)
        select_50 = voxel_coords_distance[:]>50
        
        #id_list = [i for i in range(coords.shape[0])]
        id_list_select_30 = torch.argwhere(select_30==True).reshape(torch.argwhere(select_30==True).shape[0])
        id_list_select_30to50 = torch.argwhere(select_30to50==True).reshape(torch.argwhere(select_30to50==True).shape[0])
        id_list_select_50 = torch.argwhere(select_50==True).reshape(torch.argwhere(select_50==True).shape[0])

        shuffle_id_list_select_30 = self.fisher_yates_shuffle(id_list_select_30)
        shuffle_id_list_select_30to50 = self.fisher_yates_shuffle(id_list_select_30to50)
        shuffle_id_list_select_50 = self.fisher_yates_shuffle(id_list_select_50)
                
        slect_index = torch.cat((shuffle_id_list_select_30[:int(select_ratio*len(shuffle_id_list_select_30))], 
                                 shuffle_id_list_select_30to50[:int((select_ratio+0.2)*len(shuffle_id_list_select_30to50))], 
                                 shuffle_id_list_select_50[:int((select_ratio+0.2)*len(shuffle_id_list_select_50))]
        ), 0)

        nums = voxel_features.shape[0]

        voxel_fratures_all_one = torch.ones(nums,1).to(voxel_features.device)
        voxel_features_partial, voxel_coords_partial = voxel_features[slect_index,:], voxel_coords[slect_index,:]


        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features_partial,
            indices=voxel_coords_partial.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        input_sp_tensor_ones = spconv.SparseConvTensor(
            features=voxel_fratures_all_one,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        out = self.conv_out(x_conv4)

        self.forward_re_dict['target'] = input_sp_tensor_ones.dense()
        x_up1 = self.deconv1(out.dense())
        x_up2 = self.deconv2(x_up1)
        x_up3 = self.deconv3(x_up2)
   
        self.forward_re_dict['pred'] = x_up3

        return batch_dict
