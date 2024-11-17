"""
    an RGB-D camera
"""
import pybullet as p
import glob
from collections import namedtuple
from attrdict import AttrDict
import functools
import torch
import cv2
from scipy import ndimage
import numpy as np
from PIL import Image
from normal_map import startConvert

class Camera:
    def __init__(self, near=0.1, far=100.0, size=448, fov=35, dist=5.0, fixed_position=True):
        self.width, self.height = size, size
        self.near, self.far = near, far
        self.fov = fov

        aspect = self.width / self.height

        if fixed_position:
            theta = np.pi
            phi = np.pi/10
        else:
            theta = np.random.random() * np.pi*2
            phi = (np.random.random()+1) * np.pi/6
        pos = np.array([dist*np.cos(phi)*np.cos(theta), \
                dist*np.cos(phi)*np.sin(theta), \
                dist*np.sin(phi)])
        forward = -pos / np.linalg.norm(pos)
        left = np.cross([0, 0, 1], forward)
        left = left / np.linalg.norm(left)
        up = np.cross(forward, left)
        #视图矩阵：计算世界坐标系中的物体在摄像机坐标系下的坐标
        self.view_matrix = p.computeViewMatrix(pos,
                                               forward,
                                               up)
        #投影矩阵：计算世界坐标系中的物体在相机二维平面上的坐标
        self.projection_matrix = p.computeProjectionMatrixFOV(self.fov, aspect, self.near, self.far)
        _view_matrix = np.array(self.view_matrix).reshape((4, 4), order='F')
        _projection_matrix = np.array(self.projection_matrix).reshape((4, 4), order='F')
        #@ ：相乘运算，inv：计算逆矩阵
        self.tran_pix_world = np.linalg.inv(_projection_matrix @ _view_matrix)

        mat44 = np.eye(4)
        mat44[:3, :3] = np.vstack([forward, left, up]).T
        mat44[:3, 3] = pos      # mat44 is cam2world
        self.mat44 = mat44
        # log parameters
        self.near = near
        self.far = far
        self.dist = dist
        self.theta = theta
        self.phi = phi
        self.pos = pos

    def rgbd_2_world(self, w, h, d):
        x = (2 * w - self.width) / self.width
        y = -(2 * h - self.height) / self.height
        z = 2 * d - 1
        pix_pos = np.array((x, y, z, 1))
        position = self.tran_pix_world @ pix_pos
        position /= position[3]

        return position[:3]

    def shot(self):
        # Get depth values using the OpenGL renderer
        _w, _h, rgb, depth, seg = p.getCameraImage(self.width, self.height,
                                                   self.view_matrix, self.projection_matrix, [0, 0, 0.01], lightDistance=1,
                                                   )
        return rgb, depth, seg
    '''
    批量处理深度图像数据, 将多个像素的RGBD信息转换成世界坐标系下的三维位置信息
    '''
    def rgbd_2_world_batch(self, depth):
        x = (2 * np.arange(0, self.width) - self.width) / self.width
        x = np.repeat(x[None, :], self.height, axis=0)
        y = -(2 * np.arange(0, self.height) - self.height) / self.height
        y = np.repeat(y[:, None], self.width, axis=1)
        z = 2 * depth - 1

        pix_pos = np.array([x.flatten(), y.flatten(), z.flatten(), np.ones_like(z.flatten())]).T
        position = self.tran_pix_world @ pix_pos.T
        position = position.T
        # print(position)

        position[:, :] /= position[:, 3:4]

        return position[:, :3].reshape(*x.shape, -1)
    
    def compute_camera_XYZA(self, depth):
        camera_matrix = self.tran_pix_world[:3, :3]
        y, x = np.where(depth < 1) # 输出所有为True的元素的索引
        z = self.near * self.far / (self.far + depth * (self.near - self.far))
        permutation = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
        points = (permutation @ np.dot(np.linalg.inv(camera_matrix), \
            np.stack([x, y, np.ones_like(x)] * z[y, x], 0))).T # np.ones_like(x)为 写成齐次坐标形式，*z[y, x]为了将深度值映射到一个特定的范围内
        return y, x, points

    @staticmethod
    def compute_XYZA_matrix(id1, id2, pts, size1, size2): # 将点 pts 放置在（size1, size2）矩阵的位置 (id1, id2) 上
        out = np.zeros((size1, size2, 4), dtype=np.float32)
        out[id1, id2, :3] = pts
        out[id1, id2, 3] = 1 # 将 (id1, id2) 位置上的第四个维度（A）设置为1
        return out

    def get_normal_map(self, boxId):
        _w, _h, rgb, depth, seg = p.getCameraImage(self.width, self.height,
                                                   self.view_matrix, self.projection_matrix, [0, 0, 0.1], lightDistance=0.05,
                                                   lightAmbientCoeff=0.5) # lightSpecularCoeff
        image_array = rgb[:, :, :3]
        normal_map = startConvert(image_array)
        return normal_map

    def get_grasp_regien_mask(self, id1, id2, sz1, sz2):
        link_mask = np.zeros((sz1, sz2)).astype(np.uint8)
        for i in range(id1.shape[0]): # 返回索引值和元素
            link_mask[id1[i]][id2[i]] = 1
        return link_mask

    def get_observation(self):
        _w, _h, rgba, depth, seg = p.getCameraImage(self.width, self.height,
                                                   self.view_matrix, self.projection_matrix)
        rgba = (rgba * 255).clip(0, 255).astype(np.float32) / 255
        white = np.ones((rgba.shape[0], rgba.shape[1], 3), dtype=np.float32)
        mask = np.tile(rgba[:, :, 3:4], [1, 1, 3])
        rgb = rgba[:, :, :3] * mask + white * (1 - mask)
        return rgb, depth

        # return camera parameters
    def get_metadata_json(self):
        return {
            'dist': self.dist,
            'theta': self.theta,
            'phi': self.phi,
            'near': self.near,
            'far': self.far,
            'width': self.width,
            'height': self.height,
            'fov': self.fov,
            'camera_matrix': self.view_matrix,
            'projection_matrix': self.projection_matrix,
            'model_matrix': self.tran_pix_world,
            'mat44': np.array(self.tran_pix_world.tolist()),
        }