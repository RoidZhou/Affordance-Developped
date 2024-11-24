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
from scipy.spatial.transform import Rotation as R
import open3d as o3d

class CameraIntrinsic(object):
    """Intrinsic parameters of a pinhole camera model.

    Attributes:
        width (int): The width in pixels of the camera.
        height(int): The height in pixels of the camera.
        K: The intrinsic camera matrix.
    """

    def __init__(self, width, height, fx, fy, cx, cy):
        self.width = width
        self.height = height
        self.K = np.array(
            [[fx, 0.0, cx],
             [0.0, fy, cy],
             [0.0, 0.0, 1.0]]
        )

    @property
    def fx(self):
        return self.K[0, 0]

    @property
    def fy(self):
        return self.K[1, 1]

    @property
    def cx(self):
        return self.K[0, 2]

    @property
    def cy(self):
        return self.K[1, 2]

    def to_dict(self):
        """Serialize intrinsic parameters to a dict object."""
        data = {
            "width": self.width,
            "height": self.height,
            "K": self.K.flatten().tolist(),
        }
        return data

    @classmethod
    def from_dict(cls, data):
        """Deserialize intrinisic parameters from a dict object."""
        intrinsic = cls(
            width=data["width"],
            height=data["height"],
            fx=data["K"][0],
            fy=data["K"][4],
            cx=data["K"][2],
            cy=data["K"][5],
        )
        return intrinsic

class Camera:
    def __init__(self, intrinsic, near=0.01, far=20.0, size=448, fov=35, dist=5.0, fixed_position=True):
        self.intrinsic = intrinsic
        self.width, self.height = size, size
        self.near, self.far = near, far
        self.fov = fov
        self.scale = 1000
        aspect = self.width / self.height

        self.proj_matrix = _build_projection_matrix(intrinsic, near, far)
        self.gl_proj_matrix = self.proj_matrix.flatten(order="F")

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
        print("pose", pos)
        self.view_matrix = p.computeViewMatrix(pos,
                                               forward,
                                               up)
        #投影矩阵：计算世界坐标系中的物体在相机二维平面上的坐标
        self.projection_matrix = p.computeProjectionMatrixFOV(self.fov, aspect, self.near, self.far)
        _view_matrix = np.array(self.view_matrix).reshape((4, 4), order='F')
        _projection_matrix = np.array(self.projection_matrix).reshape((4, 4), order='F')
        print("self.projection_matrix ", self.projection_matrix)
        print("self._view_matrix ", _view_matrix)
        #@ ：相乘运算，inv：计算逆矩阵
        self.tran_pix_world = np.linalg.inv(_projection_matrix @ _view_matrix)
        
        self.cx = intrinsic.K[0, 2]
        self.cy = intrinsic.K[1, 2]
        self.fx = intrinsic.K[0, 0]
        self.fy = intrinsic.K[1, 1]
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
        self.forward = forward
        self.left = left
        self.up = up
        self._view_matrix = mat44
        self.gl_view_matrix = _view_matrix.flatten(order="F")

    def render(self, extrinsic):
        """Render synthetic RGB and depth images.

        Args:
            extrinsic: Extrinsic parameters, T_cam_ref.
        """
        # Construct OpenGL compatible view and projection matrices.
        gl_view_matrix = extrinsic.copy() if extrinsic is not None else np.eye(4)
        gl_view_matrix[2, :] *= -1  # flip the Z axis
        gl_view_matrix = gl_view_matrix.flatten(order="F")

        result = p.getCameraImage(
            width=self.intrinsic.width,
            height=self.intrinsic.height,
            viewMatrix=gl_view_matrix,
            projectionMatrix=self.gl_proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )

        rgb, z_buffer = np.ascontiguousarray(result[2][:, :, :3]), result[3]
        depth = (
                1.0 * self.far * self.near / (self.far - (self.far - self.near) * z_buffer)
        )

        return Frame(rgb, depth, self.intrinsic, extrinsic), gl_view_matrix

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
                                                   self.gl_view_matrix, self.gl_proj_matrix
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
        y, x = np.where(depth < 5) # 输出所有为True的元素的索引
        z = self.near * self.far / (self.far + depth * (self.near - self.far)) # 深度图的像素值（通常是一个归一化的值，表示从相机到物体的距离与近裁剪面和远裁剪面之间距离的比例）转换为实际的Z坐标。
        permutation = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
        points = (permutation @ np.dot(np.linalg.inv(camera_matrix), \
            np.stack([x, y, np.ones_like(x)] * z[y, x], 0))).T # np.ones_like(x)为 写成齐次坐标形式，*z[y, x]为了将深度值映射到一个特定的范围内
        return y, x, points

    def create_point_cloud_from_depth_image(self, depth, organized=True):
        """ Generate point cloud using depth image only.

            Input:
                depth: [numpy.ndarray, (H,W), numpy.float32]
                    depth image
                camera: [CameraInfo]
                    camera intrinsics
                organized: bool
                    whether to keep the cloud in image shape (H,W,3)

            Output:
                cloud: [numpy.ndarray, (H,W,3)/(H*W,3), numpy.float32]
                    generated cloud, (H,W,3) for organized=True, (H*W,3) for organized=False
        """
        assert(depth.shape[0] == self.height and depth.shape[1] == self.width)
        # depImg = self.far * self.near / (self.far - (self.far - self.near) * depth)
        depImg = np.asanyarray(depth).astype(np.float32) * 1000
        depth = (depImg.astype(np.uint16))
        print(type(depImg[0, 0]))
        
        camera_matrix = self.tran_pix_world
        # xmap = np.arange(self.width)
        # ymap = np.arange(self.height)
        ymap, xmap = np.where(depth < 2000)
        # points_z[ymap, xmap]

        # xmap, ymap = np.meshgrid(xmap, ymap)
        points_z = depth[ymap, xmap] / self.scale
        points_x = (xmap - self.cx) * points_z / self.fx
        points_y = (ymap - self.cy) * points_z / self.fy
        cloud = np.stack([points_x, points_y, points_z], axis=-1)
        if not organized:
            cloud = cloud.reshape([-1, 3])
        # permutation = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
        # points = (permutation @ np.dot(np.linalg.inv(camera_matrix), \
        #     np.stack([points_x, points_y, np.ones_like(points_x)] * points_z[points_y, points_x], 0))).T 
        cloud_transformed = self.transform_point_cloud(cloud, self.mat44, format='4x4')
        # cloud_transformed = self.transform_point_cloud(cloud, camera_matrix, format='4x4')
        print("pc min and max: ", np.min(cloud_transformed[:, 2]), np.max(cloud_transformed[:, 2]))
        return ymap, xmap, cloud

    def transform_point_cloud(self, cloud, transform, format='3x3'):
        """ Transform points to new coordinates with transformation matrix.

            Input:
                cloud: [np.ndarray, (N,3), np.float32]
                    points in original coordinates
                transform: [np.ndarray, (3,3)/(3,4)/(4,4), np.float32]
                    transformation matrix, could be rotation only or rotation+translation
                format: [string, '3x3'/'3x4'/'4x4']
                    the shape of transformation matrix
                    '3x3' --> rotation matrix
                    '3x4'/'4x4' --> rotation matrix + translation matrix

            Output:
                cloud_transformed: [np.ndarray, (N,3), np.float32]
                    points in new coordinates
        """
        if not (format == '3x3' or format == '4x4' or format == '3x4'):
            raise ValueError('Unknown transformation format, only support \'3x3\' or \'4x4\' or \'3x4\'.')
        if format == '3x3':
            cloud_transformed = np.dot(transform, cloud.T).T
        elif format == '4x4' or format == '3x4':
            ones = np.ones(cloud.shape[0])[:, np.newaxis]
            cloud_ = np.concatenate([cloud, ones], axis=1)
            cloud_transformed = np.dot(transform, cloud_.T).T
            cloud_transformed = cloud_transformed[:, :3]
        return cloud_transformed

    @staticmethod
    def compute_XYZA_matrix(id1, id2, pts, size1, size2): # 将点 pts 放置在（size1, size2）矩阵的位置 (id1, id2) 上
        out = np.zeros((size1, size2, 4), dtype=np.float32)
        out[id1, id2, :3] = pts
        out[id1, id2, 3] = 1 # 将 (id1, id2) 位置上的第四个维度（A）设置为1
        return out

    def get_normal_map(self, relative_offset, cam):
        rgb, depth, _, _ = update_camera_image_to_base(relative_offset, cam)
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
                                                   self.gl_view_matrix, self.gl_proj_matrix)
        # rgba = (rgba * 255).clip(0, 255).astype(np.float32) / 255
        # white = np.ones((rgba.shape[0], rgba.shape[1], 3), dtype=np.float32)
        # mask = np.tile(rgba[:, :, 3:4], [1, 1, 3])
        # rgb = rgba[:, :, :3] * mask + white * (1 - mask)
        depImg = np.asanyarray(depth).astype(np.float32) * 1000
        depth = (depImg.astype(np.uint16))
        return rgba, depth

    def depth_image(self, depth):
        return np.asarray(depth)

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
            'mat44': self.mat44,
        }

class Frame(object):
    def __init__(self, rgb, depth, intrinsic, extrinsic=None):
        self.rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=o3d.geometry.Image(rgb),
            depth=o3d.geometry.Image(depth),
            depth_scale=1.0,
            depth_trunc=2.0,
            convert_rgb_to_intensity=False
        )

        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=intrinsic.width,
            height=intrinsic.height,
            fx=intrinsic.fx,
            fy=intrinsic.fy,
            cx=intrinsic.cx,
            cy=intrinsic.cy,
        )

        self.extrinsic = extrinsic if extrinsic is not None \
            else np.eye(4)

    def color_image(self):
        return np.asarray(self.rgbd.color)

    def depth_image(self):
        return np.asarray(self.rgbd.depth)

    def point_cloud(self):
        pc = o3d.geometry.PointCloud.create_from_rgbd_image(
            image=self.rgbd,
            intrinsic=self.intrinsic,
            extrinsic=self.extrinsic
        )
        pc = np.asarray(pc.points).reshape(448, 448, -1)

        return pc

def point_cloud_flter(pc, depth):
        row, col = np.where(depth < 2000)
        pc = pc[row, col]

        return row, col, pc

def _build_projection_matrix(intrinsic, near, far):
    perspective = np.array(
        [
            [intrinsic.fx, 0.0, -intrinsic.cx, 0.0],
            [0.0, intrinsic.fy, -intrinsic.cy, 0.0],
            [0.0, 0.0, near + far, near * far],
            [0.0, 0.0, -1.0, 0.0],
        ]
    )
    ortho = _gl_ortho(0.0, intrinsic.width, intrinsic.height, 0.0, near, far)
    return np.matmul(ortho, perspective)

'''
坐标可视化
'''
class DebugAxes(object):
    """
    可视化基于base的坐标系, 红色x轴, 绿色y轴, 蓝色z轴
    常用于检查当前关节pose或者测量关键点的距离
    用法:
    goalPosition1 = DebugAxes()
    goalPosition1.update([0,0.19,0.15
                         ],[0,0,0,1])
    """
    def __init__(self):
        self.uids = [-1, -1, -1]

    def update(self, pos, orn):
        """
        Arguments:
        - pos: len=3, position in world frame
        - orn: len=4, quaternion (x, y, z, w), world frame
        """
        pos = np.asarray(pos).reshape(3)

        rot3x3 = R.from_quat(orn).as_matrix()
        axis_x, axis_y, axis_z = rot3x3.T
        self.uids[0] = p.addUserDebugLine(pos, pos + axis_x * 0.05, [1, 0, 0], replaceItemUniqueId=self.uids[0])
        self.uids[1] = p.addUserDebugLine(pos, pos + axis_y * 0.05, [0, 1, 0], replaceItemUniqueId=self.uids[1])
        self.uids[2] = p.addUserDebugLine(pos, pos + axis_z * 0.05, [0, 0, 1], replaceItemUniqueId=self.uids[2])

def showAxes(pos, axis_x, axis_y, axis_z):
    p.addUserDebugLine(pos, pos + axis_x * 0.5, [1, 0, 0]) # red
    p.addUserDebugLine(pos, pos + axis_y * 0.5, [0, 1, 0]) # green
    p.addUserDebugLine(pos, pos + axis_z * 0.5, [0, 0, 1]) # blue

def ornshowAxes(pos, orn):
    rot3x3 = R.from_quat(orn).as_matrix()
    axis_x, axis_y, axis_z = rot3x3.T
    print("axis_x, axis_y, axis_z ", axis_x, axis_y, axis_z)
    p.addUserDebugLine(pos, pos + axis_x * 0.5, [1, 0, 0], lineWidth=2) # red
    p.addUserDebugLine(pos, pos + axis_y * 0.5, [0, 1, 0], lineWidth=2) # green
    p.addUserDebugLine(pos, pos + axis_z * 0.5, [0, 0, 1], lineWidth=2) # blue

def _gl_ortho(left, right, bottom, top, near, far):
    ortho = np.diag(
        [2.0 / (right - left), 2.0 / (top - bottom), -2.0 / (far - near), 1.0]
    )
    ortho[0, 3] = -(right + left) / (right - left)
    ortho[1, 3] = -(top + bottom) / (top - bottom)
    ortho[2, 3] = -(far + near) / (far - near)
    return ortho

'''
绑定相机位置并获取更新图像
'''
def update_camera_image(end_state, camera):
    cv2.namedWindow("image")
    end_pos = end_state[0]
    end_orn = end_state[1]
    wcT = _bind_camera_to_end(end_pos, end_orn)
    cwT = np.linalg.inv(wcT)

    frame = camera.render(cwT)
    assert isinstance(frame, Frame)

    rgb = frame.color_image()  # 这里以显示rgb图像为例, frame还包含了深度图, 也可以转化为点云
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])  # flip the rgb channel

    cv2.namedWindow("image")
    cv2.imshow("image", bgr)
    key = cv2.waitKey(10)
    time.sleep(10)

    return bgr

def update_camera_image_to_base(relative_offset, camera):

    # end_pos = end_state[0]
    # end_orn = end_state[1]
    end_pos = [0,0,0]
    end_orn = R.from_euler('XYZ', [0, 0, 0])
    end_orn = end_orn.as_quat()
    wcT = _bind_camera_to_base(end_pos, end_orn, relative_offset)
    cwT = np.linalg.inv(wcT)

    frame, gl_view_matrix = camera.render(cwT)
    assert isinstance(frame, Frame)

    rgb = frame.color_image()  # 这里以显示rgb图像为例, frame还包含了深度图, 也可以转化为点云
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])  # flip the rgb channel

    rgbd = frame.depth_image()

    pc = frame.point_cloud()

    import matplotlib
    matplotlib.use('TkAgg')  # 大小写无所谓 tkaGg ,TkAgg 都行
    import matplotlib.pyplot as plt

    # plt.figure(num=1)
    # plt.imshow(rgb)
    # plt.show()

    # plt.figure(num=2)
    # plt.imshow(rgbd)
    # plt.show()

    return rgb, rgbd, pc, cwT


def _bind_camera_to_base(end_pos, end_orn_or, relative_offset):
    """设置相机坐标系与末端坐标系的相对位置

    Arguments:
    - end_pos: len=3, end effector position
    - end_orn: len=4, end effector orientation, quaternion (x, y, z, w)

    Returns:
    - wcT: shape=(4, 4), transform matrix, represents camera pose in world frame
    """
    camera_link = DebugAxes()

    end_orn = R.from_quat(end_orn_or).as_matrix()
    end_x_axis, end_y_axis, end_z_axis = end_orn.T

    wcT = np.eye(4)  # w: world, c: camera, ^w_c T
    wcT[:3, 0] = -end_y_axis  # camera x axis
    wcT[:3, 1] = -end_z_axis  # camera y axis
    wcT[:3, 2] = end_x_axis  # camera z axis

    '''
    dist = 1
    theta = np.random.random() * np.pi*2
    phi = (np.random.random()+1) * np.pi/6
    pos = np.array([dist*np.cos(phi)*np.cos(theta), \
            dist*np.cos(phi)*np.sin(theta), \
            dist*np.sin(phi)])
    '''
    wcT[:3, 3] = end_orn.dot(relative_offset) + end_pos  # eye position
    # fg = R.from_euler('XYZ', [-np.pi/2, 0, 0]).as_matrix()

    forward = -relative_offset / np.linalg.norm(relative_offset)
    left = np.cross([0, 0, 1], forward)
    left = left / np.linalg.norm(left)
    up = np.cross(forward, left)
    fg = np.vstack([left, -up, forward]).T
    wcT[:3, :3] = fg

    camera_link.update(wcT[:3,3],R.from_matrix(wcT[:3, :3]).as_quat())

    return wcT

def _bind_camera_to_end(end_pos, end_orn_or):
    """设置相机坐标系与末端坐标系的相对位置

    Arguments:
    - end_pos: len=3, end effector position
    - end_orn: len=4, end effector orientation, quaternion (x, y, z, w)

    Returns:
    - wcT: shape=(4, 4), transform matrix, represents camera pose in world frame
    """
    relative_offset = [-0.08, 0, 0.6]  # 相机原点相对于末端执行器局部坐标系的偏移量
    end_orn = R.from_quat(end_orn_or).as_matrix()
    end_x_axis, end_y_axis, end_z_axis = end_orn.T


    wcT = np.eye(4)  # w: world, c: camera, ^w_c T
    wcT[:3, 0] = -end_y_axis  # camera x axis
    wcT[:3, 1] = -end_z_axis  # camera y axis
    wcT[:3, 2] = end_x_axis  # camera z axis
    wcT[:3, 3] = end_orn.dot(relative_offset) + end_pos  # eye position
    fg = R.from_euler('XYZ', [-0.35, 0, 0]).as_matrix()

    camera_link = DebugAxes()
    camera_link.update(wcT[:3,3],end_orn_or)

    wcT[:3, :3] = np.matmul(wcT[:3, :3], fg)
    return wcT

def get_target_part_pose(objectID, tablaID):
    cid = p.createConstraint(objectID, -1, tablaID, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 1])
    cInfo = p.getConstraintInfo(cid)
    print("info: ", cInfo[7])
    print("info: ", cInfo[9])

    # info = p.getLinkState(self.objectID, self.objLinkID)
    pose = cInfo[7]
    orie = cInfo[9]

    return pose, orie
