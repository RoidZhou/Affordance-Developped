"""
    For panda (two-finger) gripper: pushing, pushing-left, pushing-up, pulling, pulling-left, pulling-up
        50% all parts closed, 50% middle (for each part, 50% prob. closed, 50% prob. middle)
        Simulate until static before starting
"""

import os
import sys
import shutil
import numpy as np
from PIL import Image
import cv2
import json
from argparse import ArgumentParser
from utils import get_robot_ee_pose, are_parallel, get_global_position_from_camera, save_h5, rotateMatrixToEulerAngles,create_orthogonal_vectors2,create_orthogonal_vectors
from sapien.core import Pose
from env_custom import Env, ContactError
from camera import ornshowAxes, Camera, CameraIntrinsic,showAxes,_bind_camera_to_end, update_camera_image, update_camera_image_to_base, point_cloud_flter
# from robots.kinova_robot import Robot
# from robots.ur5_robot import Robot
import pyvista as pv
import pcl
import pcl.pcl_visualization
import os
import time
import pdb
import math
import pybullet as p
import pybullet_data
from collections import namedtuple
from attrdict import AttrDict
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

parser = ArgumentParser()
parser.add_argument('category', type=str) # StorageFurniture
parser.add_argument('--out_dir', type=str)
parser.add_argument('--trial_id', type=int, default=0, help='trial id')
parser.add_argument('--random_seed', type=int, default=None)
parser.add_argument('--no_gui', action='store_true', default=False, help='no_gui [default: False]')
args = parser.parse_args()

gripper_main_control_joint_name = ["j2s7s300_joint_finger_1",
                    "j2s7s300_joint_finger_2",
                    "j2s7s300_joint_finger_3",
                    ]

mimic_joint_name = ["j2s7s300_joint_finger_tip_1",
                    "j2s7s300_joint_finger_tip_2",
                    "j2s7s300_joint_finger_tip_3",
                    ]
jointInfo = namedtuple("jointInfo",
                       ["id","name","type","lowerLimit","upperLimit","maxForce","maxVelocity"])

joints = AttrDict()
print("start collect data")
trial_id = args.trial_id
if args.no_gui:
    out_dir = os.path.join(args.out_dir, '%s_%d' % (args.category, trial_id))
else:
    out_dir = os.path.join('results', '%s_%d' % (args.category, trial_id))
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
print(out_dir)
os.makedirs(out_dir)
flog = open(os.path.join(out_dir, 'log.txt'), 'w')
out_info = dict() # 创建一个空字典

# set random seed
if args.random_seed is not None:
    np.random.seed(args.random_seed)
    out_info['random_seed'] = args.random_seed

camera_config = "./setup.json"
with open(camera_config, "r") as j:
    config = json.load(j)

camera_intrinsic = CameraIntrinsic.from_dict(config["intrinsic"])  # 相机内参数据
# setup camera
cam = Camera(camera_intrinsic, dist=0.5, fixed_position=False)

# setup env
env = Env()

out_info['camera_metadata'] = cam.get_metadata_json()

# p.resetDebugVisualizerCamera(2.0, -270., -60., (0., 0., 0.))
#重置相机的位置
# p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)  # Shadows on/off
#开启光线渲染

# load shape
object_urdf_fn = './grasp/%s/model.urdf' % args.category
flog.write('object_urdf_fn: %s\n' % object_urdf_fn)
state = 'random-closed-middle'
if np.random.random() < 0.5:
    state = 'closed'
flog.write('Object State: %s\n' % state)
out_info['object_state'] = state
objectID = env.load_object(object_urdf_fn)

# simulate some steps for the object to stay rest
still_timesteps = 0
wait_timesteps = 0
while still_timesteps < 1000:
    env.step()
    still_timesteps += 1

dist = 0.5
theta = np.random.random() * np.pi*2
phi = (np.random.random()+1) * np.pi/4
pose = np.array([dist*np.cos(phi)*np.cos(theta), \
        dist*np.cos(phi)*np.sin(theta), \
        dist*np.sin(phi)])
relative_offset_pose = pose

rgb, depth, pc, cwT = update_camera_image_to_base(relative_offset_pose, cam)
cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts = point_cloud_flter(pc, depth)
### use the GT vision
# rgb, depth, _ = cam.shot()
Image.fromarray((rgb).astype(np.uint8)).save(os.path.join(out_dir, 'rgb.png'))

# 根据深度图（depth）和相机的内参矩阵来计算相机坐标系中的三维点
# cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts = cam.create_point_cloud_from_depth_image(depth, organized=True) # 返回有效深度值的像素位置（y, x）和计算出的三维点坐标（points）。
# ''' show
pv.plot(
    cam_XYZA_pts,
    scalars=cam_XYZA_pts[:, 2],
    render_points_as_spheres=True,
    point_size=5,
    show_scalar_bar=False,
)
# '''
positive_mask = cam_XYZA_pts > 0  # 创建布尔掩码
positive_numbers = cam_XYZA_pts[positive_mask] # 选择正数元素

cloud = pcl.PointCloud(cam_XYZA_pts.astype(np.float32))
# 创建SAC-IA分割对象
seg = cloud.make_segmenter()
seg.set_optimize_coefficients(True)
seg.set_model_type(pcl.SACMODEL_PLANE)
seg.set_method_type(pcl.SAC_RANSAC)
seg.set_distance_threshold(0.02)
# 执行分割
inliers, coefficients = seg.segment()
# 获取地面点云和非地面点云
ground_points = cloud.extract(inliers, negative=False)
non_ground_points = cloud.extract(inliers, negative=True)
# 转换为array
cam_XYZA_filter_pts = non_ground_points.to_array()
# ''' show
pv.plot(
    cam_XYZA_filter_pts,
    scalars=cam_XYZA_filter_pts[:, 2],
    render_points_as_spheres=True,
    point_size=5,
    show_scalar_bar=False,
)
# '''
positive_mask = cam_XYZA_filter_pts > 0  # 创建布尔掩码
positive_numbers = cam_XYZA_filter_pts[positive_mask] # 选择正数元素

cam_XYZA_pts_tmp = np.array(cam_XYZA_pts).astype(np.float32)
cam_XYZA_filter_pts_tem = np.array(cam_XYZA_filter_pts).astype(np.float32)

index_inliers_set = set(inliers)
cam_XYZA_filter_idx = []
cam_XYZA_pts_idx = np.arange(cam_XYZA_pts.shape[0])
for idx in range(len(cam_XYZA_pts_idx)):
    if idx not in index_inliers_set:
        cam_XYZA_filter_idx.append(cam_XYZA_pts_idx[idx])
cam_XYZA_filter_idx = np.array(cam_XYZA_filter_idx)
cam_XYZA_filter_idx = cam_XYZA_filter_idx.astype(int)
cam_XYZA_filter_id1 = cam_XYZA_id1[cam_XYZA_filter_idx]
cam_XYZA_filter_id2 = cam_XYZA_id2[cam_XYZA_filter_idx]

# 将计算出的三维点信息组织成一个矩阵格式。
cam_XYZA = cam.compute_XYZA_matrix(cam_XYZA_filter_id1, cam_XYZA_filter_id2, cam_XYZA_filter_pts, depth.shape[0], depth.shape[1])
save_h5(os.path.join(out_dir, 'cam_XYZA.h5'), \
        [(cam_XYZA_filter_id1.astype(np.uint64), 'id1', 'uint64'), \
         (cam_XYZA_filter_id2.astype(np.uint64), 'id2', 'uint64'), \
         (cam_XYZA_filter_pts.astype(np.float32), 'pc', 'float32'), \
        ])

# compute cloud normal
cloud2 = pcl.PointCloud(cam_XYZA_pts.astype(np.float32))
# ne = pcl.IntegralImageNormalEstimation(cloud2)
# ne.set_NormalEstimation_Method_AVERAGE_3D_GRADIENT()
# ne.set_MaxDepthChange_Factor(0.02)
# ne.set_NormalSmoothingSize(10.0)
# normals = ne.compute()

import pointcloud_normal

normalpoint = pointcloud_normal.kSearchNormalEstimation(non_ground_points)

gt_nor = cam.get_normal_map(relative_offset_pose, cam)[0]
Image.fromarray(((gt_nor+1)/2*255).astype(np.uint8)).save(os.path.join(out_dir, 'gt_nor.png'))

object_link_ids = env.movable_link_ids
# gt_movable_link_mask = cam.get_movable_link_mask(object_link_ids) # gt_movable_link_mask 表示为：像素图中可抓取link对应其link_id
# Image.fromarray((gt_movable_link_mask>0).astype(np.uint8)*255).save(os.path.join(out_dir, 'interaction_mask.png')) # 将gt_movable_link_mask转为二值图进行保存

gt_movable_link_mask = cam.get_grasp_regien_mask(cam_XYZA_filter_id1, cam_XYZA_filter_id2, depth.shape[0], depth.shape[1]) # gt_movable_link_mask 表示为：像素图中可抓取link对应其link_id
Image.fromarray((gt_movable_link_mask>0).astype(np.uint8)*255).save(os.path.join(out_dir, 'interaction_mask.png')) # 将gt_movable_link_mask转为二值图进行保存

# sample a pixel to interact
# object_mask = cam.get_object_mask()
xs, ys = np.where(gt_movable_link_mask==1)
if len(xs) == 0:
    flog.write('No Movable Pixel! Quit!\n')
    flog.close()
    env.close()
    exit(1)
idx = np.random.randint(len(xs)) # sample interaction pixels random
x, y = xs[idx], ys[idx]
out_info['pixel_locs'] = [int(x), int(y)] # 采样到的像素位置
# 随机设置一个可移动关节作为 actor_id
env.set_target_object_part_actor_id(object_link_ids[gt_movable_link_mask[x, y]-1], custom=False) # [gt_movable_link_mask[x, y]-1] represent pixel coordinate(x,y) correspond to movable link id
out_info['target_object_part_actor_id'] = env.target_object_part_actor_id
out_info['target_object_part_joint_id'] = env.target_object_part_joint_id

# get pixel 3D pulling direction (cam/world)
# direction_cam = gt_nor[x, y, :3]

idx_ = np.random.randint(cam_XYZA_filter_pts.shape[0])
x, y = cam_XYZA_filter_id1[idx_], cam_XYZA_filter_id2[idx_]
# direction_cam = normalpoint[idx_][:3]
direction_cam = gt_nor[x, y, :3]

direction_cam /= np.linalg.norm(direction_cam)
out_info['direction_camera'] = direction_cam.tolist()
flog.write('Direction Camera: %f %f %f\n' % (direction_cam[0], direction_cam[1], direction_cam[2]))

# get pixel 3D position (cam/world)
position_world_xyz1 = cam_XYZA[x, y, :3]
position_world = position_world_xyz1[:3]
p.addUserDebugLine(position_world, position_world + direction_cam*1, [1, 0, 0])
p.addUserDebugText(str("gt_nor"), position_world, [1, 0, 0])

# direction_world = cwT[:3, :3] @ direction_cam
# out_info['direction_world'] = direction_world.tolist()
# flog.write('Direction World: %f %f %f\n' % (direction_world[0], direction_world[1], direction_world[2]))
flog.write('mat44: %s\n' % str(cwT))

# sample a random direction in the hemisphere (cam/world)
action_direction_cam = np.random.randn(3).astype(np.float32)
action_direction_cam /= np.linalg.norm(action_direction_cam)
if action_direction_cam @ direction_cam > 0: # 两个向量的夹角小于90度
    action_direction_cam = -action_direction_cam
# out_info['gripper_direction_camera'] = action_direction_cam.tolist() # position p
action_direction_world = action_direction_cam
out_info['gripper_direction_world'] = action_direction_world.tolist()
robotStartPos0 = [0.1, 0, 0.2]
robotStartPos1 = [0.1, 0, 0.4]
robotStartPos2 = [0.1, 0, 0.6]

# compute final pose
# 初始化 gripper坐标系，默认gripper正方向朝向-z轴
robotStartOrn = p.getQuaternionFromEuler([0, 0, 0])
# gripper坐标系绕y轴旋转-pi/2, 使其正方向朝向+x轴
robotStartOrn1 = p.getQuaternionFromEuler([0, -np.pi/2, 0])
robotStartrot3x3 = R.from_quat(robotStartOrn).as_matrix()
robotStart2rot3x3 = R.from_quat(robotStartOrn1).as_matrix()
# gripper坐标变换
basegrippermatZTX = robotStartrot3x3@robotStart2rot3x3
robotStartOrn2 = R.from_matrix(basegrippermatZTX).as_quat()

# 建立gripper朝向向量relative_offset，[0，0，1]为+z轴方向，由于默认gripper正方向朝向-z轴，所以x轴为-relative_offset
relative_offset = np.array(action_direction_world)
p.addUserDebugLine(robotStartPos2, robotStartPos2 + relative_offset*1, [0, 1, 0])
p.addUserDebugText(str("action_direction_world"), robotStartPos2, [0, 1, 0])

# 以 -relative_offset 为x轴建立正交坐标系
forward, up, left = create_orthogonal_vectors(relative_offset)
fg = np.vstack([forward, up, left]).T
robotStartOrnfg = R.from_matrix(fg).as_quat()
print("res: ", np.cross(fg[:, 0], relative_offset))

# gripper坐标变换
basegrippermatT = fg@basegrippermatZTX
robotStartOrn3 = R.from_matrix(basegrippermatT).as_quat()
theta_x, theta_y, theta_z = p.getEulerFromQuaternion(robotStartOrn3)
ornshowAxes(robotStartPos2, robotStartOrn3)
print("res: ", np.cross(basegrippermatT[:, 2], relative_offset))

rotmat = np.eye(4).astype(np.float32) # 旋转矩阵
rotmat[:3, :3] = basegrippermatT

# final_dist = 0.13 # ur5 grasp
final_dist = 0.2

final_rotmat = np.array(rotmat, dtype=np.float32)
final_rotmat[:3, 3] = position_world - action_direction_world * final_dist # 以齐次坐标形式添加 平移向量
final_pose = Pose().from_transformation_matrix(final_rotmat) # 变换矩阵转位置和旋转（四元数）
out_info['target_rotmat_world'] = final_rotmat.tolist()
p.addUserDebugPoints([[position_world[0], position_world[1], position_world[2]]], [[0, 1, 0]], pointSize=8)

p.addUserDebugPoints([[final_rotmat[:3, 3][0], final_rotmat[:3, 3][1], final_rotmat[:3, 3][2]]], [[0, 0, 1]], pointSize=8)

start_rotmat = np.array(rotmat, dtype=np.float32)
# start_rotmat[:3, 3] = position_world - action_direction_world * 0.2 # 以齐次坐标形式添加 平移向量  ur5 grasp
start_rotmat[:3, 3] = position_world - action_direction_world * 0.18 # 以齐次坐标形式添加 平移向量
start_pose = Pose().from_transformation_matrix(start_rotmat) # 变换矩阵转位置和旋转（四元数）
out_info['start_rotmat_world'] = start_rotmat.tolist()

action_direction = None

if action_direction is not None:
    end_rotmat = np.array(rotmat, dtype=np.float32)
    end_rotmat[:3, 3] = position_world - action_direction_world * final_dist + action_direction * 0.05
    out_info['end_rotmat_world'] = end_rotmat.tolist()

# showAxes(start_pose.p, forward, left, -up) # red, green, blue
### viz the EE gripper position
# setup robot
# robot_urdf_fn = './robots/panda_gripper.urdf'
# robot_urdf_fn = './robots/robotiq85/robots/robotiq_arg85_description.urdf'
# robot_urdf_fn = './robots/Robotiq85/urdf/robotiq_85.urdf'
# robot_urdf_fn = './robots/robotiq_85/urdf/robotiq_85_gripper_simple.urdf'
# robot_urdf_fn = './robots/ur5_description/urdf/ur5_robotiq_85.urdf'
robot_urdf_fn = './robots/kinova_j2s7s300/urdf/j2s7s300.urdf'

robotID = env.load_robot(final_rotmat[:3, 3], robotStartOrn3)
pose, orie = get_robot_ee_pose(robotID, 5)
theta_x, theta_y, theta_z = p.getEulerFromQuaternion(orie)
ornshowAxes(pose, orie)
# move to the final pose
rgb_final_pose, depth, _, _ = update_camera_image_to_base(relative_offset_pose, cam)

rgb_final_pose = cv2.circle(rgb_final_pose, (y, x), radius=2, color=(255, 0, 3), thickness=5)
Image.fromarray((rgb_final_pose).astype(np.uint8)).save(os.path.join(out_dir, 'viz_target_pose.png'))

# move back
# activate contact checking
env.setup_gripper(env.robotID)
env.start_checking_contact(env.gripper_actor_ids, env.hand_actor_id)

### main steps
out_info['start_target_part_qpos'] = env.get_target_part_qpos(custom=False)

target_link_mat44 = np.eye(4)
pose, orie = env.get_target_part_pose()
object_matrix = np.array(p.getMatrixFromQuaternion(orie)).reshape(3,3)
target_link_mat44[:3, :3] = object_matrix
target_link_mat44[:3, 3] = pose
# 某一点云坐标相对于物体Link坐标发生的位姿变换, 该点云坐标不随Link运动而变化          inv: 从物体Link坐标系变换回全局坐标系的逆变换，表示将点从世界坐标系变换到物体Link坐标系
position_world_xyz2 = np.ones((4), dtype=np.float32)
position_world_xyz2[:3] = position_world_xyz1 
position_local_xyz1 = np.linalg.inv(target_link_mat44) @ position_world_xyz2 # position_world_xyz1: 世界坐标系下物体上某一点云坐标

success = True
try:
    env.open_gripper(env.robotID, env.joints, gripper_main_control_joint_name, mimic_joint_name)
    env.wait_n_steps(500)

    # approach
    env.move_to_target_pose(final_rotmat, 500, custom=False) # final_rotmat 齐次坐标形式的位姿矩阵4✖4
    print("move to start pose end")
    env.wait_n_steps(500)
#     env.end_checking_contact(custom=False)

#     #### 计算位置变化 ####
#     target_link_pose = env.get_target_part_pose(custom=False).p # 得到世界坐标系下物体Link的位姿
#     # position_world_xyz1_start = target_link_mat44 @ position_local_xyz1 # position_world_xyz1_end: 某一点云坐标相对于世界坐标系下发生的位姿变换
#     mov_dir = np.array(target_link_pose[:2].tolist(), dtype=np.float32) - \
#             np.array([0,0], dtype=np.float32)
#     mov_dir = np.linalg.norm(mov_dir, ord=2)
#     print("mov_dir", mov_dir)
#     if mov_dir > 0.02:
#         success = False
#         print("move start contact: ", mov_dir)
#         raise ContactError

#     env.close_gripper(env.robotID, env.joints, gripper_main_control_joint_name, mimic_joint_name)
#     robot.wait_n_steps(1000, custom=False)

#     # activate contact checking
#     print("move end")
#     env.start_checking_contact(robot.hand_actor_id, robot.gripper_actor_ids, custom=False)
#     robot.move_to_target_pose(start_rotmat, 500, custom=False)
#     print("move to end pose end")
#     env.end_checking_contact(custom=False)
#     robot.wait_n_steps(500, custom=False)
#     print("move finish")
    
except ContactError:
    success = False

# target_link_mat44 = env.get_target_part_pose(custom=False).to_transformation_matrix() # 得到世界坐标系下物体Link的位姿
# position_world_xyz1_end = target_link_mat44 @ position_local_xyz1 # position_world_xyz1_end: 某一点云坐标相对于世界坐标系下发生的位姿变换
# flog.write('touch_position_world_xyz_start: %s\n' % str(position_world_xyz1))
# flog.write('touch_position_world_xyz_end: %s\n' % str(position_world_xyz1_end))
# out_info['touch_position_world_xyz_start'] = position_world_xyz1[:3].tolist()
# out_info['touch_position_world_xyz_end'] = position_world_xyz1_end[:3].tolist()

# #### 计算位置变化 ####
# # mov_dir = np.array(out_info['touch_position_world_xyz_end'], dtype=np.float32) - \
# #         np.array(out_info['touch_position_world_xyz_start'], dtype=np.float32)
# # mov_dir /= np.linalg.norm(mov_dir)
# # intended_dir = -np.array(out_info['gripper_direction_world'], dtype=np.float32)
# # print("move end distance: ", intended_dir @ mov_dir)
# # success = (intended_dir @ mov_dir > 0.01) and (intended_dir @ mov_dir < 0.5) 

# if success:
#     out_info['result'] = 'VALID'
#     out_info['final_target_part_qpos'] = env.get_target_part_qpos(custom=False)
# else:
#     out_info['result'] = 'CONTACT_ERROR'

# # save results
# with open(os.path.join(out_dir, 'result.json'), 'w') as fout:
#     json.dump(out_info, fout)

# #close the file
# flog.close()

# if args.no_gui:
#     # close env
#     env.close()
# else:
#     if success:
#         print('[Successful Interaction] Done. Ctrl-C to quit.')
#         ### wait forever
#         robot.wait_n_steps(100000)
#     else:
#         print('[Unsuccessful Interaction] invalid gripper-object contact.')
#         # close env
#         env.close()

