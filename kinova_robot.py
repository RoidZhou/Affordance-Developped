"""
    Franka Panda Robot Arm
        support panda.urdf, panda_gripper.urdf
"""

from __future__ import division
import sapien.core as sapien
from sapien.core import Pose, PxrMaterial, SceneConfig
from transforms3d.quaternions import axangle2quat, qmult
import numpy as np
from utils import pose2exp_coordinate, adjoint_matrix


class Robot(object):
    def __init__(self, env, urdf, material, open_gripper=True):
        self.env = env
        self.timestep = env.scene.get_timestep()

        # load robot
        loader = env.scene.create_urdf_loader()
        loader.fix_root_link = True
        self.robot = loader.load(urdf, {"material": material})
        #self.robot = loader.load(urdf, material)
        self.robot.name = "robot"

        # hand (EE), two grippers, the rest arm joints (if any)
        self.end_effector_index, self.end_effector = \
            [(i, l) for i, l in enumerate(self.robot.get_links()) if l.name == 'j2s7s300_link_7'][0]
        self.hand_actor_id = self.end_effector.get_id()
        self.gripper_joints = [joint for joint in self.robot.get_joints() if 
                (joint.get_name().startswith("j2s7s300_joint_finger"))] # and (not joint.get_name().startswith("j2s7s300_joint_finger_tip"))
        self.gripper_drived_joints = [joint for joint in self.robot.get_joints() if 
                (joint.get_name().startswith("j2s7s300_joint_finger") and (not joint.get_name().startswith("j2s7s300_joint_finger_tip")))]
        self.gripper_tip_drived_joints = [joint for joint in self.robot.get_joints() if 
                (joint.get_name().startswith("j2s7s300_joint_finger_tip"))]

        self.gripper_actor_ids = [joint.get_child_link().get_id() for joint in self.gripper_joints] # 
        self.arm_joints = [joint for joint in self.robot.get_joints() if
                joint.get_dof() > 0 and (not joint.get_name().startswith("j2s7s300_joint_finger") and not joint.get_name().endswith("inner_finger_joint"))]

        # set drive joint property
        for joint in self.arm_joints:
            joint.set_drive_property(1000, 400)
        for joint in self.gripper_joints:
            joint.set_drive_property(500, 60)

        # open/close the gripper at start
        if open_gripper:
            joint_angles = []
            for j in self.robot.get_joints():
                if j.get_dof() == 1:
                    if j.get_name().startswith("j2s7s300_joint_finger"):
                        joint_angles.append(0)
                    else:
                        joint_angles.append(0)
            self.robot.set_qpos(joint_angles)

    def compute_joint_velocity_from_twist(self, twist: np.ndarray) -> np.ndarray:
        """
        This function is a kinematic-level calculation which do not consider dynamics.
        Pay attention to the frame of twist, is it spatial twist or body twist

        Jacobian is provided for your, so no need to compute the velocity kinematics
        ee_jacobian is the geometric Jacobian on account of only the joint of robot arm, not gripper
        Jacobian in SAPIEN is defined as the derivative of spatial twist with respect to joint velocity

        Args:
            twist: (6,) vector to represent the twist

        Returns:
            (7, ) vector for the velocity of arm joints (not include gripper)

        """
        assert twist.size == 6
        # Jacobian define in SAPIEN use twist (v, \omega) which is different from the definition in the slides
        # So we perform the matrix block operation below
        dense_jacobian = self.robot.compute_spatial_twist_jacobian()  # (num_link * 6, dof()) (96, 12)
        ee_jacobian = np.zeros([6, self.robot.dof - 6]) # 2 修改为3  (6,6)
        ee_jacobian[:3, :] = dense_jacobian[self.end_effector_index * 6 - 3: self.end_effector_index * 6, :self.robot.dof - 6] # 2 修改为6   [33:36, :6]
        ee_jacobian[3:6, :] = dense_jacobian[(self.end_effector_index - 1) * 6: self.end_effector_index * 6 - 3, :self.robot.dof - 6] # 2 修改为6   [30:33, :6]

        #numerical_small_bool = ee_jacobian < 1e-1
        #ee_jacobian[numerical_small_bool] = 0
        #inverse_jacobian = np.linalg.pinv(ee_jacobian)
        inverse_jacobian = np.linalg.pinv(ee_jacobian, rcond=1e-2)
        #inverse_jacobian[np.abs(inverse_jacobian) > 5] = 0
        #print(inverse_jacobian)
        return inverse_jacobian @ twist

    def internal_controller(self, qvel: np.ndarray) -> None:
        """Control the robot dynamically to execute the given twist for one time step

        This method will try to execute the joint velocity using the internal dynamics function in SAPIEN.
        尝试使用SAPIEN中的内部动力学函数来执行关节速度。
        Note that this function is only used for one time step, so you may need to call it multiple times in your code
        Also this controller is not perfect, it will still have some small movement even after you have finishing using
        it. Thus try to wait for some steps using self.wait_n_steps(n) like in the hw2.py after you call it multiple
        time to allow it to reach the target position

        Args:
            qvel: (7,) vector to represent the joint velocity

        """
        assert qvel.size == len(self.arm_joints)
        target_qpos = qvel * self.timestep + self.robot.get_drive_target()[:-6] # 2 修改为6
        for i, joint in enumerate(self.arm_joints):
            joint.set_drive_velocity_target(qvel[i])
            joint.set_drive_target(target_qpos[i])
        passive_force = self.robot.compute_passive_force()
        self.robot.set_qf(passive_force)

    # 计算从一个当前末端执行器（end effector, EE）姿态到目标末端执行器姿态所需的“扭转”（twist）
    def calculate_twist(self, time_to_target, target_ee_pose):
        relative_transform = self.end_effector.get_pose().inv().to_transformation_matrix() @ target_ee_pose # 从当前末端执行器姿态到目标姿态的相对变换
        unit_twist, theta = pose2exp_coordinate(relative_transform) # 获得扭转角度（theta）
        velocity = theta / time_to_target # 根据目标角度和时间计算角速度（或扭转速度）
        body_twist = unit_twist * velocity # 将单位扭转乘以速度，得到身体扭转（body twist），它表示在单位时间内末端执行器相对于其当前姿态的扭转
        current_ee_pose = self.end_effector.get_pose().to_transformation_matrix()
        return adjoint_matrix(current_ee_pose) @ body_twist # 伴随矩阵在这里用于将身体扭转从末端执行器的局部坐标系转换到全局坐标系（或参考坐标系）

    def move_to_target_pose(self, target_ee_pose: np.ndarray, num_steps: int, custom=True) -> None:
        """
        Move the robot hand dynamically to a given target pose
        Args:
            target_ee_pose: (4, 4) transformation of robot hand in robot base frame (ee2base)
            num_steps:  how much steps to reach to target pose, 
                        each step correspond to self.scene.get_timestep() seconds
                        in physical simulation
        """
        executed_time = num_steps * self.timestep

        spatial_twist = self.calculate_twist(executed_time, target_ee_pose)
        for i in range(num_steps):
            if i % 100 == 0:
                spatial_twist = self.calculate_twist((num_steps - i) * self.timestep, target_ee_pose)
            qvel = self.compute_joint_velocity_from_twist(spatial_twist)
            self.internal_controller(qvel)
            self.env.step(custom) # 报异常
            self.env.render()
        return

    def close_gripper(self):
        for joint in self.gripper_tip_drived_joints: # -0.8 - 1
            joint.set_drive_target(1)
        
        for joint in self.gripper_drived_joints: # 0-0.87
            joint.set_drive_target(0.5)

        # for joint in self.gripper_outer_knuckle_joint: # 0-0.8
        #     joint.set_drive_target(0.1)

    def open_gripper(self):
        for joint in self.gripper_tip_drived_joints: # -0.8 - 1
            joint.set_drive_target(0)
        
        # for joint in self.gripper_inner_knuckle_joint: # 0-0.87
        #     joint.set_drive_target(0.8)

        # for joint in self.gripper_outer_knuckle_joint: # 0-0.8
        #     joint.set_drive_target(0.7)

    def clear_velocity_command(self):
        for joint in self.arm_joints:
            joint.set_drive_velocity_target(0)

    def wait_n_steps(self, n: int, custom=True):
        self.clear_velocity_command()
        for i in range(n):
            passive_force = self.robot.compute_passive_force()
            self.robot.set_qf(passive_force)
            self.env.step(custom)
            self.env.render()
        self.robot.set_qf([0] * self.robot.dof)

