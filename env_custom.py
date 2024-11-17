import gym
from gym import error,spaces,utils
from gym.utils import seeding
from collections import namedtuple
from attrdict import AttrDict
import os
import pybullet as p
import pybullet_data
import math
import numpy as np
import random
from utils import pose2exp_coordinate, adjoint_matrix

class ContactError(Exception):
    pass

class Env(gym.Env):
    metadata = {'render.modes':['human']}

    def __init__(self, object_position_offset=0.0, vis=False):
        self.vis = vis
        self.current_step = 0
        self.object_position_offset = object_position_offset
        # Observation buffer
        self.control_dt=1./240.
        self.prev_observation = tuple()
        self.endeffort_link = "j2s7s300_link_7"
        self.eefID = 6
        self.objLinkID = 0
        self.check_contact = False
        self.hand_actor_id = self.eefID
        self.gripper_actor_ids = []
        self.numJoints = 6
        p.connect(p.GUI)
        
        # connect to engine servers
        self.physicsClient = p.connect(p.GUI if self.vis else p.DIRECT)
        # add search path for loadURDF
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.configureDebugVisualizer(lightPosition=[0, 0, 0.1])
        # define world
        p.setGravity(0, 0, -10) # NOTE
        p.setTimeStep = self.control_dt
        self.planeID = p.loadURDF("plane.urdf")
        self.tablaID = p.loadURDF("./urdf/objects/table.urdf",
                            [0.0, 0.0, 0],#base position
                            p.getQuaternionFromEuler([0, 0, 0]),#base orientation
                            useFixedBase=True)
        self.robotUrdfPath = "./kinova_j2s7s300/urdf/j2s7s300.urdf"
        self.robotStartPos = [0, 0, 0.35]
        self.robotStartOrn = p.getQuaternionFromEuler([0, 0, 1.57])

    def step(self):
        self.current_step += 1
        self.step_simulation()

    def step_simulation(self):
        """
        Hook p.stepSimulation()
        """
        p.stepSimulation()

    def wait_n_steps(self, n: int):
        for i in range(n):
            self.step()

    def check_depth_change(self, cur_depth):
        _, prev_depth, _ = self.prev_observation
        changed_depth = cur_depth - prev_depth
        changed_depth_counter = np.sum(np.abs(changed_depth) > self.DEPTH_CHANGE_THRESHOLD)
        print('changed depth pixel count:', changed_depth_counter)
        return changed_depth_counter > self.DEPTH_CHANGE_COUNTER_THRESHOLD

    def load_object(self, urdf):
        robotStartOrientation = p.getQuaternionFromEuler([-np.pi, np.pi, np.pi]) #[roll,pitch,yaw]
        self.objectID = p.loadURDF(urdf, [self.object_position_offset, 0, 0.05], robotStartOrientation)

        # compute link actor information
        self.movable_link_ids = []
        dummy_id = 1
        self.movable_link_ids.append(dummy_id)
        self.target_object_part_joint_id = dummy_id

        t = 0
        while True:
            p.stepSimulation()
            t += 1
            if t == 120:
                break
        return self.objectID
    
    def load_robot(self):
        self.robotID = p.loadURDF(self.robotUrdfPath, self.robotStartPos, self.robotStartOrn,
                     flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)
        
    def set_target_object_part_actor_id(self, actor_id, custom=True):
        self.target_object_part_actor_id = actor_id
            
    # 计算从一个当前末端执行器（end effector, EE）姿态到目标末端执行器姿态所需的“扭转”（twist）
    def calculate_twist(self, time_to_target, target_ee_pose):
        eefPose_mat44 = np.eye(4)
        pose, orie = self.get_robot_ee_pose()
        object_matrix = np.array(p.getMatrixFromQuaternion(orie)).reshape(3,3)
        eefPose_mat44[:3, :3] = object_matrix
        eefPose_mat44[:3, 3] = pose
        relative_transform = np.linalg.inv(eefPose_mat44) @ target_ee_pose

        unit_twist, theta = pose2exp_coordinate(relative_transform) # 获得扭转角度（theta）
        velocity = theta / time_to_target # 根据目标角度和时间计算角速度（或扭转速度）
        body_twist = unit_twist * velocity # 将单位扭转乘以速度，得到身体扭转（body twist），它表示在单位时间内末端执行器相对于其当前姿态的扭转
        current_ee_pose = eefPose_mat44
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
        executed_time = num_steps * self.control_dt

        spatial_twist = self.calculate_twist(executed_time, target_ee_pose)
        for i in range(num_steps):
            if i % 100 == 0:
                spatial_twist = self.calculate_twist((num_steps - i) * self.control_dt, target_ee_pose)
            qvel = self.compute_joint_velocity_from_twist(spatial_twist)
            # print("qvel : ", qvel)
            self.setJointPosition(self.robotID, qvel)
            self.step() # 报异常
        return

    def move_to_target_pose_onestep(self, target_ee_pose: np.ndarray) -> None:
        self.setJointPosition2(self.robotID, target_ee_pose)


    def close_gripper(self, robotID, joints, gripper_main_control_joint_name, mimic_joint_name):
        for i in range(len(gripper_main_control_joint_name)):
            joint = joints[gripper_main_control_joint_name[i]]
            p.setJointMotorControl2(robotID,
                                    joint.id,
                                    p.POSITION_CONTROL,
                                    targetPosition=0.5,
                                    force=joint.maxForce,
                                    maxVelocity=joint.maxVelocity)

        for i in range(len(mimic_joint_name)):
            joint = joints[mimic_joint_name[i]]
            p.setJointMotorControl2(robotID,
                                    joint.id,
                                    p.POSITION_CONTROL,
                                    targetPosition=1,
                                    force=joint.maxForce,
                                    maxVelocity=joint.maxVelocity)

    def open_gripper(self, robotID, joints, gripper_main_control_joint_name, mimic_joint_name):
        for i in range(len(gripper_main_control_joint_name)):
            joint = joints[gripper_main_control_joint_name[i]]
            p.setJointMotorControl2(robotID,
                                    joint.id,
                                    p.POSITION_CONTROL,
                                    targetPosition=0,
                                    force=joint.maxForce,
                                    maxVelocity=joint.maxVelocity)

        for i in range(len(mimic_joint_name)):
            joint = joints[mimic_joint_name[i]]
            p.setJointMotorControl2(robotID,
                                    joint.id,
                                    p.POSITION_CONTROL,
                                    targetPosition=0,
                                    force=joint.maxForce,
                                    maxVelocity=joint.maxVelocity)

    def get_target_part_pose(self):
        self.cid = p.createConstraint(self.objectID, -1, self.tablaID, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 1])
        cInfo = p.getConstraintInfo(self.cid)
        print("info: ", cInfo[7])
        print("info: ", cInfo[9])

        # info = p.getLinkState(self.objectID, self.objLinkID)
        pose = cInfo[7]
        orie = cInfo[9]

        return pose, orie

    def get_robot_ee_pose(self):
        cInfo = p.getLinkState(self.robotID, self.eefID)
        pose = cInfo[0]
        orie = cInfo[1]

        return pose, orie

    def start_checking_contact(self, robot_hand_actor_id, robot_gripper_actor_ids, strict=False, custom=True):
        self.check_contact = True
        self.check_contact_strict = strict
        if custom:
            self.first_timestep_check_contact = True
        self.robot_hand_actor_id = robot_hand_actor_id
        self.robot_gripper_actor_ids = robot_gripper_actor_ids

    def setup_gripper(self, robotID):
        controlJoints = ["j2s7s300_joint_finger_1", "j2s7s300_joint_finger_tip_1",
                        "j2s7s300_joint_finger_2", "j2s7s300_joint_finger_tip_2",
                        "j2s7s300_joint_finger_3", "j2s7s300_joint_finger_tip_3"]
        jointTypeList = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
        numJoints = p.getNumJoints(robotID)
        jointInfo = namedtuple("jointInfo",
                            ["id", "name", "type", "lowerLimit", "upperLimit", "maxForce", "maxVelocity",
                            "controllable"])
        self.joints = AttrDict()
        for i in range(numJoints):
            info = p.getJointInfo(robotID, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = jointTypeList[info[2]]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = True if jointName in controlJoints else False
            info = jointInfo(jointID, jointName, jointType, jointLowerLimit,
                            jointUpperLimit, jointMaxForce, jointMaxVelocity, controllable)
            if info.type == "REVOLUTE":  # set revolute joint to static
                p.setJointMotorControl2(robotID, info.id, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
            self.joints[info.name] = info
            self.gripper_actor_ids.append(info[0])

    def get_target_part_qpos(self, custom=True):
        qpos = [1,1]
        return float(qpos[self.target_object_part_joint_id])

    def getJointStates(self, robot):
        joint_states = p.getJointStates(robot, range(p.getNumJoints(robot)))
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        joint_torques = [state[3] for state in joint_states]
        return joint_positions, joint_velocities, joint_torques


    def getMotorJointStates(self, robot):
        joint_states = p.getJointStates(robot, range(p.getNumJoints(robot)))
        joint_infos = [p.getJointInfo(robot, i) for i in range(p.getNumJoints(robot))]
        joint_states = [j for j, i in zip(joint_states, joint_infos) if i[3] > -1]
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        joint_torques = [state[3] for state in joint_states]
        return joint_positions, joint_velocities, joint_torques

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
        # dense_jacobian = self.robot.compute_spatial_twist_jacobian()  # (num_link * 6, dof()) (96, 12)
        # ee_jacobian = np.zeros([6, self.robot.dof - 6]) # 2 修改为3  (6,6)
        # ee_jacobian[:3, :] = dense_jacobian[self.end_effector_index * 6 - 3: self.end_effector_index * 6, :self.robot.dof - 6] # 2 修改为6   [33:36, :6]
        # ee_jacobian[3:6, :] = dense_jacobian[(self.end_effector_index - 1) * 6: self.end_effector_index * 6 - 3, :self.robot.dof - 6] # 2 修改为6   [30:33, :6]


        mpos, mvel, mtorq = self.getMotorJointStates(self.robotID)
        zero_vec = [0.0] * len(mpos)
        result = p.getLinkState(self.robotID,
                                self.eefID,
                                computeLinkVelocity=1,
                                computeForwardKinematics=1)
        link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = result
        jac_t, jac_r = p.calculateJacobian(self.robotID, self.eefID, com_trn, mpos, zero_vec, zero_vec)
        
        ee_jacobian = np.zeros([6, 6])
        jac_t = np.array(jac_t)[:, :6]
        jac_r = np.array(jac_r)[:, :6]

        ee_jacobian[:3, :] = np.array(jac_t)
        ee_jacobian[3:6, :] =np.array(jac_r)

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
        assert qvel.size == self.numJoints
        pos, vel, torq = self.getJointStates(self.robotID)
        target_qpos = qvel * self.control_dt + pos[:6] # 2 修改为6
        for i, joint in enumerate(self.numJoints):
            joint.set_drive_velocity_target(qvel[i])
            joint.set_drive_target(target_qpos[i])
        passive_force = self.robot.compute_passive_force()
        self.robot.set_qf(passive_force)

    def setJointPosition(self, robot, qvel: np.ndarray, kp=1.0, kv=0.3):
        assert qvel.size == self.numJoints
        pos, vel, torq = self.getJointStates(self.robotID)
        target_qpos = qvel * self.control_dt + pos[:6] # 2 修改为6

        if len(target_qpos) == self.numJoints:
            p.setJointMotorControlArray(robot,
                                        range(self.numJoints),
                                        p.POSITION_CONTROL,
                                        targetPositions=target_qpos,
                                        targetVelocities=qvel,
                                        positionGains=[kp] * self.numJoints,
                                        velocityGains=[kv] * self.numJoints)
        else:
            print("Not setting torque. "
                "Expected torque vector of "
                "length {}, got {}".format(self.numJoints, len(target_qpos)))

    def setJointPosition2(self, robot, position, kp=1.0, kv=0.3):
        zero_vec = [0.0] * self.numJoints
        if len(position) == self.numJoints:
            p.setJointMotorControlArray(robot,
                                        range(self.numJoints),
                                        p.POSITION_CONTROL,
                                        targetPositions=position,
                                        targetVelocities=zero_vec,
                                        positionGains=[kp] * self.numJoints,
                                        velocityGains=[kv] * self.numJoints)
        else:
            print("Not setting torque. "
                "Expected torque vector of "
                "length {}, got {}".format(self.numJoints, self.numJoints))
