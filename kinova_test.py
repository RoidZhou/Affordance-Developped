import os
import time
import pdb
import math
import pybullet as p
import pybullet_data
from collections import namedtuple
from attrdict import AttrDict
import numpy as np

serverMode = p.GUI # GUI/DIRECT
robotUrdfPath = "./kinova_j2s7s300/urdf/j2s7s300.urdf"

# connect to engine servers
physicsClient = p.connect(serverMode)
# p.setPhysicsEngineParameter(enableFileCaching=0)
# add search path for loadURDF
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# define world
p.setGravity(0, 0, -10) # NOTE
planeID = p.loadURDF("plane.urdf")


robotStartPos = [0, 0, 0]
robotStartOrn = p.getQuaternionFromEuler([0, 0, 1.57])
robotID = p.loadURDF(robotUrdfPath, robotStartPos, robotStartOrn,
                     flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)

# load Object
ObjectID = p.loadURDF("./urdf/object_demo.urdf", [0, 0, 0.10], globalScaling=0.0030)
# Connect gripper base to robot tool.
# p.createConstraint(ObjectID, -1, -1, -1, jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0], parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, -0.07], childFrameOrientation=p.getQuaternionFromEuler([0, 0, np.pi / 2]))
cid = p.createConstraint(ObjectID, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 1])
info = p.getLinkState(ObjectID, 1)
print("info: ", info, cid)

jointTypeList = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
numJoints = p.getNumJoints(robotID)
jointInfo = namedtuple("jointInfo",
                       ["id","name","type","lowerLimit","upperLimit","maxForce","maxVelocity"])

joints = AttrDict()
dummy_center_indicator_link_index = 0

# get jointInfo and index of dummy_center_indicator_link
for i in range(numJoints):
    info = p.getJointInfo(robotID, i)
    jointID = info[0]
    jointName = info[1].decode("utf-8")
    jointType = jointTypeList[info[2]]
    jointLowerLimit = info[8]
    jointUpperLimit = info[9]
    jointMaxForce = info[10]
    jointMaxVelocity = info[11]
    singleInfo = jointInfo(jointID, jointName, jointType, jointLowerLimit, jointUpperLimit, jointMaxForce, jointMaxVelocity)
    joints[singleInfo.name] = singleInfo
    # register index of dummy center link
    if jointName == "gripper_roll":
        dummy_center_indicator_link_index = i

gripper_main_control_joint_name = ["j2s7s300_joint_finger_1",
                    "j2s7s300_joint_finger_2",
                    "j2s7s300_joint_finger_3",
                    ]

mimic_joint_name = ["j2s7s300_joint_finger_tip_1",
                    "j2s7s300_joint_finger_tip_2",
                    "j2s7s300_joint_finger_tip_3",
                    ]

mimic_multiplier = [1, 1, 1, -1, -1]

# id of gripper control user debug parameter
# angle calculation
# openning_length = 0.010 + 0.1143 * math.sin(0.7180367310119331 - theta)
# theta = 0.715 - math.asin((openning_length - 0.010) / 0.1143)
gripper_opening_length_control = p.addUserDebugParameter("gripper_opening_length",
                                                0,
                                                0.085,
                                                0.085)
control_dt=1./240.
p.setTimeStep=control_dt
cnt = 1
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
while (1):
    # gripper control

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
    p.stepSimulation()
    cnt += 1
    if cnt==240:
        break