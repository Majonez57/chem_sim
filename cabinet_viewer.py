import mujoco
import mujoco.viewer
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
import time
import math


# Load The scene
model = mujoco.MjModel.from_xml_path("chemscene.xml")
data = mujoco.MjData(model)

for i in range(model.njnt):
    print(i, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i))

def random_configuration(model):
    q = np.zeros(model.nq)

    for i in range(model.nq):
        low = model.jnt_range[i][0]
        high = model.jnt_range[i][1]
        print(i, math.degrees(low), math.degrees(high))
        q[i] = np.random.uniform(low, high)
    return q

ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "end_effector_link")

def get_ee_pos(data):
    return data.xpos[ee_id].copy()

random_configuration(model)
reachable = []
# Launch interactive viewer
FOLDED = (-0.55, 0, 0.2)
FOLDEDJOINTS = [0,
        math.radians(-135),
        math.radians(-150),
        math.radians(90),
        math.radians(145),
        0.1,0.1,0,0,0]

FLATPOS = (-0.55, 0.2, 0.2)
FLATJOINTS = [math.radians(90),
        math.radians(-135),
        math.radians(-145),
        math.radians(90),
        math.radians(90),
        0.1,0.1,0,0,0]


# FLATPOS = (-0.55, -0.2, 0.2)
# FLATJOINTS = [math.radians(90),
#         math.radians(100),
#         math.radians(145),
#         math.radians(90),
#         math.radians(159),
#         0.1,0.1,0,0,0]

TOPPOS = (0,0,0.8)
TOPQUAT = Rotation.from_euler('xyz',[0, 180, 0], degrees=True).as_quat()
TOPJOINTS = [0,
        math.radians(0),
        math.radians(-150),
        math.radians(90),
        math.radians(145),
        0.1,0.1,0,0,0]

#model.body_pos[model.body("robot_base").id] = (0,0,0)
#model.body_quat[model.body("robot_base").id] = (0, 1.5708, 3.1416)



data.qpos[:] = FOLDEDJOINTS
model.body_pos[model.body("robot_base").id] = FOLDED
#model.body_quat[model.body("robot_base").id] = TOPQUAT


with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_forward(model, data)
        
        
        viewer.sync()

viewer.close()