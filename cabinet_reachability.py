import mujoco
import mujoco.viewer
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import time
import math
from scipy.spatial.transform import Rotation
from heatmap_helpers import make_voxel_grid

# Load The scene
model = mujoco.MjModel.from_xml_path("chemscene.xml")
data = mujoco.MjData(model)

# Get useful ids
ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "end_effector_link")
base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot_base")


# Helper functions
def set_base_pose(pos, quat):
    model.body_pos[base_id] = pos 
    if quat != None: model.body_quat[base_id] = quat
    mujoco.mj_forward(model, data)

def get_ee_pos(data):
    return data.xpos[ee_id].copy()

def random_configuration(model):
    q = np.zeros(model.nq)

    for i in range(model.nq):
        low = model.jnt_range[i][0]
        high = model.jnt_range[i][1]
        q[i] = np.random.uniform(low, high)
    return q

def clamp_joints(joints):
    for i in range(model.nq):
        low, high = model.jnt_range[i]
        joints[i] = np.clip(joints[i], low, high)
    
    return joints

BASEPOS = (-0.55, 0, 0.2)
BASEJOINTS = [0,
        math.radians(-135),
        math.radians(-159),
        math.radians(90),
        math.radians(159),
        0,0,0,0,0]

FLATPOS = (-0.55, 0.2, 0.2)
FLATJOINTS = [math.radians(0),
        math.radians(-130),
        math.radians(-130),
        math.radians(90),
        math.radians(-90),
        0.1,0.1,0,0,0]

TOPPOS = (0,0,0.8)
TOPQUAT = Rotation.from_euler('xyz',[0, 180, 0], degrees=True).as_quat()
TOPJOINTS = [0,
        math.radians(-135),
        math.radians(-150),
        math.radians(90),
        math.radians(145),
        0.1,0.1,0,0,0]


def explore(n_steps=30000, step_size=0.05, start_joints=BASEJOINTS):
    joints = BASEJOINTS.copy() # Start from known valid config
    data.qpos[:] = joints
    mujoco.mj_forward(model,data)

    reachable = []
    rejected = 0

    for _ in range(n_steps):
        djoints = np.random.normal(scale=step_size, size=model.nq) # Random joint delta
        new_joints = clamp_joints(joints + djoints)

        data.qpos[:] = new_joints 
        mujoco.mj_forward(model,data)

        if data.ncon == 0: #If found location is collision-free
            joints = new_joints
            reachable.append(get_ee_pos(data))
            rejected = 0 # This new position works, so we'll keep it!
        else:
            rejected += 1
        
        if rejected > 100: #Avoid local traps:
            joints = BASEJOINTS.copy() # return to default
            data.qpos[:] = joints
            mujoco.forwards(model, data)
            rejected = 0
    
    return np.array(reachable)

import matplotlib.pyplot as plt

set_base_pose(FLATPOS, None)
reachable_points = explore(100000, 0.05, FLATJOINTS)

grid = make_voxel_grid(reachable_points, ((-0.55,0.55),(-0.3,0.3),(0,0.2)), resolution=0.05)
heatmap = np.sum(grid, axis=2)

plt.figure()
plt.imshow(heatmap.T, origin='lower')
plt.colorbar(label="Reach frequency")
plt.scatter([5],[0],s=100, c=(1,1,1))
plt.title(f"Top-down heatmap")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()



# heatmap = np.sum(grid, axis=1)

# plt.figure()
# plt.imshow(heatmap.T, origin='lower')
# plt.colorbar(label="Reach frequency")
# plt.title(f"Side-VIew heatmap")
# plt.xlabel("X")
# plt.ylabel("Z")
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# set_base_pose(FLATPOS, None)
# reachable = explore(30000, 0.05, FLATJOINTS)

# ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1.1,0.6, 0.8, 1]))
# ax.set_xlim(-0.55,0.55)
# ax.set_ylim(-0.3,0.3)  
# ax.set_zlim(0, 0.8) 


# ax.scatter(reachable[:,0],reachable[:,1],reachable[:,2], s=1)
# plt.show()