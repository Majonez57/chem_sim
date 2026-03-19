import mujoco
import mujoco.viewer
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
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
        q[i] = np.random.uniform(low, high)
    return q

ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "end_effector_link")

def get_ee_pos(data):
    return data.xpos[ee_id].copy()

FOLDED = [0,
        math.radians(-135),
        math.radians(-159),
        math.radians(90),
        math.radians(159),
        0,0,0,0,0]

data.qpos[:] = FOLDED

reachable = []
# Launch interactive viewer
#with mujoco.viewer.launch_passive(model, data) as viewer:
if True:    
    for _ in range(20000):
        #data.qpos[:] = random_configuration(model)
        mujoco.mj_forward(model, data)
        if data.ncon == 0:
            reachable.append(get_ee_pos(data))

#        time.sleep(0.1)

#        viewer.sync()

#    viewer.close()

import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
reachable = np.array(reachable)

ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1.1,0.6, 0.8, 1]))
ax.set_xlim(-0.55,0.55)
ax.set_ylim(-0.3,0.3)  
ax.set_zlim(0, 0.8) 


ax.scatter(reachable[:,0],reachable[:,1],reachable[:,2], s=1)
plt.show()