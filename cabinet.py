import mujoco
import mujoco.viewer
import numpy as np

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

# Launch interactive viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        
        viewer.sync()