#!/usr/bin/env python3
"""
Teleoperation for Kinova Gen3 Lite using direct Jacobian-based IK.

Position mode:  Cartesian XYZ control of tool_frame position.
Rotation mode:  Roll/Pitch/Yaw control of tool_frame orientation.
                 Position is HELD FIXED while rotating.
Toggle between modes with X button.

Gripper:
  LB (btn 4)  -> Toggle knuckle direction (open/close)
  LT (trigger) -> Move knuckle joints (6,8) in that direction (analog speed)
  RB (btn 5)  -> Toggle fingertip direction (open/close)
  RT (trigger) -> Move fingertip joints (7,9) in that direction (analog speed)

Controls (POSITION mode - WORLD frame):
  Left stick Y   ->  World Z  (up / down)
  Left stick X   ->  World Y  (forward / back)
  Right stick Y  ->  World X  (left / right)
  X (btn 2)      ->  Toggle to ROTATION mode

Controls (ROTATION mode):
  Left stick X   ->  Roll
  Left stick Y   ->  Pitch
  Right stick X  ->  Yaw
  X (btn 2)      ->  Toggle to POSITION mode

Other:
  Start (btn 7)  ->  Reset robot
"""

import mujoco
import mujoco.viewer
import numpy as np
import pygame
import time
import csv
from pathlib import Path
from datetime import datetime

# ── Configuration ────────────────────────────────────────────────────────
MODEL_PATH = "chemscene.xml"
OUTPUT_DIR = "recordings"

IK_BODY = "tool_frame"

POS_SPEED = 0.12
ROT_SPEED = 0.8
GRIP_SPEED = 1.5
DEADZONE = 0.10
IK_POS_DAMPING = 0.001
IK_ROT_DAMPING = 0.01
IK_MAX_JOINT_STEP = 0.05
IK_ITERS = 5

RECORD_ENABLED = True
RECORD_EVERY_N_STEPS = 5
VIEWER_FPS = 60

# Stick axes (NOT triggers - those are separate)
AXIS_LX = 0   # Left stick X
AXIS_LY = 1   # Left stick Y

AXIS_RX = 3   # Right stick X
AXIS_RY = 4   # Right stick Y

AXIS_LT = 2   # Left trigger
AXIS_RT = 5   # Right trigger

# Gripper joint indices in qpos
GRIP_KR = 6
GRIP_TR = 7
GRIP_KL = 8
GRIP_TL = 9


# ── Helpers ──────────────────────────────────────────────────────────────

def deadzone_filter(x, dz=DEADZONE):
    if abs(x) < dz:
        return 0.0
    sign = 1.0 if x > 0 else -1.0
    return sign * (abs(x) - dz) / (1.0 - dz)


def quat_to_axisangle(quat):
    w, x, y, z = quat
    sin_half = np.sqrt(x * x + y * y + z * z)
    if sin_half < 1e-10:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(sin_half, w)
    axis = np.array([x, y, z]) / sin_half
    return axis * angle


def axisangle_to_quat(aa):
    angle = np.linalg.norm(aa)
    if angle < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = aa / angle
    ha = angle / 2.0
    return np.array([np.cos(ha), *(axis * np.sin(ha))])


def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


# ── State recorder ───────────────────────────────────────────────────────

class StateRecorder:
    def __init__(self, output_dir=OUTPUT_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.recording = False
        self.data = []
        self.start_time = None
        self.writer = None
        self.csvfile = None

    def start(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = self.output_dir / f"teleop_{ts}.csv"
        self.csvfile = open(fname, "w", newline="")
        self.writer = csv.writer(self.csvfile)
        header = ["timestamp",
                  "ee_pos_x", "ee_pos_y", "ee_pos_z",
                  "ee_quat_w", "ee_quat_x", "ee_quat_y", "ee_quat_z",
                  "mode"]
        for i in range(6):
            header += [f"joint_{i}_pos", f"joint_{i}_vel"]
        header += ["grip_knuckle_r", "grip_tip_r",
                   "grip_knuckle_l", "grip_tip_l"]
        self.writer.writerow(header)
        self.start_time = time.time()
        self.recording = True
        print(f"Recording to {fname}")

    def stop(self):
        if self.recording:
            self.csvfile.close()
            self.recording = False
            print(f"Recording saved ({len(self.data)} frames).")

    def record_frame(self, data, ee_id, mode_str):
        if not self.recording:
            return
        t = time.time() - self.start_time
        ee_pos = data.xpos[ee_id].copy()
        ee_quat = data.xquat[ee_id].copy()
        jp = data.qpos[:6].copy()
        jv = data.qvel[:6].copy()
        row = [t, *ee_pos, *ee_quat, mode_str]
        for i in range(6):
            row += [jp[i], jv[i]]
        row += [data.qpos[6], data.qpos[7], data.qpos[8], data.qpos[9]]
        self.writer.writerow(row)
        self.data.append(row)


# ── Robot presets ─────────────────────────────────────────────────────────

FOLDED = np.array([
    0.0,
    np.radians(-135),
    np.radians(-150),
    np.radians(90),
    np.radians(145),
    0.1,
    0.1, 0.0, 0.0, 0.0,
])


def reset_robot(model, data, ik_body_id):
    """Reset arm joints to folded, gripper to default, return ee pose."""
    data.qpos[:] = FOLDED
    data.qvel[:] = 0
    data.ctrl[:] = FOLDED[:7]
    mujoco.mj_forward(model, data)
    pos = data.xpos[ik_body_id].copy()
    quat = data.xquat[ik_body_id].copy()
    print(f"  Reset done. {IK_BODY} pos={np.round(pos, 5)} quat={np.round(quat, 5)}")
    return pos, quat


# ── IK solvers ────────────────────────────────────────────────────────────

def ik_solve_position(model, data, ik_body_id, target_pos, iterations=IK_ITERS):
    """Position-only IK using translational Jacobian."""
    for _ in range(iterations):
        mujoco.mj_forward(model, data)
        ee_pos = data.xpos[ik_body_id].copy()
        pos_error = target_pos - ee_pos

        if np.linalg.norm(pos_error) < 1e-6:
            return np.linalg.norm(pos_error)

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, ik_body_id)
        J = jacp[:, :6]

        delta_q = J.T @ np.linalg.solve(
            J @ J.T + IK_POS_DAMPING * np.eye(3), pos_error
        )

        mx = np.max(np.abs(delta_q))
        if mx > IK_MAX_JOINT_STEP:
            delta_q *= IK_MAX_JOINT_STEP / mx

        data.qpos[:6] += delta_q
        for i in range(6):
            data.qpos[i] = np.clip(data.qpos[i],
                                    model.jnt_range[i][0],
                                    model.jnt_range[i][1])

    mujoco.mj_forward(model, data)
    return np.linalg.norm(target_pos - data.xpos[ik_body_id])


def ik_solve_position_and_orientation(model, data, ik_body_id,
                                       target_pos, target_quat,
                                       pos_weight=1.0, rot_weight=0.5,
                                       iterations=IK_ITERS):
    """Combined 6-DOF IK using stacked Jacobian."""
    for _ in range(iterations):
        mujoco.mj_forward(model, data)
        ee_pos = data.xpos[ik_body_id].copy()
        ee_quat = data.xquat[ik_body_id].copy()

        pos_error = target_pos - ee_pos

        q_current_conj = np.array([ee_quat[0], -ee_quat[1],
                                    -ee_quat[2], -ee_quat[3]])
        q_error = quat_multiply(target_quat, q_current_conj)
        q_error /= np.linalg.norm(q_error)
        if q_error[0] < 0:
            q_error = -q_error
        rot_error = quat_to_axisangle(q_error)

        pos_err_norm = np.linalg.norm(pos_error)
        rot_err_norm = np.linalg.norm(rot_error)

        if pos_err_norm < 1e-6 and rot_err_norm < 1e-6:
            return pos_err_norm, rot_err_norm

        error = np.concatenate([
            pos_error * pos_weight,
            rot_error * rot_weight,
        ])

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, ik_body_id)

        J6 = np.vstack([
            jacp[:, :6] * pos_weight,
            jacr[:, :6] * rot_weight,
        ])

        damping = IK_POS_DAMPING * np.eye(6)
        delta_q = J6.T @ np.linalg.solve(
            J6 @ J6.T + damping, error
        )

        mx = np.max(np.abs(delta_q))
        if mx > IK_MAX_JOINT_STEP:
            delta_q *= IK_MAX_JOINT_STEP / mx

        data.qpos[:6] += delta_q
        for i in range(6):
            data.qpos[i] = np.clip(data.qpos[i],
                                    model.jnt_range[i][0],
                                    model.jnt_range[i][1])

    mujoco.mj_forward(model, data)
    p_err = np.linalg.norm(target_pos - data.xpos[ik_body_id])
    q_cur = data.xquat[ik_body_id]
    q_err = quat_multiply(target_quat,
                          np.array([q_cur[0], -q_cur[1], -q_cur[2], -q_cur[3]]))
    r_err = np.linalg.norm(quat_to_axisangle(q_err))
    return p_err, r_err


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        print("ERROR: No joystick detected!")
        return

    js = pygame.joystick.Joystick(0)
    js.init()
    print(f"Joystick: {js.get_name()}  ({js.get_numbuttons()} btn, {js.get_numaxes()} axes)")

    for i in range(js.get_numaxes()):
        print(f"  Axis {i} rest value: {js.get_axis(i):.3f}")

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    ik_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, IK_BODY)
    assert ik_body_id >= 0, f"body '{IK_BODY}' not found"

    # Cache gripper joint limits
    jnt_names = {
        "kr": "right_finger_bottom_joint",
        "tr": "right_finger_tip_joint",
        "kl": "left_finger_bottom_joint",
        "tl": "left_finger_tip_joint",
    }
    jnt_limits = {}
    for key, jname in jnt_names.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid >= 0:
            jnt_limits[key] = (model.jnt_range[jid][0], model.jnt_range[jid][1])
            print(f"  {jname}: range={jnt_limits[key]}")

    print(f"IK target body: {IK_BODY} (id={ik_body_id})")
    print(f"nq={model.nq}  nv={model.nv}  nu={model.nu}")

    target_pos, target_quat = reset_robot(model, data, ik_body_id)

    mode = "position"
    mode_toggle_held = False

    knuckle_dir = 1
    fingertip_dir = 1
    lb_held = False
    rb_held = False

    naxes = js.get_numaxes()

    print()
    print("=== Controls ===")
    print()
    print("POSITION mode (default):")
    print("  Left stick Y   ->  World Z  (up / down)")
    print("  Left stick X   ->  World Y  (forward / back)")
    print("  Right stick Y  ->  World X  (left / right)")
    print()
    print("ROTATION mode (position is held fixed):")
    print("  Left stick X   ->  Roll")
    print("  Left stick Y   ->  Pitch")
    print("  Right stick X  ->  Yaw")
    print()
    print("Both modes:")
    print("  X  (btn 2)     ->  Toggle POSITION / ROTATION")
    print("  LB (btn 4)     ->  Toggle knuckle direction")
    print("  LT (trigger)   ->  Move knuckle joints at analog speed")
    print("  RB (btn 5)     ->  Toggle fingertip direction")
    print("  RT (trigger)   ->  Move fingertip joints at analog speed")
    print("  Start (btn 7)  ->  Reset robot")
    print()
    print(f"Mode: {mode.upper()}")
    print("Press Start to begin teleoperation...")

    recorder = StateRecorder()
    started = False
    reset_held = False
    step = 0

    # Debug: print raw axis values once to help diagnose
    debug_axes = True

    with mujoco.viewer.launch_passive(model, data) as viewer:
        frame_time = time.time()

        while viewer.is_running():
            pygame.event.pump()

            # ── Read ALL raw axes (NO deadzone yet) ──
            axes_raw = [js.get_axis(i) for i in range(naxes)]
            buttons = [js.get_button(i) for i in range(js.get_numbuttons())]

            # ── Apply deadzone ONLY to stick axes, NOT triggers ──
            stick = {}
            stick["lx"] = deadzone_filter(axes_raw[AXIS_LX])
            stick["ly"] = deadzone_filter(axes_raw[AXIS_LY])
            stick["rx"] = deadzone_filter(axes_raw[AXIS_RX])
            stick["ry"] = deadzone_filter(axes_raw[AXIS_RY])

            # ── Read triggers separately (NOT through deadzone_filter) ──
            # Triggers rest at -1.0 (Xbox) or 0.0, normalize to [0, 1]
            lt_raw = axes_raw[AXIS_LT] if AXIS_LT < naxes else 0.0
            rt_raw = axes_raw[AXIS_RT] if AXIS_RT < naxes else 0.0
            lt_val = np.clip((lt_raw + 1.0) / 2.0, 0.0, 1.0)
            rt_val = np.clip((rt_raw + 1.0) / 2.0, 0.0, 1.0)
            if lt_val < 0.05:
                lt_val = 0.0
            if rt_val < 0.05:
                rt_val = 0.0

            # Debug: print raw axis values for first few frames
            if debug_axes and step < 30 and step % 10 == 0:
                print(f"  [DEBUG] Axes raw: {[f'{v:.3f}' for v in axes_raw]}")
                print(f"  [DEBUG] Stick filtered: lx={stick['lx']:.3f} ly={stick['ly']:.3f} rx={stick['rx']:.3f} ry={stick['ry']:.3f}")
                print(f"  [DEBUG] Triggers: lt={lt_val:.3f} rt={rt_val:.3f}")

            # ── Start / Reset (button 7) ──
            if buttons[7]:
                if not reset_held:
                    reset_held = True
                    if not started:
                        started = True
                        recorder.start()
                        print("Teleoperation STARTED.")
                    else:
                        print("RESET...")
                        target_pos, target_quat = reset_robot(model, data, ik_body_id)
                        knuckle_dir = 1
                        fingertip_dir = 1
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(0.02)
                continue
            else:
                reset_held = False

            if not started:
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(0.02)
                continue

            # ── Frame timing (wall clock) ──
            now = time.time()
            dt = min(now - frame_time, 0.05)
            frame_time = now

            # ── Mode toggle: X (button 2) ──
            if buttons[2]:
                if not mode_toggle_held:
                    mode_toggle_held = True
                    # Snap BOTH targets to current pose
                    mujoco.mj_forward(model, data)
                    target_pos = data.xpos[ik_body_id].copy()
                    target_quat = data.xquat[ik_body_id].copy()
                    if mode == "position":
                        mode = "rotation"
                    else:
                        mode = "position"
                    print(f"Mode: {mode.upper()}")
                    print(f"  Snapped target_pos={np.round(target_pos, 5)}")
                    print(f"  Snapped target_quat={np.round(target_quat, 5)}")
            else:
                mode_toggle_held = False

            # ── Gripper direction toggles ──
            if buttons[4]:
                if not lb_held:
                    lb_held = True
                    knuckle_dir *= -1
                    d = "OPENING" if knuckle_dir > 0 else "CLOSING"
                    print(f"Knuckles: {d}")
            else:
                lb_held = False

            if buttons[5]:
                if not rb_held:
                    rb_held = True
                    fingertip_dir *= -1
                    d = "OPENING" if fingertip_dir > 0 else "CLOSING"
                    print(f"Fingertips: {d}")
            else:
                rb_held = False

            # ── Apply gripper with analog triggers ──
            kr_delta = knuckle_dir * lt_val * GRIP_SPEED * dt if lt_val > 0 else 0.0
            tr_delta = fingertip_dir * rt_val * GRIP_SPEED * dt if rt_val > 0 else 0.0

            data.qpos[GRIP_KR] = np.clip(
                data.qpos[GRIP_KR] + kr_delta,
                jnt_limits["kr"][0], jnt_limits["kr"][1])
            data.qpos[GRIP_KL] = np.clip(
                data.qpos[GRIP_KL] - kr_delta,
                jnt_limits["kl"][0], jnt_limits["kl"][1])
            data.qpos[GRIP_TR] = np.clip(
                data.qpos[GRIP_TR] + tr_delta,
                jnt_limits["tr"][0], jnt_limits["tr"][1])
            data.qpos[GRIP_TL] = np.clip(
                data.qpos[GRIP_TL] + tr_delta,
                jnt_limits["tl"][0], jnt_limits["tl"][1])

            # ── IK control ──
            if mode == "position":
                # Position mode: use ONLY stick axes for velocity
                pos_vel = np.array([
                    -stick["ry"],    # World X  (right stick Y)
                    stick["lx"],     # World Y  (left stick X)
                    -stick["ly"],    # World Z  (left stick Y)
                ]) * POS_SPEED

                if np.linalg.norm(pos_vel) > 1e-6:
                    target_pos = target_pos + pos_vel * dt

                # Only run IK if target has moved from current position
                mujoco.mj_forward(model, data)
                ee_err = np.linalg.norm(target_pos - data.xpos[ik_body_id])
                if ee_err > 1e-6:
                    ik_solve_position(model, data, ik_body_id, target_pos)

            else:
                # Rotation mode: use ONLY stick axes for rotation
                rot_vel = np.array([
                    stick["lx"],     # Roll  (left stick X)
                    -stick["ly"],    # Pitch (left stick Y)
                    stick["rx"],     # Yaw   (right stick X)
                ]) * ROT_SPEED

                # Only modify target_quat when there is actual joystick input
                if np.linalg.norm(rot_vel) > 1e-6:
                    delta_aa = rot_vel * dt
                    delta_quat = axisangle_to_quat(delta_aa)
                    target_quat = quat_multiply(delta_quat, target_quat)
                    target_quat /= np.linalg.norm(target_quat)

                # Always solve 6-DOF (position + orientation) to prevent drift
                ik_solve_position_and_orientation(
                    model, data, ik_body_id,
                    target_pos, target_quat,
                    pos_weight=1.0, rot_weight=0.5,
                    iterations=IK_ITERS
                )

            mujoco.mj_forward(model, data)

            # ── Record ──
            if RECORD_ENABLED and step % RECORD_EVERY_N_STEPS == 0:
                recorder.record_frame(data, ik_body_id, mode)

            step += 1

            elapsed = time.time() - now
            time.sleep(max(0, (1.0 / VIEWER_FPS) - elapsed))
            viewer.sync()

    if recorder.recording:
        recorder.stop()


if __name__ == "__main__":
    main()
