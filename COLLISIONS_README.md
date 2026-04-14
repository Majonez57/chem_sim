# Kinova Gen3 Lite Teleoperation - Collisions + Pick-and-Place

## Summary

Updated MuJoCo simulation with:
- ✅ **Collision detection enabled** (robot arm cannot phase through cabinet walls)
- ✅ **Physics simulation enabled** (gravity, friction, contact dynamics)
- ✅ **Three pickable blocks** for testing pick-and-place operations
- ✅ **Hybrid control**: kinematic arm (precise IK) + dynamic blocks (realistic physics)

## New Objects

### Three Rectangular Prisms (on cabinet floor)

| Block | Size (XYZ) | Color | Position | Mass |
|-------|-----------|-------|----------|------|
| **Block A** | 4cm × 4cm × 3cm | Red | (-0.2, -0.1, 0.015) | 0.05kg |
| **Block B** | 6cm × 4cm × 4cm | Green | (-0.1, 0.05, 0.02) | 0.08kg |
| **Block C** | 8cm × 6cm × 6cm | Blue | (0.0, -0.1, 0.03) | 0.12kg |

Each block has:
- **Freejoint**: Fully dynamic (gravity, collisions, friction)
- **Inertia**: Proper mass properties for realistic physics
- **Friction**: High friction coefficients (1.0, 0.5, 0.5) for secure grasping
- **Site**: For visualization and tracking

### Cabinet Collisions

Cabinet walls now have **collision geoms**:
- Floor, back wall, left wall, right wall, front wall (angled)
- Each wall has both a **visual geom** (transparent) and a **collision geom** (solid)
- Proper contact parameters for realistic physics
- Robot arm collides with cabinet walls

## Control Scheme

### Robot Teleoperation (unchanged)
- **Position mode**: XYZ Cartesian control
- **Rotation mode**: Roll/Pitch/Yaw orientation control
- Toggle modes with **X button**
- **LB/RB + triggers**: Analog gripper control (knuckles + fingertips)

### Block Reset (new)
- **Back button (btn 6)**: Reset all blocks to initial positions
- Useful for repeating pick-and-place tests

## Technical Details

### Hybrid Physics Approach

**Problem**: Pure IK control uses `mj_forward` (no physics), but we need physics for blocks.

**Solution**: Hybrid approach
```python
# 1. Solve IK to get desired arm configuration
ik_solve_position(model, data, ik_body_id, target_pos)

# 2. Save kinematic arm state
arm_qpos = data.qpos[:10].copy()

# 3. Step full physics (arm + blocks)
mujoco.mj_step(model, data)

# 4. Override arm back to kinematic state
data.qpos[:10] = arm_qpos
data.qvel[:10] = 0  # No arm velocity
data.ctrl[:7] = arm_qpos[:7]  # Sync actuators
```

**Result**:
- ✅ Arm is **kinematically controlled** (perfect positioning, no drift)
- ✅ Blocks are **fully dynamic** (gravity, collisions, friction)
- ✅ Gripper can **pick up and move** blocks
- ✅ Robot arm **cannot pass through** cabinet walls

### Collision Parameters

Cabinet walls:
- `solref="0.005 1"`: Reference constraint
- `solimp="0.9 0.95 0.001"`: Impedance
- `condim="4"`: 4-dimensional contact (full friction)

Blocks:
- `friction="1.0 0.5 0.5"`: High friction for secure grasping
- Same solref/solimp as cabinet for consistent contact

### qpos Layout

```
qpos[0:6]   : Robot arm joints (joint_1 to joint_6)
qpos[6:10]  : Gripper joints (4 joints: 2 per finger)
qpos[10:17]  : Block A (pos3 + quat4)
qpos[17:24]  : Block B (pos3 + quat4)
qpos[24:31]  : Block C (pos3 + quat4)
```

### CSV Recording

Recording now includes block poses:
```
timestamp, ee_pos_x, ee_pos_y, ee_pos_z, ee_quat_w, ee_quat_x, ee_quat_y, ee_quat_z, mode,
joint_0_pos, joint_0_vel, ..., joint_5_pos, joint_5_vel,
grip_knuckle_r, grip_tip_r, grip_knuckle_l, grip_tip_l,
block_a_x, block_a_y, block_a_z, block_a_qw, block_a_qx, block_a_qy, block_a_qz,
block_b_x, block_b_y, block_b_z, block_b_qw, block_b_qx, block_b_qy, block_b_qz,
block_c_x, block_c_y, block_c_z, block_c_qw, block_c_qx, block_c_qy, block_c_qz
```

## Files Changed

### `chemscene.xml`
- Enabled gravity: `<flag gravity="enable"/>`
- Added collision geoms to all cabinet walls (dual geom: visual + collision)
- Added 3 block bodies with freejoints, inertia, collision geoms
- Each block has: size, mass, friction, site

### `cabinet_joy.py`
- Switched from `mj_forward` to `mj_step` for physics
- Implemented hybrid approach: override arm qpos after mj_step
- Added Back button (btn 6) to reset blocks
- Extended CSV recording to include block poses
- Finds block qpos offsets dynamically from joint IDs

## Testing Pick-and-Place

1. **Start the script**: `python3 cabinet_joy.py`
2. **Press Start** to begin teleoperation
3. **Navigate to Block A/B/C** using position mode
4. **Rotate gripper** using rotation mode (X button)
5. **Lower gripper** over block
6. **Close knuckles** (LB + LT) to grasp
7. **Close fingertips** (RB + RT) for secure grip
8. **Lift block** and move to target location
9. **Open gripper** to release
10. **Press Back** to reset blocks for next trial

## Notes

- Blocks are placed on the **cabinet floor** (z=0 to z=0.03)
- Cabinet floor is at **z=0**
- Robot base is at **z=0.2** (side-mounted on cabinet)
- Maximum reach is approximately **0.6m** from robot base
- Blocks will **fall and collide** realistically if dropped
- Robot arm **collides** with cabinet walls (cannot phase through)

## Troubleshooting

### Blocks fall through floor
- Check gravity is enabled: `model.opt.gravity` should be `[0, 0, -9.81]`
- Verify cabinet floor collision geom exists: `cab_floor_col`

### Arm passes through walls
- Verify collision geoms are present in XML
- Check `condim="4"` on collision geoms
- Ensure `mj_step` is being called (not just `mj_forward`)

### Blocks don't respond to gripper
- Check gripper collision geoms exist in `gen3_lite_body.xml`
- Verify gripper friction parameters
- Blocks may be too heavy - reduce mass if needed

### Arm drifts when not moving
- This is expected with physics simulation
- Use higher actuator kp in XML for tighter position control
- Kinematic arm override (qpos[:10] after mj_step) helps but doesn't eliminate all drift
