#!/usr/bin/env python3
"""
Go2 + Piper PICK-AND-PLACE scene: two tables + a graspable box on table A.

Built for a physics grasp: collision + friction are enabled on the gripper finger
pads (and the box + tables); the rest of the arm stays collision-free so it can't
self-collide with the trunk during walking. Based on the control scene (Go2 torque
motors + Piper position actuators) so the existing teleop controller drives it.

Layout (world): robot starts at origin facing +x and walks to a "pick stand" ~0.5 m
behind each table. Tables are shallow (in x) so the robot body never overlaps them.

Output: models/go2_piper/go2_piper_pickplace.xml  (shared assets/)
Run build_go2_piper.py first to populate assets/.
"""
import os
import numpy as np
import mujoco as mj

HERE = os.path.dirname(os.path.abspath(__file__))
MEN = os.path.join(os.path.dirname(HERE), "mujoco_menagerie")
GO2_DIR = os.path.join(MEN, "unitree_go2")
PIPER_DIR = os.path.join(MEN, "agilex_piper")
OUT = os.path.join(HERE, "go2_piper_pickplace.xml")

MOUNT_POS = [-0.05, 0.0, 0.09]
LEG_HOME = {"hip": 0.0, "thigh": 0.9, "calf": -1.8}
STAND_Z = 0.27
PIPER_HOME = {"joint1": 0.0, "joint2": 1.57, "joint3": -1.3485,
              "joint4": 0.0, "joint5": 0.0, "joint6": 0.0,
              "joint7": 0.0, "joint8": 0.0}

# --- demo layout (world frame) -- the sequencer uses these too ---
TABLE_A = (1.10, 0.50)      # (x, y) of table A centre
TABLE_B = (1.10, -0.50)     # table B centre
TABLE_HALF = (0.12, 0.16, 0.12)   # half-sizes; top at 2*hz = 0.24
TABLE_TOP = 2 * TABLE_HALF[2]
# Slim (fingers close firmly around it) + light, tall and easy to grab.
ITEM_HALF = (0.010, 0.010, 0.030)        # the box to pick
ITEM_Z = TABLE_TOP + ITEM_HALF[2] + 0.001
ITEM_A = (TABLE_A[0], TABLE_A[1], ITEM_Z)


def _add_table(spec, name, cx, cy):
    g = spec.worldbody.add_geom()
    g.name = name
    g.type = mj.mjtGeom.mjGEOM_BOX
    g.size = list(TABLE_HALF)
    g.pos = [cx, cy, TABLE_HALF[2]]
    g.rgba = [0.55, 0.38, 0.22, 1.0]
    g.group = 0
    g.contype, g.conaffinity = 1, 1
    g.friction = [1.0, 0.02, 0.001]


def _add_item(spec):
    b = spec.worldbody.add_body()
    b.name = "pick_item"
    b.pos = list(ITEM_A)
    b.add_freejoint()
    g = b.add_geom()
    g.name = "pick_item_geom"
    g.type = mj.mjtGeom.mjGEOM_BOX
    g.size = list(ITEM_HALF)
    g.rgba = [0.10, 0.55, 0.90, 1.0]
    g.group = 0
    g.contype, g.conaffinity = 1, 1
    g.friction = [2.5, 0.1, 0.004]
    g.density = 250.0     # light (~0.006 kg) so a firm pinch easily holds it


def _enable_finger_collision(spec):
    """Re-enable collision (and friction) on just the gripper finger pads."""
    for ln in ("link7", "link8"):
        for g in spec.body(ln).geoms:
            if g.type == mj.mjtGeom.mjGEOM_BOX:     # the pad geoms, not the visual mesh
                g.contype, g.conaffinity = 1, 1
                g.friction = [2.0, 0.05, 0.002]
                g.group = 3


def main():
    go2 = mj.MjSpec.from_file(os.path.join(GO2_DIR, "go2_imu_scene.xml"))
    piper = mj.MjSpec.from_file(os.path.join(PIPER_DIR, "piper.xml"))
    for s in (go2, piper):
        for k in list(s.keys):
            s.delete(k)
    # clear the demo area: drop the scatter obstacles and the sports ball (keep floor+walls)
    for g in list(go2.worldbody.geoms):
        if g.name.startswith("obs_"):
            go2.delete(g)
    ball = go2.body("sports_ball")
    if ball is not None:
        go2.delete(ball)
    _enable_finger_collision(piper)
    fr = go2.body("base").add_frame()
    fr.pos = MOUNT_POS
    fr.attach_body(piper.body("base_link"), "piper_", "")
    # whole arm collision-free EXCEPT the finger pads we just turned on
    for g in go2.geoms:
        if g.name.startswith("piper_") and g.type != mj.mjtGeom.mjGEOM_BOX:
            g.contype, g.conaffinity = 0, 0
    # Reduce the arm joints' frictionloss (0.3) deadband so the position servos track
    # accurately enough to grasp the small item (the sequencer also calibrates the
    # command closed-loop). Leave the actuator gains as the menagerie defined them.
    for j in go2.joints:
        if j.name.startswith("piper_joint"):
            j.frictionloss = 0.0
            j.damping = max(j.damping, 1.0)
    # Firmer gripper squeeze so the pinch holds the item through the lift.
    for a in go2.actuators:
        if a.name == "piper_gripper":
            a.gainprm[0] = 200.0
            a.biasprm[1] = -200.0
            a.biasprm[2] = -10.0
    _add_table(go2, "table_a", *TABLE_A)
    _add_table(go2, "table_b", *TABLE_B)
    _add_item(go2)
    go2.meshdir = "assets"
    go2.texturedir = "assets"
    go2.modelname = "go2_piper_pickplace"
    m = go2.compile()
    open(OUT, "w").write(go2.to_xml())

    # home keyframe: robot standing, arm home, item resting on table A (qpos0).
    qpos = m.qpos0.copy()
    ctrl = np.zeros(m.nu)
    for j in range(m.njnt):
        nm = mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j) or ""
        adr = m.jnt_qposadr[j]
        if m.jnt_type[j] == mj.mjtJoint.mjJNT_FREE and \
           mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, m.jnt_bodyid[j]) == "base":
            qpos[adr + 2] = STAND_Z
        for key, val in LEG_HOME.items():
            if nm.endswith(f"{key}_joint"):
                qpos[adr] = val
        if nm.startswith("piper_"):
            qpos[adr] = PIPER_HOME.get(nm[len("piper_"):], 0.0)
    for a in range(m.nu):
        nm = mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, m.actuator_trnid[a, 0]) or ""
        for key, val in LEG_HOME.items():
            if nm.endswith(f"{key}_joint"):
                ctrl[a] = val
        if nm.startswith("piper_"):
            ctrl[a] = PIPER_HOME.get(nm[len("piper_"):], 0.0)
    spec = mj.MjSpec.from_file(OUT)
    k = spec.add_key()
    k.name, k.qpos, k.ctrl = "home", qpos.tolist(), ctrl.tolist()
    m = spec.compile()
    open(OUT, "w").write(spec.to_xml())
    print(f"OK  {OUT}")
    print(f"    nu={m.nu} nq={m.nq}  tables@{TABLE_A},{TABLE_B}  item@{ITEM_A}")


if __name__ == "__main__":
    main()
