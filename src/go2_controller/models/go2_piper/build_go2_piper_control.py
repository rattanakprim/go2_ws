#!/usr/bin/env python3
"""
Build the Go2 + AgileX Piper scene for the TELEOP CONTROLLER (walking sim).

Unlike build_go2_piper.py (a passive view scene), this keeps the Go2's 12 torque
motors (the controller PD-drives them) and the Piper's 7 position actuators, so the
robot walks under control with the arm commandable. Actuator order is legs[0:12] then
piper[12:19]; legs stay at qpos[7:19] -- matching simulator.py's indexing.

Output: models/go2_piper/go2_piper_control.xml  (+ shared assets/ from build_go2_piper.py)
Run build_go2_piper.py first (it populates assets/).
"""
import os
import numpy as np
import mujoco as mj

HERE = os.path.dirname(os.path.abspath(__file__))
MEN = os.path.join(os.path.dirname(HERE), "mujoco_menagerie")
GO2_DIR = os.path.join(MEN, "unitree_go2")
PIPER_DIR = os.path.join(MEN, "agilex_piper")
OUT = os.path.join(HERE, "go2_piper_control.xml")

MOUNT_POS = [-0.05, 0.0, 0.09]
LEG_HOME = {"hip": 0.0, "thigh": 0.9, "calf": -1.8}   # Go2 menagerie "home" stance
STAND_Z = 0.27
PIPER_HOME = {"joint1": 0.0, "joint2": 1.57, "joint3": -1.3485,
              "joint4": 0.0, "joint5": 0.0, "joint6": 0.0,
              "joint7": 0.0, "joint8": 0.0}


def main():
    go2 = mj.MjSpec.from_file(os.path.join(GO2_DIR, "go2_imu_scene.xml"))
    piper = mj.MjSpec.from_file(os.path.join(PIPER_DIR, "piper.xml"))
    for s in (go2, piper):
        for k in list(s.keys):
            s.delete(k)
    fr = go2.body("base").add_frame()
    fr.pos = MOUNT_POS
    fr.attach_body(piper.body("base_link"), "piper_", "")
    # Arm collision-free for now (group stays 2/3 so the lidar still ignores it); avoids
    # arm/trunk capsule explosions during the gait. Enable later for real manipulation.
    for g in go2.geoms:
        if g.name.startswith("piper_"):
            g.contype, g.conaffinity = 0, 0
    go2.meshdir = "assets"
    go2.texturedir = "assets"
    go2.modelname = "go2_piper_control"
    m = go2.compile()
    open(OUT, "w").write(go2.to_xml())

    # Full-length home keyframe so reset_to_stand poses base+legs+arm correctly.
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
    k.name = "home"
    k.qpos = qpos.tolist()
    k.ctrl = ctrl.tolist()
    m = spec.compile()
    open(OUT, "w").write(spec.to_xml())
    print(f"OK  {OUT}")
    print(f"    nu={m.nu} nq={m.nq} nv={m.nv}  (legs ctrl[0:12], piper ctrl[12:{m.nu}])")


if __name__ == "__main__":
    main()
