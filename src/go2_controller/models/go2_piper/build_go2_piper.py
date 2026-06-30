#!/usr/bin/env python3
"""
Build the Go2 + AgileX Piper (6-DOF arm + gripper) MuJoCo model.

The Piper comes from the official MuJoCo Menagerie (agilex_piper), copied into
models/mujoco_menagerie/agilex_piper/. We attach its base_link onto the Go2 `base`
body and emit a single self-contained scene with a combined "home" keyframe
(Go2 standing + Piper at its home pose).

Outputs (in this folder, models/go2_piper/):
  go2_piper_scene.xml   -- combined scene
  assets/               -- Go2 .obj + soccer.png + Piper meshes/materials

This is a separate scene; the teleop scene (go2_imu_scene.xml) is left untouched.
"""
import os, glob, shutil
import numpy as np
import mujoco as mj

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.dirname(HERE)
MEN = os.path.join(MODELS, "mujoco_menagerie")
GO2_DIR = os.path.join(MEN, "unitree_go2")
PIPER_DIR = os.path.join(MEN, "agilex_piper")
ASSETS = os.path.join(HERE, "assets")
SCENE_OUT = os.path.join(HERE, "go2_piper_scene.xml")

MOUNT_POS = [-0.05, 0.0, 0.09]          # rel. Go2 `base` frame (x back, z up)
STAND_Z = 0.30                          # trunk height for the standing keyframe
LEG_HOME = {"hip": 0.0, "thigh": 0.9, "calf": -1.8}
PIPER_HOME = {"joint1": 0.0, "joint2": 1.57, "joint3": -1.3485,
              "joint4": 0.0, "joint5": 0.0, "joint6": 0.0, "joint7": 0.0}


def step_assets():
    os.makedirs(ASSETS, exist_ok=True)
    # MjSpec.from_file(go2_imu_scene) resolves the soccer texture via meshdir=assets,
    # so the Go2 menagerie assets/ needs a copy too (the top-level one isn't found there).
    shutil.copy(os.path.join(GO2_DIR, "soccer.png"),
                os.path.join(GO2_DIR, "assets", "soccer.png"))
    for f in glob.glob(os.path.join(GO2_DIR, "assets", "*")):
        shutil.copy(f, ASSETS)
    for f in glob.glob(os.path.join(PIPER_DIR, "assets", "*")):
        shutil.copy(f, ASSETS)


def _clear_keys(spec):
    for k in list(spec.keys):
        spec.delete(k)


# Passive joint springs (stiffness, damping) that hold the home pose with ZERO ctrl,
# so the robot stands however the scene is opened (viewer starts ctrl=0, not at a keyframe).
LEG_SPRING = {  # joint-suffix -> (stiffness, damping)
    "hip_joint":   (60.0, 5.0),
    "thigh_joint": (60.0, 5.0),
    "calf_joint":  (90.0, 6.0),
}
PIPER_SPRING = (15.0, 2.0)   # arm is gravity-compensated; light spring just pins it home


def _make_passive_stance(spec):
    """Make the standing+home pose the model's passive equilibrium (no ctrl needed).

    - Drop ALL actuators (this is a static view scene), so nothing fights the springs.
    - Give each actuated joint a spring whose rest (springref) is its home angle, in the
      model's ORIGINAL numbering. (Do NOT set `ref`: on a hinge that renumbers the joint
      so the as-built straight pose becomes the home value -> the spring would rest
      straight-legged.) From the as-built (straight) start the spring then bends the legs
      down into the standing stance on the first step.
    """
    for a in list(spec.actuators):
        spec.delete(a)
    for j in spec.joints:
        home = None
        for suf in LEG_SPRING:
            if j.name.endswith(suf):
                home = LEG_HOME[suf.split("_")[0]]
                k, d = LEG_SPRING[suf]
        if j.name.startswith("piper_joint"):
            jn = j.name[len("piper_"):]
            home = PIPER_HOME.get(jn)
            k, d = PIPER_SPRING
        if home is None:
            continue
        j.springref = home
        j.stiffness = k
        j.damping = max(j.damping, d)


def step_scene():
    go2 = mj.MjSpec.from_file(os.path.join(GO2_DIR, "go2_imu_scene.xml"))
    piper = mj.MjSpec.from_file(os.path.join(PIPER_DIR, "piper.xml"))
    _clear_keys(go2)            # keyframe qpos length changes once we attach -> drop both
    _clear_keys(piper)
    frame = go2.body("base").add_frame()
    frame.pos = MOUNT_POS
    frame.attach_body(piper.body("base_link"), "piper_", "")
    _make_passive_stance(go2)   # robot stands passively at home, no ctrl needed
    go2.meshdir = "assets"
    go2.texturedir = "assets"
    go2.modelname = "go2_piper_scene"
    go2.compile()
    open(SCENE_OUT, "w").write(go2.to_xml())


def _jname(m, i):
    return mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, i) or ""


def add_home_keyframe():
    """Add a combined home keyframe (order-independent, by joint name)."""
    m = mj.MjModel.from_xml_path(SCENE_OUT)
    qpos = m.qpos0.copy()
    # Go2 trunk free joint -> standing height
    for i in range(m.njnt):
        if m.jnt_type[i] == mj.mjtJoint.mjJNT_FREE and \
           mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, m.jnt_bodyid[i]) == "base":
            qpos[m.jnt_qposadr[i] + 2] = STAND_Z
    # legs + piper joints, by name
    for i in range(m.njnt):
        nm = _jname(m, i)
        adr = m.jnt_qposadr[i]
        for key, val in LEG_HOME.items():
            if nm.endswith(f"{key}_joint"):
                qpos[adr] = val
        for jn, val in PIPER_HOME.items():
            if nm == f"piper_{jn}":
                qpos[adr] = val
    # ctrl so position servos hold that pose: map actuator -> its joint target
    ctrl = np.zeros(m.nu)
    for a in range(m.nu):
        jid = m.actuator_trnid[a, 0]
        nm = _jname(m, jid)
        for key, val in LEG_HOME.items():
            if nm.endswith(f"{key}_joint"):
                ctrl[a] = val
        for jn, val in PIPER_HOME.items():
            if nm == f"piper_{jn}":
                ctrl[a] = val

    spec = mj.MjSpec.from_file(SCENE_OUT)
    k = spec.add_key()
    k.name = "home"
    k.qpos = qpos.tolist()
    k.ctrl = ctrl.tolist()
    spec.compile()
    open(SCENE_OUT, "w").write(spec.to_xml())
    return m


def main():
    step_assets()
    step_scene()
    m = add_home_keyframe()
    piper_mass = sum(m.body_mass[i] for i in range(m.nbody)
                     if (mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, i) or "").startswith("piper_"))
    print(f"OK  {SCENE_OUT}")
    print(f"    bodies={m.nbody} joints={m.njnt} actuators={m.nu}  Piper mass={piper_mass:.3f} kg")


if __name__ == "__main__":
    main()
