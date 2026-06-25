"""
Leg kinematics for the Unitree Go2.

Each leg is a 3-DOF chain in its hip frame (x forward, y left, z up):
    q1 = hip abduction  (rotation about x)
    q2 = thigh / hip     (rotation about y)
    q3 = calf / knee     (rotation about y)

Geometry comes straight from the MuJoCo model (mujoco_menagerie/unitree_go2):
    thigh lateral offset HIP_D = 0.0955 m
    thigh length  L1 = 0.213 m
    calf length   L2 = 0.213 m

Left legs (FL, RL) have d_sign = +1, right legs (FR, RR) have d_sign = -1
(their hip-to-thigh offset points the other way along y).
"""
import math

HIP_D = 0.0955   # hip -> thigh lateral offset (m)
L1 = 0.213       # thigh length (m)
L2 = 0.213       # calf length (m)

# Order matches the model's actuators: FL, FR, RL, RR.
LEG_NAMES = ("FL", "FR", "RL", "RR")
D_SIGN = {"FL": +1.0, "FR": -1.0, "RL": +1.0, "RR": -1.0}

# Nominal foot positions in the BODY frame (x fwd, y left), ~under each hip.
# Used to map body twist (vx, vy, wz) into per-leg foot motion.
HIP_XY = {
    "FL": (0.1934,  0.0465 + HIP_D),
    "FR": (0.1934, -(0.0465 + HIP_D)),
    "RL": (-0.1934,  0.0465 + HIP_D),
    "RR": (-0.1934, -(0.0465 + HIP_D)),
}

# Abduction-joint (hip) mount positions in the BODY frame, from the model's
# FL_hip/FR_hip/... body offsets. Used for body-lean (pitch/roll/yaw) IK.
HIP_MOUNT = {
    "FL": (0.1934,  0.0465),
    "FR": (0.1934, -0.0465),
    "RL": (-0.1934,  0.0465),
    "RR": (-0.1934, -0.0465),
}


def forward_kinematics(q1, q2, q3, d_sign):
    """Foot position (x, y, z) in the hip frame given the three joint angles."""
    d = d_sign * HIP_D
    vx = -(L1 * math.sin(q2) + L2 * math.sin(q2 + q3))
    vz = -(L1 * math.cos(q2) + L2 * math.cos(q2 + q3))
    x = vx
    y = d * math.cos(q1) - vz * math.sin(q1)
    z = d * math.sin(q1) + vz * math.cos(q1)
    return x, y, z


def inverse_kinematics(x, y, z, d_sign):
    """Joint angles (q1, q2, q3) that place the foot at (x, y, z) in the hip frame.

    Knee is chosen to bend backward (q3 < 0), matching the model's home pose.
    """
    d = d_sign * HIP_D

    # --- abduction (q1) and the in-plane vertical reach (vz) ---
    R = math.hypot(y, z)
    vz = -math.sqrt(max(R * R - d * d, 1e-9))      # downward, negative
    q1 = math.atan2(z, y) - math.atan2(vz, d)

    # --- sagittal 2-link (thigh + calf) ---
    X = -x        # = L1 sin q2 + L2 sin(q2+q3)
    Z = -vz       # = L1 cos q2 + L2 cos(q2+q3), positive
    r2 = X * X + Z * Z
    c3 = (r2 - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
    c3 = max(-1.0, min(1.0, c3))
    q3 = -math.acos(c3)                            # bend knee backward
    q2 = math.atan2(X, Z) - math.atan2(L2 * math.sin(q3), L1 + L2 * math.cos(q3))
    return q1, q2, q3
