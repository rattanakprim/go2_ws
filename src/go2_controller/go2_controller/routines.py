"""
Timed scripted routines for the Go2 (expressive moves + recovery).

Each routine is a function(gait, te) -> (targets12, kp, kd, done):
  gait : the TrotGait instance (for IK + poses)
  te   : seconds elapsed since the routine started
returns the 12 joint targets, the PD gains to use, and whether it's finished.
The controller runs the active routine each physics step until done, then
returns to the previous posture.
"""
import math

from .kinematics import inverse_kinematics, D_SIGN, HIP_D

# index of each leg's first joint in the 12-vector (FL, FR, RL, RR)
LEG_SLOT = {"FL": 0, "FR": 3, "RL": 6, "RR": 9}


def _set_leg(targets, leg, q):
    s = LEG_SLOT[leg]
    targets[s:s + 3] = list(q)


def _foot_ik(leg, x, y, z):
    return inverse_kinematics(x, y, z, D_SIGN[leg])


def wave(gait, te):
    """Lift the front-right paw and wave it side to side (~3 s)."""
    # shift weight back-left so the FL/RL/RR triangle supports the body
    targets = list(gait.joint_targets(0, 0, 0, 0, tx=-0.07, ty=0.03))
    d = D_SIGN["FR"] * HIP_D
    sway = 0.06 * math.sin(2 * math.pi * 2.0 * te)        # paw side-to-side
    _set_leg(targets, "FR", _foot_ik("FR", 0.12, d + sway, -0.10))
    return targets, 300.0, 7.0, te > 3.0


def shake(gait, te):
    """Offer the front-right paw forward with a small up/down shake (~3 s)."""
    targets = list(gait.joint_targets(0, 0, 0, 0, tx=-0.07, ty=0.03))
    d = D_SIGN["FR"] * HIP_D
    bob = 0.03 * math.sin(2 * math.pi * 1.5 * te)
    _set_leg(targets, "FR", _foot_ik("FR", 0.17, d, -0.12 + bob))
    return targets, 300.0, 7.0, te > 3.0


def dance(gait, te):
    """Bob up/down and sway side to side on all four feet (~5 s)."""
    roll = 0.18 * math.sin(2 * math.pi * 1.0 * te)
    height = 0.27 + 0.05 * math.sin(2 * math.pi * 2.0 * te)
    targets = gait.joint_targets(0, 0, 0, 0, roll=roll, height=height)
    return targets, 300.0, 7.0, te > 5.0


def selfright(gait, te):
    """Best-effort self-right: tuck, then sweep legs to roll the body upright.

    Open-loop and torque-limited, so this is not guaranteed to flip a fully
    inverted robot -- it's a best-effort recovery attempt.
    """
    # phase 1: tuck every leg to the left side to break symmetry
    if te < 0.5:
        targets = []
        for leg in ("FL", "FR", "RL", "RR"):
            # abduction hard to +, knee fully folded
            targets += [1.0, 0.4, -2.6]
        return targets, 200.0, 4.0, False
    # phase 2: sweep abduction the other way + extend to push and roll over
    if te < 1.6:
        s = (te - 0.5) / 1.1
        ab = 1.0 - 2.0 * s            # +1.0 -> -1.0 sweep
        targets = []
        for leg in ("FL", "FR", "RL", "RR"):
            targets += [ab, 0.6, -1.2]
        return targets, 400.0, 3.0, False
    # phase 3: settle into a stand
    return gait.stand_pose(), 300.0, 7.0, te > 2.4


def jump(gait, te):
    """Crouch to load, then explosively extend the legs (~5 cm hop)."""
    JC_H, JC_T, JP_H, JP_T = 0.12, 0.22, 0.43, 0.06
    if te < JC_T:                                  # crouch / load
        h = 0.27 + (JC_H - 0.27) * (te / JC_T)
        return gait.stand_pose_at(h), 300.0, 7.0, False
    if te < JC_T + JP_T:                           # explosive extension
        h = JC_H + (JP_H - JC_H) * ((te - JC_T) / JP_T)
        return gait.stand_pose_at(h), 1500.0, 1.0, False
    return gait.stand_pose(), 300.0, 7.0, te > JC_T + JP_T + 0.4


ROUTINES = {
    "jump": jump,
    "wave": wave,
    "shake": shake,
    "dance": dance,
    "selfright": selfright,
}
