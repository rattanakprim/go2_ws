"""
Velocity-driven gaits + body-pose control for the Go2.

joint_targets() maps a body twist (vx, vy, wz) AND an optional body pose
(roll, pitch, yaw, ride-height) into 12 target joint angles.

Gait presets (leg phase patterns):
  crawl  - one foot at a time, 3 always down  -> statically stable (default)
  trot   - diagonal pairs together            -> dynamic (experimental)
  pace   - same-side pairs together           -> dynamic (experimental)
  bound  - front pair / rear pair             -> dynamic (experimental)
  pronk  - all four together                  -> dynamic (experimental)
Only `crawl` is reliable with this open-loop controller; the others are
provided as selectable patterns but may stumble without balance feedback.
"""
import math
import numpy as np

from .kinematics import (
    inverse_kinematics, D_SIGN, HIP_XY, HIP_MOUNT, HIP_D, LEG_NAMES,
)

# name -> (phase offset per leg, duty factor)
GAITS = {
    "crawl": ({"FL": 0.0, "RR": 0.25, "FR": 0.5, "RL": 0.75}, 0.75),
    "trot":  ({"FL": 0.0, "RR": 0.0, "FR": 0.5, "RL": 0.5}, 0.5),
    "pace":  ({"FL": 0.0, "RL": 0.0, "FR": 0.5, "RR": 0.5}, 0.5),
    "bound": ({"FL": 0.0, "FR": 0.0, "RL": 0.5, "RR": 0.5}, 0.5),
    "pronk": ({"FL": 0.0, "FR": 0.0, "RL": 0.0, "RR": 0.0}, 0.5),
}


def _body_rot_T(roll, pitch, yaw):
    """Transpose of the body rotation matrix R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return (Rz @ Ry @ Rx).T


class TrotGait:
    def __init__(self, period=0.4, duty=0.75, stand_height=0.27,
                 step_height=0.08, max_step=0.25, lift_full_step=0.03,
                 stride_gain=1.4):
        self.period = period
        self.duty = duty
        self.stand_height = stand_height
        self.step_height = step_height
        self.max_step = max_step
        self.lift_full_step = lift_full_step
        self.stride_gain = stride_gain
        self.offsets = dict(GAITS["crawl"][0])
        self.set_gait("crawl")

    def set_gait(self, name):
        if name not in GAITS:
            return False
        self.offsets, self.duty = dict(GAITS[name][0]), GAITS[name][1]
        return True

    def _foot_offset(self, phase, step_x, step_y, lift_scale):
        beta = self.duty
        if phase < beta:                       # stance: foot moves backward
            frac = phase / beta
            dx, dy, dz = step_x * (0.5 - frac), step_y * (0.5 - frac), 0.0
        else:                                  # swing: lift & swing forward
            frac = (phase - beta) / (1.0 - beta)
            dx = step_x * (-0.5 + frac)
            dy = step_y * (-0.5 + frac)
            dz = self.step_height * lift_scale * math.sin(math.pi * frac)
        return dx, dy, dz

    def joint_targets(self, vx, vy, wz, t,
                      roll=0.0, pitch=0.0, yaw=0.0, height=None,
                      tx=0.0, ty=0.0):
        """12 joint angles for the current twist + body pose.

        tx, ty shift the body's center of mass forward/left (m) by moving every
        foot the other way -- used to balance over 3 legs for expressive moves.
        """
        H = self.stand_height if height is None else height
        stance_time = self.duty * self.period
        Rt = _body_rot_T(roll, pitch, yaw)
        targets = []
        for leg in LEG_NAMES:
            mx, my = HIP_MOUNT[leg]
            # Ground contact point (relative to body center, world) -> body frame
            p0 = np.array([mx, my + D_SIGN[leg] * HIP_D, -H])
            fb = Rt @ p0
            # express in the hip frame (origin at the abduction joint), w/ CoM shift
            base_x, base_y, base_z = fb[0] - mx - tx, fb[1] - my - ty, fb[2]

            # gait step from the commanded twist
            rx, ry = HIP_XY[leg]
            step_x = (vx - wz * ry) * stance_time * self.stride_gain
            step_y = (vy + wz * rx) * stance_time * self.stride_gain
            mag = math.hypot(step_x, step_y)
            if mag > self.max_step:
                step_x *= self.max_step / mag
                step_y *= self.max_step / mag
                mag = self.max_step
            lift_scale = min(1.0, mag / self.lift_full_step)

            phase = (t / self.period + self.offsets[leg]) % 1.0
            dx, dy, dz = self._foot_offset(phase, step_x, step_y, lift_scale)

            q = inverse_kinematics(base_x + dx, base_y + dy, base_z + dz,
                                   D_SIGN[leg])
            targets.extend(q)
        return targets

    def stand_pose(self):
        return self.joint_targets(0.0, 0.0, 0.0, 0.0)

    def _straight_targets(self, heights):
        """Feet straight under their hips at the given height(s) (float or 4-list)."""
        targets = []
        for i, leg in enumerate(LEG_NAMES):
            h = heights[i] if isinstance(heights, (list, tuple)) else heights
            d = D_SIGN[leg] * HIP_D
            targets.extend(inverse_kinematics(0.0, d, -h, D_SIGN[leg]))
        return targets

    def stand_pose_at(self, height):
        return self._straight_targets(height)

    def sit_pose(self):
        """Sit: front tall, rear folded low."""
        return self._straight_targets([0.27, 0.27, 0.12, 0.12])

    def crouch_pose(self):
        return self._straight_targets(0.17)

    def liedown_pose(self):
        """Belly down: all legs folded so the body rests near the ground."""
        return self._straight_targets(0.09)
