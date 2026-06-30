"""Unit tests for go2_controller.kinematics (pure leg geometry, no ROS/MuJoCo)."""
import math

import pytest

from go2_controller.kinematics import (
    forward_kinematics, inverse_kinematics, D_SIGN, LEG_NAMES, HIP_D, L1, L2,
)


def test_home_pose_foot_straight_down():
    """q = (0, 0, 0): foot hangs straight down at full leg length, y = lateral offset."""
    for leg in LEG_NAMES:
        d = D_SIGN[leg] * HIP_D
        x, y, z = forward_kinematics(0.0, 0.0, 0.0, D_SIGN[leg])
        assert x == pytest.approx(0.0, abs=1e-9)
        assert y == pytest.approx(d, abs=1e-9)
        assert z == pytest.approx(-(L1 + L2), abs=1e-9)


@pytest.mark.parametrize("leg", LEG_NAMES)
@pytest.mark.parametrize("q1", [-0.3, 0.0, 0.25])
@pytest.mark.parametrize("q2", [-0.2, 0.3, 0.6])
@pytest.mark.parametrize("q3", [-1.4, -0.9, -0.4])
def test_fk_ik_round_trip(leg, q1, q2, q3):
    """IK must recover the foot position produced by FK (knee-backward branch)."""
    d_sign = D_SIGN[leg]
    foot = forward_kinematics(q1, q2, q3, d_sign)
    q1r, q2r, q3r = inverse_kinematics(*foot, d_sign)
    foot_again = forward_kinematics(q1r, q2r, q3r, d_sign)
    for got, want in zip(foot_again, foot):
        assert got == pytest.approx(want, abs=1e-6)


@pytest.mark.parametrize("leg", LEG_NAMES)
def test_ik_knee_bends_backward(leg):
    """The knee solution is always the backward-bending one (q3 <= 0)."""
    _, _, q3 = inverse_kinematics(0.05, D_SIGN[leg] * HIP_D, -0.27, D_SIGN[leg])
    assert q3 <= 1e-9


def test_ik_clamps_unreachable_target():
    """A target beyond full leg extension must not raise (acos arg is clamped)."""
    reach = L1 + L2 + 0.5  # well past the workspace
    q1, q2, q3 = inverse_kinematics(0.0, HIP_D, -reach, D_SIGN["FL"])
    assert all(math.isfinite(v) for v in (q1, q2, q3))
