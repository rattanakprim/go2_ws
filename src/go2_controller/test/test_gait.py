"""Unit tests for go2_controller.gait (twist -> joint targets, no ROS/MuJoCo)."""
import math

import pytest

from go2_controller.gait import TrotGait, GAITS
from go2_controller.kinematics import LEG_NAMES


def test_gait_presets_wellformed():
    """Every preset covers all four legs with valid phase offsets and duty."""
    for name, (offsets, duty) in GAITS.items():
        assert set(offsets) == set(LEG_NAMES), f"{name} missing legs"
        for off in offsets.values():
            assert 0.0 <= off < 1.0
        assert 0.0 < duty <= 1.0


def test_set_gait_valid_and_invalid():
    g = TrotGait()
    assert g.set_gait("trot") is True
    assert g.duty == GAITS["trot"][1]
    # an unknown gait is rejected and leaves state untouched
    prev_offsets, prev_duty = dict(g.offsets), g.duty
    assert g.set_gait("does_not_exist") is False
    assert g.offsets == prev_offsets and g.duty == prev_duty


def test_joint_targets_returns_twelve_angles():
    g = TrotGait()
    targets = g.joint_targets(0.2, 0.0, 0.0, t=0.0)
    assert len(targets) == 12
    assert all(math.isfinite(v) for v in targets)


def test_stand_pose_matches_zero_command():
    g = TrotGait()
    assert g.stand_pose() == g.joint_targets(0.0, 0.0, 0.0, 0.0)


def test_zero_twist_is_static_over_time():
    """With no commanded velocity the pose must not change as time advances."""
    g = TrotGait()
    a = g.joint_targets(0.0, 0.0, 0.0, t=0.0)
    b = g.joint_targets(0.0, 0.0, 0.0, t=g.period * 0.5)
    for x, y in zip(a, b):
        assert x == pytest.approx(y, abs=1e-9)


def test_forward_command_animates_legs():
    """A forward velocity must move the joints between two points in the cycle."""
    g = TrotGait()
    a = g.joint_targets(0.3, 0.0, 0.0, t=0.0)
    b = g.joint_targets(0.3, 0.0, 0.0, t=g.period * 0.5)
    assert any(abs(x - y) > 1e-3 for x, y in zip(a, b))


def test_foot_offset_stance_is_grounded_swing_lifts():
    """Stance keeps the foot on the ground (dz == 0); swing lifts it (dz >= 0)."""
    g = TrotGait()
    # mid-stance (phase < duty)
    _, _, dz_stance = g._foot_offset(g.duty * 0.5, 0.1, 0.0, lift_scale=1.0)
    assert dz_stance == pytest.approx(0.0, abs=1e-12)
    # mid-swing (phase > duty)
    swing_phase = g.duty + (1.0 - g.duty) * 0.5
    _, _, dz_swing = g._foot_offset(swing_phase, 0.1, 0.0, lift_scale=1.0)
    assert dz_swing >= 0.0
    assert dz_swing == pytest.approx(g.step_height, abs=1e-9)
