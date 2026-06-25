"""Headless validation: IK correctness + does the robot actually walk?"""
import os
import math
import numpy as np

from go2_controller.kinematics import (
    forward_kinematics, inverse_kinematics, D_SIGN, LEG_NAMES, HIP_D,
)
from go2_controller.gait import TrotGait
from go2_controller.simulator import Go2Sim

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(HERE, "models", "mujoco_menagerie", "unitree_go2",
                   "go2_imu_scene.xml")


def test_ik_roundtrip():
    print("== IK round-trip (FK -> IK -> FK) ==")
    worst = 0.0
    for leg in LEG_NAMES:
        ds = D_SIGN[leg]
        for q1, q2, q3 in [(0, 0.9, -1.8), (0.2, 1.1, -2.0), (-0.2, 0.6, -1.4)]:
            x, y, z = forward_kinematics(q1, q2, q3, ds)
            a1, a2, a3 = inverse_kinematics(x, y, z, ds)
            x2, y2, z2 = forward_kinematics(a1, a2, a3, ds)
            err = max(abs(x - x2), abs(y - y2), abs(z - z2))
            worst = max(worst, err)
    print(f"   worst foot error: {worst*1000:.3f} mm")
    assert worst < 1e-6, "IK/FK inconsistent!"
    print("   OK")


def run(vx, vy, wz, seconds=3.0, label=""):
    sim = Go2Sim(XML)
    gait = TrotGait()
    sim.set_target(gait.stand_pose())
    # settle for 0.3 s
    for _ in range(int(0.3 / sim.dt)):
        sim.step()
    p0, q0 = sim.base_pose()
    yaw0 = quat_yaw(q0)
    steps = int(seconds / sim.dt)
    t = 0.0
    for _ in range(steps):
        sim.set_target(gait.joint_targets(vx, vy, wz, t))
        sim.step()
        t += sim.dt
    p1, q1 = sim.base_pose()
    yaw1 = quat_yaw(q1)
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    dyaw = math.atan2(math.sin(yaw1 - yaw0), math.cos(yaw1 - yaw0))
    print(f"== {label} (vx={vx}, vy={vy}, wz={wz}) over {seconds}s ==")
    print(f"   moved dx={dx:+.3f} m  dy={dy:+.3f} m  dyaw={math.degrees(dyaw):+.1f} deg"
          f"  final height={p1[2]:.3f} m")
    return dx, dy, dyaw, p1[2]


def quat_yaw(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def run_all():
    test_ik_roundtrip()
    print()
    _, _, _, h = run(0.0, 0.0, 0.0, 2.0, "stand still")
    assert h > 0.20, "robot collapsed while standing"
    dx, _, _, h = run(0.3, 0.0, 0.0, 4.0, "walk forward")
    assert h > 0.18, "robot fell while walking"
    assert dx > 0.3, f"did not walk forward enough (dx={dx:.3f})"
    dxb, _, _, _ = run(-0.3, 0.0, 0.0, 4.0, "walk backward")
    assert dxb < -0.3, f"did not walk backward enough (dx={dxb:.3f})"
    _, _, dyaw, _ = run(0.0, 0.0, 0.8, 4.0, "turn left (yaw)")
    assert dyaw > 0.5, f"did not turn enough (dyaw={dyaw:.2f})"
    _, dy, _, _ = run(0.0, 0.2, 0.0, 4.0, "strafe left")
    assert dy > 0.2, f"did not strafe enough (dy={dy:.2f})"

    test_imu()
    test_sit()
    test_jump()
    print("\nALL CHECKS PASSED")


def test_imu():
    print("== IMU ==")
    sim = Go2Sim(XML)
    for _ in range(300):
        sim.step()
    imu = sim.imu()
    assert imu is not None, "no IMU sensors in model"
    quat, gyro, acc = imu
    print(f"   quat={np.round(quat,2)} gyro={np.round(gyro,2)} acc={np.round(acc,2)}")
    assert acc[2] > 8.0, f"accelerometer z should read ~gravity, got {acc[2]:.2f}"
    print("   OK")


def test_sit():
    print("== sit (rear should drop below front) ==")
    sim = Go2Sim(XML)
    gait = TrotGait()
    for _ in range(int(2.0 / sim.dt)):
        sim.set_target(gait.sit_pose())
        sim.step()
    # compare front vs rear foot/hip via pitch: sitting pitches nose up
    _, q = sim.base_pose()
    w, x, y, z = q
    pitch = math.degrees(math.asin(max(-1, min(1, 2 * (w * y - z * x)))))
    print(f"   body pitch while sitting: {pitch:+.1f} deg (nose up = positive-ish)")
    assert abs(pitch) > 8.0, "sit did not visibly tilt the body"
    print("   OK")


def test_jump():
    print("== jump (base should rise then land) ==")
    sim = Go2Sim(XML)
    gait = TrotGait()
    sim.set_target(gait.stand_pose())
    for _ in range(int(0.4 / sim.dt)):
        sim.step()
    base0 = sim.base_pose()[0][2]
    peak = base0
    t = 0.0
    JC_H, JC_T, JP_H, JP_T = 0.12, 0.22, 0.43, 0.06
    for _ in range(int(1.2 / sim.dt)):
        if t < JC_T:
            sim.set_gains(300, 7)
            h = base0 + (JC_H - base0) * (t / JC_T)
            sim.set_target(gait.stand_pose_at(h))
        elif t < JC_T + JP_T:
            sim.set_gains(1500, 1.0)
            h = JC_H + (JP_H - JC_H) * ((t - JC_T) / JP_T)
            sim.set_target(gait.stand_pose_at(h))
        else:
            sim.set_gains(300, 7)
            sim.set_target(gait.stand_pose())
        sim.step()
        t += sim.dt
        peak = max(peak, sim.base_pose()[0][2])
    final = sim.base_pose()[0][2]
    print(f"   base0={base0:.3f} peak={peak:.3f} rise={peak-base0:+.3f} final={final:.3f}")
    assert peak - base0 > 0.03, "jump produced no rise"
    assert final > 0.20, "did not land standing"
    print("   OK")


if __name__ == "__main__":
    run_all()
