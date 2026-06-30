#!/usr/bin/env python3
"""Redundancy (null-space self-motion) demo for the 7-DOF arm.

This node exists ONLY to demonstrate the kinematic redundancy of the arm: a
7-DOF arm has one extra DoF beyond the 6 needed for a full EE pose, so for a
fixed EE pose there is a 1-parameter family of joint configurations ("self
motion"). This node holds the end-effector pose FIXED and continuously moves the
joints along that null space, so the elbow visibly reconfigures while the tool
frame stays put — the textbook picture of redundancy resolution.

It reuses the authoritative kinematics (ArmKinematics from fk_arm_final), so the
FK + geometric Jacobian match RViz/Gazebo exactly. The control law each tick is

    dq_p = J^+ (K_p e_pose)        # primary: pin the EE at the captured pose
    dq_n = (I - J^+ J) g           # secondary: self-motion, projected to null space
    dq   = clamp(dq_p, dq_max) + clamp(dq_n, null_step)

with a damped pseudo-inverse J^+ = J^T (J J^T + lam^2 I)^-1. The null-space
projector (I - J^+ J) guarantees the secondary term g does not move the EE
(generically the projector has rank 1 here — the single self-motion direction).
The two terms are clamped INDEPENDENTLY: the primary cap (dq_max) keeps the
EE-hold stable near singular configs, while the null cap (null_step) keeps the
self-motion visible without ever starving the EE correction. With the defaults
the EE holds to a few mm / hundredths of a degree while the joints sweep ~1 rad.

Secondary behaviours (`mode`):
  oscillate   : g = null_gain * sin(omega t) * drive   — sweeps the elbow back and
                forth along the self-motion manifold (default; best for a figure).
  center      : g = null_gain * (q_mid - q)            — drifts toward mid-limits in
                the null space (the same secondary task ik_arm_final uses).
  limit_avoid : g = null_gain * (q_mid - q) / range^2  — descends the joint-limit
                cost H = sum(((q - q_mid)/range)^2) along the null space, weighted
                by 1/range^2 so tight-range joints (e.g. joint_6) are pushed hardest.
  manip       : g = null_gain * grad(w) / w            — ascends log-manipulability
                w = sqrt(det(J J^T)), reconfiguring away from singularities.

limit_avoid and manip work best with position_only:=true (a 3-DOF task leaves a
4-D null space, giving the secondary task room to act).

Node behaviour:
  SUB  /robot_description  std_msgs/String      (latched URDF)
  SUB  /joint_states       sensor_msgs/JointState
  PUB  /joint_commands     sensor_msgs/JointState

Drive the result through the usual rigs (relay_node -> RViz, ik_to_trajectory ->
Gazebo). Watch /ee_pose (run fk_arm_final.py): position/orientation should stay
essentially constant while the joints move. The node logs the live EE drift and
the per-joint travel so you can quote "EE held to <x mm / <y deg while joints
swept z rad" directly in the thesis.
"""

import math
import numpy as np

from arm_bot.fk_arm_final import ArmKinematics, rotm2axang_vec


def _manipulability(kin, q):
    """Yoshikawa manipulability w = sqrt(det(J J^T))."""
    J, _ = kin.jacobian(np.asarray(q, float))
    return math.sqrt(max(0.0, np.linalg.det(J @ J.T)))


def _manip_grad(kin, q, eps=1e-4):
    """Central finite-difference gradient dw/dq."""
    q = np.asarray(q, float)
    g = np.zeros(kin.n)
    for i in range(kin.n):
        dq = np.zeros(kin.n); dq[i] = eps
        g[i] = (_manipulability(kin, q + dq) - _manipulability(kin, q - dq)) / (2 * eps)
    return g


def _build_node_class():
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
    from std_msgs.msg import String
    from sensor_msgs.msg import JointState

    class NullspaceDemo(Node):
        def __init__(self):
            super().__init__('nullspace_demo')
            self.declare_parameter('base_link', 'base_link')
            self.declare_parameter('tip_link', 'ee')
            self.declare_parameter('rate_hz', 100.0)
            self.declare_parameter('mode', 'oscillate')   # oscillate | center | limit_avoid | manip
            self.declare_parameter('omega', 0.5)          # sweep speed, rad/s
            self.declare_parameter('null_gain', 1.0)      # null-space drive amplitude
            self.declare_parameter('drive', [1.0] * 7)    # joint-space direction (oscillate)
            self.declare_parameter('lam', 0.05)           # DLS damping
            self.declare_parameter('pos_gain', 6.0)       # EE-hold position feedback
            self.declare_parameter('ori_gain', 3.0)       # EE-hold orientation feedback
            self.declare_parameter('dq_max', 0.03)        # primary (EE-hold) step cap, rad/tick
            self.declare_parameter('null_step', 0.01)     # null-space (self-motion) step cap, rad/tick
            self.declare_parameter('position_only', False)  # hold EE position only (3-DOF) -> 4-D null space

            gp = lambda n: self.get_parameter(n).value
            self._base = gp('base_link'); self._tip = gp('tip_link')
            self.mode = str(gp('mode'))
            self.omega = float(gp('omega'))
            self.null_gain = float(gp('null_gain'))
            self.drive = np.array(gp('drive'), float)
            self.lam = float(gp('lam'))
            self.pos_gain = float(gp('pos_gain'))
            self.ori_gain = float(gp('ori_gain'))
            self.dq_max = float(gp('dq_max'))
            self.null_step = float(gp('null_step'))
            self.position_only = bool(gp('position_only'))

            self.kin = None
            self.idx = None
            self.q_fb = None
            self.q_cmd = None        # internally integrated command configuration
            self.T_des = None        # EE pose to hold (captured on first feedback)
            self.q_lo = None
            self.q_hi = None         # running per-joint travel envelope (for logging)
            self.t0 = None
            self._log_t = 0.0

            latched = QoSProfile(depth=1,
                                 durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                                 reliability=QoSReliabilityPolicy.RELIABLE)
            self.create_subscription(String, '/robot_description', self._cb_urdf, latched)
            self.create_subscription(JointState, '/joint_states', self._cb_js, 30)
            self.pub = self.create_publisher(JointState, '/joint_commands', 20)
            self.create_timer(1.0 / float(gp('rate_hz')), self._tick)
            self.get_logger().info(
                f'nullspace_demo waiting for /robot_description ({self._base} -> '
                f'{self._tip}); mode={self.mode} omega={self.omega} '
                f'null_gain={self.null_gain}')

        def _cb_urdf(self, msg):
            if self.kin is not None:
                return
            try:
                self.kin = ArmKinematics.from_urdf(msg.data, self._base, self._tip)
            except Exception as e:
                self.get_logger().error(f'URDF parse failed: {e}')
                return
            if self.drive.shape[0] != self.kin.n:
                self.drive = np.ones(self.kin.n)
            self.get_logger().info(
                f'URDF loaded: {self.kin.n} DoF — joints {self.kin.joint_names}')

        def _cb_js(self, msg):
            if self.kin is None:
                return
            if self.idx is None:
                try:
                    self.idx = [msg.name.index(n) for n in self.kin.joint_names]
                except ValueError:
                    return
            self.q_fb = np.array([msg.position[i] for i in self.idx], float)

        def _now(self):
            return self.get_clock().now().nanoseconds * 1e-9

        def _tick(self):
            if self.kin is None or self.q_fb is None:
                return
            # On the first valid tick, latch the current EE pose as the one to hold
            # and seed the integrated command from the live feedback.
            if self.q_cmd is None:
                self.q_cmd = self.q_fb.copy()
                self.T_des = self.kin.fk(self.q_cmd)
                self.q_lo = self.q_cmd.copy()
                self.q_hi = self.q_cmd.copy()
                self.t0 = self._now()
                self.get_logger().info('captured EE pose to hold; starting self-motion')

            q = self.q_cmd
            J, T = self.kin.jacobian(q)

            # Primary task: pin the captured EE pose (6-DOF), or just position
            # (3-DOF) when position_only -> a 4-D null space for the secondary task.
            e_pos = self.pos_gain * (self.T_des[:3, 3] - T[:3, 3])
            if self.position_only:
                Ju = J[:3, :]
                e = e_pos
            else:
                e_rot = self.ori_gain * rotm2axang_vec(self.T_des[:3, :3] @ T[:3, :3].T)
                Ju = J
                e = np.concatenate([e_pos, e_rot])

            JuT = Ju.T
            M = Ju @ JuT + (self.lam ** 2) * np.eye(Ju.shape[0])
            Jpinv = JuT @ np.linalg.inv(M)
            dq_primary = Jpinv @ e
            pn = float(np.linalg.norm(dq_primary))
            if pn > self.dq_max:                       # cap primary -> EE-hold stays stable
                dq_primary *= self.dq_max / pn

            # Secondary task: self-motion, projected into the null space so the EE
            # does not move.
            N = np.eye(self.kin.n) - Jpinv @ Ju
            if self.mode == 'center':
                g = self.null_gain * (self.kin.q_mid - q)
            elif self.mode == 'limit_avoid':
                rng = self.kin.q_max - self.kin.q_min
                g = self.null_gain * (self.kin.q_mid - q) / (rng * rng)
            elif self.mode == 'manip':
                w = _manipulability(self.kin, q)         # ascend log-manipulability
                g = self.null_gain * _manip_grad(self.kin, q) / (w + 1e-9)
            else:  # 'oscillate'
                t = self._now() - self.t0
                g = self.null_gain * math.sin(self.omega * t) * self.drive
            dq_null = N @ g
            nn = float(np.linalg.norm(dq_null))
            if nn > self.null_step:                    # cap null independently -> motion stays visible
                dq_null *= self.null_step / nn

            self.q_cmd = np.clip(q + dq_primary + dq_null, self.kin.q_min, self.kin.q_max)
            self.q_lo = np.minimum(self.q_lo, self.q_cmd)
            self.q_hi = np.maximum(self.q_hi, self.q_cmd)

            cmd = JointState()
            cmd.header.stamp = self.get_clock().now().to_msg()
            cmd.name = list(self.kin.joint_names)
            cmd.position = [float(v) for v in self.q_cmd]
            self.pub.publish(cmd)

            # Periodically report EE hold quality + accumulated joint self-motion.
            now = self._now()
            if now - self._log_t >= 1.0:
                self._log_t = now
                T_now = self.kin.fk(self.q_cmd)
                drift_pos = np.linalg.norm(T_now[:3, 3] - self.T_des[:3, 3])
                drift_ang = np.linalg.norm(
                    rotm2axang_vec(self.T_des[:3, :3] @ T_now[:3, :3].T))
                travel = float(np.max(self.q_hi - self.q_lo))
                self.get_logger().info(
                    f'EE drift: {drift_pos*1000:.4f} mm / '
                    f'{math.degrees(drift_ang):.4f} deg | '
                    f'max joint self-motion: {travel:.3f} rad')

    return NullspaceDemo


def main():
    import rclpy
    rclpy.init()
    node = _build_node_class()()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
