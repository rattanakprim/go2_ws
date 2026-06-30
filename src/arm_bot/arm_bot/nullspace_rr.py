#!/usr/bin/env python3
"""Velocity-level null-space redundancy resolution for the 7-DOF arm.

Extends the existing DLS IK (ik_arm_final) with a resolved-rate law:

    q_dot = J_dpinv @ x_dot  +  (I - J_pinv @ J) @ (k * grad_H)

Design (matches the thesis requirements):
  - Task term uses the DAMPED pseudoinverse  J_dpinv = J^T (J J^T + lam^2 I)^-1
    with the same lambda as ik_arm_final (default 0.05).
  - The null-space projector uses the EXACT (Moore-Penrose) pseudoinverse
    J_pinv = pinv(J), i.e. N = I - J_pinv J. Building N from the damped inverse
    would leak secondary motion into the task (~lambda^2); the exact projector
    keeps J @ (N @ anything) ~ 1e-15.
  - The secondary objective grad_H is pluggable. Two are provided, each returning
    the gradient of a quantity to be MAXIMISED (ascended with +k):
      'limit' : Liegeois range-centering   grad_H = (q_mid - q) / range^2
                (weighted by 1/range^2 so tight-range joints, e.g. joint_6, dominate)
      'manip' : manipulability             grad_H = d/dq sqrt(det(J J^T))  / w
                (central finite difference; divided by w to ascend log-manipulability,
                 which is well-conditioned since w ~ 1e-3 on this arm)

FK and the 6xN geometric Jacobian come from ArmKinematics (fk_arm_final) — the
single source of truth. This module imports them; it does not reimplement them.
Joint limits are read from ArmKinematics (i.e. from the URDF), not hard-coded.
"""
import math
import numpy as np

from arm_bot.fk_arm_final import ArmKinematics, rotm2axang_vec


# ─────────────────────────────────────────────────────────────────────────────
# Core linear algebra
# ─────────────────────────────────────────────────────────────────────────────
def damped_pinv(J, lam):
    """Damped (DLS) pseudoinverse  J^T (J J^T + lam^2 I)^-1  of a 6xN Jacobian."""
    m = J.shape[0]
    return J.T @ np.linalg.inv(J @ J.T + (lam ** 2) * np.eye(m))


def null_projector(J):
    """EXACT null-space projector  N = I - J_pinv J  (Moore-Penrose pseudoinverse)."""
    n = J.shape[1]
    return np.eye(n) - np.linalg.pinv(J) @ J


def manipulability(J):
    """Yoshikawa manipulability  w = sqrt(det(J J^T))."""
    return math.sqrt(max(0.0, np.linalg.det(J @ J.T)))


# ─────────────────────────────────────────────────────────────────────────────
# Pluggable secondary objectives — return grad_H (ascended with +k)
# ─────────────────────────────────────────────────────────────────────────────
def grad_limit(kin, q):
    """Liegeois range-centering. Maximises H = -sum(((q-q_mid)/range)^2), so
    grad_H = (q_mid - q)/range^2 (pulls each joint toward mid-range, weighted by
    1/range^2 so the tight-range joints are pushed hardest)."""
    rng = kin.q_max - kin.q_min
    return (kin.q_mid - q) / (rng * rng)


def grad_manip(kin, q, eps=1e-4):
    """Gradient of manipulability via central finite difference, divided by w
    (i.e. grad of log w) for good conditioning."""
    g = np.zeros(kin.n)
    for i in range(kin.n):
        dq = np.zeros(kin.n); dq[i] = eps
        wp = manipulability(kin.jacobian(q + dq)[0])
        wm = manipulability(kin.jacobian(q - dq)[0])
        g[i] = (wp - wm) / (2.0 * eps)
    w = manipulability(kin.jacobian(q)[0])
    return g / (w + 1e-9)


OBJECTIVES = {'none': None, 'limit': grad_limit, 'manip': grad_manip}


# ─────────────────────────────────────────────────────────────────────────────
# Resolved-rate step
# ─────────────────────────────────────────────────────────────────────────────
def resolved_rate(kin, q, x_dot, *, lam=0.05, objective='none', k=0.0,
                  position_only=False):
    """One velocity-level step.

    q       : current joint vector (N,)
    x_dot   : task velocity — 6-vector [v; omega], or 3-vector [v] if position_only
    lam     : DLS damping for the task term
    objective, k : null-space secondary task and its gain
    position_only : track EE position only (3-DOF task) -> 4-D null space
    Returns (q_dot, info) where info carries J, T, and the task/null split.
    """
    J, T = kin.jacobian(q)
    Ju = J[:3, :] if position_only else J          # task Jacobian (3xN or 6xN)
    q_dot_task = damped_pinv(Ju, lam) @ x_dot
    if objective != 'none' and k != 0.0:
        N = null_projector(Ju)                     # exact projector for THIS task
        q_dot_null = N @ (k * OBJECTIVES[objective](kin, q))
    else:
        q_dot_null = np.zeros(kin.n)
    return q_dot_task + q_dot_null, {
        'J': J, 'T': T, 'q_dot_task': q_dot_task, 'q_dot_null': q_dot_null,
        'manip': manipulability(J)}


def limit_cost(kin, q):
    """Liegeois joint-limit cost H = sum(((q - q_mid)/range)^2) (0 = all mid-range)."""
    z = (np.asarray(q, float) - kin.q_mid) / (kin.q_max - kin.q_min)
    return float(np.sum(z ** 2))


def circle_target(center, radius, period, t, plane='yz'):
    """Desired EE position on a circle (and its velocity feed-forward) at time t."""
    w = 2.0 * math.pi / period
    c, s = math.cos(w * t), math.sin(w * t)
    axes = {'yz': (1, 2), 'xy': (0, 1), 'xz': (0, 2)}[plane]
    p = np.array(center, float); v = np.zeros(3)
    p[axes[0]] += radius * c;  v[axes[0]] = -radius * w * s
    p[axes[1]] += radius * s;  v[axes[1]] =  radius * w * c
    return p, v


def _square_segments(a, rc):
    """Counter-clockwise rounded-square perimeter, starting at the bottom tangent
    point of the right edge (a, -e). Returns a list of segments:
      ('line', length, p0, unit_dir)   — straight edge
      ('arc',  length, centre, theta0) — quarter-circle corner (CCW, radius rc)
    e = a - rc is the half-length of the straight part of each edge."""
    e = a - rc
    q = 0.5 * math.pi * rc
    L = 2.0 * e
    return [
        ('line', L, (a, -e), (0.0, 1.0)),                 # right edge, going +b
        ('arc',  q, (e, e),  0.0),                        # top-right corner
        ('line', L, (e, a),  (-1.0, 0.0)),                # top edge, going -a
        ('arc',  q, (-e, e), 0.5 * math.pi),              # top-left corner
        ('line', L, (-a, e), (0.0, -1.0)),                # left edge, going -b
        ('arc',  q, (-e, -e), math.pi),                   # bottom-left corner
        ('line', L, (-e, -a), (1.0, 0.0)),                # bottom edge, going +a
        ('arc',  q, (e, -e), 1.5 * math.pi),              # bottom-right corner
    ]


def square_target(center, radius, period, t, plane='yz', corner=0.3):
    """Desired EE position + velocity feed-forward tracing a rounded square.

    radius = half side-length [m];  corner = corner radius as a fraction of radius.
    The perimeter is walked at CONSTANT speed (arc-length parametrised), so the
    velocity feed-forward is smooth and bounded everywhere, including through the
    rounded corners — a superellipse would spike to infinite tangential speed at
    the corners. Corners are C1-continuous (tangent matches across edge/arc joins).
    """
    a = float(radius)
    rc = max(1e-6, min(float(corner), 0.999)) * a        # corner radius
    segs = _square_segments(a, rc)
    P = sum(seg[1] for seg in segs)                       # total perimeter
    speed = P / period
    s = (speed * t) % P

    acc = 0.0
    pos2, tan2 = (a, -rc), (0.0, 1.0)                     # fallback = segment start
    for seg in segs:
        kind, length = seg[0], seg[1]
        if s <= acc + length or seg is segs[-1]:
            ls = s - acc
            if kind == 'line':
                p0, d = seg[2], seg[3]
                pos2 = (p0[0] + d[0] * ls, p0[1] + d[1] * ls)
                tan2 = d
            else:
                c, th0 = seg[2], seg[3]
                th = th0 + ls / rc
                pos2 = (c[0] + rc * math.cos(th), c[1] + rc * math.sin(th))
                tan2 = (-math.sin(th), math.cos(th))
            break
        acc += length

    axes = {'yz': (1, 2), 'xy': (0, 1), 'xz': (0, 2)}[plane]
    p = np.array(center, float); v = np.zeros(3)
    p[axes[0]] += pos2[0];  v[axes[0]] = tan2[0] * speed
    p[axes[1]] += pos2[1];  v[axes[1]] = tan2[1] * speed
    return p, v


def path_target(path, center, radius, period, t, plane='yz', corner=0.3):
    """Dispatch to the requested EE path ('circle' or 'square')."""
    if path == 'square':
        return square_target(center, radius, period, t, plane, corner)
    return circle_target(center, radius, period, t, plane)


def pose_twist(kin, q, T_des, *, kp=1.0, ko=1.0):
    """Closed-loop task velocity for pose tracking: [kp*(p_des-p); ko*axisangle(R_des R^T)]."""
    J, T = kin.jacobian(q)
    e_p = T_des[:3, 3] - T[:3, 3]
    e_o = rotm2axang_vec(T_des[:3, :3] @ T[:3, :3].T)
    return np.concatenate([kp * e_p, ko * e_o])


# ─────────────────────────────────────────────────────────────────────────────
# ROS 2 node — resolved-rate loop tracking a Cartesian circle with null-space task
# ─────────────────────────────────────────────────────────────────────────────
def _build_node_class():
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
    from std_msgs.msg import String
    from sensor_msgs.msg import JointState
    from geometry_msgs.msg import PoseStamped

    class NullspaceRRNode(Node):
        def __init__(self):
            super().__init__('nullspace_rr')
            self.declare_parameter('base_link', 'base_link')
            self.declare_parameter('tip_link', 'ee')
            self.declare_parameter('rate_hz', 100.0)
            self.declare_parameter('objective', 'none')   # none | limit | manip
            self.declare_parameter('k', 0.0)              # null-space gain (live-tunable)
            self.declare_parameter('lam', 0.05)           # DLS damping (task term)
            self.declare_parameter('path', 'square')      # circle | square | hold
            self.declare_parameter('radius', 0.06)        # circle radius / square half-side [m]
            self.declare_parameter('corner', 0.3)         # square corner radius / radius
            self.declare_parameter('period', 8.0)         # loop period [s]
            self.declare_parameter('plane', 'yz')         # yz | xy | xz
            self.declare_parameter('kp_track', 2.0)       # position tracking gain
            self.declare_parameter('dq_max', 0.08)        # per-tick TASK step cap [rad]
            self.declare_parameter('null_step', 0.02)     # per-tick NULL step cap [rad]
            # path='hold' — EE pinned in place; the redundant joints self-move along
            # the null space instead of tracking a path. hold_motion picks the shape:
            #   'sweep'  : sin() along one direction. Holds FULL 6-DOF pose, so the
            #              null space is 1-D — the joints can only move back and forth.
            #   'circle' : the joints trace a closed LOOP (like drawing a circle) and
            #              repeat. Needs a >=2-D null space, so it holds EE POSITION
            #              only (3-DOF) and lets orientation float; the joints follow a
            #              moving cos/sin target offset from the ready pose.
            self.declare_parameter('hold_motion', 'sweep')  # sweep | circle
            self.declare_parameter('position_only', False)  # hold EE position only (3-DOF)
            self.declare_parameter('ori_gain', 3.0)       # EE-hold orientation feedback gain
            self.declare_parameter('osc_omega', 0.5)      # self-motion speed [rad/s] (both)
            self.declare_parameter('osc_gain', 1.0)       # sweep: drive amplitude
            self.declare_parameter('hold_amp', 0.4)       # circle: joint amplitude [rad]
            self.declare_parameter('knull', 3.0)          # circle: null-space follow gain
            # joint-space self-motion directions (joint_6 index 5 is left at 0 — its
            # range is tight, [-0.489, 0.262], so driving it would clip and disturb EE)
            self.declare_parameter('drive',  [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0])
            self.declare_parameter('drive2', [1.0, -1.0, 1.0, -1.0, 1.0, 0.0, -1.0])
            # Joints to hold FIXED during hold mode (comma-separated, 1-based to match
            # joint_1..joint_7). E.g. lock:=7 fixes the last joint so only the others
            # self-move. Best with circle / position_only (full-pose hold + a lock can
            # leave no null space). Empty = no joint locked.
            self.declare_parameter('lock', '')
            # Dexterous "ready" pose driven to before tracking. The default start
            # (q=0) is a SINGULARITY for this arm (w=0) — tracking any path from
            # there is ill-conditioned. Homing to a bent elbow fixes it.
            self.declare_parameter('ready', [0.0, 0.7, 0.0, -1.1, 0.0, 0.0, 0.0])
            self.declare_parameter('home_rate', 0.8)      # homing joint speed [rad/s]

            gp = lambda n: self.get_parameter(n).value
            self._base = gp('base_link'); self._tip = gp('tip_link')
            self.rate = float(gp('rate_hz')); self.dt = 1.0 / self.rate
            self.objective = str(gp('objective')); self.k = float(gp('k'))
            self.lam = float(gp('lam')); self.radius = float(gp('radius'))
            self.path = str(gp('path')); self.corner = float(gp('corner'))
            self.period = float(gp('period')); self.plane = str(gp('plane'))
            self.kp = float(gp('kp_track')); self.dq_max = float(gp('dq_max'))
            self.null_step = float(gp('null_step'))
            self.hold_motion = str(gp('hold_motion'))
            self.position_only = bool(gp('position_only'))
            self.ori_gain = float(gp('ori_gain'))
            self.osc_omega = float(gp('osc_omega')); self.osc_gain = float(gp('osc_gain'))
            self.hold_amp = float(gp('hold_amp')); self.knull = float(gp('knull'))
            self.drive = np.array(gp('drive'), float)
            self.drive2 = np.array(gp('drive2'), float)
            self.lock_str = str(gp('lock'))
            self.active = None                            # resolved once URDF is known
            self.ready = np.array(gp('ready'), float)
            self.home_step = float(gp('home_rate')) * self.dt

            self.kin = None; self.idx = None
            self.q_fb = None; self.q_cmd = None; self.center = None; self.t0 = None
            self.T_hold = None; self.q_lo = None; self.q_hi = None
            self.homing = True; self._log_t = 0.0

            latched = QoSProfile(depth=1,
                                 durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                                 reliability=QoSReliabilityPolicy.RELIABLE)
            self.create_subscription(String, '/robot_description', self._cb_urdf, latched)
            self.create_subscription(JointState, '/joint_states', self._cb_js, 30)
            self.pub = self.create_publisher(JointState, '/joint_commands', 20)
            self.pub_tgt = self.create_publisher(PoseStamped, '/ee_target', 10)
            self.create_timer(self.dt, self._tick)
            self._log_params()

        def _log_params(self):
            """Print all resolved parameters to the terminal."""
            hold = (self.path == 'hold')
            lines = [
                '──────────── nullspace_rr parameters ────────────',
                f'  path            = {self.path}'
                + ('   (EE pinned; joints self-move in null space)' if hold
                   else '   (EE tracks path; null-space secondary task)'),
                f'  base_link/tip   = {self._base} -> {self._tip}',
                f'  rate_hz         = {self.rate}',
                f'  lam (DLS damp)  = {self.lam}',
                f'  dq_max          = {self.dq_max} rad/tick (task cap)',
                f'  null_step       = {self.null_step} rad/tick (null cap)',
                f'  ready pose      = {np.round(self.ready, 3).tolist()}',
                f'  home_rate       = {self.home_step / self.dt:.3f} rad/s',
            ]
            if hold:
                circle = (self.hold_motion == 'circle')
                lines += [
                    f'  hold_motion     = {self.hold_motion}'
                    + ('   (joints trace a closed loop; EE position pinned, '
                       'orientation free)' if circle
                       else '   (joints sweep back-and-forth; full EE pose pinned)'),
                    f'  lock (fixed)    = {self.lock_str or "(none)"}'
                    + '   joints kept fixed (1-based; the rest self-move)',
                    f'  kp_track        = {self.kp} (EE-hold position gain)',
                    f'  osc_omega       = {self.osc_omega} rad/s (self-motion speed)',
                ]
                if circle:
                    lines += [
                        f'  hold_amp        = {self.hold_amp} rad (joint circle amplitude)',
                        f'  knull           = {self.knull} (null-space follow gain)',
                        f'  drive           = {np.round(self.drive, 2).tolist()}',
                        f'  drive2          = {np.round(self.drive2, 2).tolist()}',
                    ]
                else:
                    lines += [
                        f'  ori_gain        = {self.ori_gain} (EE-hold orientation gain)',
                        f'  osc_gain        = {self.osc_gain} (sweep amplitude)',
                        f'  drive           = {np.round(self.drive, 2).tolist()}',
                    ]
            else:
                lines += [
                    f'  objective       = {self.objective}   k = {self.k}',
                    f'  kp_track        = {self.kp}',
                    f'  radius          = {self.radius} m   corner = {self.corner}',
                    f'  period          = {self.period} s   plane = {self.plane}',
                ]
            lines.append('──────────────────────────────────────────────────')
            self.get_logger().info('\n'.join(lines))

        def _cb_urdf(self, msg):
            if self.kin is not None:
                return
            try:
                self.kin = ArmKinematics.from_urdf(msg.data, self._base, self._tip)
                if self.ready.size != self.kin.n:
                    self.get_logger().warn(
                        f'ready pose has {self.ready.size} values but arm has '
                        f'{self.kin.n} DoF — skipping homing (tracking from start pose)')
                    self.homing = False
                else:
                    self.ready = np.clip(self.ready, self.kin.q_min, self.kin.q_max)
                if self.drive.size != self.kin.n:           # self-motion directions
                    self.drive = np.ones(self.kin.n)
                if self.drive2.size != self.kin.n:
                    self.drive2 = np.ones(self.kin.n)
                # Resolve locked joints (1-based in the param) -> 0-based active list.
                locked = []
                for tok in self.lock_str.replace(' ', '').split(','):
                    if tok:
                        j = int(tok) - 1
                        if 0 <= j < self.kin.n:
                            locked.append(j)
                self.active = [i for i in range(self.kin.n) if i not in locked]
                self.locked = locked
                self.get_logger().info(f'URDF loaded: {self.kin.n} DoF')
            except Exception as e:
                self.get_logger().error(f'URDF parse failed: {e}')

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

        def _publish(self, q):
            cmd = JointState()
            cmd.header.stamp = self.get_clock().now().to_msg()
            cmd.name = list(self.kin.joint_names)
            cmd.position = [float(v) for v in q]
            self.pub.publish(cmd)

        def _publish_target(self, p):
            """Publish the desired EE point on /ee_target (drives the cyan RViz trail)."""
            ps = PoseStamped()
            ps.header.stamp = self.get_clock().now().to_msg()
            ps.header.frame_id = self._base
            ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = \
                float(p[0]), float(p[1]), float(p[2])
            ps.pose.orientation.w = 1.0
            self.pub_tgt.publish(ps)

        def _tick_hold(self):
            """EE pinned at T_hold; the redundant joints self-move along the null
            space while the tool frame stays put — the textbook redundancy picture.
              'sweep'  : full 6-DOF pose held (1-D null space) -> back-and-forth.
              'circle' : EE POSITION held (3-DOF -> 4-D null space) -> joints follow
                         a cos/sin target offset from ready, tracing a closed loop."""
            q = self.q_cmd
            t = self._now() - self.t0
            J, T = self.kin.jacobian(q)
            circle = (self.hold_motion == 'circle')
            pos_only = self.position_only or circle

            # Primary: pin the captured EE pose (position only, or full pose).
            if pos_only:
                Ju = J[:3, :]
                x_dot = self.kp * (self.T_hold[:3, 3] - T[:3, 3])
            else:
                e_p = self.kp * (self.T_hold[:3, 3] - T[:3, 3])
                e_o = self.ori_gain * rotm2axang_vec(self.T_hold[:3, :3] @ T[:3, :3].T)
                Ju = J
                x_dot = np.concatenate([e_p, e_o])
            # Restrict to ACTIVE joints — locked joints (e.g. the last one) never
            # move: they are dropped from the task AND the self-motion.
            act = self.active
            Ja = Ju[:, act]
            dq_task = (damped_pinv(Ja, self.lam) @ x_dot) * self.dt

            # Secondary: self-motion, projected so the EE does not move.
            if circle:                                 # follow a moving circular target
                w = self.osc_omega * t
                q_des = self.ready + self.hold_amp * (
                    math.cos(w) * self.drive + math.sin(w) * self.drive2)
                g = self.knull * (q_des - q)
            else:                                      # back-and-forth sweep
                g = self.osc_gain * math.sin(self.osc_omega * t) * self.drive
            Na = np.eye(len(act)) - np.linalg.pinv(Ja) @ Ja
            dq_null = (Na @ g[act]) * self.dt
            nt = float(np.linalg.norm(dq_task))
            if nt > self.dq_max:                       # cap task -> EE-hold stays stable
                dq_task *= self.dq_max / nt
            nn = float(np.linalg.norm(dq_null))
            if nn > self.null_step:                    # cap null -> self-motion stays visible
                dq_null *= self.null_step / nn
            dq = np.zeros(self.kin.n)
            dq[act] = dq_task + dq_null                 # locked joints get exactly 0
            self.q_cmd = np.clip(q + dq, self.kin.q_min, self.kin.q_max)
            self.q_lo = np.minimum(self.q_lo, self.q_cmd)
            self.q_hi = np.maximum(self.q_hi, self.q_cmd)
            self._publish(self.q_cmd)
            self._publish_target(self.T_hold[:3, 3])

            now = self._now()
            if now - self._log_t >= 1.0:
                self._log_t = now
                T_now = self.kin.fk(self.q_cmd)
                dp = np.linalg.norm(T_now[:3, 3] - self.T_hold[:3, 3]) * 1000.0
                da = math.degrees(np.linalg.norm(
                    rotm2axang_vec(self.T_hold[:3, :3] @ T_now[:3, :3].T)))
                travel = float(np.max(self.q_hi - self.q_lo))
                self.get_logger().info(
                    f'EE drift {dp:.3f} mm / {da:.3f} deg | '
                    f'max joint self-motion {travel:.3f} rad')

        def _tick(self):
            if self.kin is None or self.q_fb is None:
                return
            if self.q_cmd is None:
                self.q_cmd = self.q_fb.copy()
                if self.homing:
                    self.get_logger().info('homing to ready pose...')

            # Phase 1 — drive smoothly to the dexterous ready pose, then start.
            if self.homing:
                d = self.ready - self.q_cmd
                if float(np.max(np.abs(d))) < 1e-3:          # arrived
                    self.homing = False
                    self.t0 = self._now()
                    T = self.kin.fk(self.q_cmd)
                    if self.path == 'hold':
                        self.T_hold = T.copy()               # full pose to pin
                        self.center = T[:3, 3].copy()        # fixed EE point (trail dot)
                        self.q_lo = self.q_cmd.copy(); self.q_hi = self.q_cmd.copy()
                        self.get_logger().info(
                            'reached ready pose; EE pinned — joints now self-move '
                            'in the null space')
                    else:
                        off0, _ = path_target(self.path, np.zeros(3), self.radius,
                                              self.period, 0.0, self.plane, self.corner)
                        self.center = T[:3, 3].copy() - off0
                        self.get_logger().info(
                            f'reached ready pose; captured {self.path} centre; tracking')
                else:
                    self.q_cmd = self.q_cmd + np.clip(d, -self.home_step, self.home_step)
                    self._publish(self.q_cmd)
                    return

            # Phase 2 (hold) — pin the EE pose; sweep the joints along the null space.
            if self.path == 'hold':
                self._tick_hold()
                return

            # Phase 2 — track the Cartesian path with null-space secondary task.
            t = self._now() - self.t0
            p_des, v_ff = path_target(self.path, self.center, self.radius, self.period,
                                      t, self.plane, self.corner)
            self._publish_target(p_des)                          # cyan reference trail
            p_cur = self.kin.fk(self.q_cmd)[:3, 3]
            x_dot = v_ff + self.kp * (p_des - p_cur)              # position task velocity
            _q_dot, info = resolved_rate(self.kin, self.q_cmd, x_dot, lam=self.lam,
                                         objective=self.objective, k=self.k,
                                         position_only=True)
            dq_task = info['q_dot_task'] * self.dt
            dq_null = info['q_dot_null'] * self.dt
            nt = float(np.linalg.norm(dq_task))
            if nt > self.dq_max:                       # cap task step
                dq_task *= self.dq_max / nt
            nn = float(np.linalg.norm(dq_null))
            if nn > self.null_step:                    # cap null step independently
                dq_null *= self.null_step / nn
            self.q_cmd = np.clip(self.q_cmd + dq_task + dq_null,
                                 self.kin.q_min, self.kin.q_max)
            self._publish(self.q_cmd)

            now = self._now()
            if now - self._log_t >= 1.0:
                self._log_t = now
                err = np.linalg.norm(self.kin.fk(self.q_cmd)[:3, 3] - p_des) * 1000.0
                w_val = info['manip']
                cost = limit_cost(self.kin, self.q_cmd)
                self.get_logger().info(
                    f'track err {err:.2f} mm | w={w_val:.5f} | limit cost {cost:.3f}')

    return NullspaceRRNode


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


# ─────────────────────────────────────────────────────────────────────────────
# Standalone verification (no ROS): rank-1 projector + EE-undisturbed check
# ─────────────────────────────────────────────────────────────────────────────
def _load_kin():
    import os, subprocess, tempfile
    pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../src/arm_bot
    xacro = os.path.join(pkg, 'urdf', 'arm_bot.urdf.xacro')
    fd, p = tempfile.mkstemp(suffix='.urdf'); os.close(fd)
    subprocess.run(['xacro', xacro], stdout=open(p, 'w'),
                   stderr=subprocess.DEVNULL, check=True)
    xml = open(p).read(); os.remove(p)
    return ArmKinematics.from_urdf(xml, 'base_link', 'ee')


def _verify():
    np.set_printoptions(precision=3, suppress=True)
    kin = _load_kin()
    q = np.array([0.3, 0.6, 0.4, -0.9, 0.5, 0.0, 0.3])      # non-singular
    J, _ = kin.jacobian(q)
    N = null_projector(J)
    eig = np.sort(np.linalg.eigvalsh(0.5 * (N + N.T)))[::-1]   # N is symmetric
    print(f'DoF={kin.n}  J shape={J.shape}  rank(J)={np.linalg.matrix_rank(J)}')
    print(f'null-space projector N: rank(tol=1e-9)={np.linalg.matrix_rank(N, tol=1e-9)} '
          f'(expect 1)   trace={np.trace(N):.6f} (expect ~1)')
    print(f'  eigenvalues of N = {eig}   (one ~1, rest ~0  ->  rank 1)')
    print(f'  idempotent? ||N@N - N|| = {np.linalg.norm(N @ N - N):.2e}')
    print()
    print('EE-undisturbed check  ||J @ (N @ k*grad_H)||  (should be ~1e-15):')
    for name in ('limit', 'manip'):
        g = OBJECTIVES[name](kin, q)
        qd_null = N @ (1.0 * g)
        leak_exact = np.linalg.norm(J @ qd_null)
        # contrast: the DAMPED projector (what the old code used) leaks
        Nd = np.eye(kin.n) - damped_pinv(J, 0.05) @ J
        leak_damped = np.linalg.norm(J @ (Nd @ (1.0 * g)))
        print(f'  {name:5s}: exact projector leak = {leak_exact:.2e}   '
              f'| damped projector leak = {leak_damped:.2e}')


if __name__ == '__main__':
    import sys
    if '--verify' in sys.argv:
        _verify()
    else:
        main()
