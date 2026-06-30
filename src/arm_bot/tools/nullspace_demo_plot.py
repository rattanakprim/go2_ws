#!/usr/bin/env python3
"""Generate the 7-DOF redundancy (null-space self-motion) figure for the thesis.

Runs the SAME control law as arm_bot/nullspace_demo.py (the node) on the live
URDF kinematics (ArmKinematics.from_urdf -> matches RViz/Gazebo exactly), holding
the end-effector pose fixed while driving the joints along the null space. It then
plots, against time:
  (a) the seven joint angles q_1..q_7 (the self-motion), and
  (b) the end-effector position error (mm) and orientation error (deg),

which stay ~0 throughout — the canonical demonstration that the extra 7th DoF
moves the arm's internal configuration without moving the tool frame.

Run (ROS 2 sourced, so xacro + urdf_parser_py are available):

    /usr/bin/python3 src/arm_bot/tools/nullspace_demo_plot.py

Writes nullspace_redundancy.png and nullspace_redundancy.pdf to the workspace root
(override with --out). The control law / defaults mirror nullspace_demo.py and the
nullspace_demo_*.launch.py defaults (oscillate mode, omega=0.6, null_gain=0.6).
"""
import argparse
import math
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)                       # .../src/arm_bot
WS = os.path.dirname(os.path.dirname(PKG))        # .../7dof_thesis_ws
sys.path.insert(0, PKG)                           # import arm_bot.fk_arm_final
from arm_bot.fk_arm_final import ArmKinematics, rotm2axang_vec


def load_kinematics(base='base_link', tip='ee'):
    """Build ArmKinematics from the package xacro (== the deployed robot)."""
    xacro_path = os.path.join(PKG, 'urdf', 'arm_bot.urdf.xacro')
    fd, urdf_path = tempfile.mkstemp(suffix='.urdf')
    os.close(fd)
    subprocess.run(['xacro', xacro_path], stdout=open(urdf_path, 'w'),
                   stderr=subprocess.DEVNULL, check=True)
    urdf_xml = open(urdf_path).read()
    os.remove(urdf_path)
    return ArmKinematics.from_urdf(urdf_xml, base, tip)


def manipulability(kin, q):
    """Yoshikawa manipulability w = sqrt(det(J J^T)) of the 6xN Jacobian."""
    J, _ = kin.jacobian(np.asarray(q, float))
    return math.sqrt(max(0.0, np.linalg.det(J @ J.T)))


def manip_grad(kin, q, eps=1e-4):
    """Central finite-difference gradient dw/dq."""
    q = np.asarray(q, float)
    g = np.zeros(kin.n)
    for i in range(kin.n):
        dq = np.zeros(kin.n); dq[i] = eps
        g[i] = (manipulability(kin, q + dq) - manipulability(kin, q - dq)) / (2 * eps)
    return g


def simulate(kin, q0, *, T=20.0, rate_hz=100.0, mode='oscillate',
             omega=0.6, null_gain=0.6, lam=0.05, pos_gain=6.0, ori_gain=3.0,
             dq_max=0.03, null_step=0.01, position_only=False):
    """Run the null-space self-motion law (mirrors nullspace_demo.py:_tick).

    Returns (t, Q, pos_err_mm, ori_err_deg) sampled each tick. The EE pose is
    captured from q0 and held; joints are driven in the null space.

    position_only=True holds only EE position (3-DOF task) -> a 4-D null space,
    giving the secondary task far more room (orientation is then free to vary)."""
    dt = 1.0 / rate_hz
    n = kin.n
    drive = np.ones(n)
    q = np.clip(np.asarray(q0, float), kin.q_min, kin.q_max)
    T_des = kin.fk(q)
    p_des, R_des = T_des[:3, 3], T_des[:3, :3]

    steps = int(round(T / dt))
    t = np.arange(steps) * dt
    Q = np.zeros((steps, n))
    pos_err = np.zeros(steps)
    ori_err = np.zeros(steps)

    for k in range(steps):
        J, Tcur = kin.jacobian(q)
        # primary task: pin the EE pose (6-DOF) or just position (3-DOF)
        e_pos = pos_gain * (p_des - Tcur[:3, 3])
        if position_only:
            Ju = J[:3, :]
            e = e_pos
        else:
            e_rot = ori_gain * rotm2axang_vec(R_des @ Tcur[:3, :3].T)
            Ju = J
            e = np.concatenate([e_pos, e_rot])
        JuT = Ju.T
        M = Ju @ JuT + (lam ** 2) * np.eye(Ju.shape[0])
        Jpinv = JuT @ np.linalg.inv(M)
        dq_p = Jpinv @ e
        pn = float(np.linalg.norm(dq_p))
        if pn > dq_max:
            dq_p *= dq_max / pn
        # secondary task: self-motion, projected into the null space
        N = np.eye(n) - Jpinv @ Ju
        if mode == 'center':
            g = null_gain * (kin.q_mid - q)
        elif mode == 'limit_avoid':
            rng = kin.q_max - kin.q_min
            g = null_gain * (kin.q_mid - q) / (rng * rng)
        elif mode == 'manip':
            w = manipulability(kin, q)                # ascend log-manipulability:
            g = null_gain * manip_grad(kin, q) / (w + 1e-9)  # scale-free, stops at the optimum
        else:  # oscillate
            g = null_gain * math.sin(omega * t[k]) * drive
        dq_n = N @ g
        nn = float(np.linalg.norm(dq_n))
        if nn > null_step:
            dq_n *= null_step / nn
        q = np.clip(q + dq_p + dq_n, kin.q_min, kin.q_max)

        # record TRUE (un-weighted) EE error of the resulting configuration
        Tk = kin.fk(q)
        Q[k] = q
        pos_err[k] = np.linalg.norm(Tk[:3, 3] - p_des) * 1000.0          # mm
        ori_err[k] = math.degrees(np.linalg.norm(
            rotm2axang_vec(R_des @ Tk[:3, :3].T)))                       # deg
    return t, Q, pos_err, ori_err


def skeleton(kin, q):
    """Kinematic skeleton (base -> the 7 revolute joint origins -> EE) in the
    base frame, for drawing the arm. Uses ArmKinematics._chain, which returns the
    EE transform and, per revolute joint, (axis, point-on-axis = joint origin)."""
    T_ee, axes = kin._chain(np.asarray(q, float))
    pts = [np.zeros(3)]                      # base_link origin
    pts += [p for (_z, p) in axes]           # joint_1 .. joint_7 origins
    pts.append(T_ee[:3, 3])                  # EE
    return np.array(pts)


def make_robot_figure(kin, t, Q, out_base, *, n_frames=9, omega=0.6):
    """3D plot of the arm at several phases of the self-motion: the links
    reconfigure (elbow sweeps an arc) while the EE point stays fixed."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d proj)

    # Sample frames from one self-motion extreme to the other: sin(omega t) runs
    # +1 -> -1 over t in [ (pi/2)/omega , (3pi/2)/omega ] — the widest fan of
    # configurations without retracing the path.
    dt = t[1] - t[0]
    t0, t1 = (math.pi / 2) / omega, (3 * math.pi / 2) / omega
    idx = np.clip(np.round(np.linspace(t0, t1, n_frames) / dt).astype(int),
                  0, len(t) - 1)

    skels = [skeleton(kin, Q[i]) for i in idx]
    ee_pts = np.array([s[-1] for s in skels])
    ee_spread_mm = float(np.linalg.norm(ee_pts - ee_pts.mean(axis=0), axis=1).max()) * 1000.0
    elbow = np.array([s[4] for s in skels])   # joint_4 origin ~ "elbow"

    fig = plt.figure(figsize=(7.5, 7.0))
    ax = fig.add_subplot(111, projection='3d')
    cmap = plt.get_cmap('viridis')
    for k, s in enumerate(skels):
        c = cmap(k / (len(skels) - 1))
        ax.plot(s[:, 0], s[:, 1], s[:, 2], '-o', color=c, lw=2.0, ms=4,
                alpha=0.85, label='_nolegend_')
    # elbow self-motion arc
    ax.plot(elbow[:, 0], elbow[:, 1], elbow[:, 2], '--', color='0.35', lw=1.3,
            label='elbow (joint_4) self-motion')
    # fixed EE point + base
    ax.scatter(*ee_pts.mean(axis=0), c='red', s=90, marker='*', depthshade=False,
               label=f'end-effector (held, spread {ee_spread_mm:.2f} mm)')
    ax.scatter(0, 0, 0, c='black', s=50, marker='s', depthshade=False, label='base')

    # equal aspect
    allpts = np.vstack(skels)
    c0 = allpts.mean(axis=0)
    r = (allpts.max(axis=0) - allpts.min(axis=0)).max() / 2.0
    ax.set_xlim(c0[0] - r, c0[0] + r)
    ax.set_ylim(c0[1] - r, c0[1] + r)
    ax.set_zlim(c0[2] - r, c0[2] + r)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_zlabel('z [m]')
    ax.set_title('7-DOF self-motion: arm reconfigures while the end-effector stays fixed\n'
                 f'({len(skels)} configurations, colour = sweep progression)', fontsize=10)
    ax.legend(loc='upper left', fontsize=8)
    ax.view_init(elev=22, azim=-60)

    fig.tight_layout()
    png, pdf = out_base + '_robot.png', out_base + '_robot.pdf'
    fig.savefig(png, dpi=150)
    fig.savefig(pdf)
    return png, pdf, ee_spread_mm


def limit_cost(kin, Q):
    """Joint-limit cost H = sum(((q-q_mid)/range)^2) per timestep, and the worst
    normalised limit proximity max_i |q_i-q_mid_i|/(0.5 range_i) (0 = mid, 1 = at limit)."""
    rng = kin.q_max - kin.q_min
    z = (Q - kin.q_mid) / rng                      # 0 at mid, +-0.5 at the limits
    H = np.sum(z ** 2, axis=1)
    proximity = np.max(np.abs(z) / 0.5, axis=1)
    return H, proximity


def make_limit_figure(kin, t, Q, pos_err, joint_names, out_base):
    """Joint-limit-avoidance figure: joints migrate away from their bounds along
    the null space (top), while the limit cost falls and the EE stays fixed (bottom)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    H, prox = limit_cost(kin, Q)
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8.5, 6.8), sharex=True)

    # (a) joint angles with their limit bands; show joints leaving the danger zone
    cmap = plt.get_cmap('tab10')
    for i in range(Q.shape[1]):
        c = cmap(i % 10)
        ax0.plot(t, Q[:, i], color=c, lw=1.6, label=joint_names[i])
        ax0.axhline(kin.q_min[i], color=c, ls=':', lw=0.8, alpha=0.5)
        ax0.axhline(kin.q_max[i], color=c, ls=':', lw=0.8, alpha=0.5)
    ax0.set_ylabel('joint angle [rad]')
    ax0.set_title('Joint-limit avoidance in the null space '
                  '(dotted lines = each joint’s limits)')
    ax0.grid(True, alpha=0.3)
    ax0.legend(ncol=4, fontsize=8, loc='upper right')

    # (b) limit cost falling (left) + EE position hold (right)
    cH, cee = 'tab:green', 'tab:blue'
    ax1.plot(t, prox * 100.0, color=cH, lw=2.0, label='worst limit proximity [%]')
    ax1.set_ylabel('worst limit proximity [%]', color=cH)
    ax1.tick_params(axis='y', labelcolor=cH)
    ax1.set_xlabel('time [s]')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(100.0, prox.max() * 110.0))

    ax1r = ax1.twinx()
    ax1r.plot(t, pos_err, color=cee, lw=1.6, label='EE position error [mm]')
    ax1r.set_ylabel('EE position error [mm]', color=cee)
    ax1r.tick_params(axis='y', labelcolor=cee)
    ax1r.set_ylim(0, max(1.0, pos_err.max() * 1.3))

    lines = ax1.get_lines() + ax1r.get_lines()
    ax1.legend(lines, [ln.get_label() for ln in lines], fontsize=8, loc='upper right')
    ax1.set_title(f'limit proximity {prox[0]*100:.0f}% → {prox[-1]*100:.0f}%   |   '
                  f'cost H {H[0]:.3f} → {H[-1]:.3f}   |   '
                  f'EE held ≤ {pos_err.max():.2f} mm', fontsize=10)

    fig.tight_layout()
    png, pdf = out_base + '.png', out_base + '.pdf'
    fig.savefig(png, dpi=150)
    fig.savefig(pdf)
    return png, pdf, (H[0], H[-1], prox[0], prox[-1])


def write_csv(path, kin, t, Q, pos_err, ori_err):
    """Dump the full per-tick time series to CSV for re-plotting in the thesis."""
    w = np.array([manipulability(kin, q) for q in Q])
    _H, prox = limit_cost(kin, Q)
    cols = [t, pos_err, ori_err, w, prox * 100.0] + [Q[:, i] for i in range(kin.n)]
    header = ('time_s,ee_pos_err_mm,ee_ori_err_deg,manip_w,limit_proximity_pct,'
              + ','.join(f'q{i+1}' for i in range(kin.n)))
    np.savetxt(path, np.column_stack(cols), delimiter=',', header=header, comments='')
    return path


def print_table(kin, t, Q, pos_err, ori_err, step_s=1.0):
    """Print the time series sampled every step_s seconds to the terminal."""
    w = np.array([manipulability(kin, q) for q in Q])
    _H, prox = limit_cost(kin, Q)
    dt = t[1] - t[0]
    stride = max(1, int(round(step_s / dt)))
    cols = ['t[s]', 'w', 'prox%', 'EEpos[mm]', 'EEori[deg]'] + [f'q{i+1}' for i in range(kin.n)]
    hdr = f'{cols[0]:>5} {cols[1]:>9} {cols[2]:>6} {cols[3]:>10} {cols[4]:>11}  ' + \
          ' '.join(f'{c:>6}' for c in cols[5:])
    print(hdr); print('-' * len(hdr))
    for k in list(range(0, len(t), stride)) + [len(t) - 1]:
        print(f'{t[k]:5.1f} {w[k]:9.5f} {prox[k]*100:6.0f} {pos_err[k]:10.3f} {ori_err[k]:11.2f}  '
              + ' '.join(f'{Q[k,i]:6.2f}' for i in range(kin.n)))


def make_manip_figure(kin, t, Q, pos_err, joint_names, out_base):
    """Manipulability-maximization figure: joints reconfigure (top) so that w =
    sqrt(det(J J^T)) rises (bottom, left) while the EE position is held (right)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    w = np.array([manipulability(kin, q) for q in Q])
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8.5, 6.8), sharex=True)

    cmap = plt.get_cmap('tab10')
    for i in range(Q.shape[1]):
        ax0.plot(t, Q[:, i], color=cmap(i % 10), lw=1.6, label=joint_names[i])
    ax0.set_ylabel('joint angle [rad]')
    ax0.set_title('Manipulability maximization in the null space '
                  '(arm reconfigures away from the singularity)')
    ax0.grid(True, alpha=0.3)
    ax0.legend(ncol=4, fontsize=8, loc='upper right')

    cw, cee = 'tab:purple', 'tab:blue'
    ax1.plot(t, w, color=cw, lw=2.0, label='manipulability  w = √det(J Jᵀ)')
    ax1.set_ylabel('manipulability w', color=cw)
    ax1.tick_params(axis='y', labelcolor=cw)
    ax1.set_xlabel('time [s]')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, w.max() * 1.15)

    ax1r = ax1.twinx()
    ax1r.plot(t, pos_err, color=cee, lw=1.6, label='EE position error [mm]')
    ax1r.set_ylabel('EE position error [mm]', color=cee)
    ax1r.tick_params(axis='y', labelcolor=cee)
    ax1r.set_ylim(0, max(1.0, pos_err.max() * 1.3))

    lines = ax1.get_lines() + ax1r.get_lines()
    ax1.legend(lines, [ln.get_label() for ln in lines], fontsize=8, loc='lower right')
    ax1.set_title(f'manipulability  w {w[0]:.4f} → {w[-1]:.4f}  '
                  f'(+{(w[-1]/w[0]-1)*100:.0f}%)   |   EE position held ≤ {pos_err.max():.2f} mm',
                  fontsize=10)

    fig.tight_layout()
    png, pdf = out_base + '.png', out_base + '.pdf'
    fig.savefig(png, dpi=150)
    fig.savefig(pdf)
    return png, pdf, (float(w[0]), float(w[-1]))


def make_figure(t, Q, pos_err, ori_err, joint_names, out_base):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    travel = float(np.max(Q.max(axis=0) - Q.min(axis=0)))
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8.5, 6.8), sharex=True)

    # (a) joint self-motion
    cmap = plt.get_cmap('tab10')
    for i in range(Q.shape[1]):
        ax0.plot(t, Q[:, i], color=cmap(i % 10), lw=1.6, label=joint_names[i])
    ax0.set_ylabel('joint angle [rad]')
    ax0.set_title('7-DOF redundancy: null-space self-motion with the end-effector pose held fixed')
    ax0.grid(True, alpha=0.3)
    ax0.legend(ncol=4, fontsize=8, loc='upper right')
    ax0.text(0.012, 0.04, f'max joint travel = {travel:.2f} rad',
             transform=ax0.transAxes, fontsize=9,
             bbox=dict(boxstyle='round', fc='white', ec='0.7', alpha=0.9))

    # (b) end-effector error (held ~0): position (mm, left) + orientation (deg, right)
    cpos, cori = 'tab:blue', 'tab:red'
    ax1.plot(t, pos_err, color=cpos, lw=1.6, label='position error [mm]')
    ax1.set_ylabel('position error [mm]', color=cpos)
    ax1.tick_params(axis='y', labelcolor=cpos)
    ax1.set_xlabel('time [s]')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(1.0, pos_err.max() * 1.3))

    ax1r = ax1.twinx()
    ax1r.plot(t, ori_err, color=cori, lw=1.6, label='orientation error [deg]')
    ax1r.set_ylabel('orientation error [deg]', color=cori)
    ax1r.tick_params(axis='y', labelcolor=cori)
    ax1r.set_ylim(0, max(0.05, ori_err.max() * 1.3))

    lines = ax1.get_lines() + ax1r.get_lines()
    ax1.legend(lines, [ln.get_label() for ln in lines], fontsize=8, loc='upper right')
    ax1.set_title(f'End-effector hold:  max {pos_err.max():.2f} mm  /  {ori_err.max():.3f}°  '
                  f'(mean {pos_err.mean():.2f} mm / {ori_err.mean():.3f}°)', fontsize=10)

    fig.tight_layout()
    png, pdf = out_base + '.png', out_base + '.pdf'
    fig.savefig(png, dpi=150)
    fig.savefig(pdf)
    return png, pdf, travel


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', default='oscillate',
                    choices=['oscillate', 'center', 'limit_avoid', 'manip'])
    ap.add_argument('--omega', type=float, default=0.6)
    ap.add_argument('--null_gain', type=float, default=0.6)
    ap.add_argument('--lam', type=float, default=None,
                    help='DLS damping (default 0.05; limit_avoid uses 0.005 — its '
                         'position-only task is well away from singularities)')
    ap.add_argument('--duration', type=float, default=20.0)
    ap.add_argument('--table', action='store_true',
                    help='print the time series (sampled per second) to the terminal')
    ap.add_argument('--out', default=None, help='output path base (no extension)')
    args = ap.parse_args()

    kin = load_kinematics()

    if args.mode == 'limit_avoid':
        out = args.out or os.path.join(WS, 'nullspace_limit_avoid')
        # Start with a wide-range joint near its limit so the null space has room
        # to retreat from it (clipped to the actual URDF limits).
        q0 = np.clip(np.array([2.7, 1.0, 0.8, -1.2, 1.0, 0.20, 0.8]),
                     kin.q_min, kin.q_max)
    elif args.mode == 'manip':
        out = args.out or os.path.join(WS, 'nullspace_manipulability')
        # Start from a near-singular (extended) posture so manipulability has
        # plenty of headroom to climb.
        q0 = np.clip(np.array([0.0, 0.1, 0.0, -0.1, 0.0, 0.0, 0.0]),
                     kin.q_min, kin.q_max)
    else:
        out = args.out or os.path.join(WS, 'nullspace_redundancy')
        # A non-singular starting posture (clipped to joint limits) so the
        # self-motion manifold is well conditioned.
        q0 = np.array([0.3, 0.6, 0.4, -0.9, 0.5, 0.0, 0.3])

    # limit_avoid and manip hold only EE position (3-DOF) -> 4-D null space, so
    # the secondary task has room to work; their well-conditioned position task
    # also lets us use lighter damping for a tighter hold.
    position_only = args.mode in ('limit_avoid', 'manip')
    lam = args.lam if args.lam is not None else (0.005 if position_only else 0.05)
    t, Q, pos_err, ori_err = simulate(
        kin, q0, T=args.duration, mode=args.mode, lam=lam,
        omega=args.omega, null_gain=args.null_gain, position_only=position_only)

    print(f'kinematics: {kin.n} DoF from URDF, joints {kin.joint_names}')
    print(f'EE hold over {args.duration:.0f}s: '
          f'position max {pos_err.max():.3f} mm (mean {pos_err.mean():.3f}), '
          f'orientation max {ori_err.max():.4f} deg (mean {ori_err.mean():.4f})')

    csv = write_csv(out + '.csv', kin, t, Q, pos_err, ori_err)
    print(f'wrote data: {csv}')
    if args.table:
        print_table(kin, t, Q, pos_err, ori_err)

    if args.mode == 'limit_avoid':
        lpng, lpdf, (H0, H1, p0, p1) = make_limit_figure(
            kin, t, Q, pos_err, kin.joint_names, out)
        print(f'limit avoidance: cost H {H0:.3f} -> {H1:.3f}, '
              f'worst proximity {p0*100:.0f}% -> {p1*100:.0f}%')
        print(f'wrote:\n  {lpng}\n  {lpdf}')
    elif args.mode == 'manip':
        mpng, mpdf, (w0, w1) = make_manip_figure(
            kin, t, Q, pos_err, kin.joint_names, out)
        print(f'manipulability: w {w0:.4f} -> {w1:.4f} (+{(w1/w0-1)*100:.0f}%)')
        print(f'wrote:\n  {mpng}\n  {mpdf}')
    else:
        png, pdf, travel = make_figure(t, Q, pos_err, ori_err, kin.joint_names, out)
        rpng, rpdf, ee_spread = make_robot_figure(kin, t, Q, out, omega=args.omega)
        print(f'max joint self-motion: {travel:.3f} rad')
        print(f'robot-pose figure: EE spread across configurations = {ee_spread:.3f} mm')
        print(f'wrote:\n  {png}\n  {pdf}\n  {rpng}\n  {rpdf}')


if __name__ == '__main__':
    main()
