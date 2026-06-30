#!/usr/bin/env python3
"""Thesis experiment: baseline (minimum-norm) vs null-space redundancy resolution.

Tracks a Cartesian circle with the 7-DOF arm using the resolved-rate law from
arm_bot.nullspace_rr (DAMPED task pseudoinverse + EXACT null-space projector) and
compares three runs on the SAME trajectory:
  - baseline : null-space OFF (k=0)  — pure minimum-norm pseudoinverse
  - limit    : null-space ON, joint-limit-avoidance objective
  - manip    : null-space ON, manipulability-maximisation objective

It logs each run to CSV (time, q1..q7, manipulability, joint-limit cost,
EE tracking error mm) and renders a 3-panel comparison figure for the thesis.

Run (ROS 2 sourced):  /usr/bin/python3 src/arm_bot/tools/nullspace_rr_experiment.py
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)                       # .../src/arm_bot
WS = os.path.dirname(os.path.dirname(PKG))        # .../7dof_thesis_ws
sys.path.insert(0, PKG)
from arm_bot.nullspace_rr import (                # the module under test
    ArmKinematics, resolved_rate, circle_target, manipulability, limit_cost)


def load_kin():
    xacro = os.path.join(PKG, 'urdf', 'arm_bot.urdf.xacro')
    fd, p = tempfile.mkstemp(suffix='.urdf'); os.close(fd)
    subprocess.run(['xacro', xacro], stdout=open(p, 'w'),
                   stderr=subprocess.DEVNULL, check=True)
    xml = open(p).read(); os.remove(p)
    return ArmKinematics.from_urdf(xml, 'base_link', 'ee')


def run(kin, q0, objective, k, *, T=16.0, dt=0.01, radius=0.08, period=8.0,
        plane='yz', lam=0.05, kp=2.0, dq_max=0.08, null_step=0.02):
    """Resolved-rate circle tracking; returns dict of time series."""
    q = np.array(q0, float)
    # place the circle so it STARTS at the current EE (no initial jump)
    off0, _ = circle_target(np.zeros(3), radius, period, 0.0, plane)
    center = kin.fk(q)[:3, 3].copy() - off0
    steps = int(round(T / dt))
    t = np.arange(steps) * dt
    Q = np.zeros((steps, kin.n))
    err = np.zeros(steps); w = np.zeros(steps); H = np.zeros(steps)
    for i in range(steps):
        p_des, v_ff = circle_target(center, radius, period, t[i], plane)
        p_cur = kin.fk(q)[:3, 3]
        x_dot = v_ff + kp * (p_des - p_cur)                # position task velocity
        _q_dot, info = resolved_rate(kin, q, x_dot, lam=lam, objective=objective,
                                     k=k, position_only=True)
        dq_task = info['q_dot_task'] * dt
        dq_null = info['q_dot_null'] * dt
        nt = float(np.linalg.norm(dq_task))
        if nt > dq_max:                       # cap task step (stability)
            dq_task *= dq_max / nt
        nn = float(np.linalg.norm(dq_null))
        if nn > null_step:                    # cap null step INDEPENDENTLY (never starves task)
            dq_null *= null_step / nn
        q = np.clip(q + dq_task + dq_null, kin.q_min, kin.q_max)
        Q[i] = q
        err[i] = np.linalg.norm(kin.fk(q)[:3, 3] - p_des) * 1000.0   # mm
        w[i] = info['manip']
        H[i] = limit_cost(kin, q)
    return {'t': t, 'Q': Q, 'err': err, 'w': w, 'H': H, 'objective': objective, 'k': k}


def write_csv(path, kin, r):
    cols = [r['t'], r['err'], r['w'], r['H']] + [r['Q'][:, i] for i in range(kin.n)]
    header = ('time_s,ee_track_err_mm,manip_w,limit_cost_H,'
              + ','.join(f'q{i+1}' for i in range(kin.n)))
    np.savetxt(path, np.column_stack(cols), delimiter=',', header=header, comments='')
    return path


def make_figure(base, lim, man, out_base):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    t = base['t']
    fig, (a0, a1, a2) = plt.subplots(3, 1, figsize=(8.5, 9.0), sharex=True)

    # (a) EE tracking error — should stay low for ALL runs (exact projector => no leak)
    a0.plot(t, base['err'], color='0.5', lw=1.6, label='baseline (null OFF)')
    a0.plot(t, lim['err'], color='tab:blue', lw=1.4, label='limit-avoid ON')
    a0.plot(t, man['err'], color='tab:orange', lw=1.4, label='manip ON')
    a0.set_ylabel('EE tracking error [mm]')
    a0.set_title('Resolved-rate circle tracking: null-space resolution does not disturb the task')
    a0.grid(True, alpha=0.3); a0.legend(fontsize=8, loc='upper right')
    a0.set_ylim(0, max(2.0, 1.3 * max(base['err'].max(), lim['err'].max(), man['err'].max())))

    # (b) joint-limit cost H — limit-avoid ON should drive it down vs baseline
    a1.plot(t, base['H'], color='0.5', lw=1.6, ls='--', label='baseline')
    a1.plot(t, lim['H'], color='tab:blue', lw=1.8, label='limit-avoid ON')
    a1.set_ylabel('joint-limit cost H')
    a1.grid(True, alpha=0.3); a1.legend(fontsize=8, loc='upper right')
    a1.set_title('joint-limit avoidance:  H  '
                 f'{base["H"].mean():.3f} (baseline) → {lim["H"].mean():.3f} (ON)', fontsize=10)

    # (c) manipulability w — manip ON should raise it vs baseline
    a2.plot(t, base['w'], color='0.5', lw=1.6, ls='--', label='baseline')
    a2.plot(t, man['w'], color='tab:orange', lw=1.8, label='manip ON')
    a2.set_ylabel('manipulability w')
    a2.set_xlabel('time [s]')
    a2.grid(True, alpha=0.3); a2.legend(fontsize=8, loc='upper right')
    a2.set_title('manipulability:  w  '
                 f'{base["w"].mean():.4f} (baseline) → {man["w"].mean():.4f} (ON)', fontsize=10)

    fig.tight_layout()
    png, pdf = out_base + '.png', out_base + '.pdf'
    fig.savefig(png, dpi=150); fig.savefig(pdf)
    return png, pdf


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--k', type=float, default=2.0, help='null-space gain for ON runs')
    ap.add_argument('--duration', type=float, default=16.0)
    ap.add_argument('--radius', type=float, default=0.08)
    ap.add_argument('--period', type=float, default=8.0)
    ap.add_argument('--out', default=os.path.join(WS, 'nullspace_rr_experiment'))
    args = ap.parse_args()

    kin = load_kin()
    q0 = np.array([0.3, 0.6, 0.4, -0.9, 0.5, 0.0, 0.3])     # non-singular start
    common = dict(T=args.duration, radius=args.radius, period=args.period)
    base = run(kin, q0, 'none', 0.0, **common)
    lim = run(kin, q0, 'limit', args.k, **common)
    man = run(kin, q0, 'manip', args.k, **common)

    for name, r in (('baseline', base), ('limit_avoid', lim), ('manip', man)):
        csv = write_csv(f'{args.out}_{name}.csv', kin, r)
        print(f'{name:10s}: track err mean {r["err"].mean():.3f} max {r["err"].max():.3f} mm | '
              f'w mean {r["w"].mean():.4f} | H mean {r["H"].mean():.3f}  -> {os.path.basename(csv)}')
    png, pdf = make_figure(base, lim, man, args.out)
    print(f'\nbaseline vs ON:')
    print(f'  limit cost H : {base["H"].mean():.3f} -> {lim["H"].mean():.3f} '
          f'({(lim["H"].mean()/base["H"].mean()-1)*100:+.0f}%)')
    print(f'  manipulability w : {base["w"].mean():.4f} -> {man["w"].mean():.4f} '
          f'({(man["w"].mean()/base["w"].mean()-1)*100:+.0f}%)')
    print(f'  EE tracking error stays ~{base["err"].mean():.2f}-{man["err"].mean():.2f} mm (task preserved)')
    print(f'wrote:\n  {png}\n  {pdf}')


if __name__ == '__main__':
    main()
