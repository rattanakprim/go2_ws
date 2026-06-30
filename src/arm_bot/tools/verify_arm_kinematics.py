#!/usr/bin/env python3
"""Verify arm_bot/arm_kinematics.py against (a) the MATLAB script and (b) the URDF.

Run (ROS 2 sourced, so xacro + urdf_parser_py are available):

    python3 src/arm_bot/tools/verify_arm_kinematics.py

(a) MATLAB check: independent re-implementations of robot_arm_7dof.mlx's fk,
    jacobian_geom, ik_dls, rotm2axang_vec, homogenoeus are compared against
    ArmKinematics.from_mdh(...) on the same Modified-DH table. Expected ~1e-12.

(b) URDF check: ArmKinematics.from_urdf(<current URDF>) FK is compared against an
    independent ElementTree URDF FK (== robot_state_publisher / RViz / Gazebo),
    and its Jacobian against a finite-difference Jacobian. Expected ~1e-13 / ~1e-7.
"""
import math, os, sys, subprocess, tempfile
import numpy as np
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)                       # .../src/arm_bot
sys.path.insert(0, PKG)                           # import arm_bot.fk_arm_final / ik_arm_final
from arm_bot.fk_arm_final import (
    ArmKinematics, matlab_mdh_table, MDH_REV_ROWS, rotm2axang_vec, mdh_transform)
from arm_bot.ik_arm_final import ik_update


# ── independent MATLAB replica (literal port of robot_arm_7dof.mlx) ──────────
def m_homogenoeus(a, alpha, d, theta):
    Tx = np.eye(4); Tx[0, 3] = a
    Rx = np.eye(4); ca, sa = math.cos(alpha), math.sin(alpha)
    Rx[1, 1], Rx[1, 2], Rx[2, 1], Rx[2, 2] = ca, -sa, sa, ca
    Tz = np.eye(4); Tz[2, 3] = d
    Rz = np.eye(4); ct, st = math.cos(theta), math.sin(theta)
    Rz[0, 0], Rz[0, 1], Rz[1, 0], Rz[1, 1] = ct, -st, st, ct
    return Tx @ Rx @ Tz @ Rz                       # Tx*Rx*Tz*Rz, as in MATLAB

def m_fk(q, dh0):
    dh = [list(r) for r in dh0]
    for k, r in enumerate(MDH_REV_ROWS):
        dh[r][3] += q[k]
    T = np.eye(4)
    for row in dh:
        T = T @ m_homogenoeus(*row)
    return T

def m_jacobian(q, dh0):
    dh = [list(r) for r in dh0]
    for k, r in enumerate(MDH_REV_ROWS):
        dh[r][3] += q[k]
    Tc = [np.eye(4)]
    for row in dh:
        Tc.append(Tc[-1] @ m_homogenoeus(*row))
    p_ee = Tc[-1][:3, 3]
    J = np.zeros((6, 7))
    for k, r in enumerate(MDH_REV_ROWS):
        Tr = Tc[r + 1]
        z, p = Tr[:3, 2], Tr[:3, 3]
        J[:3, k] = np.cross(z, p_ee - p); J[3:, k] = z
    return J

def m_rotm2axang(R):
    c = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0)); th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    if abs(th - math.pi) < 1e-6:
        M = (R + np.eye(3)) / 2.0; i = int(np.argmax(np.diag(M)))
        return th * (M[:, i] / math.sqrt(M[i, i]))
    return th * (1.0/(2.0*math.sin(th))) * np.array(
        [R[2, 1]-R[1, 2], R[0, 2]-R[2, 0], R[1, 0]-R[0, 1]])

def m_ik_dls(q0, T_des, dh, lam=0.05, step_clamp=0.2, max_iter=200,
             tol_pos=1e-4, tol_rot=1e-3):
    q = np.array(q0, float)
    for _ in range(max_iter):
        T = m_fk(q, dh)
        e_pos = T_des[:3, 3] - T[:3, 3]
        e_rot = m_rotm2axang(T_des[:3, :3] @ T[:3, :3].T)
        if np.linalg.norm(e_pos) < tol_pos and np.linalg.norm(e_rot) < tol_rot:
            break
        J = m_jacobian(q, dh)
        e = np.concatenate([e_pos, e_rot])
        dq = J.T @ np.linalg.solve(J @ J.T + lam*lam*np.eye(6), e)
        mx = np.max(np.abs(dq))
        if mx > step_clamp:
            dq *= step_clamp / mx
        q = q + dq
    return q


# ── independent URDF FK (== robot_state_publisher) ──────────────────────────
def load_urdf_fk(urdf_path):
    root = ET.parse(urdf_path).getroot()
    J = {}
    for j in root.findall('joint'):
        o = j.find('origin')
        xyz = np.array([float(x) for x in o.get('xyz', '0 0 0').split()]) if o is not None else np.zeros(3)
        rpy = np.array([float(x) for x in o.get('rpy', '0 0 0').split()]) if o is not None else np.zeros(3)
        ax = j.find('axis')
        axis = np.array([float(x) for x in ax.get('xyz').split()]) if ax is not None else None
        J[j.get('name')] = dict(type=j.get('type'), xyz=xyz, rpy=rpy, axis=axis,
                                parent=j.find('parent').get('link'), child=j.find('child').get('link'))
    # walk base_link -> ee
    by_child = {d['child']: (n, d) for n, d in J.items()}
    chain = []
    link = 'ee'
    while link != 'base_link':
        n, d = by_child[link]; chain.append((n, d)); link = d['parent']
    chain.reverse()
    def rpyR(r, p, y):
        cr, sr = math.cos(r), math.sin(r); cp, sp = math.cos(p), math.sin(p); cy, sy = math.cos(y), math.sin(y)
        return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                         [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                         [-sp, cp*sr, cp*cr]])
    def axR(a, q):
        a = a/np.linalg.norm(a); x, y, z = a; c, s = math.cos(q), math.sin(q); C = 1-c
        return np.array([[c+x*x*C, x*y*C-z*s, x*z*C+y*s],
                         [y*x*C+z*s, c+y*y*C, y*z*C-x*s],
                         [z*x*C-y*s, z*y*C+x*s, c+z*z*C]])
    def fk(q):
        M = np.eye(4); qi = 0
        for n, d in chain:
            R = rpyR(*d['rpy'])
            if d['type'] == 'revolute':
                R = R @ axR(d['axis'], q[qi]); qi += 1
            T = np.eye(4); T[:3, :3] = R; T[:3, 3] = d['xyz']; M = M @ T
        return M
    return fk


def geo(A, B):
    c = max(-1, min(1, (np.trace(A@B.T)-1)*0.5)); return math.degrees(math.acos(c))


def main():
    ok = True

    # ---- (a) MATLAB equivalence ----
    print('=== (a) MATH vs MATLAB (robot_arm_7dof.mlx) ===')
    dh = matlab_mdh_table()
    kin = ArmKinematics.from_mdh(dh)
    np.random.seed(1)
    d_h = d_fk = d_jac = 0.0
    for _ in range(300):
        a, al, d, th = np.random.uniform(-2, 2, 4)
        d_h = max(d_h, np.max(np.abs(mdh_transform(a, al, d, th) - m_homogenoeus(a, al, d, th))))
        q = np.random.uniform(-2.5, 2.5, 7)
        d_fk = max(d_fk, np.max(np.abs(kin.fk(q) - m_fk(q, dh))))
        Jk, _ = kin.jacobian(q)
        d_jac = max(d_jac, np.max(np.abs(Jk - m_jacobian(q, dh))))
    # IK update law: the shipped ik_update DLS step == MATLAB ik_dls step.
    # null_k=0, lam_min=lam (constant damping), step_clamp huge -> bare DLS step.
    d_ik = 0.0
    for _ in range(200):
        q = np.random.uniform(-2, 2, 7)
        T_des = m_fk(np.random.uniform(-2, 2, 7), dh)
        dq_mine, _, _ = ik_update(kin, q, T_des, null_k=0.0, lam=0.05,
                                  lam_min=0.05, step_clamp=1e9)
        J = m_jacobian(q, dh); T = m_fk(q, dh)
        e = np.concatenate([T_des[:3, 3] - T[:3, 3],
                            m_rotm2axang(T_des[:3, :3] @ T[:3, :3].T)])
        dq_m = J.T @ np.linalg.solve(J @ J.T + 0.05**2 * np.eye(6), e)
        d_ik = max(d_ik, np.max(np.abs(dq_mine - dq_m)))
    for nm, v, tol in [('homogenoeus/mdh_transform', d_h, 1e-12),
                       ('fk == MATLAB fk', d_fk, 1e-12),
                       ('jacobian == MATLAB jacobian_geom', d_jac, 1e-12),
                       ('ik_update DLS step == MATLAB ik_dls', d_ik, 1e-12)]:
        p = v < tol; ok &= p
        print(f'  [{"PASS" if p else "FAIL"}] {nm:42s} max|diff| = {v:.2e}')

    # ---- (b) URDF equivalence (current robot) ----
    print('\n=== (b) MODEL vs current URDF (robot_state_publisher / RViz / Gazebo) ===')
    xacro_path = os.path.join(PKG, 'urdf', 'arm_bot.urdf.xacro')
    fd, urdf_path = tempfile.mkstemp(suffix='.urdf')
    os.close(fd)
    subprocess.run(['xacro', xacro_path], stdout=open(urdf_path, 'w'),
                   stderr=subprocess.DEVNULL, check=True)
    urdf_xml = open(urdf_path).read()
    ku = ArmKinematics.from_urdf(urdf_xml, 'base_link', 'ee')
    ufk = load_urdf_fk(urdf_path)
    mp = mo = 0.0
    np.random.seed(2)
    for _ in range(500):
        q = np.random.uniform(-2.5, 2.5, 7)
        Tk = ku.fk(q); Tu = ufk(q)
        mp = max(mp, np.linalg.norm(Tk[:3, 3] - Tu[:3, 3]))
        mo = max(mo, geo(Tk[:3, :3], Tu[:3, :3]))
    # Jacobian vs finite difference
    dj = 0.0
    for _ in range(40):
        q = np.random.uniform(-1.5, 1.5, 7)
        Jk, _ = ku.jacobian(q)
        Jf = np.zeros((6, 7)); T0 = ku.fk(q)
        for i in range(7):
            dq = np.zeros(7); dq[i] = 1e-6; T1 = ku.fk(q + dq)
            Jf[:3, i] = (T1[:3, 3] - T0[:3, 3]) / 1e-6
            Jf[3:, i] = rotm2axang_vec(T1[:3, :3] @ T0[:3, :3].T) / 1e-6
        dj = max(dj, np.max(np.abs(Jk - Jf)))
    os.remove(urdf_path)
    for nm, v, tol in [('fk == URDF (RViz/Gazebo) position [m]', mp, 1e-9),
                       ('fk == URDF orientation [deg]', mo, 1e-4),
                       ('jacobian == finite-difference', dj, 1e-5)]:
        p = v < tol; ok &= p
        print(f'  [{"PASS" if p else "FAIL"}] {nm:42s} max = {v:.2e}')

    print('\n' + ('ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
