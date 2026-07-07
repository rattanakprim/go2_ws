"""
Thin MuJoCo wrapper for the Go2.

Holds the model/data, drives the 12 torque motors with a PD controller toward
target joint angles, and exposes the robot state (base pose, joint states).
Used by the ROS 2 node, but has no ROS dependency so it can be tested alone.
"""
import math
import threading

import numpy as np
import mujoco

# Joint / actuator order in the model: FL, FR, RL, RR x (hip, thigh, calf)
JOINT_NAMES = [
    f"{leg}_{j}" for leg in ("FL", "FR", "RL", "RR")
    for j in ("hip", "thigh", "calf")
]
TAU_LIMIT = 23.7   # motor torque limit (N*m), from the model


class Go2Sim:
    def __init__(self, xml_path, kp=300.0, kd=7.0, timestep=None):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        if timestep is not None:
            self.model.opt.timestep = timestep
        self.data = mujoco.MjData(self.model)
        self.kp = kp
        self.kd = kd
        self.target_q = np.zeros(12)
        self.viewer = None
        self._renderer = None
        self._cam_size = None
        # Separate offscreen renderer + tracking camera for the third-person
        # "scene" view streamed to the web panel (kept apart from the onboard
        # camera so the two different sizes don't thrash a single renderer).
        self._scene_renderer = None
        self._scene_size = None
        self._scene_cam = None
        # Thread-safe snapshot of the sim state so camera/scene rendering can run
        # on a background thread without racing the physics step (see snapshot()).
        self._render_lock = threading.Lock()
        self._snap = mujoco.MjData(self.model)
        self._lidar_sid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "lidar")
        # qvel/qpos address of the sports ball's free joint (GUI ball-driving / reset)
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "sports_ball")
        self.ball_dofadr = int(self.model.body_dofadr[bid]) if bid >= 0 else -1
        self.ball_qposadr = (int(self.model.jnt_qposadr[self.model.body_jntadr[bid]])
                             if bid >= 0 else -1)
        self._ball_vel = None        # (vx, vy) to hold this step, or None = free physics
        # Optional manipulator arm: any actuators beyond the 12 leg motors (e.g. the
        # AgileX Piper) are position-controlled -- ctrl[12:] = arm_target each step.
        self.n_arm = max(0, self.model.nu - 12)
        self.arm_target = np.zeros(self.n_arm)
        # Online-grasp helpers (Piper): finger pad geoms + a scratch MjData for IK.
        self._arm_qadr = [self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"piper_joint{i}")]
            for i in range(1, 7)] if self.n_arm else []
        l7 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "piper_link7")
        l8 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "piper_link8")
        self._l7 = l7
        self._pad_geoms = [g for g in range(self.model.ngeom)
                           if self.model.geom_bodyid[g] in (l7, l8)
                           and self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX]
        self._ik_data = mujoco.MjData(self.model) if self.n_arm else None
        self.reset_to_stand()
        self.snapshot()                                        # seed render buffer

    def snapshot(self):
        """Copy the live sim state into the render buffer. Call once per control
        tick from the control thread; camera/scene renders then read this copy on
        their own thread, so the GPU work never blocks the control loop. We copy
        qpos/qvel and recompute the derived geometry (xpos/xmat/...) with a single
        mj_forward -- everything the renderer needs, without racing mj_step."""
        with self._render_lock:
            self._snap.qpos[:] = self.data.qpos
            self._snap.qvel[:] = self.data.qvel
            mujoco.mj_forward(self.model, self._snap)

    # --- arm forward/inverse kinematics for grasping (Piper) ---
    def body_xpos(self, name):
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        return self.data.xpos[bid].copy() if bid >= 0 else None

    def grasp_point(self):
        """World position of the gripper's finger-pad centre (live state)."""
        return np.mean([self.data.geom_xpos[g] for g in self._pad_geoms], axis=0)

    _ARM_LIM = {0: (-2.6, 2.6), 1: (0.0, 3.1), 2: (-2.95, 0.0),
                3: (-1.7, 1.7), 4: (-1.2, 1.2), 5: (-2.0, 2.0)}

    def grasp_ik(self, world_target, q_seed=None, iters=4000):
        """Solve the 6 arm joints so the finger-pad centre reaches world_target,
        keeping the closing axis horizontal. Uses a scratch MjData seeded from the
        live state (so the base pose is whatever the robot is standing at)."""
        dat = self._ik_data
        dat.qpos[:] = self.data.qpos
        dat.qvel[:] = 0
        l7 = self._l7
        rng = np.random.default_rng(0)

        def grasp_pt():
            return np.mean([dat.geom_xpos[g] for g in self._pad_geoms], axis=0)

        def cost(q):
            for k, adr in enumerate(self._arm_qadr):
                dat.qpos[adr] = q[k]
            mujoco.mj_forward(self.model, dat)
            R = dat.xmat[l7].reshape(3, 3)
            return np.sum((grasp_pt() - world_target) ** 2) * 100 + 0.2 * (1 - abs(R[1, 2]))

        q = np.array(q_seed if q_seed is not None else
                     [self.data.qpos[a] for a in self._arm_qadr], dtype=float)
        best = cost(q)
        for _ in range(iters):
            i = rng.integers(6)
            q2 = q.copy()
            lo, hi = self._ARM_LIM[i]
            q2[i] = min(hi, max(lo, q2[i] + rng.standard_normal() * 0.12))
            c = cost(q2)
            if c < best:
                best, q = c, q2
        return q

    @property
    def dt(self):
        return self.model.opt.timestep

    def reset_to_stand(self):
        mujoco.mj_resetData(self.model, self.data)
        # Use the model's "home" keyframe if present (qpos + ctrl pre-set)
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
            # A short "home" keyframe poses only the robot; any free-floating body (the
            # sports ball) gets zeroed and snaps onto the robot at the origin -> explodes.
            # Restore just the ball to its spawn (qpos0) and zero its velocity. The arm
            # joints (between the legs and the ball) are left at the keyframe's home pose.
            if self.ball_qposadr >= 0:
                a = self.ball_qposadr
                self.data.qpos[a:a + 7] = self.model.qpos0[a:a + 7]
            if self.ball_dofadr >= 0:
                self.data.qvel[self.ball_dofadr:self.ball_dofadr + 6] = 0.0
            # Hold the arm at its keyframe home pose (keyframe ctrl pre-sets the targets).
            if self.n_arm > 0:
                self.arm_target = self.data.ctrl[12:12 + self.n_arm].copy()
        self.target_q = self.data.qpos[7:19].copy()
        mujoco.mj_forward(self.model, self.data)

    def set_target(self, q12):
        self.target_q = np.asarray(q12, dtype=float)

    def _apply_pd(self):
        q = self.data.qpos[7:19]
        dq = self.data.qvel[6:18]
        tau = self.kp * (self.target_q - q) - self.kd * dq
        self.data.ctrl[:12] = np.clip(tau, -TAU_LIMIT, TAU_LIMIT)
        # Position-control the arm (if any) toward its target joint angles.
        if self.n_arm > 0:
            self.data.ctrl[12:12 + self.n_arm] = self.arm_target

    def set_arm_target(self, q):
        """Set arm joint targets (6 Piper joints + gripper); extra/short inputs clamped."""
        if self.n_arm == 0:
            return
        q = np.asarray(q, dtype=float).ravel()
        n = min(len(q), self.n_arm)
        self.arm_target[:n] = q[:n]

    def arm_state(self):
        """(positions, velocities) of the arm joints, or None if no arm."""
        if self.n_arm == 0:
            return None
        # Piper qpos starts right after the 12 legs (qpos[7:19]) -> qpos[19:].
        return (self.data.qpos[19:19 + self.n_arm].copy(),
                self.data.qvel[18:18 + self.n_arm].copy())

    def set_ball_velocity(self, vel):
        """Drive the sports ball: vel=(vx, vy) world m/s, or None to release to physics."""
        self._ball_vel = vel

    def step(self):
        self._apply_pd()
        # Apply interactive drags from the passive viewer: select a body (double-click)
        # then Ctrl+drag to push it. launch_passive doesn't feed perturbations into the
        # physics itself -- we own the stepping, so we must apply them here.
        if self.viewer is not None:
            self.data.xfrc_applied[:] = 0.0      # clear last tick's perturb force
            mujoco.mjv_applyPerturbForce(self.model, self.data, self.viewer.perturb)
        # GUI ball-driving: hold the commanded planar velocity (overrides physics) so the
        # ball rolls exactly where steered; re-set each step so mj_step doesn't decay it.
        if self.ball_dofadr >= 0 and self._ball_vel is not None:
            a = self.ball_dofadr
            self.data.qvel[a], self.data.qvel[a + 1] = self._ball_vel
        mujoco.mj_step(self.model, self.data)

    # --- state accessors ---
    def base_pose(self):
        """(x, y, z) position and (w, x, y, z) quaternion of the trunk."""
        return self.data.qpos[0:3].copy(), self.data.qpos[3:7].copy()

    def base_twist(self):
        """Linear (vx, vy, vz) and angular (wx, wy, wz) velocity of the trunk."""
        return self.data.qvel[0:3].copy(), self.data.qvel[3:6].copy()

    def joint_states(self):
        """Names, positions, velocities of the 12 leg joints."""
        return JOINT_NAMES, self.data.qpos[7:19].copy(), self.data.qvel[6:18].copy()

    def set_gains(self, kp, kd):
        self.kp, self.kd = kp, kd

    def _sensor(self, name):
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sid < 0:
            return None
        adr, dim = self.model.sensor_adr[sid], self.model.sensor_dim[sid]
        return self.data.sensordata[adr:adr + dim].copy()

    def imu(self):
        """IMU reading or None if the model has no IMU sensors.

        Returns (quat[w,x,y,z], gyro[x,y,z], accel[x,y,z]); accel includes gravity.
        """
        quat = self._sensor("imu_quat")
        if quat is None:
            return None
        return quat, self._sensor("imu_gyro"), self._sensor("imu_acc")

    # --- camera ---
    def has_camera(self, name="front_camera"):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name) >= 0

    def render_camera(self, name="front_camera", width=320, height=240):
        """Render an RGB image (H x W x 3, uint8) from a named model camera."""
        if self._renderer is None or self._cam_size != (height, width):
            self._renderer = mujoco.Renderer(self.model, height, width)
            self._cam_size = (height, width)
        with self._render_lock:
            self._renderer.update_scene(self._snap, camera=name)
        return self._renderer.render()

    def render_scene(self, width=480, height=360, distance=2.2,
                     azimuth=135.0, elevation=-20.0):
        """Render a third-person chase view (H x W x 3, uint8) that follows the
        robot. Uses a free tracking camera locked onto the trunk body, so it
        needs no camera in the model XML. Streamed to the web panel as the
        live "simulation" view."""
        if self._scene_renderer is None or self._scene_size != (height, width):
            self._scene_renderer = mujoco.Renderer(self.model, height, width)
            self._scene_size = (height, width)
        if self._scene_cam is None:
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
            cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
            self._scene_cam = cam
        with self._render_lock:
            self._scene_renderer.update_scene(self._snap, camera=self._scene_cam)
        return self._scene_renderer.render()

    # --- lidar (horizontal 360 deg scan via ray casting) ---
    def lidar_scan(self, n_rays=180, range_max=10.0):
        """Cast n_rays in the horizontal plane; return distances (range_max = miss).

        Robot geoms (groups 2 & 3) are masked out so rays see only the world.
        """
        if self._lidar_sid < 0:
            return None
        pnt = np.ascontiguousarray(self.data.site_xpos[self._lidar_sid], dtype=np.float64)
        Rm = self.data.site_xmat[self._lidar_sid].reshape(3, 3)
        # Use only the trunk YAW and cast in the WORLD-horizontal plane, so body
        # roll/pitch during the gait can't tilt the scan. Tilted scans corrupt 2D
        # SLAM (causes large map drift). lidar_link is published level (yaw-only)
        # in teleop_node to match this convention.
        yaw = math.atan2(Rm[1, 0], Rm[0, 0])
        geomgroup = np.array([1, 1, 0, 0, 0, 0], dtype=np.uint8)  # exclude robot
        geomid = np.zeros(1, dtype=np.int32)
        ranges = np.empty(n_rays, dtype=np.float32)
        for i in range(n_rays):
            ang = yaw - math.pi + (2.0 * math.pi) * i / n_rays
            dvec = np.array([math.cos(ang), math.sin(ang), 0.0], dtype=np.float64)
            dist = mujoco.mj_ray(self.model, self.data, pnt, dvec,
                                 geomgroup, 1, -1, geomid)
            ranges[i] = dist if 0.0 <= dist <= range_max else range_max
        return ranges

    def lidar_cloud(self, n_az=120, elevations_deg=(-15, -7, 0, 7, 15),
                    range_max=10.0):
        """3D point cloud: rays over azimuth x elevation. Returns Nx3 (lidar frame)."""
        if self._lidar_sid < 0:
            return None
        pnt = np.ascontiguousarray(self.data.site_xpos[self._lidar_sid], dtype=np.float64)
        Rm = self.data.site_xmat[self._lidar_sid].reshape(3, 3)
        geomgroup = np.array([1, 1, 0, 0, 0, 0], dtype=np.uint8)  # exclude robot
        geomid = np.zeros(1, dtype=np.int32)
        pts = []
        for el in (math.radians(e) for e in elevations_deg):
            ce, se = math.cos(el), math.sin(el)
            for i in range(n_az):
                az = -math.pi + (2.0 * math.pi) * i / n_az
                local = np.array([ce * math.cos(az), ce * math.sin(az), se])
                dvec = np.ascontiguousarray(Rm @ local, dtype=np.float64)
                dist = mujoco.mj_ray(self.model, self.data, pnt, dvec,
                                     geomgroup, 1, -1, geomid)
                if 0.0 <= dist <= range_max:
                    pts.append(local * dist)        # point in the lidar frame
        return np.array(pts, dtype=np.float32) if pts else np.empty((0, 3), np.float32)

    # --- visualization ---
    def launch_viewer(self):
        import mujoco.viewer
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        return self.viewer

    def sync_viewer(self):
        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()
            return True
        return self.viewer is None  # True if no viewer (headless), keep running

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
