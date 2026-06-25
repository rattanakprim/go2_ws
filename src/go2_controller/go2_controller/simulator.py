"""
Thin MuJoCo wrapper for the Go2.

Holds the model/data, drives the 12 torque motors with a PD controller toward
target joint angles, and exposes the robot state (base pose, joint states).
Used by the ROS 2 node, but has no ROS dependency so it can be tested alone.
"""
import math

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
        self._lidar_sid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "lidar")
        # qvel address of the sports ball's free joint (for GUI ball-driving), if present
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "sports_ball")
        self.ball_dofadr = int(self.model.body_dofadr[bid]) if bid >= 0 else -1
        self._ball_vel = None        # (vx, vy) to hold this step, or None = free physics
        self.reset_to_stand()

    @property
    def dt(self):
        return self.model.opt.timestep

    def reset_to_stand(self):
        mujoco.mj_resetData(self.model, self.data)
        # Use the model's "home" keyframe if present (qpos + ctrl pre-set)
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
            # The "home" keyframe only poses the robot (first 19 qpos). Any extra free
            # bodies in the scene (e.g. the sports ball) get zeroed by the keyframe ->
            # they snap to the world origin onto the robot and explode. Restore those
            # trailing DOF to their model spawn (qpos0) and zero their velocity.
            if self.model.nq > 19:
                self.data.qpos[19:] = self.model.qpos0[19:]
                self.data.qvel[18:] = 0.0
        self.target_q = self.data.qpos[7:19].copy()
        mujoco.mj_forward(self.model, self.data)

    def set_target(self, q12):
        self.target_q = np.asarray(q12, dtype=float)

    def _apply_pd(self):
        q = self.data.qpos[7:19]
        dq = self.data.qvel[6:18]
        tau = self.kp * (self.target_q - q) - self.kd * dq
        self.data.ctrl[:] = np.clip(tau, -TAU_LIMIT, TAU_LIMIT)

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
        self._renderer.update_scene(self.data, camera=name)
        return self._renderer.render()

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
