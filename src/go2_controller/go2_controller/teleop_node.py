"""
ROS 2 teleop controller node for the Go2 (simulated in MuJoCo).

Subscribes : /cmd_vel   (geometry_msgs/Twist)   -- drive command
Publishes  : /joint_states (sensor_msgs/JointState)
             /odom         (nav_msgs/Odometry)   -- simulated base pose & twist

The node owns a MuJoCo simulation of the Go2, turns the incoming twist into a
crawl gait (see gait.py), and PD-controls the legs (see simulator.py).
Drive it with, e.g.:  ros2 run teleop_twist_keyboard teleop_twist_keyboard
"""
import math
import os

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import Twist, TransformStamped, PoseStamped, PoseArray
from tf2_ros import (TransformBroadcaster, StaticTransformBroadcaster,
                     Buffer, TransformListener)
from sensor_msgs.msg import (
    JointState, Imu, Image, CameraInfo, LaserScan, PointCloud2,
)
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Header, Float64MultiArray, Bool

try:
    from sensor_msgs_py import point_cloud2 as pc2
    _HAVE_PC2 = True
except ImportError:
    _HAVE_PC2 = False
from ament_index_python.packages import get_package_share_directory

from .gait import TrotGait, GAITS
from .routines import ROUTINES
from .simulator import Go2Sim


def default_model_path():
    # Go2 with the AgileX Piper arm mounted on its back. To run the bare Go2, override:
    #   --ros-args -p model_path:=<.../mujoco_menagerie/unitree_go2/go2_imu_scene.xml>
    share = get_package_share_directory("go2_controller")
    return os.path.join(share, "models", "go2_piper", "go2_piper_control.xml")


class Go2TeleopNode(Node):
    def __init__(self):
        super().__init__("go2_controller")

        # --- parameters ---
        self.declare_parameter("model_path", default_model_path())
        self.declare_parameter("use_viewer", True)
        self.declare_parameter("control_rate", 50.0)   # Hz
        self.declare_parameter("cmd_timeout", 0.5)     # s, stop if no cmd_vel
        self.declare_parameter("max_lin_acc", 1.5)     # m/s^2, command slew limit
        self.declare_parameter("max_ang_acc", 5.0)     # rad/s^2, command slew limit
        self.declare_parameter("publish_tf", True)     # broadcast odom->base_link TF
        self.declare_parameter("goal_pos_tol", 0.2)    # m, when to start final-heading turn
        self.declare_parameter("goal_yaw_tol", 0.1)    # rad, final-heading tolerance
        self.declare_parameter("kp", 300.0)
        self.declare_parameter("kd", 7.0)
        self.declare_parameter("use_camera", True)
        self.declare_parameter("use_lidar", True)
        self.declare_parameter("camera_rate", 15.0)    # Hz
        self.declare_parameter("camera_width", 320)
        self.declare_parameter("camera_height", 240)
        self.declare_parameter("lidar_rate", 10.0)     # Hz
        self.declare_parameter("lidar_rays", 360)      # 1° resolution (better SLAM)
        self.declare_parameter("lidar_range", 10.0)    # m
        self.declare_parameter("lidar_cloud_rate", 8.0)  # Hz (3D PointCloud2)

        model_path = self.get_parameter("model_path").value
        use_viewer = self.get_parameter("use_viewer").value
        rate = self.get_parameter("control_rate").value
        self.rate = rate
        self.cmd_timeout = self.get_parameter("cmd_timeout").value
        self.max_lin_acc = self.get_parameter("max_lin_acc").value
        self.max_ang_acc = self.get_parameter("max_ang_acc").value
        kp = self.get_parameter("kp").value
        kd = self.get_parameter("kd").value

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Go2 model not found: {model_path}")

        self.sim = Go2Sim(model_path, kp=kp, kd=kd)
        self.gait = TrotGait()
        self.sim.set_target(self.gait.stand_pose())
        self.sim_time = 0.0

        # walking PD gains (restored after a routine)
        self.kp_walk, self.kd_walk = kp, kd

        # posture mode: "walk" | "sit" | "crouch" | "liedown"
        self.mode = "walk"
        # active timed routine (jump/wave/shake/dance/selfright) or None
        self.routine = None
        self.routine_start = 0.0
        # body pose applied while walking/standing (radians, height in m)
        self.body_pose = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0, "height": 0.0}

        self._postures = {
            "sit": self.gait.sit_pose,
            "crouch": self.gait.crouch_pose,
            "liedown": self.gait.liedown_pose,
        }

        # how many physics steps per control tick (keeps it ~real-time)
        self.steps_per_tick = max(1, int(round((1.0 / rate) / self.sim.dt)))

        # --- command state ---
        # vx/vy/wz = latest commanded target; cvx/cvy/cwz = slew-limited value
        # actually applied to the gait (smoothed so network jitter / abrupt
        # stops don't jerk the legs).
        self.vx = self.vy = self.wz = 0.0
        self.cvx = self.cvy = self.cwz = 0.0
        self.last_cmd_time = self.get_clock().now()

        # --- ROS interfaces ---
        self.create_subscription(Twist, "cmd_vel", self.on_cmd_vel, 10)
        self.create_subscription(String, "go2/action", self.on_action, 10)
        self.create_subscription(Twist, "go2/body_pose", self.on_body_pose, 10)
        # GUI ball-driving: robot-frame (forward, left) velocity for the sports ball
        # (easier than Ctrl-dragging in the viewer). Held while fresh, else free physics.
        self._ball_cmd = (0.0, 0.0)
        self._ball_cmd_t = None
        self.create_subscription(Twist, "go2/ball_cmd", self.on_ball_cmd, 10)
        # Arm (Piper) joint targets: Float64MultiArray of 7 = [j1..j6, gripper] (radians,
        # gripper in metres). Only active when the model has an arm (sim.n_arm > 0).
        self._task = None          # active pick/place arm routine (list of phases)
        if self.sim.n_arm > 0:
            self.create_subscription(Float64MultiArray, "go2/arm_cmd",
                                     self.on_arm_cmd, 10)
            # Autonomous grasp/place: the controller has sim access so it solves IK to
            # the actual item / table (robust to where the robot stopped walking).
            self.create_subscription(Bool, "go2/pick", self.on_pick, 1)
            self.create_subscription(Float64MultiArray, "go2/place", self.on_place, 1)
            self.create_subscription(Bool, "go2/run_pickplace", self.on_run_pickplace, 1)
            self.get_logger().info(
                f"Arm detected ({self.sim.n_arm} joints); listening on /go2/arm_cmd, "
                "/go2/pick, /go2/place, /go2/run_pickplace.")
        self._demo = None          # active one-button pick-place demo plan
        self.joint_pub = self.create_publisher(JointState, "joint_states", 10)
        self.odom_pub = self.create_publisher(Odometry, "odom", 10)
        self.imu_pub = self.create_publisher(Imu, "imu", 10)

        # --- TF: needed by SLAM / Nav2 (map->odom comes from slam_toolbox) ---
        self.publish_tf = self.get_parameter("publish_tf").value
        if self.publish_tf:
            self.tf_broadcaster = TransformBroadcaster(self)
            self.static_tf = StaticTransformBroadcaster(self)
            self._publish_static_tf()

        # Final-heading alignment: Nav2/RPP reaches the goal position but not an
        # arbitrary final orientation, so we finish the in-place turn ourselves.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.nav_goal = None                                # (x, y, yaw) in map, or None
        self._yaw_ok = 0                                    # consecutive on-target ticks
        self.goal_pos_tol = self.get_parameter("goal_pos_tol").value
        self.goal_yaw_tol = self.get_parameter("goal_yaw_tol").value
        self.create_subscription(PoseStamped, "goal_pose", self.on_goal_pose, 10)

        # Waypoint route: a PoseArray (map frame) of ordered goals. We visit them
        # one at a time by republishing each as /goal_pose (the same path single
        # goals use) and advancing to the next once the current one is reached.
        self.goal_pub = self.create_publisher(PoseStamped, "goal_pose", 10)
        self.create_subscription(PoseArray, "go2/waypoints", self.on_waypoints, 10)
        self._route = []            # list of (x, y, yaw) in the map frame
        self._route_i = 0
        self._route_active = False

        # --- onboard sensors (camera + lidar) ---
        self.cam_w = self.get_parameter("camera_width").value
        self.cam_h = self.get_parameter("camera_height").value
        self._cam_failed = False
        if self.get_parameter("use_camera").value and self.sim.has_camera():
            self.img_pub = self.create_publisher(Image, "camera/image_raw", 10)
            self.caminfo_pub = self.create_publisher(CameraInfo, "camera/camera_info", 10)
            self.create_timer(1.0 / self.get_parameter("camera_rate").value,
                              self.camera_tick)
        if self.get_parameter("use_lidar").value:
            self.lidar_rays = self.get_parameter("lidar_rays").value
            self.lidar_range = self.get_parameter("lidar_range").value
            self.scan_pub = self.create_publisher(LaserScan, "scan", 10)
            self.create_timer(1.0 / self.get_parameter("lidar_rate").value,
                              self.lidar_tick)
            if _HAVE_PC2:                          # 3D point cloud for RViz
                self.points_pub = self.create_publisher(PointCloud2, "points", 5)
                self.create_timer(1.0 / self.get_parameter("lidar_cloud_rate").value,
                                  self.cloud_tick)
            else:
                self.get_logger().warn("sensor_msgs_py missing; /points (3D) disabled")

        if use_viewer:
            try:
                self.sim.launch_viewer()
                self.get_logger().info("MuJoCo viewer launched.")
            except Exception as exc:  # headless / no display
                self.get_logger().warn(f"Could not launch viewer ({exc}); "
                                       "running headless.")

        self.create_timer(1.0 / rate, self.control_tick)
        self.get_logger().info(
            f"Go2 controller up. Listening on /cmd_vel, {self.steps_per_tick} "
            f"physics steps/tick @ {rate:.0f} Hz.")

    def _publish_static_tf(self):
        """Static base_link -> lidar_link (lidar mount from go2.xml: pos 0.05 0 0.12)."""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "base_link"
        t.child_frame_id = "lidar_link"
        t.transform.translation.x = 0.05
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.12
        t.transform.rotation.w = 1.0
        self.static_tf.sendTransform([t])

    def on_cmd_vel(self, msg: Twist):
        self.vx = msg.linear.x
        self.vy = msg.linear.y
        self.wz = msg.angular.z
        self.last_cmd_time = self.get_clock().now()

    def on_goal_pose(self, msg: PoseStamped):
        q = msg.pose.orientation
        yaw = math.atan2(2.0 * (q.w*q.z + q.x*q.y), 1.0 - 2.0 * (q.y*q.y + q.z*q.z))
        self.nav_goal = (msg.pose.position.x, msg.pose.position.y, yaw)

    def on_waypoints(self, msg: PoseArray):
        """Receive an ordered route (map frame). Empty array cancels the route."""
        pts = [(p.position.x, p.position.y) for p in msg.poses]
        if not pts:
            self._route, self._route_active, self.nav_goal = [], False, None
            self.get_logger().info("waypoint route: cancelled")
            return
        route = []
        for k, (x, y) in enumerate(pts):
            if k < len(pts) - 1:                  # face toward the next waypoint
                nx, ny = pts[k + 1]
                yaw = math.atan2(ny - y, nx - x)
            elif k > 0:                           # last point: keep arrival heading
                px, py = pts[k - 1]
                yaw = math.atan2(y - py, x - px)
            else:                                 # single waypoint
                yaw = 0.0
            route.append((x, y, yaw))
        self._route, self._route_i, self._route_active = route, 0, True
        self.get_logger().info(f"waypoint route: {len(route)} point(s)")
        self._send_waypoint()

    def _send_waypoint(self):
        """Publish the current waypoint as a /goal_pose so Nav2 drives to it."""
        x, y, yaw = self._route[self._route_i]
        m = PoseStamped()
        m.header.frame_id = "map"
        m.header.stamp = self.get_clock().now().to_msg()
        m.pose.position.x, m.pose.position.y = x, y
        m.pose.orientation.z = math.sin(yaw / 2.0)
        m.pose.orientation.w = math.cos(yaw / 2.0)
        self.goal_pub.publish(m)
        self.nav_goal = (x, y, yaw)               # drive + final-heading target
        self.get_logger().info(
            f"waypoint {self._route_i + 1}/{len(self._route)} -> ({x:.2f}, {y:.2f})")

    def _pose_in_map(self):
        """Current base_link pose (x, y, yaw) in the map frame, or None."""
        try:
            t = self.tf_buffer.lookup_transform("map", "base_link", Time())
        except Exception:
            return None
        tr, q = t.transform.translation, t.transform.rotation
        yaw = math.atan2(2.0 * (q.w*q.z + q.x*q.y), 1.0 - 2.0 * (q.y*q.y + q.z*q.z))
        return tr.x, tr.y, yaw

    def _final_heading(self, vx, vy, wz):
        """Once at the goal position, rotate in place to the commanded final yaw.

        Proportional + gentle speed so the open-loop crawl gait (which keeps
        stepping a moment after the command changes) doesn't overshoot, and we
        only release the goal once the heading has *settled* on target.
        """
        pose = self._pose_in_map()
        if pose is None:
            return vx, vy, wz
        mx, my, myaw = pose
        gx, gy, gyaw = self.nav_goal
        if math.hypot(gx - mx, gy - my) > self.goal_pos_tol:
            self._yaw_ok = 0
            return vx, vy, wz                  # still approaching -> let Nav2 drive
        err = math.atan2(math.sin(gyaw - myaw), math.cos(gyaw - myaw))
        if abs(err) <= self.goal_yaw_tol:
            self._yaw_ok += 1
            if self._yaw_ok >= 8:              # ~0.16 s on target -> done
                self.nav_goal = None
                return 0.0, 0.0, 0.0
        else:
            self._yaw_ok = 0
        wz = max(-0.4, min(0.4, 1.5 * err))    # gentle, proportional -> low overshoot
        self.cvx = self.cvy = 0.0
        self.cwz = wz
        return 0.0, 0.0, wz

    def on_action(self, msg: String):
        a = msg.data.strip().lower()
        if a in ROUTINES:                         # timed move (jump/wave/dance/...)
            if self.routine != a:
                self.routine = a
                self.routine_start = self.sim_time
                self.get_logger().info(f"routine: {a}")
        elif a in ("stand", "walk"):
            self.routine, self.mode = None, "walk"
            self.get_logger().info("action: stand")
        elif a in ("liedown", "lie_down", "lay"):
            self.routine, self.mode = None, "liedown"
            self.get_logger().info("action: liedown")
        elif a in ("sit", "crouch"):
            self.routine, self.mode = None, a
            self.get_logger().info(f"action: {a}")
        elif a.startswith("gait_") or a.startswith("gait:"):
            name = a.replace("gait_", "").replace("gait:", "")
            if self.gait.set_gait(name):
                self.get_logger().info(f"gait: {name}")
            else:
                self.get_logger().warn(f"unknown gait '{name}' (have {list(GAITS)})")
        else:
            self.get_logger().warn(f"unknown action '{a}'")

    def on_body_pose(self, msg: Twist):
        self.body_pose["roll"] = msg.angular.x
        self.body_pose["pitch"] = msg.angular.y
        self.body_pose["yaw"] = msg.angular.z
        self.body_pose["height"] = msg.linear.z      # 0 => default stand height

    def on_ball_cmd(self, msg: Twist):
        self._ball_cmd = (msg.linear.x, msg.linear.y)   # robot-frame (forward, left) m/s
        self._ball_cmd_t = self.get_clock().now()

    def _target_for_mode(self, vx, vy, wz):
        """Compute the 12 joint targets for the current behavior."""
        # 1) an active timed routine overrides everything
        if self.routine is not None:
            te = self.sim_time - self.routine_start
            targets, kp, kd, done = ROUTINES[self.routine](self.gait, te)
            self.sim.set_gains(kp, kd)
            if done:
                self.routine = None
            return targets

        moving = abs(vx) + abs(vy) + abs(wz) > 1e-6

        # 2) held postures (driving stands sit/crouch back up; lie-down is sticky)
        if self.mode in self._postures:
            if moving and self.mode != "liedown":
                self.mode = "walk"
            else:
                self.sim.set_gains(self.kp_walk, self.kd_walk)
                return self._postures[self.mode]()

        # 3) walk + body pose
        self.sim.set_gains(self.kp_walk, self.kd_walk)
        bp = self.body_pose
        height = bp["height"] if bp["height"] > 0.05 else None
        return self.gait.joint_targets(vx, vy, wz, self.sim_time,
                                       roll=bp["roll"], pitch=bp["pitch"],
                                       yaw=bp["yaw"], height=height)

    @staticmethod
    def _slew(current, target, max_delta):
        """Move `current` toward `target` by at most `max_delta` per call."""
        delta = target - current
        if delta > max_delta:
            return current + max_delta
        if delta < -max_delta:
            return current - max_delta
        return target

    def control_tick(self):
        self._demo_tick()           # autonomous pick-place demo drives vx/vy/wz + arm

        # Safety (deadman watchdog): target zero if cmd_vel went stale.
        dt_cmd = (self.get_clock().now() - self.last_cmd_time).nanoseconds * 1e-9
        tx, ty, tz = (self.vx, self.vy, self.wz) if dt_cmd < self.cmd_timeout \
            else (0.0, 0.0, 0.0)

        # Slew-rate limit toward the target so jitter / abrupt stops are smooth.
        dt = 1.0 / self.rate
        self.cvx = self._slew(self.cvx, tx, self.max_lin_acc * dt)
        self.cvy = self._slew(self.cvy, ty, self.max_lin_acc * dt)
        self.cwz = self._slew(self.cwz, tz, self.max_ang_acc * dt)
        vx, vy, wz = self.cvx, self.cvy, self.cwz

        # After Nav2 delivers us to the goal position, finish the heading turn.
        if self.nav_goal is not None:
            vx, vy, wz = self._final_heading(vx, vy, wz)

        # Waypoint route: the current goal was reached (nav_goal cleared) -> next.
        if self._route_active and self.nav_goal is None:
            self._route_i += 1
            if self._route_i < len(self._route):
                self._send_waypoint()
            else:
                self._route_active = False
                self.get_logger().info("waypoint route: complete")

        # Drive the ball while a fresh command is held; otherwise let physics roll it.
        # The command is ROBOT-FRAME (forward, left); rotate it by the base yaw so the
        # buttons feel intuitive from the robot's point of view regardless of heading.
        if (self._ball_cmd_t is not None and
                (self.get_clock().now() - self._ball_cmd_t).nanoseconds * 1e-9 < 0.3):
            fwd, left = self._ball_cmd
            _, q = self.sim.base_pose()          # MuJoCo quat [w, x, y, z]
            yaw = math.atan2(2.0 * (q[0]*q[3] + q[1]*q[2]),
                             1.0 - 2.0 * (q[2]*q[2] + q[3]*q[3]))
            c, s = math.cos(yaw), math.sin(yaw)
            self.sim.set_ball_velocity((fwd*c - left*s, fwd*s + left*c))
        else:
            self.sim.set_ball_velocity(None)

        self._task_tick()           # advance an autonomous pick/place arm routine

        for _ in range(self.steps_per_tick):
            self.sim.set_target(self._target_for_mode(vx, vy, wz))
            self.sim.step()
            self.sim_time += self.sim.dt

        if not self.sim.sync_viewer():      # viewer window was closed
            self.get_logger().info("Viewer closed; continuing headless "
                                   "(sensors & commands still active).")
            self.sim.close()               # drop the viewer, keep the node running

        self.publish_state()

    def on_arm_cmd(self, msg: Float64MultiArray):
        """Set the arm (Piper) joint targets from a Float64MultiArray."""
        if self._task is None:                 # don't fight an autonomous pick/place
            self.sim.set_arm_target(msg.data)

    HOME_ARM = [0.0, 1.57, -1.3485, 0.0, 0.0, 0.0]
    GRIP_OPEN, GRIP_CLOSE, LIFT_DJ2 = 0.030, 0.0, -0.40

    def on_pick(self, msg: Bool):
        if msg.data:
            self._do_pick()

    def _do_pick(self):
        """Autonomously grasp the 'pick_item' wherever it is in front of the robot."""
        if self._task is not None:
            return False
        item = self.sim.body_xpos("pick_item")
        if item is None:
            return False
        self._grasp_q = self.sim.grasp_ik(item)
        pre = list(self._grasp_q); pre[1] += self.LIFT_DJ2
        self._start_task([
            {"arm": pre + [self.GRIP_OPEN], "dwell": 2.0},
            {"arm": list(self._grasp_q) + [self.GRIP_OPEN], "dwell": 2.0},
            {"arm": "OPEN", "dwell": 1.5, "recal": "pick_item"},   # re-aim on the item
            {"arm": "CLOSE", "dwell": 2.0},
            {"arm": "LIFT", "dwell": 2.0},
        ])
        self.get_logger().info("pick: grasping item")
        return True

    def on_place(self, msg: Float64MultiArray):
        if len(msg.data) >= 3:
            self._do_place(np.array(msg.data[:3]))

    def _do_place(self, tb):
        """Lower the held item onto a target [x, y, z] (table surface) and release."""
        if self._task is not None:
            return False
        self._grasp_q = self.sim.grasp_ik(tb)
        pre = list(self._grasp_q); pre[1] += self.LIFT_DJ2
        self._start_task([
            {"arm": pre + [self.GRIP_CLOSE], "dwell": 2.0},
            {"arm": "CLOSE", "dwell": 2.0},                        # lower to the table
            {"arm": "OPEN", "dwell": 1.5},                         # release
            {"arm": pre + [self.GRIP_OPEN], "dwell": 1.5},         # retract up
            {"arm": self.HOME_ARM + [self.GRIP_OPEN], "dwell": 1.0},
        ])
        self.get_logger().info(f"place: lowering onto {tuple(round(v, 2) for v in tb)}")
        return True

    # --- one-button autonomous pick-place demo (A -> B -> A) ---
    DEMO_STANDS = {"A": (0.60, 0.50, 0.0), "B": (0.60, -0.50, 0.0)}
    DEMO_TOPS = {"A": (1.10, 0.50, 0.30), "B": (1.10, -0.50, 0.30)}
    DEMO_PLAN = [("walk", "A"), ("pick", None), ("walk", "B"), ("place", "B"),
                 ("pick", None), ("walk", "A"), ("place", "A")]

    def on_run_pickplace(self, msg: Bool):
        if not msg.data or self._demo is not None:
            return
        if self.sim.body_xpos("pick_item") is None:
            self.get_logger().warn("run_pickplace ignored: not a pick-place scene.")
            return
        self._demo = list(self.DEMO_PLAN)
        self._demo_i = 0
        self.get_logger().info("pick-place demo: started")

    def _demo_tick(self):
        """Drive the demo: walk to a stand, then pick/place; called each control tick."""
        if self._demo is None:
            return
        if self._task is not None:           # an arm routine is running -> hold still
            self.vx = self.vy = self.wz = 0.0
            self.last_cmd_time = self.get_clock().now()
            return
        if self._demo_i >= len(self._demo):
            self.vx = self.vy = self.wz = 0.0
            self._demo = None
            self.get_logger().info("pick-place demo: done")
            return
        kind, arg = self._demo[self._demo_i]
        if kind == "walk":
            if self._demo_walk(self.DEMO_STANDS[arg]):
                self._demo_i += 1
        elif kind == "pick":
            self._do_pick(); self._demo_i += 1
        elif kind == "place":
            self._do_place(np.array(self.DEMO_TOPS[arg])); self._demo_i += 1

    def _demo_walk(self, target):
        """Set drive command toward (x, y, yaw); return True when arrived."""
        pos, quat = self.sim.base_pose()
        x, y = pos[0], pos[1]
        yaw = math.atan2(2 * (quat[0]*quat[3] + quat[1]*quat[2]),
                         1 - 2 * (quat[2]*quat[2] + quat[3]*quat[3]))
        ex, ey = target[0] - x, target[1] - y
        dist = math.hypot(ex, ey)
        yaw_err = math.atan2(math.sin(target[2] - yaw), math.cos(target[2] - yaw))
        self.last_cmd_time = self.get_clock().now()
        V, W = 0.22, 0.5
        if dist > 0.07:
            fx = math.cos(yaw) * ex + math.sin(yaw) * ey
            fy = -math.sin(yaw) * ex + math.cos(yaw) * ey
            self.vx = max(-V, min(V, 0.8 * fx))
            self.vy = max(-V, min(V, 0.8 * fy))
            self.wz = max(-W, min(W, 1.0 * yaw_err))
            return False
        if abs(yaw_err) > 0.06:
            self.vx = self.vy = 0.0
            self.wz = max(-W, min(W, 1.2 * yaw_err))
            return False
        self.vx = self.vy = self.wz = 0.0
        return True

    def _start_task(self, phases):
        self._task = phases
        self._task_i = -1
        self._advance_phase()

    def _resolve_arm(self, a):
        if a is None:
            return None
        if isinstance(a, str):
            q = list(self._grasp_q)
            if a == "OPEN":
                return q + [self.GRIP_OPEN]
            if a == "CLOSE":
                return q + [self.GRIP_CLOSE]
            if a == "LIFT":
                q[1] += self.LIFT_DJ2
                return q + [self.GRIP_CLOSE]
        return a

    def _advance_phase(self):
        self._task_i += 1
        if self._task_i >= len(self._task):
            self._task = None
            return
        ph = self._task[self._task_i]
        if "recal" in ph:                       # re-aim IK at the item's actual position
            item = self.sim.body_xpos(ph["recal"])
            if item is not None:
                err = item - self.sim.grasp_point()
                self._grasp_q = self.sim.grasp_ik(item + err, q_seed=self._grasp_q,
                                                  iters=2500)
        arm = self._resolve_arm(ph["arm"])
        if arm is not None:
            self.sim.set_arm_target(arm)
        self._phase_end = self.get_clock().now() + rclpy.duration.Duration(seconds=ph["dwell"])

    def _task_tick(self):
        if self._task is not None and self.get_clock().now() >= self._phase_end:
            self._advance_phase()

    def publish_state(self):
        now = self.get_clock().now().to_msg()

        names, qpos, qvel = self.sim.joint_states()
        names = list(names)
        qpos = list(qpos)
        qvel = list(qvel)
        arm = self.sim.arm_state()                 # append Piper joints if present
        if arm is not None:
            aq, av = arm
            names += [f"piper_joint{i}" for i in range(1, self.sim.n_arm + 1)]
            qpos += list(aq)
            qvel += list(av)
        js = JointState()
        js.header.stamp = now
        js.name = names
        js.position = qpos
        js.velocity = qvel
        self.joint_pub.publish(js)

        pos, quat = self.sim.base_pose()
        lin, ang = self.sim.base_twist()
        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = float(pos[0])
        odom.pose.pose.position.y = float(pos[1])
        odom.pose.pose.position.z = float(pos[2])
        odom.pose.pose.orientation.w = float(quat[0])
        odom.pose.pose.orientation.x = float(quat[1])
        odom.pose.pose.orientation.y = float(quat[2])
        odom.pose.pose.orientation.z = float(quat[3])
        # Odometry.twist is in the child (base_link) frame, but MuJoCo's free-joint
        # linear velocity is WORLD-frame -> rotate the planar part into the body by -yaw
        # (Nav2's controller reads /odom for velocity feedback). Angular is already body.
        yaw = math.atan2(2.0 * (quat[0]*quat[3] + quat[1]*quat[2]),
                         1.0 - 2.0 * (quat[2]**2 + quat[3]**2))
        cy, sy = math.cos(yaw), math.sin(yaw)
        odom.twist.twist.linear.x = float(lin[0] * cy + lin[1] * sy)
        odom.twist.twist.linear.y = float(-lin[0] * sy + lin[1] * cy)
        odom.twist.twist.linear.z = float(lin[2])
        odom.twist.twist.angular.z = float(ang[2])
        self.odom_pub.publish(odom)

        # TF: odom -> base_link (same pose), so the SLAM/Nav2 transform tree is live
        if self.publish_tf:
            tf = TransformStamped()
            tf.header.stamp = now
            tf.header.frame_id = "odom"
            tf.child_frame_id = "base_link"
            tf.transform.translation.x = float(pos[0])
            tf.transform.translation.y = float(pos[1])
            tf.transform.translation.z = float(pos[2])
            # Level the frame: yaw only (drop roll/pitch) so lidar_link stays
            # horizontal -> consistent with the yaw-only lidar scan, stable 2D SLAM.
            # (yaw computed above for the odom twist.)
            tf.transform.rotation.z = math.sin(yaw / 2.0)
            tf.transform.rotation.w = math.cos(yaw / 2.0)
            self.tf_broadcaster.sendTransform(tf)

        imu = self.sim.imu()
        if imu is not None:
            quat, gyro, acc = imu
            m = Imu()
            m.header.stamp = now
            m.header.frame_id = "imu_link"
            m.orientation.w = float(quat[0])
            m.orientation.x = float(quat[1])
            m.orientation.y = float(quat[2])
            m.orientation.z = float(quat[3])
            m.angular_velocity.x = float(gyro[0])
            m.angular_velocity.y = float(gyro[1])
            m.angular_velocity.z = float(gyro[2])
            m.linear_acceleration.x = float(acc[0])
            m.linear_acceleration.y = float(acc[1])
            m.linear_acceleration.z = float(acc[2])
            self.imu_pub.publish(m)

    def camera_tick(self):
        if self._cam_failed:
            return
        try:
            img = self.sim.render_camera("front_camera", self.cam_w, self.cam_h)
        except Exception as exc:                 # GL context issue, headless, etc.
            self._cam_failed = True
            self.get_logger().warn(f"camera disabled ({exc})")
            return
        now = self.get_clock().now().to_msg()
        h, w = img.shape[0], img.shape[1]

        msg = Image()
        msg.header.stamp = now
        msg.header.frame_id = "camera_link"
        msg.height, msg.width = h, w
        msg.encoding = "rgb8"
        msg.is_bigendian = 0
        msg.step = w * 3
        msg.data = img.tobytes()
        self.img_pub.publish(msg)

        # minimal CameraInfo (pinhole from the camera fovy = 58 deg)
        info = CameraInfo()
        info.header = msg.header
        info.height, info.width = h, w
        f = h / (2.0 * math.tan(math.radians(58.0) / 2.0))
        cx, cy = w / 2.0, h / 2.0
        info.k = [f, 0.0, cx, 0.0, f, cy, 0.0, 0.0, 1.0]
        info.p = [f, 0.0, cx, 0.0, 0.0, f, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        info.distortion_model = "plumb_bob"
        self.caminfo_pub.publish(info)

    def lidar_tick(self):
        ranges = self.sim.lidar_scan(self.lidar_rays, self.lidar_range)
        if ranges is None:
            return
        n = len(ranges)
        scan = LaserScan()
        scan.header.stamp = self.get_clock().now().to_msg()
        scan.header.frame_id = "lidar_link"
        scan.angle_min = -math.pi
        scan.angle_max = math.pi - (2.0 * math.pi / n)
        scan.angle_increment = 2.0 * math.pi / n
        scan.range_min = 0.1
        scan.range_max = float(self.lidar_range)
        scan.ranges = ranges.tolist()
        self.scan_pub.publish(scan)

    def cloud_tick(self):
        pts = self.sim.lidar_cloud(range_max=self.lidar_range)
        if pts is None:
            return
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "lidar_link"
        self.points_pub.publish(pc2.create_cloud_xyz32(header, pts.tolist()))


def main(args=None):
    rclpy.init(args=args)
    node = Go2TeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.sim.close()
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
