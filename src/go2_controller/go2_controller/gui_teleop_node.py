"""
GUI control panel for the Go2 -- a tkinter window that drives the controller.

Publishes:
  /cmd_vel       (geometry_msgs/Twist)  drive (buttons / joystick / keys)
  /go2/action    (std_msgs/String)      postures, gaits, expressive moves
  /go2/body_pose (geometry_msgs/Twist)  roll/pitch/yaw lean + ride height

Keyboard (panel focused): WASD/QE move, Space stop, j jump, x sit, z stand.

Run:  ros2 run go2_controller gui_teleop
"""
import math
import os
import threading
import time
import tkinter as tk

import numpy as np
import rclpy
import rclpy.time
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import Image, LaserScan
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import String, Bool
from tf2_ros import Buffer, TransformListener
from PIL import Image as PILImage, ImageTk

GAITS = ["crawl", "trot", "pace", "bound", "pronk"]


class Go2GuiTeleop(Node):
    def __init__(self):
        super().__init__("go2_gui_teleop")
        self.pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.action_pub = self.create_publisher(String, "go2/action", 10)
        self.pose_pub = self.create_publisher(Twist, "go2/body_pose", 10)
        self.goal_pub = self.create_publisher(PoseStamped, "goal_pose", 10)
        self.explore_pub = self.create_publisher(Bool, "go2/explore_enable", 1)
        self.follow_pub = self.create_publisher(Bool, "go2/follow_enable", 1)
        self.ball_pub = self.create_publisher(Twist, "go2/ball_cmd", 10)
        self.fx = self.fy = self.fz = 0.0      # latched drive (unit, scaled by sliders)
        self.latest_frame = None               # newest /camera/image_raw message
        self.latest_scan = None                # newest /scan message
        self.latest_map = None                 # newest /map OccupancyGrid
        self.create_subscription(Image, "camera/image_raw", self._on_image, 5)
        self.create_subscription(LaserScan, "scan", self._on_scan, 5)
        # /map is latched (transient_local): slam_toolbox AND map_server both publish
        # it that way, and map_server emits it only ONCE on activation -- a volatile
        # subscriber would miss it in localize mode. Match the standard "map QoS".
        map_qos = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                             reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, "map", self._on_map, map_qos)
        # TF so we can place the robot on the map (map -> base_link)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def _on_image(self, msg):
        self.latest_frame = msg

    def _on_scan(self, msg):
        self.latest_scan = msg

    def _on_map(self, msg):
        self.latest_map = msg

    def send_goal(self, x, y, yaw=0.0):
        m = PoseStamped()
        m.header.frame_id = "map"
        m.header.stamp = self.get_clock().now().to_msg()
        m.pose.position.x = float(x)
        m.pose.position.y = float(y)
        m.pose.orientation.z = math.sin(yaw / 2.0)
        m.pose.orientation.w = math.cos(yaw / 2.0)
        self.goal_pub.publish(m)

    def robot_in_map(self):
        """(x, y, yaw) of base_link in the map frame, or None if TF not ready."""
        try:
            t = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time())
        except Exception:
            return None
        tr, q = t.transform.translation, t.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return tr.x, tr.y, yaw

    def publish(self, lin, ang):
        m = Twist()
        m.linear.x, m.linear.y, m.angular.z = self.fx * lin, self.fy * lin, self.fz * ang
        self.pub.publish(m)
        return m

    def send_action(self, action):
        self.action_pub.publish(String(data=action))

    def send_explore(self, enable):
        self.explore_pub.publish(Bool(data=bool(enable)))

    def send_follow(self, enable):
        self.follow_pub.publish(Bool(data=bool(enable)))

    def send_ball(self, vx, vy):
        m = Twist()
        m.linear.x, m.linear.y = vx, vy      # world-frame planar velocity
        self.ball_pub.publish(m)

    def publish_pose(self, roll, pitch, yaw, height):
        m = Twist()
        m.angular.x, m.angular.y, m.angular.z = roll, pitch, yaw
        m.linear.z = height
        self.pose_pub.publish(m)


class ControlPanel:
    def __init__(self, node: Go2GuiTeleop):
        self.node = node
        self.root = tk.Tk()
        self.root.title("Go2 Control Panel")

        # ---------- sensors: camera + lidar, side by side, same size ----------
        self.sensor_size = 260
        f = tk.LabelFrame(self.root, text="Sensors", padx=6, pady=4)
        f.pack(padx=8, pady=4)

        camcol = tk.Frame(f)
        camcol.grid(row=0, column=0, padx=4)
        tk.Label(camcol, text="Camera  (/camera/image_raw)").pack()
        self.cam_label = tk.Label(camcol, bg="#222", bd=1, relief="sunken")
        self.cam_label.pack()
        # placeholder so the box is the right size before frames arrive
        self._placeholder = ImageTk.PhotoImage(
            PILImage.new("RGB", (self.sensor_size, self.sensor_size), (34, 34, 34)))
        self.cam_label.config(image=self._placeholder)
        self.cam_photo = None

        lidcol = tk.Frame(f)
        lidcol.grid(row=0, column=1, padx=4)
        tk.Label(lidcol, text="Lidar  (/scan, top-down, forward = up)").pack()
        self.lidar_size = self.sensor_size
        self.lidar_view_m = 5.0          # metres shown from center to edge
        self.lidar_canvas = tk.Canvas(lidcol, width=self.lidar_size,
                                      height=self.lidar_size, bg="#0b0b0b",
                                      highlightthickness=1, highlightbackground="#444")
        self.lidar_canvas.pack()

        # ---------- map (SLAM top-down) ----------
        f = tk.LabelFrame(self.root, text="Map  (SLAM + Nav2, forward = up)",
                          padx=6, pady=4)
        f.pack(padx=8, pady=4)
        self.nav_mode = tk.BooleanVar(value=False)
        tk.Checkbutton(f, text="Autonomous mode — click the map to set a Nav2 goal",
                       variable=self.nav_mode, command=self._on_navmode).pack(anchor="w")
        self.map_size = 320
        self.map_canvas = tk.Canvas(f, width=self.map_size, height=self.map_size,
                                    bg="#0b0b0b", highlightthickness=1,
                                    highlightbackground="#444", cursor="crosshair")
        self.map_canvas.pack()
        self.map_canvas.bind("<Button-1>", self._map_press)
        self.map_canvas.bind("<B1-Motion>", self._map_drag)
        self.map_canvas.bind("<ButtonRelease-1>", self._map_release)
        self.map_photo = None
        self._map_view = None        # (ox, oy, res, w, h, scale, opx, opy)
        self._goal = None            # (x, y, yaw|None)
        self._goal_drag = None       # goal position while press-dragging
        self._was_moving = False     # for idle-silence of /cmd_vel
        self._map_tick = 0
        tk.Button(f, text="■ Stop navigation", command=self._stop_nav).pack(pady=(4, 0))
        # auto-explore: the go2_explorer node maps unknown space on its own; this
        # checkbox just gates it via /go2/explore_enable. Needs explorer running.
        self.explore = tk.BooleanVar(value=False)
        tk.Checkbutton(f, text="🔍 Auto-explore (frontier mapping — needs explore node)",
                       variable=self.explore, command=self._on_explore).pack(anchor="w")
        self.follow = tk.BooleanVar(value=False)
        tk.Checkbutton(f, text="🎯 Follow object (camera — needs follow node)",
                       variable=self.follow, command=self._on_follow).pack(anchor="w")
        tk.Button(f, text="💾 Save map", command=self._save_map).pack(pady=(2, 0))

        # ---------- move the ball (drag the joystick, or press & hold buttons) ----------
        f = tk.LabelFrame(self.root, text="Move ball (drag the joystick, or hold buttons)",
                          padx=6, pady=4)
        f.pack(fill="x", padx=8, pady=4)
        self._ball_dir = (0.0, 0.0)      # (vx, vy) world while a button is held
        self._ball_active = False        # idle-silence: stop publishing when released
        bs = 0.6                         # ball drive speed (m/s)

        def hold(btn, vx, vy):
            btn.bind("<ButtonPress-1>", lambda _e: setattr(self, "_ball_dir", (vx, vy)))
            btn.bind("<ButtonRelease-1>", lambda _e: setattr(self, "_ball_dir", (0.0, 0.0)))

        b_fwd = tk.Button(f, text="↑ ahead", width=7, height=1)
        b_fwd.grid(row=0, column=1, padx=2, pady=2);  hold(b_fwd, +bs, 0.0)
        b_left = tk.Button(f, text="← left", width=7, height=1)
        b_left.grid(row=1, column=0, padx=2, pady=2); hold(b_left, 0.0, +bs)
        b_right = tk.Button(f, text="right →", width=7, height=1)
        b_right.grid(row=1, column=2, padx=2, pady=2); hold(b_right, 0.0, -bs)
        b_back = tk.Button(f, text="↓ back", width=7, height=1)
        b_back.grid(row=2, column=1, padx=2, pady=2); hold(b_back, -bs, 0.0)

        # mouse joystick: drag = roll ball in that direction at proportional speed
        self.ball_joy_size, self.ball_knob_r = 110, 16
        self.ball_joy_speed = 0.8                # max ball speed at full deflection (m/s)
        bjr = self.ball_joy_r = self.ball_joy_size // 2
        bcv = tk.Canvas(f, width=self.ball_joy_size, height=self.ball_joy_size,
                        bg="#f7eef7", highlightthickness=1, highlightbackground="#c39bd3")
        bcv.grid(row=0, column=3, rowspan=3, padx=(12, 0))
        bcv.create_oval(8, 8, self.ball_joy_size - 8, self.ball_joy_size - 8, outline="#c39bd3")
        bkr = self.ball_knob_r
        self.ball_knob = bcv.create_oval(bjr - bkr, bjr - bkr, bjr + bkr, bjr + bkr,
                                         fill="#cc33aa")
        self.ball_joy_canvas = bcv
        bcv.bind("<Button-1>", self._ball_joy_drag)
        bcv.bind("<B1-Motion>", self._ball_joy_drag)
        bcv.bind("<ButtonRelease-1>", self._ball_joy_release)
        tk.Label(f, text="↑ ahead (mouse joystick)").grid(row=3, column=3)
        tk.Label(f, text="(relative to the robot: ↑ rolls the ball ahead of it, etc.)",
                 fg="#888").grid(row=4, column=0, columnspan=4, sticky="w")

        # ---------- speed ----------
        f = tk.LabelFrame(self.root, text="Speed", padx=6, pady=4)
        f.pack(fill="x", padx=8, pady=4)
        self.lin = tk.DoubleVar(value=0.3)
        self.ang = tk.DoubleVar(value=0.8)
        tk.Label(f, text="linear m/s").grid(row=0, column=0)
        tk.Scale(f, variable=self.lin, from_=0.0, to=0.5, resolution=0.05,
                 orient=tk.HORIZONTAL, length=150).grid(row=0, column=1)
        tk.Label(f, text="angular rad/s").grid(row=1, column=0)
        tk.Scale(f, variable=self.ang, from_=0.0, to=1.5, resolution=0.1,
                 orient=tk.HORIZONTAL, length=150).grid(row=1, column=1)

        # ---------- drive pad + joystick ----------
        f = tk.LabelFrame(self.root, text="Drive", padx=6, pady=4)
        f.pack(fill="x", padx=8, pady=4)

        def db(text, r, c, **kw):
            tk.Button(f, text=text, width=8, height=2, **kw).grid(
                row=r, column=c, padx=2, pady=2)
        db("⟲ Turn L", 0, 0, command=lambda: self.set(fz=+1))
        db("↑ Fwd", 0, 1, command=lambda: self.set(fx=+1))
        db("⟳ Turn R", 0, 2, command=lambda: self.set(fz=-1))
        db("← Str L", 1, 0, command=lambda: self.set(fy=+1))
        db("■ STOP", 1, 1, command=self.stop, bg="#d9534f", fg="white")
        db("→ Str R", 1, 2, command=lambda: self.set(fy=-1))
        db("↓ Back", 2, 1, command=lambda: self.set(fx=-1))

        self.joy_size, self.knob_r = 130, 18
        r = self.joy_r = self.joy_size // 2
        cv = tk.Canvas(f, width=self.joy_size, height=self.joy_size, bg="#f0f0f0",
                       highlightthickness=1, highlightbackground="#aaa")
        cv.grid(row=0, column=3, rowspan=3, padx=(10, 0))
        cv.create_oval(8, 8, self.joy_size - 8, self.joy_size - 8, outline="#bbb")
        kr = self.knob_r
        self.knob = cv.create_oval(r - kr, r - kr, r + kr, r + kr, fill="#0d6efd")
        self.joy_canvas = cv
        cv.bind("<Button-1>", self._joy_drag)
        cv.bind("<B1-Motion>", self._joy_drag)
        cv.bind("<ButtonRelease-1>", self._joy_release)
        tk.Label(f, text="virtual joystick").grid(row=3, column=3)

        # ---------- postures + actions ----------
        f = tk.LabelFrame(self.root, text="Posture / Actions", padx=6, pady=4)
        f.pack(fill="x", padx=8, pady=4)

        def ab(text, r, c, action, **kw):
            tk.Button(f, text=text, width=8,
                      command=lambda: self.action(action), **kw).grid(
                row=r, column=c, padx=2, pady=2)
        ab("⮝ Stand", 0, 0, "stand")
        ab("⤓ Sit", 0, 1, "sit", bg="#6f42c1", fg="white")
        ab("Lie down", 0, 2, "liedown")
        ab("Crouch", 0, 3, "crouch")
        ab("⤒ Jump", 1, 0, "jump", bg="#0d6efd", fg="white")
        ab("👋 Wave", 1, 1, "wave")
        ab("🤝 Shake", 1, 2, "shake")
        ab("💃 Dance", 1, 3, "dance")
        ab("Self-right*", 2, 0, "selfright")
        tk.Label(f, text="*experimental (open-loop self-right rarely succeeds)",
                 fg="#888").grid(row=2, column=1, columnspan=3, sticky="w")

        # ---------- gait ----------
        f = tk.LabelFrame(self.root, text="Gait (crawl is stable; others experimental)",
                          padx=6, pady=4)
        f.pack(fill="x", padx=8, pady=4)
        self.gait = tk.StringVar(value="crawl")
        tk.OptionMenu(f, self.gait, *GAITS,
                      command=lambda name: self.node.send_action(f"gait_{name}")
                      ).pack(side="left")

        # ---------- body pose ----------
        f = tk.LabelFrame(self.root, text="Body pose (lean + height)", padx=6, pady=4)
        f.pack(fill="x", padx=8, pady=4)
        self.roll = tk.DoubleVar(value=0.0)
        self.pitch = tk.DoubleVar(value=0.0)
        self.yaw = tk.DoubleVar(value=0.0)
        self.height = tk.DoubleVar(value=0.0)        # 0 => default stand height
        for i, (lbl, var, lo, hi, res) in enumerate([
                ("roll", self.roll, -0.3, 0.3, 0.02),
                ("pitch", self.pitch, -0.3, 0.3, 0.02),
                ("yaw", self.yaw, -0.3, 0.3, 0.02),
                ("height", self.height, 0.0, 0.4, 0.02)]):
            tk.Label(f, text=lbl).grid(row=i, column=0)
            tk.Scale(f, variable=var, from_=lo, to=hi, resolution=res,
                     orient=tk.HORIZONTAL, length=180).grid(row=i, column=1)
        tk.Button(f, text="reset pose", command=self._reset_pose).grid(
            row=4, column=0, columnspan=2, pady=(4, 0))

        self.status = tk.Label(self.root, text="stopped")
        self.status.pack(pady=(2, 6))

        self._bind_keys()
        self.root.after(50, self._tick)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------- drive state ----------
    def set(self, fx=0.0, fy=0.0, fz=0.0):
        self.node.fx, self.node.fy, self.node.fz = fx, fy, fz

    def stop(self):
        self.set()

    def action(self, name):
        self.stop()
        self.node.send_action(name)

    def _reset_pose(self):
        for v in (self.roll, self.pitch, self.yaw, self.height):
            v.set(0.0)

    def _joy_drag(self, event):
        r, reach, kr = self.joy_r, self.joy_r - self.knob_r, self.knob_r
        dx, dy = event.x - r, event.y - r
        dist = math.hypot(dx, dy)
        if dist > reach:
            dx, dy = dx * reach / dist, dy * reach / dist
        self.joy_canvas.coords(self.knob, r + dx - kr, r + dy - kr,
                               r + dx + kr, r + dy + kr)
        self.set(fx=-dy / reach, fz=-dx / reach)

    def _joy_release(self, _e):
        r, kr = self.joy_r, self.knob_r
        self.joy_canvas.coords(self.knob, r - kr, r - kr, r + kr, r + kr)
        self.stop()

    def _bind_keys(self):
        km = {"w": dict(fx=+1), "s": dict(fx=-1), "a": dict(fy=+1), "d": dict(fy=-1),
              "q": dict(fz=+1), "e": dict(fz=-1), "Up": dict(fx=+1),
              "Down": dict(fx=-1), "Left": dict(fz=+1), "Right": dict(fz=-1)}
        for k, c in km.items():
            self.root.bind(f"<{k}>", lambda _e, c=c: self.set(**c))
        self.root.bind("<space>", lambda _e: self.stop())
        self.root.bind("<j>", lambda _e: self.action("jump"))
        self.root.bind("<x>", lambda _e: self.action("sit"))
        self.root.bind("<z>", lambda _e: self.action("stand"))

    def _update_camera(self):
        msg = self.node.latest_frame
        if msg is None or msg.encoding != "rgb8":
            return
        try:
            img = PILImage.frombytes("RGB", (msg.width, msg.height), bytes(msg.data))
        except (ValueError, BufferError):
            return
        img = img.resize((self.sensor_size, self.sensor_size))   # match lidar size
        self.cam_photo = ImageTk.PhotoImage(img)     # keep a reference
        self.cam_label.config(image=self.cam_photo, text="")
        if not getattr(self, "_cam_seen", False):
            self._cam_seen = True
            self.node.get_logger().info("GUI: first camera frame displayed.")

    def _update_lidar(self):
        msg = self.node.latest_scan
        if msg is None:
            return
        c = self.lidar_canvas
        c.delete("pts")
        s = self.lidar_size
        cx = cy = s // 2
        scale = (s // 2 - 6) / self.lidar_view_m
        # range rings every 1 m
        for ring in range(1, int(self.lidar_view_m) + 1):
            rr = ring * scale
            c.create_oval(cx - rr, cy - rr, cx + rr, cy + rr,
                          outline="#1f3b1f", tags="pts")
        # robot marker (triangle pointing up = forward)
        c.create_polygon(cx, cy - 7, cx - 5, cy + 5, cx + 5, cy + 5,
                         fill="#33ccff", tags="pts")
        a = msg.angle_min
        for r in msg.ranges:
            if 0.0 < r < self.lidar_view_m:
                x = cx - r * scale * math.sin(a)      # forward up, +y(left) -> left
                y = cy - r * scale * math.cos(a)
                c.create_oval(x - 1.5, y - 1.5, x + 1.5, y + 1.5,
                              fill="#ff4040", outline="", tags="pts")
            a += msg.angle_increment

    def _ball_joy_drag(self, event):
        r, kr = self.ball_joy_r, self.ball_knob_r
        reach = r - kr
        dx, dy = event.x - r, event.y - r
        dist = math.hypot(dx, dy)
        if dist > reach:
            dx, dy = dx * reach / dist, dy * reach / dist
        self.ball_joy_canvas.coords(self.ball_knob, r + dx - kr, r + dy - kr,
                                    r + dx + kr, r + dy + kr)
        bs = self.ball_joy_speed
        self._ball_dir = (-dy / reach * bs, -dx / reach * bs)   # up=ahead(+x), left=+y

    def _ball_joy_release(self, _e):
        r, kr = self.ball_joy_r, self.ball_knob_r
        self.ball_joy_canvas.coords(self.ball_knob, r - kr, r - kr, r + kr, r + kr)
        self._ball_dir = (0.0, 0.0)

    def _on_explore(self):
        on = self.explore.get()
        self.node.send_explore(on)
        if on:                                # exploration drives via Nav2 ->
            self._set_follow(False)           # only one /cmd_vel owner at a time
            self.nav_mode.set(True)           # force autonomous so manual cmd_vel
            self._on_navmode()                # pauses (no /cmd_vel contention)

    def _on_follow(self):
        on = self.follow.get()
        self.node.send_follow(on)
        if on:                                # follower drives /cmd_vel from detections
            self._set_explore(False)          # mutually exclusive with explore
            self.nav_mode.set(True)           # force autonomous so manual cmd_vel pauses
            self._on_navmode()

    def _set_explore(self, on):
        self.explore.set(on)
        self.node.send_explore(on)

    def _set_follow(self, on):
        self.follow.set(on)
        self.node.send_follow(on)

    def _on_navmode(self):
        if self.nav_mode.get():
            self.stop()                       # halt manual motion; _tick then pauses
        else:                                 # leaving autonomous -> stop autonomy modes
            if self.explore.get():
                self._set_explore(False)
            if self.follow.get():
                self._set_follow(False)

    def _canvas_to_world(self, event):
        ox, oy, res, w, h, scale, opx, opy = self._map_view
        col = (event.x - opx) / scale
        row = h - (event.y - opy) / scale     # display is Y-flipped (forward up)
        return ox + col * res, oy + row * res

    def _map_press(self, event):
        if not self.nav_mode.get() or self._map_view is None:
            return
        gx, gy = self._canvas_to_world(event)
        self._goal_drag = (gx, gy)
        self._goal = (gx, gy, None)           # heading set by dragging

    def _map_drag(self, event):
        if self._goal_drag is None or self._map_view is None:
            return
        wx, wy = self._canvas_to_world(event)
        gx, gy = self._goal_drag
        dx, dy = wx - gx, wy - gy
        yaw = math.atan2(dy, dx) if math.hypot(dx, dy) > 0.15 else None
        self._goal = (gx, gy, yaw)

    def _map_release(self, event):
        if self._goal_drag is None:
            return
        gx, gy = self._goal_drag
        yaw = self._goal[2] if self._goal else None
        if yaw is None:                       # tap: face the travel direction
            rp = self.node.robot_in_map()
            yaw = math.atan2(gy - rp[1], gx - rp[0]) if rp is not None else 0.0
            self._goal = (gx, gy, yaw)
        self.node.send_goal(gx, gy, yaw)
        self._goal_drag = None

    def _save_map(self):
        """Write the current /map to ~/go2_ws/maps as Nav2 .pgm + .yaml.

        Self-contained (no map_saver process/service): the GUI already has the
        OccupancyGrid, so we render it in Nav2's trinary convention -- free=254,
        occupied=0, unknown=205 -- with the image flipped so row 0 is max-y (top).
        """
        m = self.node.latest_map
        if m is None or m.info.width == 0 or m.info.height == 0:
            self.status.config(text="no map to save yet")
            return
        w, h, res = m.info.width, m.info.height, m.info.resolution
        o = m.info.origin.position
        grid = np.asarray(m.data, dtype=np.int16).reshape(h, w)
        img = np.full((h, w), 205, dtype=np.uint8)       # unknown
        img[(grid >= 0) & (grid <= 25)] = 254            # free  (<= free_thresh)
        img[grid >= 65] = 0                              # occupied (>= occupied_thresh)
        img = np.flipud(img)                             # pgm row 0 = top = max y

        maps_dir = os.path.expanduser("~/go2_ws/maps")
        os.makedirs(maps_dir, exist_ok=True)
        name = time.strftime("gui_map_%Y%m%d_%H%M%S")
        pgm, yaml = os.path.join(maps_dir, name + ".pgm"), os.path.join(maps_dir, name + ".yaml")
        try:
            PILImage.fromarray(img).save(pgm)        # uint8 2D -> 'L' (P5 .pgm)
            with open(yaml, "w") as f:
                f.write(f"image: {name}.pgm\nmode: trinary\nresolution: {res}\n"
                        f"origin: [{o.x}, {o.y}, 0]\nnegate: 0\n"
                        f"occupied_thresh: 0.65\nfree_thresh: 0.25\n")
        except OSError as e:
            self.node.get_logger().error(f"Map save failed: {e}")
            self.status.config(text="map save FAILED (see log)")
            return
        self.node.get_logger().info(f"Saved map to {pgm}")
        self.status.config(text=f"map saved: {name}")

    def _stop_nav(self):
        rp = self.node.robot_in_map()         # goal = current pose -> Nav2 halts here
        if rp is not None:
            self.node.send_goal(rp[0], rp[1], rp[2])
        self._goal = None
        self._goal_drag = None
        self.nav_mode.set(False)
        if self.explore.get():                # stopping nav also stops autonomy modes
            self._set_explore(False)
        if self.follow.get():
            self._set_follow(False)

    def _update_map(self):
        msg = self.node.latest_map
        if msg is None or msg.info.width == 0 or msg.info.height == 0:
            return
        w, h, res = msg.info.width, msg.info.height, msg.info.resolution
        o = msg.info.origin.position
        # occupancy -> grayscale: unknown(-1)=gray, free(0)=white, occupied(100)=black
        grid = np.asarray(msg.data, dtype=np.int16).reshape(h, w)
        shade = np.where(grid < 0, 80, 255 - (grid * 255 // 100)).astype(np.uint8)
        shade = np.flipud(shade)              # map +y is up -> image top
        pim = PILImage.fromarray(shade, mode="L").convert("RGB")
        scale = min(self.map_size / w, self.map_size / h)
        dw, dh = max(1, int(w * scale)), max(1, int(h * scale))
        opx, opy = (self.map_size - dw) // 2, (self.map_size - dh) // 2
        self.map_photo = ImageTk.PhotoImage(pim.resize((dw, dh), PILImage.NEAREST))
        self._map_view = (o.x, o.y, res, w, h, scale, opx, opy)

        c = self.map_canvas
        c.delete("m")
        c.create_image(opx, opy, anchor="nw", image=self.map_photo, tags="m")
        if self._goal is not None:            # goal marker + heading
            gx, gy, gyaw = self._goal
            px = opx + ((gx - o.x) / res) * scale
            py = opy + (h - (gy - o.y) / res) * scale
            c.create_oval(px - 7, py - 7, px + 7, py + 7,
                          outline="#3fb950", width=2, tags="m")
            if gyaw is None:                  # not aimed yet -> crosshair
                c.create_line(px - 11, py, px + 11, py, fill="#3fb950", width=2, tags="m")
                c.create_line(px, py - 11, px, py + 11, fill="#3fb950", width=2, tags="m")
            else:                             # heading arrow
                tx = opx + ((gx + 0.7 * math.cos(gyaw) - o.x) / res) * scale
                ty = opy + (h - (gy + 0.7 * math.sin(gyaw) - o.y) / res) * scale
                c.create_line(px, py, tx, ty, fill="#3fb950", width=2,
                              arrow="last", tags="m")
        rp = self.node.robot_in_map()         # robot arrow
        if rp is not None:
            rx, ry, ryaw = rp
            px = opx + ((rx - o.x) / res) * scale
            py = opy + (h - (ry - o.y) / res) * scale
            ex, ey = px + 11 * math.cos(ryaw), py - 11 * math.sin(ryaw)
            c.create_line(px, py, ex, ey, fill="#33ccff", width=2, tags="m")
            c.create_oval(px - 4, py - 4, px + 4, py + 4,
                          fill="#0d6efd", outline="#fff", tags="m")

    def _tick(self):
        if not rclpy.ok():
            self.root.destroy()
            return
        self._update_camera()
        self._update_lidar()
        self._map_tick = (self._map_tick + 1) % 5
        if self._map_tick == 0:               # refresh map ~4 Hz (cheaper)
            self._update_map()

        # ball driving: publish while a button is held, one stop on release, then silent
        bx, by = self._ball_dir
        if bx or by:
            self.node.send_ball(bx, by)
            self._ball_active = True
        elif self._ball_active:
            self.node.send_ball(0.0, 0.0)
            self._ball_active = False

        if self.nav_mode.get():
            # Autonomous: don't publish /cmd_vel — let Nav2 drive (no contention).
            self.status.config(
                text="🎯 following object" if self.follow.get()
                else "🔍 auto-exploring — Nav2 driving" if self.explore.get()
                else "autonomous mode — Nav2 driving")
        else:
            self.node.publish_pose(self.roll.get(), self.pitch.get(),
                                   self.yaw.get(), self.height.get())
            # Publish /cmd_vel only while actively commanding; when idle go silent
            # (one final stop) so an open panel never fights Nav2. Watchdog holds stop.
            commanding = any(abs(v) > 1e-6 for v in
                             (self.node.fx, self.node.fy, self.node.fz))
            if commanding:
                m = self.node.publish(self.lin.get(), self.ang.get())
                self._was_moving = True
                self.status.config(
                    text=f"vx={m.linear.x:+.2f} vy={m.linear.y:+.2f} wz={m.angular.z:+.2f}")
            elif self._was_moving:
                self.node.publish(self.lin.get(), self.ang.get())   # final zero
                self._was_moving = False
                self.status.config(text="stopped")
            else:
                self.status.config(text="idle (Nav2 free to drive)")
        self.root.after(50, self._tick)

    def _on_close(self):
        self.set()
        self.node.publish(0.0, 0.0)
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main(args=None):
    rclpy.init(args=args)
    node = Go2GuiTeleop()
    # Spin ROS in a background thread so subscriptions (camera, scan) are
    # processed continuously; the tk main loop only reads the latest data.
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    panel = ControlPanel(node)
    try:
        panel.run()
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
