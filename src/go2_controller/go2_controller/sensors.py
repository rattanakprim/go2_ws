"""Onboard sensor publishing for the Go2 sim node (camera, 2D lidar, 3D cloud).

Extracted from teleop_node.py to keep the controller focused on locomotion.
A SensorPublisher owns the camera / lidar / point-cloud publishers and timers
and reads the MuJoCo sim each tick. Construct one with the owning Node and the
Go2Sim; it declares its own parameters and wires up its own publishers/timers,
so the controller node doesn't have to.

Parameters (declared on the owning node):
    use_camera, camera_rate, camera_width, camera_height
    use_lidar,  lidar_rate,  lidar_rays,   lidar_range, lidar_cloud_rate

Publishes:
    camera/image_raw    (sensor_msgs/Image)        -- front RGB camera
    camera/camera_info  (sensor_msgs/CameraInfo)
    sim/scene_image     (sensor_msgs/Image)        -- third-person chase view
    scan                (sensor_msgs/LaserScan)     -- 360deg 2D lidar
    points              (sensor_msgs/PointCloud2)   -- 3D lidar (if sensor_msgs_py present)
"""
import math
import threading
import time

import cv2

from std_msgs.msg import Header
from sensor_msgs.msg import (Image, CameraInfo, CompressedImage,
                             LaserScan, PointCloud2)

try:
    from sensor_msgs_py import point_cloud2 as pc2
    _HAVE_PC2 = True
except ImportError:
    _HAVE_PC2 = False


class SensorPublisher:
    def __init__(self, node, sim):
        self.node = node
        self.sim = sim

        node.declare_parameter("use_camera", True)
        node.declare_parameter("use_lidar", True)
        node.declare_parameter("camera_rate", 15.0)    # Hz
        node.declare_parameter("camera_width", 320)
        node.declare_parameter("camera_height", 240)
        # Third-person orbit view streamed to the web panel. Rendered at high res
        # with anti-aliasing and published JPEG-COMPRESSED (encoded on the render
        # thread, ~50 KB/frame) instead of raw (~1.4 MB): the tiny payload keeps
        # per-frame serialisation off the GIL long enough that the 50 Hz control
        # loop stays smooth even at 800x600@12. web_video_server streams it via
        # the compressed transport (web panel uses default_transport=compressed).
        node.declare_parameter("use_scene_cam", True)
        node.declare_parameter("scene_rate", 12.0)     # Hz
        node.declare_parameter("scene_width", 800)
        node.declare_parameter("scene_height", 600)
        node.declare_parameter("scene_quality", 85)    # JPEG quality (1..100)
        node.declare_parameter("lidar_rate", 10.0)     # Hz
        node.declare_parameter("lidar_rays", 360)      # 1deg resolution (better SLAM)
        node.declare_parameter("lidar_range", 10.0)    # m
        node.declare_parameter("lidar_cloud_rate", 8.0)  # Hz (3D PointCloud2)

        # --- camera + third-person "scene" view ---
        # Both are GPU renders (~6-8 ms/frame). Run them on a dedicated background
        # thread (see _render_loop) so rendering never blocks the 50 Hz control /
        # physics loop -- that blocking is what made teleop feel laggy. The thread
        # reads a thread-safe snapshot of the sim (sim.snapshot(), taken each
        # control tick), so it doesn't race the physics stepping.
        self.cam_w = node.get_parameter("camera_width").value
        self.cam_h = node.get_parameter("camera_height").value
        self._cam_failed = False
        self._cam_on = node.get_parameter("use_camera").value and sim.has_camera()
        if self._cam_on:
            self.img_pub = node.create_publisher(Image, "camera/image_raw", 10)
            self.caminfo_pub = node.create_publisher(CameraInfo, "camera/camera_info", 10)
            self.cam_period = 1.0 / node.get_parameter("camera_rate").value

        self.scene_w = node.get_parameter("scene_width").value
        self.scene_h = node.get_parameter("scene_height").value
        self.scene_q = int(node.get_parameter("scene_quality").value)
        self._scene_failed = False
        self._scene_on = node.get_parameter("use_scene_cam").value
        if self._scene_on:
            # image_transport compressed-topic naming convention: <base>/compressed
            self.scene_pub = node.create_publisher(
                CompressedImage, "sim/scene_image/compressed", 10)
            self.scene_period = 1.0 / node.get_parameter("scene_rate").value

        self._stop = None
        if self._cam_on or self._scene_on:
            self._stop = threading.Event()
            self._render_thread = threading.Thread(
                target=self._render_loop, name="sim_render", daemon=True)
            self._render_thread.start()

        # --- lidar (2D scan + optional 3D cloud) ---
        if node.get_parameter("use_lidar").value:
            self.lidar_rays = node.get_parameter("lidar_rays").value
            self.lidar_range = node.get_parameter("lidar_range").value
            self.scan_pub = node.create_publisher(LaserScan, "scan", 10)
            node.create_timer(1.0 / node.get_parameter("lidar_rate").value,
                              self.lidar_tick)
            if _HAVE_PC2:                          # 3D point cloud for RViz
                self.points_pub = node.create_publisher(PointCloud2, "points", 5)
                node.create_timer(1.0 / node.get_parameter("lidar_cloud_rate").value,
                                  self.cloud_tick)
            else:
                node.get_logger().warn("sensor_msgs_py missing; /points (3D) disabled")

    def camera_tick(self):
        if self._cam_failed:
            return
        try:
            img = self.sim.render_camera("front_camera", self.cam_w, self.cam_h)
        except Exception as exc:                 # GL context issue, headless, etc.
            self._cam_failed = True
            self.node.get_logger().warn(f"camera disabled ({exc})")
            return
        now = self.node.get_clock().now().to_msg()
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

    def scene_tick(self):
        if self._scene_failed:
            return
        try:
            img = self.sim.render_scene(self.scene_w, self.scene_h)
            # Encode to JPEG here (cv2 releases the GIL) so we publish ~50 KB, not
            # ~1.4 MB of raw pixels -- that's what keeps the control loop smooth.
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_JPEG_QUALITY, self.scene_q])
        except Exception as exc:                 # GL context issue, headless, etc.
            self._scene_failed = True
            self.node.get_logger().warn(f"scene camera disabled ({exc})")
            return
        if not ok:
            return
        msg = CompressedImage()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.format = "jpeg"
        msg.data = buf.tobytes()
        self.scene_pub.publish(msg)

    def _render_loop(self):
        """Render camera + scene off the control thread, each at its own rate."""
        next_cam = next_scene = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            if self._cam_on and not self._cam_failed and now >= next_cam:
                next_cam = now + self.cam_period
                self.camera_tick()
            if self._scene_on and not self._scene_failed and now >= next_scene:
                next_scene = now + self.scene_period
                self.scene_tick()
            # sleep until the next render is due (bounded so stop() stays snappy)
            due = []
            if self._cam_on and not self._cam_failed:
                due.append(next_cam)
            if self._scene_on and not self._scene_failed:
                due.append(next_scene)
            wait = (min(due) - time.monotonic()) if due else 0.1
            self._stop.wait(max(0.001, min(wait, 0.1)))

    def stop(self):
        """Stop the background render thread (call on shutdown)."""
        if self._stop is not None:
            self._stop.set()
            self._render_thread.join(timeout=1.0)

    def lidar_tick(self):
        ranges = self.sim.lidar_scan(self.lidar_rays, self.lidar_range)
        if ranges is None:
            return
        n = len(ranges)
        scan = LaserScan()
        scan.header.stamp = self.node.get_clock().now().to_msg()
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
        header.stamp = self.node.get_clock().now().to_msg()
        header.frame_id = "lidar_link"
        self.points_pub.publish(pc2.create_cloud_xyz32(header, pts.tolist()))
