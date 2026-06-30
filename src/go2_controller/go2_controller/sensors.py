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
    scan                (sensor_msgs/LaserScan)     -- 360deg 2D lidar
    points              (sensor_msgs/PointCloud2)   -- 3D lidar (if sensor_msgs_py present)
"""
import math

from std_msgs.msg import Header
from sensor_msgs.msg import Image, CameraInfo, LaserScan, PointCloud2

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
        node.declare_parameter("lidar_rate", 10.0)     # Hz
        node.declare_parameter("lidar_rays", 360)      # 1deg resolution (better SLAM)
        node.declare_parameter("lidar_range", 10.0)    # m
        node.declare_parameter("lidar_cloud_rate", 8.0)  # Hz (3D PointCloud2)

        # --- camera ---
        self.cam_w = node.get_parameter("camera_width").value
        self.cam_h = node.get_parameter("camera_height").value
        self._cam_failed = False
        if node.get_parameter("use_camera").value and sim.has_camera():
            self.img_pub = node.create_publisher(Image, "camera/image_raw", 10)
            self.caminfo_pub = node.create_publisher(CameraInfo, "camera/camera_info", 10)
            node.create_timer(1.0 / node.get_parameter("camera_rate").value,
                              self.camera_tick)

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
