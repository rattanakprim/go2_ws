"""Object / person following for the Go2 -- coco_detector detections to /cmd_vel.

Subscribes /detected_objects (vision_msgs/Detection2DArray) published by coco_detector
(which runs MobileNet on /camera/image_raw) and drives the gait to keep a target class
centred and at a set apparent size:
  * horizontal bbox offset  -> turn   (angular.z)   -- centre the target
  * bbox height vs setpoint -> fwd/back(linear.x)   -- apparent size is a distance proxy
A deadman stops the robot if the target is lost for lost_timeout seconds. Following only
publishes /cmd_vel while it has a fresh target (then one stop) -- same idle-silence rule as
the panels, so it never fights manual control when there is nothing to follow.

NOTE: the MuJoCo sim contains no COCO objects, so this is meant for a real camera (or a sim
scene with a recognisable rendered asset). Do NOT run Nav2 at the same time (both drive
/cmd_vel -- one source at a time).

Run:    ros2 run go2_controller follow            # needs coco_detector publishing detections
Toggle: ros2 topic pub --once /go2/follow_enable std_msgs/Bool '{data: false}'
"""
import math

import rclpy
import rclpy.time
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CameraInfo, LaserScan
from std_msgs.msg import Bool
from vision_msgs.msg import Detection2DArray


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


class Go2Follower(Node):
    def __init__(self):
        super().__init__("go2_follower")
        # --- params ---
        self.declare_parameter("target_class", "person")
        self.declare_parameter("autostart", False)   # wait for /go2/follow_enable (GUI toggle)
        self.declare_parameter("desired_height_frac", 0.7)  # target bbox_h / image_h (bigger = closer)
        self.declare_parameter("deadband_frac", 0.08)       # ignore small centring / size errors
        self.declare_parameter("k_ang", 1.6)                # turn gain on horizontal error
        self.declare_parameter("k_lin", 1.5)                # drive gain (reach max speed quickly)
        self.declare_parameter("max_ang", 0.7)
        self.declare_parameter("max_lin", 0.2)              # crawl gait's speed sweet spot (~0.17 m/s
                                                            # actual); higher commands SLOW it (feet slip)
        self.declare_parameter("lost_timeout", 0.7)         # s without target -> stop
        self.declare_parameter("image_width", 640)          # fallback until camera_info arrives
        self.declare_parameter("image_height", 480)
        # reactive obstacle avoidance (uses /scan)
        self.declare_parameter("avoid_enable", True)
        self.declare_parameter("avoid_dist", 0.8)           # m: start slowing/steering inside this
        self.declare_parameter("avoid_stop_dist", 0.35)     # m: forward speed -> 0 at this range
        self.declare_parameter("avoid_gain", 1.2)           # steering-away strength
        self.declare_parameter("ball_exclude_deg", 25.0)    # ignore /scan within this of the ball bearing
        self.declare_parameter("avoid_arc_deg", 75.0)       # only avoid obstacles within this forward arc
        self.declare_parameter("hfov_deg", 73.0)            # camera horizontal FOV (320x240 @ fovy 58)

        g = self.get_parameter
        self.target_class = g("target_class").value
        # accept a comma-separated list: up close MobileNet often mis-labels the ball
        # ("frisbee"/"mouse"), so include those to keep the lock as it fills the frame.
        self.target_classes = {c.strip() for c in str(self.target_class).split(",")}
        self.enabled = bool(g("autostart").value)
        self.desired_h = float(g("desired_height_frac").value)
        self.deadband = float(g("deadband_frac").value)
        self.k_ang = float(g("k_ang").value)
        self.k_lin = float(g("k_lin").value)
        self.max_ang = float(g("max_ang").value)
        self.max_lin = float(g("max_lin").value)
        self.lost_timeout = float(g("lost_timeout").value)
        self.img_w = float(g("image_width").value)
        self.img_h = float(g("image_height").value)
        self.avoid_enable = bool(g("avoid_enable").value)
        self.avoid_dist = float(g("avoid_dist").value)
        self.avoid_stop_dist = float(g("avoid_stop_dist").value)
        self.avoid_gain = float(g("avoid_gain").value)
        self.ball_exclude = math.radians(float(g("ball_exclude_deg").value))
        self.avoid_arc = math.radians(float(g("avoid_arc_deg").value))
        self.hfov = math.radians(float(g("hfov_deg").value))
        self.scan = None

        # --- state ---
        self.target_cmd = (0.0, 0.0)     # (lin, ang) from the most recent detection
        self.last_seen = None            # Time of last target detection
        self._was_active = False         # for idle-silence (one final stop, then quiet)

        # --- ROS ---
        self.pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.create_subscription(Detection2DArray, "detected_objects", self._on_det, 10)
        self.create_subscription(CameraInfo, "camera/camera_info", self._on_caminfo, 1)
        self.create_subscription(LaserScan, "scan", self._on_scan, 5)
        self.create_subscription(Bool, "go2/follow_enable", self._on_enable, 1)
        self.create_timer(0.05, self._tick)   # 20 Hz publish + deadman
        self.add_on_set_parameters_callback(self._on_params)   # live tuning
        self.get_logger().info(
            f"Follower up (target='{self.target_class}', autostart={self.enabled}, "
            f"hold@height_frac={self.desired_h}).")

    # ------------------------------------------------------------------ #
    def _on_params(self, params):
        """Apply runtime `ros2 param set` changes live (e.g. desired_height_frac, gains)."""
        from rcl_interfaces.msg import SetParametersResult
        fields = {"desired_height_frac": "desired_h", "deadband_frac": "deadband",
                  "k_ang": "k_ang", "k_lin": "k_lin", "max_ang": "max_ang",
                  "max_lin": "max_lin", "lost_timeout": "lost_timeout",
                  "avoid_dist": "avoid_dist", "avoid_stop_dist": "avoid_stop_dist",
                  "avoid_gain": "avoid_gain"}
        try:
            for p in params:
                if p.name in fields:
                    setattr(self, fields[p.name], float(p.value))
                elif p.name == "avoid_enable":
                    self.avoid_enable = bool(p.value)
                elif p.name == "target_class":
                    self.target_class = str(p.value)
                    self.target_classes = {c.strip() for c in self.target_class.split(",")}
        except (ValueError, TypeError) as e:        # reject bad values cleanly, no traceback
            return SetParametersResult(successful=False, reason=str(e))
        return SetParametersResult(successful=True)

    def _on_caminfo(self, msg):
        if msg.width:
            self.img_w = float(msg.width)
        if msg.height:
            self.img_h = float(msg.height)

    def _on_enable(self, msg):
        self.enabled = bool(msg.data)
        if not self.enabled:
            self.target_cmd = (0.0, 0.0)
        self.get_logger().info(f"Following {'enabled' if self.enabled else 'disabled'}.")

    def _best_target(self, msg):
        """Largest bbox of the target class (largest area ~= nearest)."""
        best, best_area = None, 0.0
        for d in msg.detections:
            if not d.results or d.results[0].hypothesis.class_id not in self.target_classes:
                continue
            area = d.bbox.size_x * d.bbox.size_y
            if area > best_area:
                best, best_area = d, area
        return best

    def _on_det(self, msg):
        if not self.enabled:
            return
        t = self._best_target(msg)
        if t is None:
            return
        ex = (t.bbox.center.position.x - self.img_w / 2.0) / (self.img_w / 2.0)  # [-1,1], + = right
        hfrac = t.bbox.size_y / self.img_h                                       # apparent size

        ang = 0.0 if abs(ex) < self.deadband else -self.k_ang * ex               # target right -> turn right (wz<0)
        herr = self.desired_h - hfrac
        lin = 0.0 if abs(herr) < self.deadband else self.k_lin * herr            # smaller than setpoint -> forward
        if abs(ex) > 0.4 and lin > 0.0:      # face the target before driving into it
            lin = 0.0

        # reactive obstacle avoidance: ball bearing (robot frame, +left) so we can ignore
        # the ball's own lidar return and steer around everything else.
        ball_bearing = -ex * (self.hfov / 2.0)
        lin, ang = self._avoid(lin, ang, ball_bearing)

        self.target_cmd = (_clamp(lin, -self.max_lin, self.max_lin),
                           _clamp(ang, -self.max_ang, self.max_ang))
        self.last_seen = self.get_clock().now()

    def _on_scan(self, msg):
        self.scan = msg

    def _avoid(self, lin, ang, ball_bearing):
        """Slow forward motion and steer away from the nearest /scan obstacle that is in
        the forward arc but NOT within ball_exclude of the ball's bearing."""
        if not self.avoid_enable or self.scan is None:
            return lin, ang
        s = self.scan
        nearest, nearest_ang = float("inf"), 0.0
        a = s.angle_min
        for r in s.ranges:
            if s.range_min < r < min(self.avoid_dist, s.range_max):
                # |a| within forward arc, and a not near the ball direction (wrap the
                # bearing difference to [-pi,pi] so it's correct near the +-pi seam)
                d_ball = abs(math.atan2(math.sin(a - ball_bearing), math.cos(a - ball_bearing)))
                if abs(a) <= self.avoid_arc and d_ball > self.ball_exclude:
                    if r < nearest:
                        nearest, nearest_ang = r, a
            a += s.angle_increment
        if nearest == float("inf"):
            return lin, ang
        # closeness in [0,1]: 1 at stop_dist, 0 at avoid_dist
        span = max(1e-3, self.avoid_dist - self.avoid_stop_dist)
        close = _clamp((self.avoid_dist - nearest) / span, 0.0, 1.0)
        if lin > 0.0:                                   # brake when obstacle is roughly ahead
            if abs(nearest_ang) < math.radians(45):
                lin *= (1.0 - close)
        # steer away from the obstacle (obstacle on left (+ang) -> turn right (-wz))
        ang -= math.copysign(self.avoid_gain * close, nearest_ang)
        return lin, ang

    def _tick(self):
        if not self.enabled:
            return
        fresh = (self.last_seen is not None and
                 (self.get_clock().now() - self.last_seen).nanoseconds * 1e-9 <= self.lost_timeout)
        if fresh:
            lin, ang = self.target_cmd
            m = Twist()
            m.linear.x, m.angular.z = lin, ang
            self.pub.publish(m)
            self._was_active = True
        elif self._was_active:           # target just lost -> one stop, then go silent
            self.pub.publish(Twist())
            self._was_active = False


def main():
    rclpy.init()
    node = Go2Follower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
