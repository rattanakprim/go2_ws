"""
Pick-and-place demo sequencer for the Go2 + Piper (A -> B -> A return trip).

Walks the robot (odom-feedback waypoints over /cmd_vel) between two tables and
triggers the controller's autonomous grasp/place. The plan is a simple list of
("walk", stand) / ("pick",) / ("place", xyz) steps, executed in order. The arm IK +
grasp run in the controller (sim access, re-aims on the item), so the walk only has
to get the robot roughly in front of a table.

Run the pick-place controller, then:  ros2 run go2_controller pick_place
"""
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float64MultiArray

STAND_A = (0.60, 0.50, 0.0)        # in front of table A
STAND_B = (0.60, -0.50, 0.0)       # in front of table B
# Release a few cm above the table so the fingers clear the edge; the box drops on.
TOP_A = (1.10, 0.50, 0.30)         # place target on table A
TOP_B = (1.10, -0.50, 0.30)        # place target on table B
V_MAX, W_MAX = 0.22, 0.5
POS_TOL, YAW_TOL = 0.07, 0.06
PICK_SECS, PLACE_SECS = 10.5, 9.0

# A -> B -> A:  grab at A, carry to B, set down, grab back from B, carry to A, set down.
PLAN = [
    ("walk", STAND_A), ("pick", None),
    ("walk", STAND_B), ("place", TOP_B),
    ("pick", None),                              # robot is already at table B
    ("walk", STAND_A), ("place", TOP_A),
]


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class PickPlace(Node):
    def __init__(self):
        super().__init__("pick_place")
        self.cmd_pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.pick_pub = self.create_publisher(Bool, "go2/pick", 1)
        self.place_pub = self.create_publisher(Float64MultiArray, "go2/place", 1)
        self.create_subscription(Odometry, "odom", self._on_odom, 10)
        self.pose = None
        self.step_i = 0
        self.waiting_until = None
        self.create_timer(0.1, self._tick)
        self.get_logger().info("Pick-place sequencer (A->B->A) up. Waiting for /odom…")

    def _on_odom(self, m):
        q = m.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.pose = (m.pose.pose.position.x, m.pose.pose.position.y, yaw)

    def _drive(self, vx, vy, wz):
        t = Twist(); t.linear.x, t.linear.y, t.angular.z = vx, vy, wz
        self.cmd_pub.publish(t)

    def _goto(self, target):
        x, y, yaw = self.pose
        ex, ey = target[0] - x, target[1] - y
        dist = math.hypot(ex, ey)
        ye = wrap(target[2] - yaw)
        if dist > POS_TOL:
            fx = math.cos(yaw) * ex + math.sin(yaw) * ey
            fy = -math.sin(yaw) * ex + math.cos(yaw) * ey
            self._drive(max(-V_MAX, min(V_MAX, 0.8 * fx)),
                        max(-V_MAX, min(V_MAX, 0.8 * fy)),
                        max(-W_MAX, min(W_MAX, 1.0 * ye)))
            return False
        if abs(ye) > YAW_TOL:
            self._drive(0.0, 0.0, max(-W_MAX, min(W_MAX, 1.2 * ye)))
            return False
        self._drive(0.0, 0.0, 0.0)
        return True

    def _wait(self, secs):
        self.waiting_until = self.get_clock().now() + rclpy.duration.Duration(seconds=secs)

    def _tick(self):
        if self.pose is None:
            return
        if self.step_i >= len(PLAN):
            self._drive(0.0, 0.0, 0.0)
            return
        if self.waiting_until is not None:           # mid arm-routine: hold still
            self._drive(0.0, 0.0, 0.0)
            if self.get_clock().now() >= self.waiting_until:
                self.waiting_until = None
                self.step_i += 1
            return
        kind, arg = PLAN[self.step_i]
        if kind == "walk":
            if self._goto(arg):
                self.step_i += 1
        elif kind == "pick":
            self.pick_pub.publish(Bool(data=True))
            self.get_logger().info("pick")
            self._wait(PICK_SECS)
        elif kind == "place":
            self.place_pub.publish(Float64MultiArray(data=[float(v) for v in arg]))
            self.get_logger().info(f"place at {arg}")
            self._wait(PLACE_SECS)


def main():
    rclpy.init()
    node = PickPlace()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
