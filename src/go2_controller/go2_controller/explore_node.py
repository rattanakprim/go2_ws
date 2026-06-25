"""Frontier-based autonomous exploration for the Go2 sim.

Maps an unknown space by itself: finds the boundary between known-free and
unknown cells in /map (a "frontier"), picks one, and sends it to Nav2's
navigate_to_pose action. Repeats until no reachable frontier remains, then
reports the map complete.

Pipeline (must already be running): sim (teleop_controller) -> slam_toolbox
-> nav2.  Nav2 publishes /cmd_vel, teleop's gait consumes it, so manual panels
must be in autonomous / idle-silent mode or they will fight Nav2 for /cmd_vel.

Run:    ros2 run go2_controller explore
Pause:  ros2 topic pub --once /go2/explore_enable std_msgs/Bool '{data: false}'
Resume: ros2 topic pub --once /go2/explore_enable std_msgs/Bool '{data: true}'
"""
import math
from collections import deque

import numpy as np
import rclpy
import rclpy.time
from rclpy.action import ActionClient
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Bool
from nav2_msgs.action import NavigateToPose
from tf2_ros import Buffer, TransformListener


class Go2Explorer(Node):
    def __init__(self):
        super().__init__("go2_explorer")

        # --- params (override with --ros-args -p name:=value) ---
        self.declare_parameter("autostart", True)
        self.declare_parameter("plan_period", 3.0)        # s between planning ticks
        self.declare_parameter("min_frontier_size", 8)    # min cells in a frontier cluster
        self.declare_parameter("blacklist_radius", 0.5)   # m; skip near a failed goal
        self.declare_parameter("min_nav_dist", 0.4)       # m; ignore frontiers this close
        self.enabled = self.get_parameter("autostart").value
        self.min_frontier_size = int(self.get_parameter("min_frontier_size").value)
        self.blacklist_radius = float(self.get_parameter("blacklist_radius").value)
        self.min_nav_dist = float(self.get_parameter("min_nav_dist").value)
        plan_period = float(self.get_parameter("plan_period").value)

        # --- state ---
        self.map_msg = None
        self.goal_active = False
        self.current_goal = None        # (x, y) world of the in-flight goal
        self.blacklist = []             # world points of frontiers Nav2 couldn't reach

        # --- ROS plumbing ---
        self.create_subscription(OccupancyGrid, "map", self._on_map, 1)
        self.create_subscription(Bool, "go2/explore_enable", self._on_enable, 1)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.nav = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.timer = self.create_timer(plan_period, self._plan)

        self.get_logger().info(
            f"Explorer up (autostart={self.enabled}). Waiting for /map + Nav2...")

    # ------------------------------------------------------------------ #
    # callbacks
    # ------------------------------------------------------------------ #
    def _on_map(self, msg):
        self.map_msg = msg

    def _on_enable(self, msg):
        self.enabled = msg.data
        self.get_logger().info(f"Exploration {'enabled' if msg.data else 'paused'}.")

    # ------------------------------------------------------------------ #
    # planning tick
    # ------------------------------------------------------------------ #
    def _plan(self):
        if not self.enabled or self.goal_active or self.map_msg is None:
            return
        if not self.nav.server_is_ready():
            self.get_logger().warn("Nav2 navigate_to_pose not ready yet...", throttle_duration_sec=5.0)
            return

        robot = self._robot_xy()
        if robot is None:
            self.get_logger().warn("No map->base_link TF yet...", throttle_duration_sec=5.0)
            return

        clusters = self._frontier_clusters()
        if not clusters:
            self.get_logger().info("No frontiers left -- map complete. Exploration done.")
            self.enabled = False
            return

        # Goal = the frontier CELL nearest the robot (a point on the known/unknown
        # boundary), NOT the cluster centroid: for a frontier ring enclosing the
        # robot the centroid sits on the robot itself, so the robot never moves.
        # Skip cells closer than min_nav_dist (already-here) and blacklisted ones.
        best = None     # (dist, gx, gy, size)
        for cells, size in clusters:
            d = np.hypot(cells[:, 0] - robot[0], cells[:, 1] - robot[1])
            i = int(np.argmin(d))
            gx, gy, gd = float(cells[i, 0]), float(cells[i, 1]), float(d[i])
            if gd < self.min_nav_dist or self._blacklisted(gx, gy):
                continue
            if best is None or gd < best[0]:
                best = (gd, gx, gy, size)

        if best is None:
            self.get_logger().info(
                "No reachable frontier beyond min distance -- map complete.")
            self.enabled = False
            return
        self._send_goal(best[1], best[2], robot, best[3])

    # ------------------------------------------------------------------ #
    # frontier detection
    # ------------------------------------------------------------------ #
    def _frontier_clusters(self):
        """Return [(cells_world, size), ...] for each frontier cluster in /map.

        cells_world is an (N, 2) array of the cluster's cell centres in the map
        frame, so the caller can aim at the nearest boundary cell, not the centroid.
        """
        m = self.map_msg
        w, h = m.info.width, m.info.height
        res = m.info.resolution
        ox, oy = m.info.origin.position.x, m.info.origin.position.y
        grid = np.asarray(m.data, dtype=np.int16).reshape(h, w)

        free = grid == 0
        unknown = grid == -1
        # a free cell is a frontier if any 4-neighbour is unknown
        neigh_unknown = np.zeros_like(unknown)
        neigh_unknown[1:, :] |= unknown[:-1, :]
        neigh_unknown[:-1, :] |= unknown[1:, :]
        neigh_unknown[:, 1:] |= unknown[:, :-1]
        neigh_unknown[:, :-1] |= unknown[:, 1:]
        frontier = free & neigh_unknown

        # connected components (8-connectivity) over the frontier mask
        visited = np.zeros_like(frontier)
        clusters = []
        rows, cols = np.nonzero(frontier)
        for r0, c0 in zip(rows, cols):
            if visited[r0, c0]:
                continue
            q = deque([(r0, c0)])
            visited[r0, c0] = True
            cells = []
            while q:
                r, c = q.popleft()
                cells.append((r, c))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and frontier[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            q.append((nr, nc))
            if len(cells) >= self.min_frontier_size:
                arr = np.array(cells)                      # (N, 2) rows, cols
                wx = ox + (arr[:, 1] + 0.5) * res
                wy = oy + (arr[:, 0] + 0.5) * res
                clusters.append((np.stack([wx, wy], axis=1), len(cells)))
        return clusters

    # ------------------------------------------------------------------ #
    # goal handling
    # ------------------------------------------------------------------ #
    def _send_goal(self, x, y, robot, size):
        yaw = math.atan2(y - robot[1], x - robot[0])   # approach facing the frontier
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = "map"
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(x)
        goal.pose.pose.position.y = float(y)
        goal.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(yaw / 2.0)

        self.goal_active = True
        self.current_goal = (x, y)
        self.get_logger().info(
            f"Exploring frontier at ({x:.2f}, {y:.2f}) [{size} cells, "
            f"{math.dist(robot, (x, y)):.1f} m away]")
        self.nav.send_goal_async(goal).add_done_callback(self._on_goal_response)

    def _on_goal_response(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().warn("Nav2 rejected the goal; blacklisting.")
            self._blacklist_current()
            self.goal_active = False
            return
        handle.get_result_async().add_done_callback(self._on_goal_result)

    def _on_goal_result(self, future):
        status = future.result().status   # 4 == SUCCEEDED
        if status == 4:
            self.get_logger().info("Reached frontier; replanning.")
        else:
            self.get_logger().warn(f"Goal ended (status {status}); blacklisting that spot.")
            self._blacklist_current()
        self.goal_active = False   # next _plan tick picks the next frontier

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _robot_xy(self):
        try:
            t = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time())
            return (t.transform.translation.x, t.transform.translation.y)
        except Exception:
            return None

    def _blacklist_current(self):
        if self.current_goal is not None:
            self.blacklist.append(self.current_goal)

    def _blacklisted(self, x, y):
        return any(math.dist((x, y), b) < self.blacklist_radius for b in self.blacklist)


def main():
    rclpy.init()
    node = Go2Explorer()
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
