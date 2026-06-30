#!/usr/bin/env python3
"""Bridge IK output (/joint_commands) to the Damiao CAN-FD motor stack.

Pattern lifted from ~/scara_bot_ws/src/SCARA_pkg/SCARA_pkg/joint_states_bridge.py
and adapted for the 7-DOF arm + Damiao CAN-FD hardware (DM4340 joints 1-4,
DM4310 joints 5-7) driven by tanerb_canfd_sub/pos_motor_sub.py.

  /joint_commands (sensor_msgs/JointState, ~200 Hz from ik_arm_dh_urdf)
        │
        │  per-joint deadband (skip noise)
        │  global rate cap   (don't flood CAN-FD bus)
        │  per-joint sign flip (matches inverted_motors in pos_motor_sub.py)
        │  optional joint-velocity clamp (safety)
        ▼
  /joint_states (sensor_msgs/JointState — already what pos_motor_sub.py reads)

The CAN-FD subscriber (pos_motor_sub.py) is unchanged — it sees the same
/joint_states topic it always saw, just with deadband/rate filtering applied.

Default parameters err on the safe side. Tune up only after smooth motion is
confirmed on the bench.

Parameters
----------
  output_topic        ('/joint_states') — final topic to publish on
  deadband_rad        (5e-4)            — skip publish if max joint change < this
  max_publish_hz      (50.0)            — cap publish rate
  max_joint_velocity  (3.0 rad/s)       — clamp commanded joint velocity (uses dt
                                          since previous publish for the limit;
                                          set to 0 to disable)
  invert              ([])              — joint indices (0-based) whose sign to flip
                                          to match motor direction. Defaults are set
                                          for the 7-DOF arm based on
                                          pos_motor_sub.inverted_motors which lists
                                          can_id 3 and 5  (joints 3 & 5 → indices 2, 4)
  publish_initial     (True)            — emit one frame immediately on first command
                                          to seed pos_motor_sub
"""

import time
import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


JOINT_NAMES = [f'joint_{i}' for i in range(1, 8)]


class JointCommandsBridge(Node):
    def __init__(self):
        super().__init__('joint_commands_bridge')

        self.declare_parameter('output_topic',       '/joint_states')
        self.declare_parameter('deadband_rad',       5.0e-4)   # 0.5 mrad
        self.declare_parameter('max_publish_hz',     50.0)
        self.declare_parameter('max_joint_velocity', 3.0)      # rad/s; 0 disables
        # hard per-joint angle limit to protect motor cables; clamp |q| <= this
        self.declare_parameter('max_joint_angle',    math.pi)  # rad; 0 disables
        # joint indices (0-based) whose sign to flip; defaults match the
        # inverted_motors set in tanerb_canfd_sub/pos_motor_sub.py
        # (can_id 3 -> joint_3 -> index 2;  can_id 5 -> joint_5 -> index 4)
        self.declare_parameter('invert',             [2, 4])
        self.declare_parameter('publish_initial',    True)
        # Seed pose published once at startup so that downstream nodes (FK ->
        # /ee_pose -> demo -> /ee_target -> IK -> /joint_commands -> us) can
        # bootstrap before any /joint_commands has arrived. Set rate_hz=0 to
        # disable. With rate_hz>0, the seed pose is republished at that rate
        # until the first real /joint_commands message arrives.
        self.declare_parameter('seed_positions',     [0.0]*7)
        self.declare_parameter('seed_rate_hz',       20.0)

        gp = lambda n: self.get_parameter(n).value
        self.output_topic   = gp('output_topic')
        self.deadband       = float(gp('deadband_rad'))
        self.max_hz         = max(1.0, float(gp('max_publish_hz')))
        self.min_period     = 1.0 / self.max_hz
        self.v_max          = float(gp('max_joint_velocity'))
        self.q_max          = float(gp('max_joint_angle'))
        self.invert         = set(int(i) for i in gp('invert'))
        self.publish_initial = bool(gp('publish_initial'))

        self._last_published = None     # np.ndarray (7,) of last sent joint positions
        self._last_pub_t     = 0.0      # monotonic seconds

        # publisher first so subscribers attached before us see it
        self.pub = self.create_publisher(JointState, self.output_topic, 10)
        self.create_subscription(JointState, '/joint_commands',
                                 self.cb, qos_profile=10)

        self.get_logger().info(
            f'joint_commands_bridge up. /joint_commands -> {self.output_topic}; '
            f'deadband={self.deadband*1000:.2f} mrad; '
            f'rate cap={self.max_hz:.1f} Hz; '
            f'v_max={self.v_max:.2f} rad/s; '
            f'q_max=±{math.degrees(self.q_max):.1f}° ({self.q_max:.3f} rad); '
            f'invert indices={sorted(self.invert)}'
        )

        seed_positions = [float(v) for v in gp('seed_positions')]
        if len(seed_positions) != 7:
            self.get_logger().warn(
                f'seed_positions has {len(seed_positions)} items; expected 7. '
                f'Using zeros.')
            seed_positions = [0.0]*7
        self._seed_q = seed_positions
        seed_rate = float(gp('seed_rate_hz'))
        if seed_rate > 0:
            self._seed_timer = self.create_timer(1.0/seed_rate, self._seed_tick)
        else:
            self._seed_timer = None
            self._publish_seed()  # one-shot

    def _publish_seed(self):
        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.name = list(JOINT_NAMES)
        out.position = list(self._seed_q)
        self.pub.publish(out)

    def _seed_tick(self):
        # Republish seed until the first real /joint_commands frame arrives,
        # at which point we cancel ourselves so we don't fight the IK output.
        if self._last_published is not None:
            self._seed_timer.cancel()
            self._seed_timer = None
            return
        self._publish_seed()

    def cb(self, msg: JointState):
        # extract positions in canonical joint_1..joint_7 order; missing → 0
        try:
            q = np.array(
                [msg.position[msg.name.index(n)] for n in JOINT_NAMES],
                dtype=float
            )
        except (ValueError, IndexError):
            return

        # apply per-joint sign flip
        for idx in self.invert:
            if 0 <= idx < 7:
                q[idx] = -q[idx]

        # hard per-joint angle clamp (cable-protection)
        if self.q_max > 0:
            clamped = np.clip(q, -self.q_max, self.q_max)
            if not np.array_equal(clamped, q):
                over = np.where(np.abs(q) > self.q_max)[0]
                self.get_logger().warn(
                    f'joint angle clamp hit on indices {over.tolist()}: '
                    f'{[f"{math.degrees(q[i]):+.1f}°" for i in over]} '
                    f'-> ±{math.degrees(self.q_max):.1f}°',
                    throttle_duration_sec=1.0,
                )
                q = clamped

        now = time.monotonic()
        first = self._last_published is None

        # rate cap
        if not first and (now - self._last_pub_t) < self.min_period:
            return

        # deadband — must always have a valid baseline first
        if not first:
            if float(np.max(np.abs(q - self._last_published))) < self.deadband:
                return

        # velocity clamp (per-joint)
        if not first and self.v_max > 0:
            dt = max(now - self._last_pub_t, 1e-3)
            dq_max = self.v_max * dt
            dq = q - self._last_published
            np.clip(dq, -dq_max, dq_max, out=dq)
            q = self._last_published + dq

        if first and not self.publish_initial:
            self._last_published = q
            self._last_pub_t = now
            return

        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.name = list(JOINT_NAMES)
        out.position = [float(v) for v in q]
        # leave velocity / effort empty — pos_motor_sub falls back to default_vel
        self.pub.publish(out)

        self._last_published = q
        self._last_pub_t = now


def main(args=None):
    rclpy.init(args=args)
    node = JointCommandsBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
