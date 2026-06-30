#!/usr/bin/env python3
"""
ik_to_trajectory.py — bridge /joint_commands (sensor_msgs/JointState)
to /arm_controller/joint_trajectory (trajectory_msgs/JointTrajectory).

Each incoming JointState is forwarded as a single-point trajectory with
time_from_start = ~step_horizon_s, letting JointTrajectoryController
interpolate between the robot's current state and the IK target.
"""
from builtin_interfaces.msg import Duration

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class IKToTrajectory(Node):
    def __init__(self):
        super().__init__("ik_to_trajectory")

        self.declare_parameter("step_horizon_s", 0.08)
        self._horizon = float(self.get_parameter("step_horizon_s").value)

        self._pub = self.create_publisher(
            JointTrajectory, "/arm_controller/joint_trajectory", 10
        )
        self.create_subscription(
            JointState, "/joint_commands", self._cb, 20
        )
        self.get_logger().info(
            f"ik_to_trajectory: horizon={self._horizon*1000:.0f} ms"
        )

    def _cb(self, msg: JointState):
        if not msg.name or not msg.position:
            return

        traj = JointTrajectory()
        traj.joint_names = list(msg.name)

        pt = JointTrajectoryPoint()
        pt.positions = list(msg.position)
        sec = int(self._horizon)
        nsec = int((self._horizon - sec) * 1e9)
        pt.time_from_start = Duration(sec=sec, nanosec=nsec)

        traj.points.append(pt)
        self._pub.publish(traj)


def main():
    rclpy.init()
    rclpy.spin(IKToTrajectory())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
