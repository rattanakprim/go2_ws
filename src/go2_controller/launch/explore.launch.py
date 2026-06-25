"""Autonomous frontier exploration for the Go2 sim.

The robot maps an unknown space by itself instead of you driving it first.
Requires sim + slam + nav2 already up (see bringup.launch.py):

    ros2 run go2_controller teleop_controller     # sim + TF
    ros2 launch go2_controller slam.launch.py       # map + map->odom
    ros2 launch go2_controller nav2.launch.py        # autonomy
    ros2 launch go2_controller explore.launch.py      # this -- drives exploration

Pause/resume any time:
    ros2 topic pub --once /go2/explore_enable std_msgs/Bool '{data: false}'
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="go2_controller",
            executable="explore",
            name="go2_explorer",
            output="screen",
        ),
    ])
