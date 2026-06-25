"""Online SLAM for the Go2 sim using slam_toolbox.

Consumes /scan + the TF tree (odom->base_link->lidar_link) that teleop_node
broadcasts, and produces:
  * /map        (nav_msgs/OccupancyGrid)  -- the live map
  * map -> odom (TF)                       -- the SLAM correction

Run the sim (teleop_controller) first, then:
    ros2 launch go2_controller slam.launch.py
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    params = os.path.join(get_package_share_directory("go2_controller"),
                          "config", "slam_params.yaml")
    return LaunchDescription([
        Node(
            package="slam_toolbox",
            executable="async_slam_toolbox_node",
            name="slam_toolbox",
            output="screen",
            parameters=[params],
        ),
    ])
