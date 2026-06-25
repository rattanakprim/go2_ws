"""AMCL localization for the Go2 sim against a pre-built map.

The payoff of saving a map: instead of slam_toolbox rebuilding the map every run,
map_server serves a saved map and amcl localizes the robot in it (publishing
map->odom). Then nav2 can navigate the known space immediately -- no exploration.

Order (do NOT run slam at the same time):
    ros2 run go2_controller teleop_controller        # sim + odom->base_link TF
    ros2 launch go2_controller localization.launch.py  # map_server + amcl -> map->odom
    ros2 launch go2_controller nav2.launch.py           # autonomy

    # use a different map:
    ros2 launch go2_controller localization.launch.py map:=/path/to/map.yaml
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    nav2_bringup = get_package_share_directory("nav2_bringup")
    params = os.path.join(get_package_share_directory("go2_controller"),
                          "config", "amcl_params.yaml")
    default_map = "/home/nak/go2_ws/maps/go2_explored.yaml"

    return LaunchDescription([
        DeclareLaunchArgument("map", default_value=default_map,
                              description="Path to the map YAML to localize against."),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(nav2_bringup, "launch", "localization_launch.py")),
            launch_arguments={
                "map": LaunchConfiguration("map"),
                "use_sim_time": "false",
                "autostart": "true",
                "params_file": params,
            }.items(),
        ),
    ])
