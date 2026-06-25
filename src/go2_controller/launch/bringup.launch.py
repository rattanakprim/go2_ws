"""One-shot bringup for the full Go2 autonomy stack.

Starts everything in the right order so you don't juggle four terminals:

  t+0s  sim (teleop_controller)  -> robot + TF (odom->base_link->lidar_link)
  t+0s  phone stack              -> rosbridge + web_video_server + web panel
  t+3s  slam_toolbox             -> /map + map->odom   (waits for TF to exist)
  t+10s Nav2                     -> autonomy           (waits for SLAM + CPU to settle)
  t+16s explorer (opt-in)        -> frontier auto-map  (waits for Nav2 to activate)

The staggered start matters: SLAM needs the sim's TF first, and Nav2 needs the
map frame from SLAM. Each piece can be toggled off with an argument.

Usage (source the workspace with conda disabled first):
    ros2 launch go2_controller bringup.launch.py
    ros2 launch go2_controller bringup.launch.py use_viewer:=false   # headless
    ros2 launch go2_controller bringup.launch.py nav2:=false          # SLAM only
    ros2 launch go2_controller bringup.launch.py explore:=true         # auto-map the space
    ros2 launch go2_controller bringup.launch.py localize:=true         # AMCL on saved map
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            TimerAction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory("go2_controller")
    launch_dir = os.path.join(pkg, "launch")

    use_viewer = LaunchConfiguration("use_viewer")
    enable_phone = LaunchConfiguration("phone")
    enable_slam = LaunchConfiguration("slam")
    enable_nav2 = LaunchConfiguration("nav2")
    enable_explore = LaunchConfiguration("explore")
    localize = LaunchConfiguration("localize")
    map_yaml = LaunchConfiguration("map")
    # SLAM runs only when asked AND not localizing (they'd both broadcast map->odom).
    slam_on = PythonExpression(["'", enable_slam, "' == 'true' and '",
                                localize, "' == 'false'"])

    def include(name):
        return PythonLaunchDescriptionSource(os.path.join(launch_dir, name))

    sim = Node(
        package="go2_controller",
        executable="teleop_controller",
        name="go2_controller",
        output="screen",
        parameters=[{"use_viewer": use_viewer}],
    )

    phone = IncludeLaunchDescription(
        include("phone_teleop.launch.py"),
        condition=IfCondition(enable_phone),
    )

    slam = TimerAction(period=3.0, actions=[          # wait for the sim's TF
        IncludeLaunchDescription(include("slam.launch.py"),
                                 condition=IfCondition(slam_on)),
    ])

    localization = TimerAction(period=3.0, actions=[  # alt to slam: map_server + amcl
        IncludeLaunchDescription(
            include("localization.launch.py"),
            launch_arguments={"map": map_yaml}.items(),
            condition=IfCondition(localize)),
    ])

    nav2 = TimerAction(period=10.0, actions=[         # wait for SLAM + CPU to settle
        IncludeLaunchDescription(include("nav2.launch.py"),
                                 condition=IfCondition(enable_nav2)),
    ])

    explore = TimerAction(period=16.0, actions=[      # wait for Nav2 to activate
        IncludeLaunchDescription(include("explore.launch.py"),
                                 condition=IfCondition(enable_explore)),
    ])

    return LaunchDescription([
        DeclareLaunchArgument("use_viewer", default_value="true",
                              description="Show the MuJoCo 3D viewer."),
        DeclareLaunchArgument("phone", default_value="true",
                              description="Start rosbridge + web_video_server + web panel."),
        DeclareLaunchArgument("slam", default_value="true",
                              description="Start slam_toolbox mapping."),
        DeclareLaunchArgument("nav2", default_value="true",
                              description="Start the Nav2 autonomy stack."),
        DeclareLaunchArgument("explore", default_value="false",
                              description="Start frontier auto-exploration (robot maps on its own)."),
        DeclareLaunchArgument("localize", default_value="false",
                              description="Use AMCL on a saved map instead of SLAM (no mapping)."),
        DeclareLaunchArgument("map", default_value="/home/nak/go2_ws/maps/go2_explored.yaml",
                              description="Map YAML for localize mode."),
        sim,
        phone,
        slam,
        localization,
        nav2,
        explore,
    ])
