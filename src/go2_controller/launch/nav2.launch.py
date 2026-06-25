"""Nav2 (no localization) for the Go2 sim.

slam_toolbox provides the map + map->odom, so this brings up only the
navigation stack (planner, controller, behaviors, bt_navigator, smoothers).
Nav2 publishes /cmd_vel, which teleop_node consumes to drive the gait.

Order:
    ros2 run go2_controller teleop_controller      # sim + TF
    ros2 launch go2_controller slam.launch.py       # map + map->odom
    ros2 launch go2_controller nav2.launch.py        # autonomy

Send a goal from the phone panel (tap the map) or with:
    ros2 topic pub --once /goal_pose geometry_msgs/PoseStamped '{...}'
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    nav2_bringup = get_package_share_directory("nav2_bringup")
    params = os.path.join(get_package_share_directory("go2_controller"),
                          "config", "nav2_params.yaml")
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(nav2_bringup, "launch", "navigation_launch.py")),
            launch_arguments={
                "use_sim_time": "false",
                "autostart": "true",
                "params_file": params,
            }.items(),
        ),
    ])
