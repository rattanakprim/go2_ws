# cartesian_demo.launch.py
#
# One-command recreation of the textbook Cartesian-waypoint figures:
#   - brings up the MoveIt demo (move_group + RViz + fake controllers)
#   - after move_group is ready, runs cartesian_waypoint_demo, which moves to
#     the `bend` pose, draws the 3 green waypoints + red path line, and executes
#     straight-line Cartesian segments 1->2->3->1.
#
# Enable "Show Trail" in the MotionPlanning display (already on in moveit.rviz)
# to reproduce "Cartesian Path Planning Trail Enabled".
#
# Tunables (pass as e.g. dy:=0.05 loops:=2):
#   dy, dx, dz, eef_step, sphere_size, vel_scale, start_state, loops

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    args = {
        "dy": "0.09",
        "dx": "0.0",
        "dz": "0.0",
        "eef_step": "0.005",
        "sphere_size": "0.02",
        "vel_scale": "0.1",
        "start_state": "bend",
        "loops": "1",
    }

    declared = [DeclareLaunchArgument(k, default_value=v) for k, v in args.items()]

    demo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("arm_moveit_config"), "launch", "demo.launch.py"]
            )
        )
    )

    demo_node = Node(
        package="arm_bot",
        executable="cartesian_waypoint_demo",
        name="cartesian_waypoint_demo",
        output="screen",
        parameters=[{k: LaunchConfiguration(k) for k in args}],
    )

    # Give move_group + RViz a few seconds to come up before planning.
    delayed_node = TimerAction(period=10.0, actions=[demo_node])

    return LaunchDescription(declared + [demo, delayed_node])
