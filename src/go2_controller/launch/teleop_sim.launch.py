"""Launch the Go2 MuJoCo controller and a keyboard teleop driver.

The keyboard node runs in its own terminal (prefix xterm) so it can read keys.
If you don't have xterm, run teleop_twist_keyboard yourself in a second terminal:
    ros2 run teleop_twist_keyboard teleop_twist_keyboard
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_viewer = LaunchConfiguration("use_viewer")
    keyboard = LaunchConfiguration("keyboard")

    return LaunchDescription([
        DeclareLaunchArgument("use_viewer", default_value="true",
                              description="Show the MuJoCo 3D viewer window."),
        DeclareLaunchArgument("keyboard", default_value="true",
                              description="Also start teleop_twist_keyboard in an xterm."),

        Node(
            package="go2_controller",
            executable="teleop_controller",
            name="go2_controller",
            output="screen",
            parameters=[{"use_viewer": use_viewer}],
        ),
        Node(
            package="teleop_twist_keyboard",
            executable="teleop_twist_keyboard",
            name="teleop_twist_keyboard",
            output="screen",
            prefix="xterm -e",
            condition=IfCondition(keyboard),
        ),
    ])
