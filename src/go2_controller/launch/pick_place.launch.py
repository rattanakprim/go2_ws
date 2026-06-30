"""Pick-and-place demo: Go2 + Piper walks to table A, grabs the box, carries it to
table B, and places it.

  ros2 launch go2_controller pick_place.launch.py

Starts the MuJoCo controller on the pick-place scene (two tables + a graspable box)
and the sequencer node. Pass start_delay to give the sim time to come up before the
sequence begins.
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("go2_controller")
    scene = os.path.join(share, "models", "go2_piper", "go2_piper_pickplace.xml")
    use_viewer = LaunchConfiguration("use_viewer")
    start_delay = LaunchConfiguration("start_delay")

    controller = Node(
        package="go2_controller", executable="teleop_controller",
        name="go2_controller", output="screen",
        parameters=[{"model_path": scene, "use_viewer": use_viewer}],
    )
    # The demo runs inside the controller (sim access). Kick it off once at startup;
    # the GUI / web "Run pick-place" buttons publish the same topic on demand.
    trigger = ExecuteProcess(
        cmd=["ros2", "topic", "pub", "--once", "/go2/run_pickplace",
             "std_msgs/msg/Bool", "{data: true}"],
        output="screen",
    )
    return LaunchDescription([
        DeclareLaunchArgument("use_viewer", default_value="true"),
        DeclareLaunchArgument("start_delay", default_value="6.0",
                              description="seconds to wait before auto-starting the demo"),
        controller,
        TimerAction(period=start_delay, actions=[trigger]),
    ])
