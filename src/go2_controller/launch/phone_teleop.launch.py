"""Phone/web control for the Go2.

Brings up everything a phone browser needs to drive the simulated Go2 over your
LAN, mirroring the desktop ``gui_teleop`` panel:

  * rosbridge_server  (ws://<pc-ip>:9090)  -- topics over a WebSocket
  * web_video_server  (http://<pc-ip>:8080) -- /camera/image_raw as MJPEG
  * a static HTTP server (http://<pc-ip>:8000) -- serves web/index.html

On your phone (same WiFi/LAN) open:  http://<pc-ip>:8000

The simulation node (teleop_controller) is NOT started by default -- run it
separately, or pass ``controller:=true`` to start it here too.

    ros2 launch go2_controller phone_teleop.launch.py
    ros2 launch go2_controller phone_teleop.launch.py controller:=true
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    web_dir = os.path.join(get_package_share_directory("go2_controller"), "web")

    web_port = LaunchConfiguration("web_port")
    controller = LaunchConfiguration("controller")
    use_viewer = LaunchConfiguration("use_viewer")

    return LaunchDescription([
        DeclareLaunchArgument("web_port", default_value="8000",
                              description="Port for the static web panel."),
        DeclareLaunchArgument("controller", default_value="false",
                              description="Also start the MuJoCo sim (teleop_controller)."),
        DeclareLaunchArgument("use_viewer", default_value="true",
                              description="Show the MuJoCo viewer (only if controller:=true)."),

        # ROS topics <-> WebSocket bridge for the phone.
        Node(
            package="rosbridge_server",
            executable="rosbridge_websocket",
            name="rosbridge_websocket",
            output="screen",
            parameters=[{"port": 9090}],
        ),

        # Camera as an MJPEG stream the phone can show in an <img>.
        Node(
            package="web_video_server",
            executable="web_video_server",
            name="web_video_server",
            output="screen",
            parameters=[{"port": 8080, "address": "0.0.0.0"}],
        ),

        # Serve the static mobile panel.
        ExecuteProcess(
            cmd=["python3", "-m", "http.server",
                 LaunchConfiguration("web_port"),
                 "--bind", "0.0.0.0", "--directory", web_dir],
            output="screen",
        ),

        # Optional: also start the simulation here.
        Node(
            package="go2_controller",
            executable="teleop_controller",
            name="go2_controller",
            output="screen",
            parameters=[{"use_viewer": use_viewer}],
            condition=IfCondition(controller),
        ),
    ])
