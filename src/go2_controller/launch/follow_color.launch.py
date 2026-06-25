"""Object following via reliable COLOUR-BLOB tracking (sim) -- tracker + follow controller.

color_tracker segments the vivid magenta ball from /camera/image_raw and publishes
/detected_objects (class_id "ball"); go2_follower chases it. Reliable at any distance with
no GPU and no MobileNet mis-labelling -- the robust alternative to follow.launch.py for sim.
Obstacle avoidance and the GUI ball-driving work unchanged.

    ros2 launch go2_controller follow_color.launch.py
    ros2 launch go2_controller follow_color.launch.py hold:=0.7   # closer hold
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("autostart", default_value="false",
                              description="Start following immediately; else wait for the GUI toggle."),
        DeclareLaunchArgument("hold", default_value="0.6",
                              description="Target ball height fraction (bigger = holds closer)."),
        Node(package="go2_controller", executable="color_tracker",
             name="color_tracker", output="screen"),
        Node(package="go2_controller", executable="follow",
             name="go2_follower", output="screen",
             parameters=[{"target_class": "ball",
                          "autostart": LaunchConfiguration("autostart"),
                          "desired_height_frac": LaunchConfiguration("hold")}]),
    ])
