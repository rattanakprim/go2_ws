"""Object / person following: coco_detector (MobileNet) + the follow controller.

coco_detector runs MobileNet on /camera/image_raw and publishes /detected_objects;
go2_follower turns the target's bbox into /cmd_vel to chase it.

NOTE: the MuJoCo sim has no COCO objects to detect, so this is for a real camera (or a
sim scene with a recognisable rendered asset). torch.cuda is currently unavailable, so
device defaults to cpu. Do NOT run Nav2 at the same time (both publish /cmd_vel).

    ros2 launch go2_controller follow.launch.py
    ros2 launch go2_controller follow.launch.py target_class:='sports ball'
    ros2 launch go2_controller follow.launch.py device:=cuda      # if CUDA becomes available
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    target_class = LaunchConfiguration("target_class")
    device = LaunchConfiguration("device")
    threshold = LaunchConfiguration("threshold")
    return LaunchDescription([
        DeclareLaunchArgument("target_class", default_value="sports ball",
                              description="COCO class to follow (e.g. person, 'sports ball')."),
        DeclareLaunchArgument("device", default_value="cpu",
                              description="coco_detector torch device: cpu or cuda."),
        DeclareLaunchArgument("threshold", default_value="0.5",
                              description="Detection confidence cutoff (lower for non-photoreal sim)."),
        DeclareLaunchArgument("autostart", default_value="false",
                              description="Start following immediately; else wait for the GUI toggle."),
        DeclareLaunchArgument("hold", default_value="0.7",
                              description="Target ball height fraction (bigger = robot holds closer)."),
        Node(
            package="coco_detector", executable="coco_detector_node",
            name="coco_detector", output="screen",
            parameters=[{"device": device,
                         "detection_threshold": threshold,
                         "publish_annotated_image": True}],
        ),
        Node(
            package="go2_controller", executable="follow",
            name="go2_follower", output="screen",
            parameters=[{"target_class": target_class,
                         "autostart": LaunchConfiguration("autostart"),
                         "desired_height_frac": LaunchConfiguration("hold")}],
        ),
    ])
