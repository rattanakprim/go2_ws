#!/usr/bin/env python3
"""Forward-kinematics test rig (RViz, no Gazebo) for fk_arm_final.

Runs ONLY the forward kinematics — drive the joints, watch the EE pose.

  rviz.launch.py           — robot_state_publisher (latches /robot_description) + RViz
  joint_state_publisher    — zero defaults for every URDF joint, and merges
                             /joint_commands into /joint_states
  fk_arm_final.py          — /joint_states -> /ee_pose

fk_arm_final builds FK from the live URDF, so /ee_pose matches robot_state_publisher
(tf base_link->ee) exactly. This is the FK ground truth.

Usage:
  ros2 launch arm_bot fk_arm_final_rviz.launch.py
  # drive the joints from another terminal:
  ros2 topic pub -r 30 /joint_commands sensor_msgs/msg/JointState "{
    name: ['joint_1','joint_2','joint_3','joint_4','joint_5','joint_6','joint_7'],
    position: [0.0, 0.3, 0.0, -0.5, 0.0, 0.1, 0.0]}"

Verify the FK output (must agree with what RViz draws):
  ros2 topic echo /ee_pose
  ros2 run tf2_ros tf2_echo base_link ee

To run inverse kinematics instead, use ik_arm_final_rviz.launch.py.

Args: base_link (base_link), tip_link (ee).
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    base_link = LaunchConfiguration('base_link')
    tip_link = LaunchConfiguration('tip_link')

    rviz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare('arm_bot'), 'launch', 'rviz.launch.py'])))
    jsp = Node(
        package='joint_state_publisher', executable='joint_state_publisher',
        name='joint_state_publisher', output='screen',
        parameters=[{'rate': 30, 'source_list': ['joint_commands']}])
    fk = Node(
        package='arm_bot', executable='fk_arm_final.py', name='fk_arm_final',
        output='screen', parameters=[{'base_link': base_link, 'tip_link': tip_link}])
    return LaunchDescription([
        DeclareLaunchArgument('base_link', default_value='base_link'),
        DeclareLaunchArgument('tip_link', default_value='ee'),
        rviz, jsp, fk,
    ])
