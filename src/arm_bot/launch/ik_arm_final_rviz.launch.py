#!/usr/bin/env python3
"""Inverse-kinematics rig (RViz, no Gazebo) for ik_arm_final.

Runs ONLY the inverse kinematics — publish a target, the arm goes there.

  rviz.launch.py     — robot_state_publisher (latches /robot_description) + RViz
  relay_node.py      — /joint_commands -> /joint_states (transparent fake feedback)
  ik_arm_final.py    — /ee_target -> /joint_commands  (robust DLS IK + restarts)

ik_arm_final builds its FK + Jacobian from the live URDF (so it matches RViz),
follows the MATLAB formulation, and uses a position-first warm-up + random
restarts to reach arbitrary positions AND orientations.

Usage:
  ros2 launch arm_bot ik_arm_final_rviz.launch.py
  ros2 topic pub --once /ee_target geometry_msgs/msg/PoseStamped \\
    "{header: {frame_id: 'base_link'},
      pose: {position: {x: 0.0, y: 0.0, z: 0.56},
             orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}"

Verify it lands on target (tf is what RViz draws):
  ros2 run tf2_ros tf2_echo base_link ee

To also see /ee_pose, overlay the FK node in another terminal:
  ros2 run arm_bot fk_arm_final.py

Args: position_only (false).
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    position_only = LaunchConfiguration('position_only')
    rviz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare('arm_bot'), 'launch', 'rviz.launch.py'])))
    relay = Node(package='arm_bot', executable='relay_node.py',
                 name='joint_relay', output='screen')
    ik = Node(package='arm_bot', executable='ik_arm_final.py',
              name='ik_arm_final', output='screen',
              parameters=[{'position_only': position_only, 'closed_loop': True}])
    return LaunchDescription([
        DeclareLaunchArgument('position_only', default_value='false'),
        rviz, relay, ik,
    ])
