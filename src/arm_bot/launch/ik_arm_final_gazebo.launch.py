#!/usr/bin/env python3
"""Inverse-kinematics rig (Gazebo / Ignition, end-to-end) for ik_arm_final.

Runs ONLY the inverse kinematics into the physics sim.

  gazebo.launch.py      — Ignition + spawned robot (embeds gz_ros2_control)
  spawners              — joint_state_broadcaster + arm_controller
  rviz.launch.py        — robot_state_publisher + RViz
  ik_arm_final.py       — /ee_target -> /joint_commands  (robust DLS IK + restarts)
  ik_to_trajectory.py   — /joint_commands -> /arm_controller/joint_trajectory

/joint_states comes from joint_state_broadcaster (true simulated feedback), so
closed_loop=True warm-starts the IK from the physical sim. FK/Jacobian are built
from the live URDF -> 0 error vs the Gazebo/RViz model; IK math follows MATLAB.

Usage:
  ros2 launch arm_bot ik_arm_final_gazebo.launch.py
  ros2 topic pub --once /ee_target geometry_msgs/msg/PoseStamped \\
    "{header: {frame_id: 'base_link'},
      pose: {position: {x: 0.0, y: 0.0, z: 0.56},
             orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}"

Verify:
  ros2 run tf2_ros tf2_echo base_link ee    # Gazebo/URDF ground truth
To also see /ee_pose, overlay the FK node:  ros2 run arm_bot fk_arm_final.py

Args: base_link, tip_link (ee), dq_max (0.10), position_only.
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
    dq_max = LaunchConfiguration('dq_max')
    position_only = LaunchConfiguration('position_only')
    pkg = FindPackageShare('arm_bot')

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([pkg, 'launch', 'gazebo.launch.py'])))
    jsb = Node(package='controller_manager', executable='spawner',
               arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
               output='screen')
    arm = Node(package='controller_manager', executable='spawner',
               arguments=['arm_controller', '--controller-manager', '/controller_manager'],
               output='screen')
    rviz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([pkg, 'launch', 'rviz.launch.py'])))
    ik = Node(
        package='arm_bot', executable='ik_arm_final.py', name='ik_arm_final', output='screen',
        parameters=[{'base_link': base_link, 'tip_link': tip_link, 'dq_max': dq_max,
                     'position_only': position_only, 'closed_loop': True}])
    bridge = Node(package='arm_bot', executable='ik_to_trajectory.py',
                  name='ik_to_trajectory', output='screen',
                  parameters=[{'step_horizon_s': 0.08}])
    return LaunchDescription([
        DeclareLaunchArgument('base_link', default_value='base_link'),
        DeclareLaunchArgument('tip_link', default_value='ee'),
        DeclareLaunchArgument('dq_max', default_value='0.10'),
        DeclareLaunchArgument('position_only', default_value='false'),
        gazebo, jsb, arm, rviz, ik, bridge,
    ])
