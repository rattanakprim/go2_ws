#!/usr/bin/env python3
"""Redundancy (null-space self-motion) demo in RViz — no Gazebo.

Demonstrates the 7-DOF kinematic redundancy ONLY: the end-effector pose is held
fixed while the joints continuously move along the self-motion manifold, so the
elbow visibly reconfigures while the tool frame stays put.

  rviz.launch.py     — robot_state_publisher (latches /robot_description) + RViz
  relay_node.py      — /joint_commands -> /joint_states (transparent fake feedback)
  fk_arm_final.py    — /joint_states -> /ee_pose  (so you can watch the EE hold)
  nullspace_demo.py  — pins the EE pose, drives joints in the null space

Usage:
  ros2 launch arm_bot nullspace_demo_rviz.launch.py
  ros2 launch arm_bot nullspace_demo_rviz.launch.py mode:=center
  ros2 launch arm_bot nullspace_demo_rviz.launch.py omega:=0.4 null_gain:=0.8

Watch the EE stay fixed while joints move:
  ros2 topic echo /ee_pose          # position/orientation ~ constant
  ros2 run tf2_ros tf2_echo base_link ee
The nullspace_demo node also logs the live EE drift (mm/deg) and the accumulated
per-joint self-motion (rad) once a second.

Args: mode (oscillate|center), omega (0.6), null_gain (0.6).
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    mode = LaunchConfiguration('mode')
    omega = LaunchConfiguration('omega')
    null_gain = LaunchConfiguration('null_gain')
    position_only = LaunchConfiguration('position_only')
    lam = LaunchConfiguration('lam')
    use_rviz = LaunchConfiguration('use_rviz')
    rviz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare('arm_bot'), 'launch', 'rviz.launch.py'])),
        launch_arguments={'use_rviz': use_rviz}.items())
    relay = Node(package='arm_bot', executable='relay_node.py',
                 name='joint_relay', output='screen')
    fk = Node(package='arm_bot', executable='fk_arm_final.py',
              name='fk_arm_final', output='screen')
    demo = Node(package='arm_bot', executable='nullspace_demo.py',
                name='nullspace_demo', output='screen',
                parameters=[{'mode': mode, 'omega': omega, 'null_gain': null_gain,
                             'position_only': position_only, 'lam': lam}])
    return LaunchDescription([
        DeclareLaunchArgument('mode', default_value='oscillate'),
        DeclareLaunchArgument('omega', default_value='0.6'),
        DeclareLaunchArgument('null_gain', default_value='0.6'),
        DeclareLaunchArgument('position_only', default_value='false',
                              description='hold EE position only (3-DOF) — use with mode:=limit_avoid'),
        DeclareLaunchArgument('lam', default_value='0.05',
                              description='DLS damping; lower (e.g. 0.005) tightens the hold'),
        DeclareLaunchArgument('use_rviz', default_value='true',
                              description='set false to run headless (no display)'),
        rviz, relay, fk, demo,
    ])
