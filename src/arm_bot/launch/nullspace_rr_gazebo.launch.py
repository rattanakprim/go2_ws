#!/usr/bin/env python3
"""Resolved-rate null-space redundancy resolution — end-to-end in Gazebo (Ignition).

Same as nullspace_rr_rviz.launch.py but driven into the physics sim via the
existing ign_ros2_control stack and JointTrajectoryController.

  gazebo.launch.py    — Ignition + spawned robot (ign_ros2_control) + robot_state_publisher
  spawners            — joint_state_broadcaster (/joint_states) + arm_controller
  rviz.launch.py      — RViz (use_rviz:=false to skip)
  nullspace_rr.py     — resolved-rate loop -> /joint_commands
  ik_to_trajectory.py — /joint_commands -> /arm_controller/joint_trajectory

Usage:
  ros2 launch arm_bot nullspace_rr_gazebo.launch.py objective:=limit k:=2.0
  ros2 launch arm_bot nullspace_rr_gazebo.launch.py objective:=manip k:=2.0 use_rviz:=false

Args: objective (none|limit|manip), k, lam, radius, period, plane, use_rviz.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    objective = LaunchConfiguration('objective')
    k = LaunchConfiguration('k')
    lam = LaunchConfiguration('lam')
    path = LaunchConfiguration('path')
    radius = LaunchConfiguration('radius')
    corner = LaunchConfiguration('corner')
    period = LaunchConfiguration('period')
    plane = LaunchConfiguration('plane')
    osc_omega = LaunchConfiguration('osc_omega')
    osc_gain = LaunchConfiguration('osc_gain')
    ori_gain = LaunchConfiguration('ori_gain')
    hold_motion = LaunchConfiguration('hold_motion')
    hold_amp = LaunchConfiguration('hold_amp')
    lock = LaunchConfiguration('lock')
    use_rviz = LaunchConfiguration('use_rviz')
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
        PythonLaunchDescriptionSource(PathJoinSubstitution([pkg, 'launch', 'rviz.launch.py'])),
        launch_arguments={'use_rviz': use_rviz}.items())
    rr = Node(package='arm_bot', executable='nullspace_rr.py', name='nullspace_rr', output='screen',
              parameters=[{'objective': objective, 'k': k, 'lam': lam, 'path': path,
                           'radius': radius, 'corner': corner,
                           'period': period, 'plane': plane,
                           'osc_omega': osc_omega, 'osc_gain': osc_gain,
                           'ori_gain': ori_gain, 'hold_motion': hold_motion,
                           'hold_amp': hold_amp,
                           'lock': ParameterValue(lock, value_type=str)}])
    bridge = Node(package='arm_bot', executable='ik_to_trajectory.py',
                  name='ik_to_trajectory', output='screen',
                  parameters=[{'step_horizon_s': 0.08}])
    fk = Node(package='arm_bot', executable='fk_arm_final.py', name='fk_arm_final', output='screen')
    trail = Node(package='arm_bot', executable='ee_trail_marker.py', name='ee_trail_marker',
                 output='screen')
    return LaunchDescription([
        DeclareLaunchArgument('objective', default_value='none'),
        DeclareLaunchArgument('k', default_value='0.0'),
        DeclareLaunchArgument('lam', default_value='0.05'),
        DeclareLaunchArgument('path', default_value='square', description='circle | square'),
        DeclareLaunchArgument('radius', default_value='0.06'),
        DeclareLaunchArgument('corner', default_value='0.3'),
        DeclareLaunchArgument('period', default_value='8.0'),
        DeclareLaunchArgument('plane', default_value='yz'),
        DeclareLaunchArgument('osc_omega', default_value='0.5',
                              description='hold mode: self-motion speed [rad/s]'),
        DeclareLaunchArgument('osc_gain', default_value='1.0',
                              description='hold sweep: self-motion amplitude'),
        DeclareLaunchArgument('ori_gain', default_value='3.0',
                              description='hold sweep: EE-hold orientation gain'),
        DeclareLaunchArgument('hold_motion', default_value='sweep',
                              description='hold mode shape: sweep | circle'),
        DeclareLaunchArgument('hold_amp', default_value='0.4',
                              description='hold circle: joint amplitude [rad]'),
        DeclareLaunchArgument('lock', default_value='',
                              description='hold mode: joints to fix, 1-based e.g. 7'),
        DeclareLaunchArgument('use_rviz', default_value='true'),
        gazebo, jsb, arm, rviz, rr, bridge, fk, trail,
    ])
