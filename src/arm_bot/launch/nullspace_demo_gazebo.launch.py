#!/usr/bin/env python3
"""Redundancy (null-space self-motion) demo in Gazebo / Ignition — end-to-end.

Same demonstration as nullspace_demo_rviz.launch.py, but driven into the physics
sim: the end-effector pose is held fixed while the joints move along the
self-motion manifold, so the elbow reconfigures in Gazebo while the tool frame
stays put.

  gazebo.launch.py      — Ignition + spawned robot (embeds gz_ros2_control,
                          runs robot_state_publisher -> latched /robot_description)
  spawners              — joint_state_broadcaster (+ /joint_states) + arm_controller
  rviz.launch.py        — RViz (set use_rviz:=false to skip)
  nullspace_demo.py     — pins the EE pose, drives joints in the null space
                          -> /joint_commands
  ik_to_trajectory.py   — /joint_commands -> /arm_controller/joint_trajectory

The demo generates a smooth self-motion command stream open-loop (it integrates
its own configuration from the pose captured at start), and ik_to_trajectory
feeds it to the JointTrajectoryController, which tracks it in the sim.
/joint_states is the true simulated feedback from joint_state_broadcaster.

Usage:
  ros2 launch arm_bot nullspace_demo_gazebo.launch.py
  ros2 launch arm_bot nullspace_demo_gazebo.launch.py mode:=center
  ros2 launch arm_bot nullspace_demo_gazebo.launch.py omega:=0.4 null_gain:=0.8

Verify the EE holds while joints move:
  ros2 run tf2_ros tf2_echo base_link ee     # Gazebo/URDF ground truth
  ros2 topic echo /ee_pose                    # overlay: ros2 run arm_bot fk_arm_final.py
The nullspace_demo node logs commanded EE drift (mm/deg) + joint self-motion (rad).

Args: mode (oscillate|center), omega (0.6), null_gain (0.6), use_rviz (true),
      base_link, tip_link (ee).
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
    mode = LaunchConfiguration('mode')
    omega = LaunchConfiguration('omega')
    null_gain = LaunchConfiguration('null_gain')
    position_only = LaunchConfiguration('position_only')
    lam = LaunchConfiguration('lam')
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
    demo = Node(
        package='arm_bot', executable='nullspace_demo.py', name='nullspace_demo', output='screen',
        parameters=[{'base_link': base_link, 'tip_link': tip_link,
                     'mode': mode, 'omega': omega, 'null_gain': null_gain,
                     'position_only': position_only, 'lam': lam}])
    bridge = Node(package='arm_bot', executable='ik_to_trajectory.py',
                  name='ik_to_trajectory', output='screen',
                  parameters=[{'step_horizon_s': 0.08}])
    return LaunchDescription([
        DeclareLaunchArgument('base_link', default_value='base_link'),
        DeclareLaunchArgument('tip_link', default_value='ee'),
        DeclareLaunchArgument('mode', default_value='oscillate'),
        DeclareLaunchArgument('omega', default_value='0.6'),
        DeclareLaunchArgument('null_gain', default_value='0.6'),
        DeclareLaunchArgument('position_only', default_value='false',
                              description='hold EE position only (3-DOF) — use with mode:=limit_avoid'),
        DeclareLaunchArgument('lam', default_value='0.05',
                              description='DLS damping; lower (e.g. 0.005) tightens the hold'),
        DeclareLaunchArgument('use_rviz', default_value='true',
                              description='set false to skip RViz'),
        gazebo, jsb, arm, rviz, demo, bridge,
    ])
