#!/usr/bin/env python3
"""Inverse-kinematics rig (Damiao CAN-FD hardware) for ik_arm_final.

Runs ONLY the inverse kinematics into the motor stack.

  rviz.launch.py            — robot_state_publisher + RViz (visualise the command)
  ik_arm_final.py           — /ee_target -> /joint_commands  (robust DLS IK + restarts)
  joint_commands_bridge.py  — /joint_commands -> /joint_states  (deadband + rate cap +
                              per-joint sign flip for motor sense: joints 3 & 5)

NOT launched here (start manually once motors are powered + CAN-FD dongle is in):
  ros2 run arm_bot_hw pos_motor_sub      # reads /joint_states, drives the motors

IMPORTANT — feedback: joint_commands_bridge sign-flips joints 3 & 5 onto
/joint_states (motor convention). So this rig runs ik_arm_final with
closed_loop:=false — it warm-starts from its own last command, not the
sign-flipped echo. FK/Jacobian come from the live URDF, so commanded poses are
correct for the real arm; the IK math follows the MATLAB formulation.

Workflow:
  ros2 launch arm_bot ik_arm_final_hardware.launch.py      # terminal A
  ros2 run arm_bot_hw pos_motor_sub                        # terminal B (hardware live)
  ros2 topic pub --once /ee_target geometry_msgs/msg/PoseStamped \\
    "{header: {frame_id: 'base_link'},
      pose: {position: {x: 0.0, y: 0.0, z: 0.56},
             orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}"   # terminal C

For a no-motor dry run use ik_arm_final_rviz.launch.py (includes a relay).
To monitor /ee_pose, overlay the FK node:  ros2 run arm_bot fk_arm_final.py

Args: position_only, deadband_rad (5e-4), max_publish_hz (50), max_joint_velocity (3.0).
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    position_only = LaunchConfiguration('position_only')
    deadband = LaunchConfiguration('deadband_rad')
    max_hz = LaunchConfiguration('max_publish_hz')
    v_max = LaunchConfiguration('max_joint_velocity')

    rviz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare('arm_bot'), 'launch', 'rviz.launch.py'])))
    ik = Node(
        package='arm_bot', executable='ik_arm_final.py', name='ik_arm_final', output='screen',
        parameters=[{'position_only': position_only, 'closed_loop': False}])
    bridge = Node(
        package='arm_bot', executable='joint_commands_bridge.py',
        name='joint_commands_bridge', output='screen',
        parameters=[{'deadband_rad': deadband, 'max_publish_hz': max_hz,
                     'max_joint_velocity': v_max}])
    return LaunchDescription([
        DeclareLaunchArgument('position_only', default_value='false'),
        DeclareLaunchArgument('deadband_rad', default_value='5e-4'),
        DeclareLaunchArgument('max_publish_hz', default_value='50.0'),
        DeclareLaunchArgument('max_joint_velocity', default_value='3.0'),
        rviz, ik, bridge,
    ])
