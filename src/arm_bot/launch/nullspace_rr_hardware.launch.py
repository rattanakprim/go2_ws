#!/usr/bin/env python3
"""Null-space redundancy demo on the REAL Damiao CAN-FD arm + live RViz.

Runs the resolved-rate null-space node in EE-hold mode (the tool frame is pinned
while the redundant joints self-move) and drives the physical motors through the
CAN-FD stack, while RViz shows exactly the same command stream — so what you see
on screen is what the motors are doing.

  rviz.launch.py            — robot_state_publisher (+ RViz) -> displays /joint_states
  nullspace_rr.py           — EE-hold self-motion -> /joint_commands  (true angles)
  fk_arm_final.py           — /joint_states -> /ee_pose  (monitor the EE hold)
  joint_commands_bridge.py  — /joint_commands -> /joint_states  (deadband + rate cap
                              + velocity/angle clamp). NO sign flip here.

NOT launched here (start manually once motors are powered + CAN-FD dongle is in):
  ros2 run arm_bot_hw pos_motor_sub      # reads /joint_states, drives the motors

IMPORTANT — sign convention:
  pos_motor_sub flips joints 3 & 5 INTERNALLY (its `inverted_motors` set). So this
  rig leaves the bridge sign-flip DISABLED (invert:=-1) and publishes the TRUE
  kinematic angles on /joint_states. That keeps BOTH right at once:
    • RViz shows the true configuration (EE pinned, joints reconfiguring), and
    • pos_motor_sub, applying its own flip, drives the motors correctly.
  Enabling the bridge flip here as well would double-flip joints 3 & 5.
  If your bench moves joints 3/5 the wrong way, override with invert:=2,4 ... no —
  instead check pos_motor_sub.inverted_motors; the flip must happen exactly once.

SAFETY — the arm starts at q=0 and homes SLOWLY to the dexterous ready pose
(home_rate, default 0.8 rad/s) before any self-motion. Keep an e-stop ready and
start with small osc_gain / hold_amp on the first bench run.

Workflow:
  ros2 launch arm_bot nullspace_rr_hardware.launch.py            # terminal A (RViz + commands)
  ros2 run arm_bot_hw pos_motor_sub                              # terminal B (motors live)

Defaults to the "tool fully fixed, no tilt" sweep. Switch shapes:
  ros2 launch arm_bot nullspace_rr_hardware.launch.py hold_motion:=circle      # joints loop, EE position pinned
  ros2 launch arm_bot nullspace_rr_hardware.launch.py hold_motion:=circle lock:=7  # also fix the last joint

Args: hold_motion (sweep|circle), osc_omega, osc_gain, ori_gain, hold_amp, lock,
      home_rate, deadband_rad, max_publish_hz, max_joint_velocity, use_rviz.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    hold_motion = LaunchConfiguration('hold_motion')
    osc_omega = LaunchConfiguration('osc_omega')
    osc_gain = LaunchConfiguration('osc_gain')
    ori_gain = LaunchConfiguration('ori_gain')
    hold_amp = LaunchConfiguration('hold_amp')
    lock = LaunchConfiguration('lock')
    home_rate = LaunchConfiguration('home_rate')
    deadband = LaunchConfiguration('deadband_rad')
    max_hz = LaunchConfiguration('max_publish_hz')
    v_max = LaunchConfiguration('max_joint_velocity')
    use_rviz = LaunchConfiguration('use_rviz')
    pkg = FindPackageShare('arm_bot')

    rviz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([pkg, 'launch', 'rviz.launch.py'])),
        launch_arguments={'use_rviz': use_rviz}.items())
    rr = Node(
        package='arm_bot', executable='nullspace_rr.py', name='nullspace_rr', output='screen',
        parameters=[{'path': 'hold', 'hold_motion': hold_motion,
                     'osc_omega': osc_omega, 'osc_gain': osc_gain, 'ori_gain': ori_gain,
                     'hold_amp': hold_amp, 'home_rate': home_rate,
                     'lock': ParameterValue(lock, value_type=str)}])
    fk = Node(package='arm_bot', executable='fk_arm_final.py', name='fk_arm_final', output='screen')
    bridge = Node(
        package='arm_bot', executable='joint_commands_bridge.py',
        name='joint_commands_bridge', output='screen',
        parameters=[{'deadband_rad': deadband, 'max_publish_hz': max_hz,
                     'max_joint_velocity': v_max,
                     # disable the bridge sign-flip: pos_motor_sub flips 3 & 5 itself,
                     # and /joint_states must stay TRUE so RViz is correct. -1 = none.
                     'invert': [-1]}])
    return LaunchDescription([
        DeclareLaunchArgument('hold_motion', default_value='sweep',
                              description='sweep (full pose pinned) | circle (position pinned)'),
        DeclareLaunchArgument('osc_omega', default_value='0.5'),
        DeclareLaunchArgument('osc_gain', default_value='1.0', description='sweep amplitude'),
        DeclareLaunchArgument('ori_gain', default_value='3.0'),
        DeclareLaunchArgument('hold_amp', default_value='0.4', description='circle joint amplitude'),
        DeclareLaunchArgument('lock', default_value='',
                              description='joints to fix, 1-based e.g. 7'),
        DeclareLaunchArgument('home_rate', default_value='0.8',
                              description='homing joint speed [rad/s] (slow = safe)'),
        DeclareLaunchArgument('deadband_rad', default_value='5e-4'),
        DeclareLaunchArgument('max_publish_hz', default_value='50.0'),
        DeclareLaunchArgument('max_joint_velocity', default_value='3.0'),
        DeclareLaunchArgument('use_rviz', default_value='true'),
        rviz, rr, fk, bridge,
    ])
