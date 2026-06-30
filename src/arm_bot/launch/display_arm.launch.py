#!/usr/bin/env python3
"""Display arm_bot on its own in RViz, with joint sliders.

Standalone viewer -- does NOT require building/installing the arm_bot package.
Loads urdf/arm_display.urdf (absolute mesh paths) into robot_state_publisher,
runs joint_state_publisher_gui for live joint control, and opens RViz.
"""
import os
from launch import LaunchDescription
from launch_ros.actions import Node

PKG = "/home/nak/go2_ws/src/arm_bot"
URDF = os.path.join(PKG, "urdf", "arm_display.urdf")
RVIZ = os.path.join(PKG, "rviz", "view.rviz")


def generate_launch_description():
    robot_description = {"robot_description": open(URDF).read()}
    nodes = [
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             output="screen", parameters=[robot_description]),
        Node(package="joint_state_publisher_gui", executable="joint_state_publisher_gui",
             output="screen"),
        Node(package="rviz2", executable="rviz2", output="screen",
             arguments=(["-d", RVIZ] if os.path.exists(RVIZ) else [])),
    ]
    return LaunchDescription(nodes)
