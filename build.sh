#!/usr/bin/env bash
# Build script for the go2_ros2_sdk workspace.
# Disables conda so the system Python 3.10 (the one ROS 2 Humble uses) is used.
set -o pipefail

echo "########## 1/5  Disabling conda ##########"
conda deactivate 2>/dev/null || true
conda deactivate 2>/dev/null || true
export PATH="$(echo "$PATH" | tr ':' '\n' | grep -v miniconda | paste -sd:)"
hash -r
echo "python3 -> $(command -v python3)  ($(python3 --version 2>&1))"
if ! python3 --version 2>&1 | grep -q "3.10"; then
  echo "ERROR: expected system Python 3.10, aborting."; exit 1
fi

echo "########## 2/5  Sourcing ROS 2 Humble ##########"
source /opt/ros/humble/setup.bash
echo "ROS_DISTRO=$ROS_DISTRO"

echo "########## 3/5  Installing Python requirements (large: torch/open3d) ##########"
cd /home/nak/go2_ws/src/go2_ros2_sdk
python3 -m pip install --user -r requirements.txt || { echo "PIP FAILED"; exit 1; }

echo "########## 4/5  Resolving ROS dependencies (rosdep) ##########"
cd /home/nak/go2_ws
rosdep update || true
rosdep install --from-paths src --ignore-src -r -y || { echo "ROSDEP FAILED"; exit 1; }

echo "########## 5/5  colcon build ##########"
colcon build --symlink-install || { echo "COLCON BUILD FAILED"; exit 1; }

echo "########## BUILD COMPLETE ##########"
