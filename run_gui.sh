#!/usr/bin/env bash
# Launch the Go2 MuJoCo sim + the tkinter GUI control panel.
# Disables conda so system Python 3.10 (ROS 2 Humble) is used.
set -o pipefail

conda deactivate 2>/dev/null || true
conda deactivate 2>/dev/null || true
export PATH="$(echo "$PATH" | tr ':' '\n' | grep -v miniconda | paste -sd:)"
hash -r
source /opt/ros/humble/setup.bash
source /home/nak/go2_ws/install/setup.bash
export DISPLAY=:0

echo "python3 -> $(command -v python3) ($(python3 --version 2>&1))  ROS_DISTRO=$ROS_DISTRO"

# Start the simulation node (MuJoCo viewer + sensors).
ros2 run go2_controller teleop_controller --ros-args -p use_viewer:=true \
    > /home/nak/go2_ws/sim.log 2>&1 &
SIM_PID=$!
echo "sim node pid=$SIM_PID  (log: ~/go2_ws/sim.log)"

# Give the sim a moment to spin up, then start the GUI panel.
sleep 4
ros2 run go2_controller gui_teleop \
    > /home/nak/go2_ws/gui.log 2>&1 &
GUI_PID=$!
echo "gui panel pid=$GUI_PID  (log: ~/go2_ws/gui.log)"

echo "Both launched. To stop: kill $SIM_PID $GUI_PID"
