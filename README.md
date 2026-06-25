# Go2 quadruped — sim-first ROS 2 workspace

A learning project: a Unitree **Go2** quadruped simulated in **MuJoCo**, driven by a
self-authored ROS 2 controller, with manual teleop, full SLAM/Nav2 autonomy, autonomous
exploration, map-based localization, and reliable camera object-following with reactive
obstacle avoidance. Built sim-first; hardware later.

- **ROS 2:** Humble · **Python:** 3.10 (system, *not* conda) · **Sim:** MuJoCo

## Packages

| Package | What it is |
|---|---|
| `go2_controller` | **The project.** Custom MuJoCo sim + velocity/teleop controller, gaits, sensors, GUIs, autonomy nodes. |
| `go2_ros2_sdk` | Third-party Unitree SDK (real-robot bridge + `coco_detector`). *Not* in this repo — clone separately (below). |

## First-time setup

```bash
# 1. clone the third-party SDK into src/ (gitignored here)
cd ~/go2_ws/src
git clone https://github.com/abizovnuralem/go2_ros2_sdk.git

# 2. build (disables conda so system Python 3.10 / ROS Humble is used)
cd ~/go2_ws
./build.sh                       # or: colcon build --symlink-install
source install/setup.bash
```

> Build/run use **system Python 3.10**, not conda — the helper scripts disable conda. See
> the project memory notes for env gotchas.

## Running things

Source first: `source /opt/ros/humble/setup.bash && source install/setup.bash`

| Goal | Command |
|---|---|
| Sim + GUI control panel | `./run_gui.sh` |
| Sim only (with viewer) | `ros2 run go2_controller teleop_controller --ros-args -p use_viewer:=true` |
| Full autonomy stack (sim→SLAM→Nav2) | `ros2 launch go2_controller bringup.launch.py` |
| …with autonomous exploration | `ros2 launch go2_controller bringup.launch.py explore:=true` |
| …on a saved map (AMCL, no SLAM) | `ros2 launch go2_controller bringup.launch.py localize:=true` |
| SLAM only | `ros2 launch go2_controller slam.launch.py` |
| Nav2 only | `ros2 launch go2_controller nav2.launch.py` |
| Phone / web control panel | `ros2 launch go2_controller phone_teleop.launch.py` |
| Object following (colour-blob, reliable in sim) | `ros2 launch go2_controller follow_color.launch.py` |
| Object following (MobileNet, for a real camera) | `ros2 launch go2_controller follow.launch.py` |

Desktop GUI: `ros2 run go2_controller gui_teleop`. The Map section has Autonomous-mode,
Auto-explore, Follow toggles + Save-map; the Move-ball panel (joystick/buttons) steers the
sim ball for the follow demo.

## Capabilities

- **Locomotion:** stable crawl gait (+ experimental trot/pace/bound/pronk), postures
  (stand/sit/crouch/liedown), body lean, timed routines (jump/wave/shake/dance).
- **Control:** desktop tkinter GUI, phone/web panel (rosbridge), gamepad, keyboard.
- **Sensors:** front RGB camera, 360° 2D lidar (`/scan`), 3D lidar (`/points`), IMU.
- **Autonomy:** SLAM (`slam_toolbox`), Nav2 navigation, frontier auto-exploration,
  AMCL localization on a saved map, tap-to-go from the phone/desktop map.
- **Following:** colour-blob tracker → velocity controller with reactive lidar obstacle
  avoidance; ball is drivable from the GUI (mouse joystick).

## Not yet / roadmap

- **RL walking policy** (fast dynamic gaits, balance recovery) — see
  `src/go2_controller/docs/RL_ROADMAP.md`. Gated on installing the NVIDIA GPU driver.
- **Sim-to-real** deployment via `go2_ros2_sdk`.
