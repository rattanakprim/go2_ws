# go2_controller

Your own velocity/teleop controller for the Unitree Go2, simulated in **MuJoCo**.

It subscribes to `/cmd_vel` (`geometry_msgs/Twist`) and walks the robot using a
statically-stable **crawl gait** (one foot swings at a time → 3 feet always on the
ground, so it walks reliably with no balance controller). It also takes discrete
**actions** on `/go2/action` (`std_msgs/String`): `jump`, `sit`, `stand`. It
publishes `/joint_states`, `/odom`, and `/imu` (`sensor_msgs/Imu`, from a simulated
IMU on the trunk), and opens a live 3D viewer.

### Topics
| Topic | Type | Dir | Notes |
|-------|------|-----|-------|
| `/cmd_vel` | geometry_msgs/Twist | in | drive (vx, vy, wz) |
| `/go2/action` | std_msgs/String | in | postures, gaits, moves (below) |
| `/go2/body_pose` | geometry_msgs/Twist | in | lean: angular x/y/z = roll/pitch/yaw; linear.z = ride height (0 = default) |
| `/imu` | sensor_msgs/Imu | out | orientation, gyro, accel (incl. gravity) |
| `/camera/image_raw` | sensor_msgs/Image | out | front RGB camera (rgb8) |
| `/camera/camera_info` | sensor_msgs/CameraInfo | out | pinhole intrinsics |
| `/scan` | sensor_msgs/LaserScan | out | 360° 2D lidar (ray-cast) |
| `/points` | sensor_msgs/PointCloud2 | out | 3D lidar (multi-ring ray-cast) |
| `/odom` | nav_msgs/Odometry | out | simulated base pose & twist |
| `/joint_states` | sensor_msgs/JointState | out | 12 leg joints |

The GUI panel shows the **live camera feed** and a **top-down lidar view**; for the
full **3D point cloud** use RViz (add a PointCloud2 display on `/points`, fixed
frame `lidar_link`).

### Actions (`/go2/action`, std_msgs/String)
- **Postures:** `stand`, `sit`, `crouch`, `liedown`
- **Gaits:** `gait_crawl` (stable), `gait_trot` / `gait_pace` / `gait_bound` /
  `gait_pronk` (experimental — dynamic, may stumble open-loop)
- **Moves:** `jump` (~5 cm hop), `wave`, `shake`, `dance`,
  `selfright` (experimental — open-loop self-right rarely succeeds; see
  `docs/RL_ROADMAP.md`)

### Ways to control it
- **GUI panel:** `ros2 run go2_controller gui_teleop` — live **camera feed** at the
  top, plus buttons + sliders, virtual joystick, postures, gaits, body-pose, and
  Jump/Sit/Stand. Keys (panel focused): WASD/QE move, `Space` stop, `j` jump,
  `x` sit, `z` stand.
- **View sensors externally:** `rqt_image_view` (camera) and `rviz2` (add a
  LaserScan display on `/scan`, fixed frame `lidar_link`).
- **Keyboard:** `ros2 run teleop_twist_keyboard teleop_twist_keyboard`.
- **Joystick:** plug in a gamepad, then `ros2 run joy joy_node` +
  `ros2 run go2_controller joy_teleop` (left stick = move, right stick = turn,
  A = jump, B = sit, X = stand).

> Note: the jump is a ~5 cm hop — the menagerie model's motors are torque-limited
> (±23.7 N·m). The real Go2 jumps far higher thanks to much stronger actuators.

```
go2_controller/
├── go2_controller/
│   ├── kinematics.py   # leg forward/inverse kinematics (matches the model)
│   ├── gait.py         # crawl gait: /cmd_vel twist -> 12 foot/joint targets
│   ├── simulator.py    # MuJoCo wrapper + PD torque control (no ROS deps)
│   └── teleop_node.py  # the ROS 2 node (cmd_vel -> sim -> joint_states/odom)
├── launch/teleop_sim.launch.py
├── models/             # Go2 MuJoCo model (mujoco_menagerie/unitree_go2)
└── test_walk.py        # headless check: IK + walks fwd/back/turn/strafe
```

## Build

```bash
conda deactivate                  # use system Python 3.10, not conda
source /opt/ros/humble/setup.bash
cd ~/go2_ws
colcon build --packages-select go2_controller
```

## Run

Terminal 1 — the controller + viewer:
```bash
source ~/go2_ws/install/setup.bash
ros2 run go2_controller teleop_controller
```

Terminal 2 — drive it with the keyboard:
```bash
source /opt/ros/humble/setup.bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```
Use `i`/`,` to go forward/back, `j`/`l` to turn, etc. (`teleop_twist_keyboard`
prints its key map). Stop the robot with `k`.

Or launch both at once (needs `xterm`):
```bash
ros2 launch go2_controller teleop_sim.launch.py
```

### GUI control panel (instead of the keyboard)

A point-and-click window with direction buttons + speed sliders:
```bash
ros2 run go2_controller gui_teleop      # in its own terminal
```
Buttons drive the robot (latched until you press **STOP**); sliders set speed.
Keyboard while the panel is focused: `W`/`S` fwd/back, `A`/`D` strafe,
`Q`/`E` turn, `Space` stop.

Headless (no viewer, e.g. over SSH): add `-p use_viewer:=false`.

## Tuning (where to learn)

| What | Where |
|------|-------|
| Walking speed / step size | `max_step`, `period`, `duty` in `gait.py` |
| Foot lift height | `step_height` in `gait.py` |
| Leg stiffness / tracking | `kp`, `kd` (params, or defaults in `simulator.py`) |
| Standing height | `stand_height` in `gait.py` |

## Test (no ROS needed)

```bash
cd ~/go2_ws/src/go2_controller
MUJOCO_GL=egl python3 test_walk.py
```
