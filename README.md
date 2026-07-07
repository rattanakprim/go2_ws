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

## Remote control over the internet (tunnel)

The web control panel is also published as a static page at
**<https://rattanakprim.github.io/quadruped/>**. Loading it there shows the full UI, but the
controls only work once its **"Robot host"** box points at a *reachable* rosbridge. Because the
page is served over HTTPS, browsers block plain `ws://`/`http://` to a LAN IP — you need a
secure `wss://` endpoint. A [Cloudflare tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/)
gives you one for free with no account.

```bash
# 0. one-time: install cloudflared (Debian/Ubuntu)
#    see https://pkg.cloudflare.com/ — or: sudo apt install cloudflared

# 1. start the panel + rosbridge + camera on the robot/PC (as usual)
ros2 launch go2_controller phone_teleop.launch.py

# 2. expose rosbridge (:9090) over a public wss:// URL — leave this running
cloudflared tunnel --url http://localhost:9090
#    prints e.g.  https://random-words-1234.trycloudflare.com
```

Then, on the github.io page, paste the tunnel host into the **Robot host** box as a full
websocket URL and hit **Connect** (`https://` → `wss://`):

```
wss://random-words-1234.trycloudflare.com
```

Notes:
- The URL is entered once and saved in the browser (`localStorage`); it changes each time you
  restart `cloudflared` unless you set up a [named tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/do-more-with-tunnels/named-tunnels/).
- The camera stream (`web_video_server` on `:8080`) needs its **own** tunnel if you want live
  video remotely — run a second `cloudflared tunnel --url http://localhost:8080`. Control alone
  only needs the `:9090` tunnel above.
- **On the same WiFi/LAN you don't need any of this** — just open `http://<robot-ip>:8000`
  directly (the `phone_teleop` panel), which talks to rosbridge over plain `ws://`.
- Anyone with the tunnel URL can drive the robot while it's running. Stop `cloudflared`
  (Ctrl-C) when you're done.

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
