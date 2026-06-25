# Roadmap: the "hard" Go2 capabilities (reinforcement learning)

The scripted controller in this package (IK + fixed gaits + PD) is great for
postures, body lean, a stable crawl, hops, and expressive moves. But a whole
class of behaviors **cannot** be done well open-loop, because they need the robot
to *react* to its own state (it's falling, the ground tilted, it got shoved):

- Reliable **dynamic gaits** (fast trot, pace, bound, gallop), run / sprint
- **Balance recovery** after a push or slip
- **Stairs, slopes, uneven terrain**, stepping over / around obstacles
- **Self-right** after falling, agile **leaps onto platforms**

The modern, proven way to get these on a Go2 is **reinforcement learning (RL)**:
train a neural-net policy in massively-parallel simulation, then deploy it. This
is exactly how Unitree/ANYmal/MIT and the research community do legged locomotion.

## The standard pipeline

1. **Simulator with GPU parallelism — NVIDIA Isaac Lab** (successor to Isaac Gym).
   Runs thousands of Go2s in parallel on your RTX 3050. The community repos
   `legged_gym` / `rsl_rl` (ETH RSL) and Isaac Lab's built-in locomotion tasks
   already include a **Unitree Go2** robot config — start from those, don't
   build from scratch.

2. **Policy**: a small MLP. Inputs = proprioception you ALREADY publish here
   (IMU orientation + angular velocity, joint positions/velocities, last action,
   commanded velocity). Outputs = the 12 joint targets — fed to the *same kind of
   PD law* this package uses. So your sim-side interface is conceptually identical.

3. **Algorithm**: PPO (on-policy). `rsl_rl` provides a fast PPO tuned for legged
   robots; ~minutes-to-hours on your GPU for a walking policy.

4. **Reward**: track commanded velocity + stay upright + energy/effort penalties
   + foot-clearance/air-time terms. Terrain skills come from a **terrain
   curriculum** (Isaac Lab's `TerrainImporter`: flats → slopes → stairs → rough),
   gradually hardened as the policy improves.

5. **Robustness for recovery/push-resistance**: **domain randomization** (mass,
   friction, motor strength, latency) + random shove forces during training.
   This is what produces "self-corrects after a push."

6. **Sim-to-real (later, on hardware)**: keep observations to what a real Go2
   exposes, randomize dynamics, then deploy via `unitree_ros2` / the SDK.

## Concrete next steps (when you want to start)

1. Install **Isaac Lab** (Omniverse/Isaac Sim + Isaac Lab) — your RTX 3050 works,
   though more VRAM trains faster.
2. Run the stock **Go2 flat-terrain velocity task**, confirm it learns to walk.
3. Add the **rough-terrain / stairs curriculum** → terrain traversal.
4. Add **push events + domain randomization** → balance recovery.
5. Optionally bridge the trained policy into THIS ROS 2 package: swap the gait in
   `teleop_node` for "query the policy net each tick" — `/cmd_vel`, `/imu`,
   `/joint_states` are already the right I/O.

## Honest scope

- Walking policy in Isaac Lab: a weekend.
- Robust terrain + push-recovery: weeks of reward/curriculum tuning.
- Backflip: trajectory optimization or specialized RL **and** it's likely
  infeasible on the menagerie model anyway (motors capped at ±23.7 N·m; the real
  Go2's actuators are far stronger).

## Pointers

- Isaac Lab: https://isaac-sim.github.io/IsaacLab/
- legged_gym (ETH RSL): https://github.com/leggedrobotics/legged_gym
- rsl_rl (PPO): https://github.com/leggedrobotics/rsl_rl
- Paper: Rudin et al., "Learning to Walk in Minutes Using Massively Parallel
  Deep RL" (CoRL 2021).
