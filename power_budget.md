# Power Budget — DIY 50 kg-Payload Quadruped

Companion to `quadruped_50kg_BOM.csv`. All figures are planning ballparks — re-run with your own mass/speed/efficiency.

## Assumptions
- Total mass: **90 kg** (50 kg payload + ~40 kg structure)
- Pack voltage: **14S** → 51.8 V nominal (~58.8 V full, ~42 V empty)
- Motor + drivetrain efficiency: **~65%**
- Cost-of-Transport (CoT): **~1.0** (typical heavy quadruped)
- Usable battery capacity: **85%** of nameplate (don't deep-discharge)
- Nominal walking speed: **1 m/s**

## Power draw by mode

| Mode | Locomotion / holding | + Electronics | Total draw |
|---|---|---|---|
| Standing — low-ratio QDD (custom / B2) | ~400 W (motors holding static torque) | ~60 W | **~450 W** |
| Standing — high-ratio (geometry-opt, AK80-64) | ~120 W (gearing holds torque cheaply) | ~60 W | **~180 W** |
| Walking @ 1 m/s | ~1,300 W (CoT x m x g x v / 0.65) | ~60 W | **~1,350 W** |
| Peak (fast trot / accel / climb, brief) | 2,500–3,500 W | — | **~3,000 W spike** |

**Electronics breakdown (~60 W):** Jetson AGX Orin ~40 W (full RL + perception), Livox Mid-360 ~6.5 W, 2x RealSense D435i ~4 W, IMU / MCU / comms ~10 W.

## Runtime per battery tier

| Pack | Nameplate | Usable (85%) | Standing | Walking cont. | Mixed (~900 W) |
|---|---|---|---|---|---|
| Budget — 14S 10Ah LiPo | 518 Wh | 440 Wh | ~1.0 h | ~20 min | ~0.5 h |
| Premium — 14S 30Ah Li-ion | 1,554 Wh | 1,320 Wh | ~2.9 h | ~1.0 h | ~1.5 h |
| GeomOpt — 14S 15Ah | 777 Wh | 660 Wh | ~3.7 h | ~36 min | ~0.7 h |

## Electrical sizing (feeds the BOM)
- **Continuous current** (walking): 1,350 W / 51.8 V ≈ **26 A**
- **Peak current**: 3,000 W / 51.8 V ≈ **58 A** → wiring / contactor / BMS must handle ~60 A.
  - AS150 connectors (150 A) cover peaks; restrict XT90 (~90 A) to lower-current branches.
- **Battery C-rate**: Budget 10Ah @ 58 A = 5.8C (easy for LiPo). Premium 30Ah @ 58 A = ~2C → use high-drain Li-ion cells (Molicel P42A / P45B class).
- **Charge time**: Premium 1.55 kWh at a 5 A charger (~260 W) ≈ **6 h**; size charger up for faster turnaround.

## Takeaways
1. **20 min walking on the budget pack is too little** for the autonomy stack to be useful — plan on the ~1.5 kWh premium pack (or larger) for >1 h missions.
2. **Standing power is the silent battery (and heat) killer for low-ratio QDD** — ~450 W just to stand holding 50 kg. Another strong argument for the geometry-optimized high-ratio design (~180 W standing) if the robot holds the load for long periods.
3. **Thermal limit, not just battery**: low-ratio QDD motors holding 50 kg statically can overheat regardless of charge — design for a "sit / rest" posture or duty-cycle the standing.
