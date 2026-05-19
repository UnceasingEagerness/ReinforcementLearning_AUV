"""
scenario.py
-----------
HoloOcean scenario configuration for 2 ASVs in PierHarbor.

Agent  : SurfaceVessel (water surface agent)
Control: Scheme 0 — Thrusters: [Left, Right] floats
Sensors: IMU (6-DOF accel + gyro), GPS (lat/lon/alt), LocationSensor (x,y,z)
World  : PierHarbor

POSG mapping (from Ch.3 reading):
  I  = {asv0, asv1}
  S  = full HoloOcean world state (both positions, velocities, obstacles)
  Oi = IMU + GPS + Location of agent i  (local observation only)
  Ai = [left_thrust, right_thrust] in [-1, 1]
  T  = HoloOcean Fossen physics
  Ri = computed externally (goal distance + collision penalty)
"""

# ---------------------------------------------------------------------------
# Agent spawn positions — separated enough to avoid initial collision
# PierHarbor uses NED-like coordinates; z=0 is water surface
# ---------------------------------------------------------------------------
ASV0_SPAWN = [0.0,  5.0, 0.0]   # [x, y, z] in metres
ASV1_SPAWN = [0.0, -5.0, 0.0]

# Goal positions each ASV must reach
ASV0_GOAL  = [30.0,  5.0, 0.0]
ASV1_GOAL  = [30.0, -5.0, 0.0]

# Collision distance threshold (metres)
COLLISION_DIST = 2.0

# ---------------------------------------------------------------------------
# Sensor definitions — reused for both agents
# ---------------------------------------------------------------------------
def _imu_sensor(name="IMUSensor"):
    return {
        "sensor_type": "IMUSensor",
        "sensor_name": name,
        "socket": "Platform",
        "configuration": {
            "AccelSigma": 0.00277,
            "AngVelSigma": 0.00123
        }
    }

def _gps_sensor(name="GPSSensor"):
    return {
        "sensor_type": "GPSSensor",
        "sensor_name": name,
        "socket": "Platform"
    }

def _location_sensor(name="LocationSensor"):
    return {
        "sensor_type": "LocationSensor",
        "sensor_name": name,
        "socket": "Platform"
    }

def _velocity_sensor(name="VelocitySensor"):
    return {
        "sensor_type": "VelocitySensor",
        "sensor_name": name,
        "socket": "Platform"
    }

# ---------------------------------------------------------------------------
# Full scenario dict — passed directly to holoocean.make(scenario_cfg=...)
# Keeping simulator open across episodes (no re-make) is handled in env.py
# ---------------------------------------------------------------------------
SCENARIO = {
    "name":         "PierHarbor-MultiASV",
    "world":        "PierHarbor",
    "package_name": "Ocean",
    "ticks_per_sec": 20,
    "frames_per_sec": False,          # run as fast as possible (headless)
    "agents": [
        {
            "agent_name":    "asv0",
            "agent_type":    "SurfaceVessel",
            "control_scheme": 0,      # thruster control [Left, Right]
            "location":      ASV0_SPAWN,
            "rotation":      [0.0, 0.0, 0.0],
            "sensors": [
                _imu_sensor("IMU0"),
                _gps_sensor("GPS0"),
                _location_sensor("Loc0"),
                _velocity_sensor("Vel0"),
            ]
        },
        {
            "agent_name":    "asv1",
            "agent_type":    "SurfaceVessel",
            "control_scheme": 0,
            "location":      ASV1_SPAWN,
            "rotation":      [0.0, 0.0, 0.0],
            "sensors": [
                _imu_sensor("IMU1"),
                _gps_sensor("GPS1"),
                _location_sensor("Loc1"),
                _velocity_sensor("Vel1"),
            ]
        }
    ]
}