SCENARIO = {
    "name": "TorpedoNav",
    "world": "SimpleUnderwater",
    "main_agent": "auv0",
    "agents": [
        {
            "agent_name": "auv0",
            "agent_type": "TorpedoAUV",
            "sensors": [
                {
                    # Returns shape (18,) in v2.3.0, layout:
                    # [accel(3), vel(3), pos(3), ang_accel(3), ang_vel(3), rpy(3)]
                    # We use: pos=[6:9], vel=[3:6], rpy=[15:18]
                    "sensor_type": "DynamicsSensor",
                    "sensor_name": "DynamicsSensor"
                },
                {
                    # Returns (7,):
                    # [vel_x, vel_y, vel_z, range_x_fwd, range_y_fwd,
                    #  range_x_back, range_y_back]
                    # Doppler Velocity Log — realistic velocity + seafloor ranges
                    "sensor_type": "DVLSensor",
                    "sensor_name": "DVLSensor",
                    "configuration": {
                        "ReturnRange": True
                    }
                },
                {
                    # Returns (1,): depth in metres (pressure-based)
                    "sensor_type": "DepthSensor",
                    "sensor_name": "DepthSensor"
                },
                {
                    # Returns shape (NumBeams,) — distance to nearest
                    # surface per beam, in metres.
                    # MaxRange is returned when no surface is hit.
                    # 12 beams at 30-degree spacing = full 360-degree coverage
                    "sensor_type": "SinglebeamSonar",
                    "sensor_name": "SinglebeamSonar",
                    "configuration": {
                        "MaxRange": 10.0,
                        "MinRange": 0.5,
                        "NumBeams": 12,
                        "Azimuth":  360
                    }
                }
            ],
            # control_scheme 0 = AUV Fins:
            # act() accepts [RightFin, TopFin, LeftFin, BottomFin, Thruster]
            # Fins: -45 to 45 deg, Thruster: -100 to 100 (% of max thrust)
            "control_scheme": 0,
            # Initial spawn location [x, y, z] in metres
            # Negative z = underwater in HoloOcean convention
            "location": [0, 0, -5],
            "rotation": [0, 0, 0]
        }
    ],
    # How many Unreal ticks per holoocean tick() call.
    "ticks_per_sec": 200,
    "frames_per_sec": False     # headless — no rendering
}