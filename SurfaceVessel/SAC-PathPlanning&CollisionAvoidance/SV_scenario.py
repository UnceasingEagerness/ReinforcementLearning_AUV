# scenario.py
# -----------------------------------------------------------------
# HoloOcean scenario config for SurfaceVessel navigation.
# This tells HoloOcean:
#   - which Unreal world to load
#   - which agent to spawn and where
#   - which sensors to attach
# -----------------------------------------------------------------

SCENARIO = {
    "name": "SurfaceVesselNav",
    "package_name": "Ocean",
    "world": "SimpleUnderwater",  # change to a world with water surface
    "main_agent": "vessel0",
    "show_viewport": True,
    "agents": [
        {
            "agent_name": "vessel0",
            # SurfaceVessel: 2 rear thrusters, moves in XY plane only
            # Cannot dive — stays on water surface at z=0
            "agent_type": "SurfaceVessel",
            "sensors": [
                {
                    # Returns shape (18,) in v2.3.0, layout:
                    # [accel(3), vel(3), pos(3), ang_accel(3), ang_vel(3), rpy(3)]
                    # We use: pos=[6:9], vel=[3:6], rpy=[15:18]
                    # Note: z is always ~0 for surface vessel
                    "sensor_type": "DynamicsSensor",
                    "sensor_name": "DynamicsSensor"
                },
                {
                    # Returns (7,):
                    # [vel_x, vel_y, vel_z, range_x_fwd, range_y_fwd,
                    #  range_x_back, range_y_back]
                    # Doppler Velocity Log — realistic velocity measurement
                    "sensor_type": "DVLSensor",
                    "sensor_name": "DVLSensor",
                    "configuration": {
                        "ReturnRange": True
                    }
                },
                {
                    # Returns shape (NumBeams,) — distance to nearest -NO ITS NOT RETURNING THIS, ITS ALWAYS RETURNING (200,) WHEN I  print(sonar_raw.shape)
                    #So the problem is this is a singleBeam Sonor so only one beam and (200,) is like the echo intensity at 200 different distance slices along that single beam.
                    # surface/obstacle per beam, in metres.
                    # 64 beams at 30-degree spacing = full 360-degree coverage
                    "sensor_type": "SinglebeamSonar",
                    "sensor_name": "SinglebeamSonar",
                    "configuration": {
                        "MaxRange": 70.0, #PaperValue
                        "MinRange": 0.5,
                        "NumBeams": 64,    
                        "Azimuth":  360
                        #"OpeningAngle": 30
                        #"RangeMin": 0.5,
                        #"RangeMax": 30,
                        #"RangeBins": 200,
                        #"AddSigma": 0,
                        #"MultSigma": 0,
                        #"RangeSigma": 0.1,
                        #"ShowWarning": true,
                        #"InitOctreeRange": 40,
                        #"ViewRegion": false,
                        #"ViewOctree": -10,
                        #"WaterDensity": 997,
                        #"WaterSpeedSound": 1480,
                        #"UseApprox": true
                    }
                },
                
                {
                        "sensor_type": "IMUSensor",
                        "sensor_name": "IMUSensor",
                        "socket": "IMUSocket",
                        "Hz": 200,
                        "configuration": {
                            "AccelSigma": 0.00277,
                            "AngVelSigma": 0.00123,
                            #"AccelBiasSigma": 0.00141,
                            #"AngVelBiasSigma": 0.00388,
                            #"ReturnBias": true
                        }
                },
                
                {
                    "sensor_type": "RaycastLidar",
                    "sensor_name": "Lidar",
                    # "socket": "CameraSocket", # You may not need this socket definition for an ASV unless you have a specific mount point
                    "configuration": {
                        "Channels": 1,                   # 1 vertical layer (makes it a 2D LiDAR instead of 3D)
                        "Range": 70.0,                   # Matches the paper's 70m maximum range
                        "HorizontalFov": 360.0,          # Full 360-degree sweep
                        "RotationFrequency": 10,         # 10 full sweeps per second
                        "PointsPerSecond": 640,          # 10 sweeps/sec * 64 beams/sweep = 640 total points
                        "UpperFovLimit": 0,              # 0 degrees (keeps the beam perfectly flat/horizontal)
                        "LowerFovLimit": 0,              # 0 degrees 
                        "ShowDebugPoints": False,        # Set to True if you want to visually see the lasers in the viewport!
                        "NoiseStdDev": 0.0,              # Keep at 0.0 for clean training data initially
                        
                        # Optional realism parameters (you can leave these as 0 for faster RL training)
                        "AtmospAttenRate": 0.0,
                        "DropOffGenRate": 0.0,
                        "DropOffIntensityLimit": 0.0,
                        "DropOffAtZeroIntensity": 0.0
                    },
                    "Hz": 10  # Matches the RotationFrequency so you get one full 64-beam sweep per RL step
                }
            ],
            # control_scheme 0 = Thrusters:
            # act() accepts [LeftThruster, RightThruster]
            # Differential thrust controls both speed and turning.
            # Both positive  = forward
            # Both negative  = backward
            # Opposite signs = spin/turn on the spot
            "control_scheme": 0,
            # Surface vessel spawns at z=0 (water surface)
            "location": [0, 0, 0],
            "rotation": [0, 0, 0]
        }
    ],
    "ticks_per_sec": 200,
    "frames_per_sec": True     # headless — no rendering
}