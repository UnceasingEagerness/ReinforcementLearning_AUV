import copy

NUM_AGENTS = 2

BASE_AGENT = {
    "agent_type": "SurfaceVessel",

    "sensors": [

        {
            "sensor_type": "DynamicsSensor",
            "sensor_name": "DynamicsSensor"
        },

        {
            "sensor_type": "DVLSensor",
            "sensor_name": "DVLSensor",
            "configuration": {
                "ReturnRange": True
            }
        },

        {
            "sensor_type": "SinglebeamSonar",
            "sensor_name": "SinglebeamSonar",
            "configuration": {
                "MaxRange": 10.0,
                "MinRange": 0.5,
                "NumBeams": 12,
                "Azimuth": 360,
                "OpeningAngle": 30
            }
        },

        {
            "sensor_type": "IMUSensor",
            "sensor_name": "IMUSensor",
            "socket": "IMUSocket",
            "Hz": 200,
            "configuration": {
                "AccelSigma": 0.00277,
                "AngVelSigma": 0.00123
            }
        }
    ],

    # differential thrust
    "control_scheme": 0
}


spawn_locations = [
    [0.0, 0.0, 0.0],
    [10.0, 0.0, 0.0],
    [-10.0, 0.0, 0.0],
    [0.0, 10.0, 0.0]
]


agents = []

for i in range(NUM_AGENTS):

    agent = copy.deepcopy(BASE_AGENT)

    agent["agent_name"] = f"vessel{i}"

    agent["location"] = spawn_locations[i]

    agent["rotation"] = [0.0, 0.0, 0.0]

    agents.append(agent)


SCENARIO = {

    "name": "MultiASVScenario",

    "package_name": "Ocean",

    "world": "SimpleUnderwater",

    "main_agent": "vessel0",

    "show_viewport": True,

    "agents": agents,

    "ticks_per_sec": 200,

    "frames_per_sec": False
}