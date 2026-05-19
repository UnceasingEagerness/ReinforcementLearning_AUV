# torpedo_nav_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import holoocean
from scenario_torpedo import SCENARIO


class TorpedoNavEnv(gym.Env):
    metadata = {"render_modes": [], "render_fps": 10}


    def __init__(self, max_steps=500):
        super().__init__()

        self.max_steps  = max_steps
        self.step_count = 0

        self.goal_pos            = np.array([10.0, 0.0, -5.0], dtype=np.float32)
        self.goal_radius         = 1.0
        self.collision_threshold = 0.5
        self.stall_speed         = 0.5

        # 4 computed + 3 vel + 3 rpy + 1 depth + 3 dvl_vel + 4 dvl_ranges + 12 sonar = 30
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(30,), dtype=np.float32
        )

        self.action_space = spaces.Box(
            low  = np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high = np.array([ 1.0,  1.0,  1.0], dtype=np.float32),
            dtype = np.float32
        )

        self.sim = holoocean.make(SCENARIO)
        self._last_sensors = {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        sx = float(self.np_random.uniform(-3.0, 3.0))
        sy = float(self.np_random.uniform(-3.0, 3.0))
        sz = -5.0

        self.sim.reset()
        self.sim.teleport("auv0", location=[sx, sy, sz])
        sensors = self.sim.tick()
        self._last_sensors = sensors

        return self._get_obs(sensors), self._get_info(sensors)

    def step(self, action):
        self.step_count += 1

        thrust    = (float(action[0]) +1) * 50.0   # +/-100% max thrust
        yaw_fin   = float(action[1]) * 45.0    # +/-45 deg
        pitch_fin = float(action[2]) * 45.0    # +/-45 deg

        # [RightFin, TopFin, LeftFin, BottomFin, Thruster]
        # yaw:   right and left fins opposed
        # pitch: top and bottom fins symmetric
        cmd = [yaw_fin, -pitch_fin, -yaw_fin, pitch_fin, thrust]
        self.sim.act("auv0", cmd)
        sensors = self.sim.tick()
        self._last_sensors = sensors

        obs                = self._get_obs(sensors)
        reward, terminated = self._get_reward(sensors)
        truncated          = self.step_count >= self.max_steps

        return obs, reward, terminated, truncated, self._get_info(sensors)

    def close(self):
        if hasattr(self, "sim") and self.sim is not None:
            self.sim.stop()

    def _get_obs(self, sensors):
        dyn   = sensors["DynamicsSensor"]                        # (18,)
        dvl   = sensors["DVLSensor"]                             # (7,)
        depth = sensors["DepthSensor"]                           # (1,)
        sonar = sensors["SinglebeamSonar"].astype(np.float32)    # (12,)

        pos   = dyn[6:9].astype(np.float32)    # x, y, z metres
        vel   = dyn[3:6].astype(np.float32)    # vx,vy,vz m/s
        rpy   = dyn[15:18].astype(np.float32)  # roll,pitch,yaw degrees
        speed = float(np.linalg.norm(vel))

        dist = float(np.linalg.norm(self.goal_pos - pos))

        # yaw error wrapped to [-pi, pi]
        goal_dir_yaw = np.arctan2(
            self.goal_pos[1] - pos[1],
            self.goal_pos[0] - pos[0]
        )
        angle_yaw = (goal_dir_yaw - np.deg2rad(rpy[2]) + np.pi) % (2*np.pi) - np.pi

        # pitch error wrapped to [-pi, pi]
        horiz_dist     = float(np.linalg.norm(self.goal_pos[:2] - pos[:2]))
        goal_dir_pitch = np.arctan2(self.goal_pos[2] - pos[2], horiz_dist)
        angle_pitch    = (goal_dir_pitch - np.deg2rad(rpy[1]) + np.pi) % (2*np.pi) - np.pi

        dvl_vel    = dvl[0:3].astype(np.float32)   # DVL velocity
        dvl_ranges = dvl[3:7].astype(np.float32)   # seafloor beam ranges
        depth_val  = depth[0:1].astype(np.float32) # pressure depth

        # shape: 4 + 3 + 3 + 1 + 3 + 4 + 12 = 30
        return np.concatenate([
            [dist, float(angle_yaw), float(angle_pitch), speed],  # [0:4]
            vel,        # [4:7]
            rpy,        # [7:10]
            depth_val,  # [10:11]
            dvl_vel,    # [11:14]
            dvl_ranges, # [14:18]
            sonar       # [18:30]
        ], dtype=np.float32)

    def _get_reward(self, sensors):
        dyn  = sensors["DynamicsSensor"]
        pos  = dyn[6:9]
        vel  = dyn[3:6]
        dist = float(np.linalg.norm(self.goal_pos - pos))

        reward = -0.01 * dist

        if dist < self.goal_radius:
            return reward + 10.0, True

        if float(np.min(sensors["SinglebeamSonar"])) < self.collision_threshold:
            return reward - 5.0, True

        if float(np.linalg.norm(vel)) < self.stall_speed:
            reward -= 2.0

        return reward, False

    def _get_info(self, sensors):
        dyn = sensors["DynamicsSensor"]
        pos = dyn[6:9]
        vel = dyn[3:6]
        return {
            "dist_to_goal": float(np.linalg.norm(self.goal_pos - pos)),
            "speed":        float(np.linalg.norm(vel)),
            "min_sonar":    float(np.min(sensors["SinglebeamSonar"])),
            "depth":        float(sensors["DepthSensor"][0]),
            "step_count":   self.step_count
        }