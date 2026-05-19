import numpy as np
import gymnasium as gym
from gymnasium import spaces
import holoocean
from SV_scenario import SCENARIO
import copy
from ppo_agent import Agent


class SurfaceVesselEnv(gym.Env):
    
    metadata = {"render_modes": [], "render_fps": 10}

    
    def __init__(self, max_steps=500):
        super().__init__()

        self.max_steps  = max_steps
        self.step_count = 0
        self.action_diff = 0.0

        # goal is on the surface — z=0
        # randomise goal, ensure it's not too close to spawn
        '''
        while True:
            gx = float(self.np_random.uniform(-10.0, 10.0))
            gy = float(self.np_random.uniform(-10.0, 10.0))
            if np.linalg.norm([gx - sx , gy - sy]) > 3.0:  # min 3m apart
                break
        '''
        self.goal_pos = np.array([10.0, 0.0, 0.0], dtype=np.float32)  #This is the default goal
        #self.goal_pos            = np.array([10.0, 0.0, 0.0], dtype=np.float32)
        self.goal_radius         = 1.0   # metres — counts as reached
        self.collision_threshold = 0.5   # metres — sonar below this = collision
        self._prev_dist = None

        # observation space (25,):
        # 3 computed + 2 vel_xy + 1 yaw + 3 dvl_vel + 4 dvl_ranges + 12 sonar
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(25,), dtype=np.float32
        )

        # action space (2,): [left_thruster, right_thruster] in [-1, 1]
        self.action_space = spaces.Box(
            low  = np.array([-1.0, -1.0], dtype=np.float32),
            high = np.array([ 1.0,  1.0], dtype=np.float32),
            dtype = np.float32
        )

        self.sim = holoocean.make(scenario_cfg=SCENARIO,show_viewport=False)
        '''
        self._base_scenario = SCENARIO
        self.sim = None
        self._make_sim([0.0, 0.0, 0.0])  # initial spawn
        '''

        self._last_sensors = {}

    def _make_sim(self, spawn_loc):
        # kill old sim if exists
        if self.sim is not None:
            del self.sim
            self.sim = None

        # deep copy scenario and update spawn location
        scenario = copy.deepcopy(self._base_scenario)
        scenario["agents"][0]["location"] = spawn_loc

        self.sim = holoocean.make(scenario_cfg=scenario,show_viewport=False)

    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self._prev_action = np.zeros(self.action_space.shape[0], dtype=np.float32)

        '''
        #Randomising the spawn may make the episode 5-10sec slower as it restarts the unreal process every episode
        # randomise XY start, z=0 always (surface)
        sx = float(self.np_random.uniform(-5.0, 5.0))
        sy = float(self.np_random.uniform(-5.0, 5.0))
        
        # randomise goal, min 3m from spawn
        while True:
            gx = float(self.np_random.uniform(-10.0, 10.0))
            gy = float(self.np_random.uniform(-10.0, 10.0))
            if np.linalg.norm([gx - 0, gy - 0]) > 3.0:
                break
        self.goal_pos = np.array([gx, gy, 0.0], dtype=np.float32)
        '''
        self.goal_pos = np.array([10.0, 0.0, 0.0], dtype=np.float32)

        # rebuild sim with new spawn location
        #self._make_sim([sx, sy, 0.0])
        

        self._prev_dist = None
        self.sim.reset()
        #self.sim.teleport("vessel0", location=[sx, sy, 0.0])
        sensors = self.sim.tick()
        self._last_sensors = sensors

        return self._get_obs(sensors), self._get_info(sensors)

    
    def step(self, action):
        self.step_count += 1

        self.action_diff = np.linalg.norm(action - self._prev_action)
        self._prev_action = action.copy()

        # scale [-1,1] → thruster forces in Newtons
        left_thrust  = float(action[0]) * 5000.0   # ±100 N
        right_thrust = float(action[1]) * 5000.0   # ±100 N
        #thruster_diff = abs(left_thrust - right_thrust)

        # act() for control_scheme 0: [LeftThruster, RightThruster]
        self.sim.act("vessel0", [left_thrust, right_thrust])
        sensors = self.sim.tick()
        self._last_sensors = sensors

        obs                = self._get_obs(sensors)
        reward, terminated  = self._get_reward(sensors)
        truncated          = self.step_count >= self.max_steps

        return obs, reward, terminated, truncated, self._get_info(sensors)

    
    def close(self):
        if hasattr(self, "sim") and self.sim is not None:
            del self.sim   # correct way to close in HoloOcean v2.3.0
            self.sim = None

    
    def _get_obs(self, sensors):
        #Dynamics sensor : (linear acceleration, velocity, and position, angular acceleration, velocity, and orientation
        dyn   = sensors["DynamicsSensor"]                        # (18,)
        dvl   = sensors["DVLSensor"]                             # (7,)
        sonar = sensors["SinglebeamSonar"].astype(np.float32) 
        print(sonar.shape)   # (12,) but its returning (200,) - why? and still the obs_space didnt complain
        imu_val = sensors["IMUSensor"]   #Lets use Angular velocity(Gyroscope) and Linear Accelaration(Accelerometer)

        pos   = dyn[6:9].astype(np.float32)    # x, y, z — z always ~0
        vel   = dyn[3:6].astype(np.float32)    # vx, vy, vz
        rpy   = dyn[15:18].astype(np.float32)  # roll, pitch, yaw (deg)

        vel_xy = vel[0:2]                       # only XY matters for surface
        yaw    = rpy[2:3]                       # only yaw matters for surface
        speed  = float(np.linalg.norm(vel_xy))

        # distance to goal in XY plane only
        dist = float(np.linalg.norm(self.goal_pos[:2] - pos[:2]))

        # yaw error: direction to goal minus current yaw, wrapped [-pi, pi]
        goal_dir_yaw = np.arctan2(
            self.goal_pos[1] - pos[1],
            self.goal_pos[0] - pos[0]
        )
        angle_yaw = (goal_dir_yaw - np.deg2rad(rpy[2]) + np.pi) \
                    % (2 * np.pi) - np.pi
        
        angle_yaw /= np.pi

        # DVL
        dvl_vel    = dvl[0:3].astype(np.float32)   # velocity from DVL
        dvl_ranges = dvl[3:7].astype(np.float32)   # seafloor beam ranges

        dist /= 20.0
        angle_yaw /= np.pi
        speed /= 5.0
        sonar /= 10.0
        dvl_ranges /= 10.0

        # shape: 3 + 2 + 1 + 3 + 4 + 12 = 25
        return np.concatenate([
            [dist, float(angle_yaw), speed],    # [0:3]   navigation
            vel_xy,                             # [3:5]   XY velocity
            yaw,                                # [5:6]   heading
            dvl_vel,                            # [6:9]   DVL velocity
            dvl_ranges,                         # [9:13]  seafloor ranges
            sonar                               # [13:25] obstacle distances
        ], dtype=np.float32)

    def _get_reward(self, sensors):
        dyn  = sensors["DynamicsSensor"]
        pos  = dyn[6:9]
        vel  = dyn[3:6]
        rpy  = dyn[15:18]

        dist  = float(np.linalg.norm(self.goal_pos[:2] - pos[:2]))
        speed = float(np.linalg.norm(vel[:2]))

        reward = 0.0
        reward -= 10.0 * dist      # always penalise being far from goal
        # ── 1. PROGRESS (potential-based) ─────────────────────────────────────
        if self._prev_dist is not None:
            progress = self._prev_dist - dist      # +ve = got closer
            reward  += 10.0 * progress
            if progress < 0:
                reward -= 0.5 * abs(progress)
        self._prev_dist = dist

        # ── 2. HEADING ALIGNMENT ──────────────────────────────────────────────
        goal_dir_yaw = np.arctan2(
            self.goal_pos[1] - pos[1],
            self.goal_pos[0] - pos[0]
        )
        yaw_error = abs(
            (goal_dir_yaw - np.deg2rad(rpy[2]) + np.pi) % (2 * np.pi) - np.pi
        )
        reward += 0.05 * np.cos(yaw_error)
        #reward -= 0.001 * thruster_diff

        '''
        # ── 3. FORWARD SPEED (only when roughly aligned) ──────────────────────
        forward_alignment = np.cos(yaw_error)

        if forward_alignment > 0.7:
            reward += 0.5 * speed * forward_alignment
        '''

        # ── 4. STEP PENALTY ───────────────────────────────────────────────────
        reward -= 0.005

        # ── 5. COLLISION (re-enable once obstacles exist) ─────────────────────
        # if float(np.min(sensors["SinglebeamSonar"])) < self.collision_threshold:
        #     return reward - 10.0, True

        # ── 6. GOAL REACHED ───────────────────────────────────────────────────
        if dist < self.goal_radius:
            time_bonus = max(0.0, 1.0 - self.step_count / self.max_steps)
            return reward + 50.0 + 5.0 * time_bonus, True
        
        #reward -= 0.1 * self.action_diff 


        return reward, False

    def _get_info(self, sensors):
        dyn = sensors["DynamicsSensor"]
        pos = dyn[6:9]
        vel = dyn[3:6]
        return {
            "Goal_Location" :self.goal_pos,
            "Spawn_Location":[0,0,0],
            "dist_to_goal": float(np.linalg.norm(self.goal_pos[:2] - pos[:2])),
            "speed":        float(np.linalg.norm(vel[:2])),
            "min_sonar":    float(np.min(sensors["SinglebeamSonar"])),
            "yaw_deg":      float(dyn[17]),
            "step_count":   self.step_count
        }