import numpy as np
import gymnasium as gym
from gymnasium import spaces
import holoocean
from SV_scenario import SCENARIO
import copy
from sac_agent import Agent
from reward import SurfaceVesselReward
from collections import deque


class SurfaceVesselEnv(gym.Env):
    
    metadata = {"render_modes": [], "render_fps": 10}

    
    def __init__(self, max_steps=10000,show_viewport=False):
        super().__init__()

        self.action_history = deque(maxlen=20)

        self.max_steps  = max_steps
        self.step_count = 0
        self.action_diff = 0.0
        self.reward_calculator = SurfaceVesselReward()
        # goal is on the surface — z=0
        # randomise goal, ensure it's not too close to spawn
        '''
        while True:
            gx = float(self.np_random.uniform(-10.0, 10.0))
            gy = float(self.np_random.uniform(-10.0, 10.0))
            if np.linalg.norm([gx - sx , gy - sy]) > 3.0:  # min 3m apart
                break
        '''
        #self.goal_pos = np.array([10.0, 0.0, 0.0], dtype=np.float32)  #This is the default goal
        #self.goal_pos            = np.array([10.0, 0.0, 0.0], dtype=np.float32)
        self.goal_radius         = 1.0   # metres — counts as reached
        self.collision_threshold = 0.5   # metres — sonar below this = collision
        self._prev_dist = None

        # observation space (25,):
        # 3 computed + 2 vel_xy + 1 yaw + 3 dvl_vel + 4 dvl_ranges + 12 sonar
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(6,), dtype=np.float32
        )

        # action space (2,): [left_thruster, right_thruster] in [-1, 1]
        self.action_space = spaces.Box(
            low  = np.array([-1.0, -1.0], dtype=np.float32),
            high = np.array([ 1.0,  1.0], dtype=np.float32),
            dtype = np.float32
        )

        self.sim = holoocean.make(scenario_cfg=SCENARIO,show_viewport=show_viewport)
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

        self.sim = holoocean.make(scenario_cfg=scenario,show_viewport=True)

    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self._prev_action = np.zeros(self.action_space.shape[0], dtype=np.float32)

        '''
        #Randomising the spawn may make the episode 5-10sec slower as it restarts the unreal process every episode
        # randomise XY start, z=0 always (surface)
        sx = float(self.np_random.uniform(-5.0, 5.0))
        sy = float(self.np_random.uniform(-5.0, 5.0))
        '''
        # randomise goal, min 3m from spawn
        self.action_history.clear()
        
        while True:
            gx = float(self.np_random.uniform(-10.0, 10.0))
            gy = float(self.np_random.uniform(-10.0, 10.0))
            if np.linalg.norm([gx - 0, gy - 0]) > 3.0:
                break
        self.goal_pos = np.array([gx, gy, 0.0], dtype=np.float32)
        
        #self.goal_pos = np.array([10.0,0.0, 0.0], dtype=np.float32)

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
        left_thrust  = float(action[0]) * 1000.0   # ±100 N
        right_thrust = float(action[1]) * 1000.0   # ±100 N
        #thruster_diff = abs(left_thrust - right_thrust)

        # act() for control_scheme 0: [LeftThruster, RightThruster]
        self.sim.act("vessel0", [left_thrust, right_thrust])
        #sensors = self.sim.tick()
        frame_skip = 20 
        for _ in range(frame_skip):
            sensors = self.sim.tick()
        self._last_sensors = sensors

        obs                = self._get_obs(sensors)
        reward, terminated  = self._get_reward(sensors)

        
        # distance truncation
        dyn = sensors["DynamicsSensor"]
        pos = dyn[6:9]
        dist = float(np.linalg.norm(self.goal_pos[:2] - pos[:2]))
        #max_allowed_dist = float(np.linalg.norm(self.goal_pos[:2])) * 2.5
        max_allowed_dist = 25
        
        truncated = (self.step_count >= self.max_steps) or (dist > max_allowed_dist)
        

        return obs, reward, terminated, truncated, self._get_info(sensors)

    
    def close(self):
        if hasattr(self, "sim") and self.sim is not None:
            del self.sim   # correct way to close in HoloOcean v2.3.0
            self.sim = None

    
    def _get_obs(self, sensors):
        #Dynamics sensor : (linear acceleration, velocity, and position, angular acceleration, velocity, and orientation
        dyn   = sensors["DynamicsSensor"]                        # (18,)
        dvl   = sensors["DVLSensor"]                             # (7,)
        sonar_raw= sensors["SinglebeamSonar"].astype(np.float32) 
        #print(sonar_raw.shape)   # (12,) but is showing (200,) -> [intensity_bin1, intensity_bin2, ... intensity_bin200]
        # compress sonar image into nearest obstacle per beam
        '''
        if sonar_raw.ndim > 1:
            sonar = np.min(sonar_raw, axis=1)
        else:
            sonar = sonar_raw
        '''
        imu_val = sensors["IMUSensor"]   #Lets use Angular velocity(Gyroscope) and Linear Accelaration(Accelerometer)

        pos   = dyn[6:9].astype(np.float32)    # x, y, z — z always ~0
        vel   = dyn[3:6].astype(np.float32)    # vx, vy, vz
        vel_x = dyn[3]
        vel_y = dyn[4]
        rpy   = dyn[15:18].astype(np.float32)  # roll, pitch, yaw (deg)
        yaw_rate = dyn[14]

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
        
        # DVL
        dvl_vel    = dvl[0:3].astype(np.float32)   # velocity from DVL
        dvl_ranges = dvl[3:7].astype(np.float32)   # seafloor beam ranges

        dist /= 20.0
        angle_yaw /= np.pi
        speed /= 5.0
        #sonar /= 10.0
        dvl_ranges /= 10.0

        # shape: 3 + 2 + 1 + 3 + 4 + 12 = 25
        '''
        return np.concatenate([
            [dist, float(angle_yaw), speed],    # [0:3]   navigation
            vel_xy,                             # [3:5]   XY velocity
            yaw,                                # [5:6]   heading
            dvl_vel,                            # [6:9]   DVL velocity
            dvl_ranges,                         # [9:13]  seafloor ranges
            sonar                               # [13:25] obstacle distances
        ], dtype=np.float32)
        '''
        return np.array([
            dist,
            float(angle_yaw),
            speed,
            vel_xy[0],
            vel_xy[1],
            yaw_rate
        ], dtype=np.float32)

    
    def _get_reward(self, sensors):
        dyn  = sensors["DynamicsSensor"]
        pos  = dyn[6:9]
        vel  = dyn[3:6]
        rpy  = dyn[15:18]

        reward, terminated = self.reward_calculator.get_rewardS3(
            pos=pos, 
            vel=vel, 
            rpy=rpy, 
            goal_pos=self.goal_pos, 
            #action_diff=self.action_diff,
            action=self._prev_action,
            goal_radius = self.goal_radius
        )
        
        return reward, terminated

        '''
        dist  = float(np.linalg.norm(self.goal_pos[:2] - pos[:2]))
        speed = float(np.linalg.norm(vel[:2]))

        reward = 0.0
        # 1. Linear distance penalty (Scaled so it doesn't overpower)
        # Keeps a constant gradient across the entire map
        reward -= 0.1 * dist

        # 2. Progress reward (Clipped instead of Tanh)
        if self._prev_dist is not None:
            progress = self._prev_dist - dist
            # Clip progress to prevent physics glitch spikes, but keep it linear inside the normal operating range
            safe_progress = np.clip(progress, -2.0, 2.0) 
            reward += 50.0 * safe_progress
        self._prev_dist = dist

        # 3. Heading alignment 
        goal_dir_yaw = np.arctan2(
            self.goal_pos[1] - pos[1],
            self.goal_pos[0] - pos[0]
        )
        
        # Assuming rpy[2] is in degrees based on your code
        vessel_yaw_rad = np.deg2rad(rpy[2]) 
        yaw_error = abs((goal_dir_yaw - vessel_yaw_rad + np.pi) % (2 * np.pi) - np.pi)
        
        # forward_alignment is 1.0 at perfect aim, -1.0 at opposite aim
        forward_alignment = np.cos(yaw_error)

        # Linear speed gate: Ramp up linearly from 0 to 1 as speed goes from 0 to 2 m/s
        # This rewards alignment even at low speeds, but prevents spinning in place
        speed_gate = np.clip(speed / 2.0, 0.0, 1.0)
        
        # Only reward alignment if facing generally forward (> 0)
        align_gate = 1.0 if forward_alignment > 0 else 0.0
        
        reward += 10.0 * forward_alignment * speed_gate * align_gate

        # 4. Existential time penalty
        reward -= 1.0

        # 5. Action smoothness
        reward -= 0.5 * self.action_diff

        return reward, False


        

        # 1. Smooth distance penalty using sigmoid
        # Instead of raw dist penalty, maps distance to smooth [-1, 0] range
        # sigmoid centered at 5m — far = -1, close = 0
        dist_penalty = -(1.0 / (1.0 + np.exp(-0.3 * (dist - 5.0))))
        reward += 0.5 * dist_penalty

        # 2. Progress reward — kept but scaled down
        if self._prev_dist is not None:
            progress = self._prev_dist - dist
            # Smooth progress using tanh — caps large progress values
            # prevents reward spikes that destabilise Q-values
            reward += 100.0 * np.tanh(progress * 2.0)
        self._prev_dist = dist

        # 3. Heading alignment with logistic smoothing
        goal_dir_yaw = np.arctan2(
            self.goal_pos[1] - pos[1],
            self.goal_pos[0] - pos[0]
        )
        yaw_error = abs(
            (goal_dir_yaw - np.deg2rad(rpy[2]) + np.pi) % (2 * np.pi) - np.pi
        )
        forward_alignment = np.cos(yaw_error)

        # Logistic gate — smoothly activates when alignment > 0 and speed > 0.5
        # Instead of hard if-statement, smooth transition
        speed_gate = 1.0 / (1.0 + np.exp(-5.0 * (speed - 3)))
        align_gate = 1.0 / (1.0 + np.exp(-5.0 * forward_alignment))
        reward += 10.0 * speed * forward_alignment * speed_gate * align_gate

        # 4. Existential time penalty
        reward -= 1

        # 5. Action smoothness
        reward -= 0.5 * self.action_diff

        '''
        # 6. Distance truncation — goal_dist * 2.5
        max_allowed_dist = float(np.linalg.norm(self.goal_pos[:2])) * 2.5
        if dist > max_allowed_dist:
            return -50.0, True
        '''

        # 7. Goal reached
        if dist < self.goal_radius:
            return 1000.0, True

        return reward, False

        '''

    '''
    def _get_reward(self, sensors):
    
        dyn  = sensors["DynamicsSensor"]
        pos  = dyn[6:9]
        vel  = dyn[3:6]
        rpy  = dyn[15:18]

        dist  = float(np.linalg.norm(self.goal_pos[:2] - pos[:2]))
        speed = float(np.linalg.norm(vel[:2]))
        
        reward = 0.0
        reward -= 0.01 * dist
        # 1. Progress Reward (Keep this, it provides a dense gradient)
        if self._prev_dist is not None:
            progress = self._prev_dist - dist
            reward += 10.0 * progress
        self._prev_dist = dist

        # 2. Heading Alignment (Tied to Velocity!)
        goal_dir_yaw = np.arctan2(
            self.goal_pos[1] - pos[1],
            self.goal_pos[0] - pos[0]
        )
        yaw_error = abs(
            (goal_dir_yaw - np.deg2rad(rpy[2]) + np.pi) % (2 * np.pi) - np.pi
        )
        forward_alignment = np.cos(yaw_error)

        # ONLY reward looking at the goal if we are actually moving towards it.
        # This completely breaks the "sit still and farm" exploit.
        if forward_alignment > 0.0 and speed > 0.5:
            reward += 1.0 * speed * forward_alignment

        # 3. Existential Time Penalty
        # Increased from -0.002 to -0.05 to make loitering highly unprofitable.
        reward -= 0.05

        # 4. Action Smoothness Penalty
        reward -= 0.005 * self.action_diff

        # 5. Out of Bounds Termination (CRITICAL)
        # If the agent drifts aimlessly or gets pushed away, kill the episode.
        # This prevents the replay buffer from filling up with useless data.
        if dist > 15.0:
            return -50.0, True

        # 6. Massive Success Bonus
        # Needs to be significantly higher than any potential progress reward.
        if dist < self.goal_radius:
            return 100.0, True

        return reward, False
        
        
        
        reward = 0.0

        # progress reward
        if self._prev_dist is not None:
            progress = self._prev_dist - dist
            reward += 10.0 * progress

        self._prev_dist = dist
        goal_dir_yaw = np.arctan2(
            self.goal_pos[1] - pos[1],
            self.goal_pos[0] - pos[0]
        )
        yaw_error = abs(
            (goal_dir_yaw - np.deg2rad(rpy[2]) + np.pi) % (2 * np.pi) - np.pi
        )

        # heading alignment
        reward += 0.05 * np.cos(yaw_error)

        # small time penalty
        reward -= 0.002

        # smooth actions
        reward -= 0.005 * self.action_diff

        # success
        if dist < self.goal_radius:
            return 10.0, True

        return reward, False
        '''

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