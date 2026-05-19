import numpy as np
import gymnasium as gym
from gymnasium import spaces
import holoocean

from multiagent.multi_asv_scenario import SCENARIO, NUM_AGENTS


class MultiASVEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(self, max_steps=500):

        super().__init__()

        self.num_agents = NUM_AGENTS

        self.agent_names = [
            f"vessel{i}" for i in range(self.num_agents)
        ]

        self.max_steps = max_steps

        self.step_count = 0

        self.goal_radius = 1.0

        self.collision_distance = 1.5

        self.max_thrust = 15000.0

        # ---------------------------------------------------
        # GOALS
        # ---------------------------------------------------

        self.goal_positions = []

        # ---------------------------------------------------
        # ACTION SPACE
        # each agent:
        # [left_thruster, right_thruster]
        # ---------------------------------------------------

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )

        # ---------------------------------------------------
        # OBSERVATION SPACE
        #
        # own:
        #   dist_goal
        #   angle_goal
        #   vx
        #   vy
        #   yaw
        #
        # other agents:
        #   rel_x
        #   rel_y
        #   rel_vx
        #   rel_vy
        #
        # for 2 agents:
        # 5 + 4 = 9
        # ---------------------------------------------------

        obs_dim = 5 + (self.num_agents - 1) * 4

        self.single_observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        self.sim = holoocean.make(
            scenario_cfg=SCENARIO,
            show_viewport=True      )

        self.prev_distances = np.zeros(self.num_agents)

    # =========================================================
    # RESET
    # =========================================================

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.step_count = 0

        self.sim.reset()

        self.goal_positions = []

        # ---------------------------------------------------
        # RANDOM GOALS
        # ---------------------------------------------------

        for i in range(self.num_agents):

            while True:

                gx = np.random.uniform(-10, 10)

                gy = np.random.uniform(-10, 10)

                goal = np.array([gx, gy, 0.0], dtype=np.float32)

                if np.linalg.norm(goal[:2]) > 3.0:
                    break

            self.goal_positions.append(goal)

        sensors = self.sim.tick()

        obs_n = []

        for i in range(self.num_agents):

            obs = self._get_obs(sensors, i)

            obs_n.append(obs)

            self.prev_distances[i] = obs[0]

        return obs_n, {}

    # =========================================================
    # STEP
    # =========================================================

    def step(self, actions):

        self.step_count += 1

        # ---------------------------------------------------
        # APPLY ACTIONS
        # ---------------------------------------------------

        for _ in range(10):

            for i, agent_name in enumerate(self.agent_names):

                action = actions[i]

                #left = float(action[0]) * self.max_thrust

                #right = float(action[1]) * self.max_thrust
                left =  right = 1.0

                self.sim.act(agent_name, [left, right])

            sensors = self.sim.tick()

        # ---------------------------------------------------
        # BUILD OUTPUTS
        # ---------------------------------------------------

        obs_n = []

        reward_n = []

        done_n = []

        info_n = []

        for i in range(self.num_agents):

            obs = self._get_obs(sensors, i)

            reward, done = self._get_reward(sensors, i)

            info = self._get_info(sensors, i)

            obs_n.append(obs)

            reward_n.append(reward)

            done_n.append(done)

            info_n.append(info)

        truncated = self.step_count >= self.max_steps

        return (
            obs_n,
            reward_n,
            done_n,
            truncated,
            info_n
        )

    # =========================================================
    # OBSERVATION
    # =========================================================

    def _get_obs(self, sensors, agent_idx):

        agent_name = self.agent_names[agent_idx]

        dyn = sensors[agent_name]["DynamicsSensor"]

        pos = dyn[6:9].astype(np.float32)

        vel = dyn[3:6].astype(np.float32)

        rpy = dyn[15:18].astype(np.float32)

        yaw = np.deg2rad(rpy[2])

        goal = self.goal_positions[agent_idx]

        # ---------------------------------------------------
        # GOAL FEATURES
        # ---------------------------------------------------

        rel_goal = goal[:2] - pos[:2]

        dist_goal = np.linalg.norm(rel_goal)

        goal_angle = np.arctan2(
            rel_goal[1],
            rel_goal[0]
        )

        heading_error = (
            goal_angle - yaw + np.pi
        ) % (2 * np.pi) - np.pi

        heading_error /= np.pi

        dist_goal /= 20.0

        vx = vel[0] / 5.0

        vy = vel[1] / 5.0

        yaw_norm = yaw / np.pi

        obs = [
            dist_goal,
            heading_error,
            vx,
            vy,
            yaw_norm
        ]

        # ---------------------------------------------------
        # OTHER AGENTS
        # ---------------------------------------------------

        for j in range(self.num_agents):

            if j == agent_idx:
                continue

            other_name = self.agent_names[j]

            other_dyn = sensors[other_name]["DynamicsSensor"]

            other_pos = other_dyn[6:9]

            other_vel = other_dyn[3:6]

            rel_pos = other_pos[:2] - pos[:2]

            rel_vel = other_vel[:2] - vel[:2]

            obs.extend([
                rel_pos[0] / 20.0,
                rel_pos[1] / 20.0,
                rel_vel[0] / 5.0,
                rel_vel[1] / 5.0
            ])

        obs = np.array(obs, dtype=np.float32)

        if np.isnan(obs).any():
            print("OBS NAN")
            print(obs)
            raise ValueError("Observation contains NaN")

        if np.isinf(obs).any():
            print("OBS INF")
            print(obs)
            raise ValueError("Observation contains INF")

        return obs

    # =========================================================
    # REWARD
    # =========================================================

    def _get_reward(self, sensors, agent_idx):

        agent_name = self.agent_names[agent_idx]

        dyn = sensors[agent_name]["DynamicsSensor"]

        pos = dyn[6:9]

        goal = self.goal_positions[agent_idx]

        dist = np.linalg.norm(goal[:2] - pos[:2])

        reward = 0.0

        # ---------------------------------------------------
        # PROGRESS REWARD
        # ---------------------------------------------------

        progress = self.prev_distances[agent_idx] - dist

        reward += 10.0 * progress

        self.prev_distances[agent_idx] = dist

        # ---------------------------------------------------
        # STEP PENALTY
        # ---------------------------------------------------

        reward -= 0.01

        # ---------------------------------------------------
        # COLLISION PENALTY
        # ---------------------------------------------------

        for j in range(self.num_agents):

            if j == agent_idx:
                continue

            other_name = self.agent_names[j]

            other_dyn = sensors[other_name]["DynamicsSensor"]

            other_pos = other_dyn[6:9]

            d = np.linalg.norm(pos[:2] - other_pos[:2])

            if d < self.collision_distance:

                reward -= 20.0

                return reward, True

        # ---------------------------------------------------
        # GOAL REACHED
        # ---------------------------------------------------

        if dist < self.goal_radius:

            reward += 100.0

            return reward, True

        return reward, False

    # =========================================================
    # INFO
    # =========================================================

    def _get_info(self, sensors, agent_idx):

        agent_name = self.agent_names[agent_idx]

        dyn = sensors[agent_name]["DynamicsSensor"]

        pos = dyn[6:9]

        vel = dyn[3:6]

        goal = self.goal_positions[agent_idx]

        return {

            "agent": agent_name,

            "distance_to_goal":
                float(np.linalg.norm(goal[:2] - pos[:2])),

            "speed":
                float(np.linalg.norm(vel[:2])),

            "position":
                pos.tolist()
        }

    # =========================================================
    # CLOSE
    # =========================================================

    def close(self):

        if self.sim is not None:

            del self.sim

            self.sim = None