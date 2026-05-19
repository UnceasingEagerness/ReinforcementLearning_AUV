import numpy as np
import torch

from multiagent.multi_asv_env import MultiASVEnv
from algorithms.sac.masac import MASAC

from collections import deque
import random


# =========================================================
# SIMPLE REPLAY BUFFER
# =========================================================

class ReplayBuffer:

    def __init__(self, capacity=100000):

        self.buffer = deque(maxlen=capacity)

    def add(
        self,
        obs,
        actions,
        rewards,
        next_obs,
        dones
    ):

        self.buffer.append((
            obs,
            actions,
            rewards,
            next_obs,
            dones
        ))

    def sample(self, batch_size):

        batch = random.sample(self.buffer, batch_size)

        obs_b = []
        actions_b = []
        rewards_b = []
        next_obs_b = []
        dones_b = []

        for sample in batch:

            obs, actions, rewards, next_obs, dones = sample

            obs_b.append(obs)

            actions_b.append(actions)

            rewards_b.append(rewards)

            next_obs_b.append(next_obs)

            dones_b.append(dones)

        return (
            obs_b,
            actions_b,
            rewards_b,
            next_obs_b,
            dones_b
        )

    def __len__(self):

        return len(self.buffer)


# =========================================================
# TRAINING
# =========================================================

NUM_AGENTS = 2

MAX_EPISODES = 10000

MAX_STEPS = 500

BATCH_SIZE = 128

START_STEPS = 5000

UPDATE_EVERY = 50

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


env = MultiASVEnv(max_steps=MAX_STEPS)

# =========================================================
# MASAC
# =========================================================

masac = MASAC(
    num_agents=NUM_AGENTS,
    device=DEVICE,
    rnn=False
)

replay_buffer = ReplayBuffer()

total_steps = 0


# =========================================================
# TRAIN LOOP
# =========================================================

for episode in range(MAX_EPISODES):

    obs_n, _ = env.reset()

    episode_rewards = np.zeros(NUM_AGENTS)

    for step in range(MAX_STEPS):

        total_steps += 1

        # -------------------------------------------------
        # ACTION SELECTION
        # -------------------------------------------------

        actions = []

        if total_steps < START_STEPS:

            # random exploration

            for _ in range(NUM_AGENTS):

                a = env.action_space.sample()

                actions.append(a)

        else:

            # MASAC policy

            for agent_idx in range(NUM_AGENTS):

                obs_tensor = torch.FloatTensor(
                    obs_n[agent_idx]
                ).unsqueeze(0).to(DEVICE)

                # no RNN history initially
                dummy_history = torch.zeros(
                    (1, 1, 1)
                ).to(DEVICE)

                action = masac.masac_agent[
                    agent_idx
                ].act(
                    dummy_history,
                    obs_tensor,
                    noise=0.1
                )

                actions.append(action.squeeze())

        # -------------------------------------------------
        # ENV STEP
        # -------------------------------------------------

        next_obs_n, reward_n, done_n, truncated, info_n = env.step(actions)

        # -------------------------------------------------
        # STORE
        # -------------------------------------------------

        replay_buffer.add(
            obs_n,
            actions,
            reward_n,
            next_obs_n,
            done_n
        )

        obs_n = next_obs_n

        episode_rewards += np.array(reward_n)

        # -------------------------------------------------
        # UPDATE
        # -------------------------------------------------

        if (
            len(replay_buffer) > BATCH_SIZE
            and total_steps % UPDATE_EVERY == 0
        ):

            for agent_idx in range(NUM_AGENTS):

                samples = replay_buffer.sample(BATCH_SIZE)

                obs_b, actions_b, rewards_b, next_obs_b, dones_b = samples

                # -------------------------------------------------
                # CONVERT TO MASAC FORMAT
                # -------------------------------------------------

                obs_agents = [
                    [] for _ in range(NUM_AGENTS)
                ]

                action_agents = [
                    [] for _ in range(NUM_AGENTS)
                ]

                reward_agents = [
                    [] for _ in range(NUM_AGENTS)
                ]

                next_obs_agents = [
                    [] for _ in range(NUM_AGENTS)
                ]

                done_agents = [
                    [] for _ in range(NUM_AGENTS)
                ]

                for b in range(BATCH_SIZE):

                    for a in range(NUM_AGENTS):

                        obs_agents[a].append(
                            obs_b[b][a]
                        )

                        action_agents[a].append(
                            actions_b[b][a]
                        )

                        reward_agents[a].append(
                            rewards_b[b][a]
                        )

                        next_obs_agents[a].append(
                            next_obs_b[b][a]
                        )

                        done_agents[a].append(
                            dones_b[b][a]
                        )

                # -------------------------------------------------
                # DUMMY HISTORY
                # -------------------------------------------------

                masac_samples = (
                    obs_agents,
                    action_agents,
                    reward_agents,
                    next_obs_agents,
                    done_agents
                )

                masac.update(
                    masac_samples,
                    agent_idx,
                    logger=None
                )

            masac.update_targets()

        # -------------------------------------------------
        # TERMINATION
        # -------------------------------------------------

        if all(done_n) or truncated:

            break

    # -----------------------------------------------------
    # LOGGING
    # -----------------------------------------------------

    print(
        f"Episode: {episode} | "
        f"Rewards: {episode_rewards}"
    )

    # -----------------------------------------------------
    # SAVE MODELS
    # -----------------------------------------------------

    if episode % 100 == 0:

        for i, agent in enumerate(masac.masac_agent):

            torch.save(
                agent.actor.state_dict(),
                f"actor_agent_{i}.pth"
            )

            torch.save(
                agent.critic.state_dict(),
                f"critic_agent_{i}.pth"
            )

print("Training Complete")

env.close()