from surfacevessel_env import SurfaceVesselEnv
from sac_agent import Agent
import numpy as np

env = SurfaceVesselEnv()

agent = Agent(
    input_dims=env.observation_space.shape,
    env=env,
    n_actions=env.action_space.shape[0]
)

n_episodes = 500

for episode in range(n_episodes):

    observation, info = env.reset()

    done = False
    score = 0

    while not done:

        # 1. Agent chooses action
        action = agent.choose_action(observation)

        # Tensor -> numpy
        action = action.numpy()

        # 2. Environment executes action
        observation_, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        # 3. Store transition
        agent.remember(
            observation,
            action,
            reward,
            observation_,
            done
        )

        # 4. Learn
        agent.learn()

        observation = observation_

        score += reward

    print(
        f"Episode {episode} Score {score}"
        f"Dist: {info['dist_to_goal']:.2f} | "
        
        )