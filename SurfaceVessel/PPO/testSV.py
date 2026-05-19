import numpy as np
import tensorflow as tf

from surfacevessel_env import SurfaceVesselEnv
from ppo_agent import Agent


if __name__ == "__main__":

    env = SurfaceVesselEnv(max_steps=500)

    agent = Agent(
        n_actions=env.action_space.shape[0],
        input_dims=env.observation_space.shape,
        gamma=0.99,
        alpha=3e-4,
        gae_lambda=0.95,
        policy_clip=0.2,
        batch_size=64,
        n_epochs=10
    )

    # -------------------------------------------------
    # Build networks once before loading weights
    # -------------------------------------------------
    dummy_state = tf.random.normal(
        (1, env.observation_space.shape[0])
    )

    agent.actor(dummy_state)
    agent.critic(dummy_state)

    # -------------------------------------------------
    # Load trained weights
    # -------------------------------------------------
    agent.actor.load_weights("models/actor.weights.h5")
    agent.critic.load_weights("models/critic.weights.h5")

    print("Weights loaded successfully!")

    n_test_episodes = 5

    for episode in range(n_test_episodes):

        observation, info = env.reset()

        done = False
        truncated = False

        score = 0

        while not (done or truncated):

            # -------------------------------------------------
            # Deterministic action (NO exploration)
            # -------------------------------------------------
            state = tf.convert_to_tensor([observation], dtype=tf.float32)

            mu = agent.actor(state)

            action = tf.squeeze(mu).numpy()

            observation_, reward, done, truncated, info = env.step(action)

            score += reward

            print(
                f"Action: {action} | "
                f"Dist: {info['dist_to_goal']:.2f}"
            )

            observation = observation_

        print(
            f"\nTEST EPISODE {episode} "
            f"| SCORE: {score:.2f}"
        )

    env.close()