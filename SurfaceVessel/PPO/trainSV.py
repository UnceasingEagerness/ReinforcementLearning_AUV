import numpy as np
from surfacevessel_env import SurfaceVesselEnv
from ppo_agent import Agent




if __name__ == "__main__":

    env = SurfaceVesselEnv(max_steps=500)

    agent = Agent(
        n_actions=env.action_space.shape[0],   # 2 thrusters
        input_dims=env.observation_space.shape,
        gamma=0.99,
        alpha=3e-4,
        gae_lambda=0.95,
        policy_clip=0.2,
        batch_size=64,
        n_epochs=10
    )

    N = 1024        # rollout size before PPO update
    n_games = 1000 # 1000

    best_score = -np.inf

    learn_iters = 0
    total_steps = 0

    score_history = []

    for episode in range(n_games):

        observation, info = env.reset()

        done = False
        truncated = False

        score = 0

        while not (done or truncated):

            # -------------------------------------------------
            # Choose continuous action from PPO policy
            # -------------------------------------------------
            action, prob, val = agent.choose_action(observation)

            # -------------------------------------------------
            # Step environment
            # -------------------------------------------------
            observation_, reward, done, truncated, info = env.step(action)

            total_steps += 1
            score += reward

            # -------------------------------------------------
            # Store rollout transition
            # -------------------------------------------------
            agent.store_transition(
                observation,
                action,
                prob,
                val,
                reward,
                done
            )

            # -------------------------------------------------
            # PPO learning step
            # -------------------------------------------------
            if total_steps % N == 0:
                agent.learn()
                learn_iters += 1

                print(f"\n=== PPO UPDATE {learn_iters} ===")

            observation = observation_

        score_history.append(score)

        avg_score = np.mean(score_history[-100:])

        # -------------------------------------------------
        # Save best models
        # -------------------------------------------------
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print(
            f"Start: ({info['Spawn_Location'][0]:.2f}, {info['Spawn_Location'][1]:.2f}) | "
            f"Goal: ({info['Goal_Location'][0]:.2f}, {info['Goal_Location'][1]:.2f}) | "
            f"Episode {episode} | "
            f"Score: {score:.2f} | "
            f"Avg Score: {avg_score:.2f} | "
            f"Steps: {total_steps} | "
            f"Dist: {info['dist_to_goal']:.2f} | "
            #f"Min Sonar: {info['min_sonar']:.2f}"
            f"Yaw: {info['yaw_deg']:.1f} | "
        )

    env.close()