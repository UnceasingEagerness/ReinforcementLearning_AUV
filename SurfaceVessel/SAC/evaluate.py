import torch
import numpy as np
import time
from surfacevessel_env import SurfaceVesselEnv
from cleanrl_sac import Actor

# ── CONFIG ────────────────────────────────────────────────────────
GOAL = [10.0, 5.0, 0.0]        # change this
MODEL_PATH = "runs/3/actor.pth"
MAX_STEPS = 5000
SHOW_VIEWPORT = True
# ─────────────────────────────────────────────────────────────────

env = SurfaceVesselEnv(max_steps=MAX_STEPS, show_viewport=SHOW_VIEWPORT)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EnvWrapper:
    def __init__(self, env):
        self.single_observation_space = env.observation_space
        self.single_action_space = env.action_space

actor = Actor(EnvWrapper(env)).to(device)
actor.load_state_dict(torch.load(MODEL_PATH, map_location=device))
actor.eval()

obs, _ = env.reset()
env.goal_pos = np.array(GOAL, dtype=np.float32)
env._prev_dist = None

done = False
total_reward = 0
step = 0

print(f"Goal: {GOAL}")
print(f"Starting dist: {np.linalg.norm(np.array(GOAL[:2])):.2f}m")
print("-" * 50)

while not done:
    with torch.no_grad():
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        _, _, mean_action = actor.get_action(obs_tensor)
        action = mean_action.cpu().numpy()[0]

    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated
    step += 1
    time.sleep(0.01)
    print(f"step={step} | dist={info['dist_to_goal']:.2f} | reward={reward:.2f}")

print("-" * 50)
success = info['dist_to_goal'] < 1.0
print(f"Result: {'SUCCESS' if success else 'FAILED'}")
print(f"Total reward: {total_reward:.2f}")
print(f"Final dist: {info['dist_to_goal']:.2f}")
print(f"Steps taken: {step}")

env.close()