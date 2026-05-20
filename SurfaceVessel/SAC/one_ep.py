# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

import pandas as pd                        # ← ADDED for CSV logging
import matplotlib.pyplot as plt            # ← ADDED for plots
import matplotlib.gridspec as gridspec     # ← ADDED for plot layout

from buffers import ReplayBuffer


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    checkpoint: str = ""               # ← ADDED: path to actor .pth to load, e.g. runs/.../actor_best.pth
    """path to a saved actor checkpoint to load before running"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True  #This ensures the DL model gives same reproducible results when initialised with same random seed.
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "SurfaceVessel-v0"
    """the environment id of the task"""
    total_timesteps: int = 10000        # ← CHANGED: 1 episode max (matches max_steps in make_env)
    """total timesteps of the experiments"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = int(1e5)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005          #Used for soft update of the Value target network - To avoid the problem of moving target
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 0            # ← CHANGED: no random phase, use actor from step 1
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-4 #1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""


from surfacevessel_env import SurfaceVesselEnv

def make_env(seed, idx):
    def thunk():
        env = SurfaceVesselEnv(max_steps=10000)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed + idx)
        return env
    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()  #This initialises the base framework
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape),
            256,
        ) #nn.Linear(inFeatures, outFeatures) where we pass in the state-action pair
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1) #Glues the state and the action on the feature axis=1, other axis is the batch dimension
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()     #Used whenever we derive our class from the base class
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)  #The state tensor is the input
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))   #np.prod multiplies all the elements in the tuple for ex:(3,)= 3
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x = F.silu(self.fc1(x))  #We have changed to SILU from RELU
        x = F.silu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)   #This is used to squeeze the value from [-1,1]
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))  This is made to make it differentiable, this is extracting the number x_t
        y_t = torch.tanh(x_t)  #Making sure x_t in [-1,1]
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)  #What was the probability of spitting out x_t from the constructed normal distribution
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)  #Converts the set of probabilities of various action into a single one by sum as they are in log
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


# ── ADDED: reward breakdown (mirrors _get_reward in reward.py) ────
def _reward_breakdown(env_unwrapped, sensors, action_diff):
    dyn   = sensors["DynamicsSensor"]
    pos   = dyn[6:9]
    vel   = dyn[3:6]
    rpy   = dyn[15:18]
    goal  = env_unwrapped.goal_pos

    dist  = float(np.linalg.norm(goal[:2] - pos[:2]))
    speed = float(np.linalg.norm(vel[:2]))

    # 1. Linear distance penalty
    r_dist = -1.0 * dist

    # 2. Progress reward
    prev_dist  = env_unwrapped._prev_dist
    r_progress = 0.0
    if prev_dist is not None:
        progress = prev_dist - dist
        safe_progress = np.clip(progress, -2.0, 2.0)
        r_progress = 50.0 * safe_progress

    # 3. Heading alignment
    goal_dir   = np.arctan2(goal[1] - pos[1], goal[0] - pos[0])
    yaw_err    = abs((goal_dir - np.deg2rad(rpy[2]) + np.pi) % (2 * np.pi) - np.pi)
    fwd        = np.cos(yaw_err)
    
    speed_gate = np.clip(speed / 2.0, 0.0, 1.0)
    align_gate = 1.0 if fwd > 0 else 0.0
    r_align    = 10.0 * fwd * speed_gate * align_gate

    # 4. Existential time penalty
    r_time     = -1.0

    # 5. Action smoothness
    r_action   = -0.5 * action_diff

    return {
        "dist":          dist,
        "speed":         speed,
        "yaw_error_deg": float(np.rad2deg(yaw_err)),
        "r_dist":        float(r_dist),
        "r_progress":    float(r_progress),
        "r_align":       float(r_align),
        "r_time":        float(r_time),
        "r_action":      float(r_action),
        "spawn_x":       0.0,
        "spawn_y":       0.0,
        "goal_x":        float(env_unwrapped.goal_pos[0]),
        "goal_y":        float(env_unwrapped.goal_pos[1]),
    }


def _save_csv(records, path):
    df = pd.DataFrame(records)
    cols = ["step", "spawn_x", "spawn_y", "goal_x", "goal_y",
            "dist", "speed", "yaw_error_deg",
            "left_action", "right_action",
            "r_dist", "r_progress", "r_align", "r_time", "r_action",
            "r_total", "cumulative", "goal_reached"]
    df[cols].to_csv(path, index=False, float_format="%.5f")
    print(f"[analyze] CSV saved → {path}")
    return df


def _plot_results(df, path):
    steps  = df["step"].values
    colors = {"r_progress":"#1D9E75","r_align":"#378ADD",
               "r_dist":"#E24B4A","r_time":"#888780","r_action":"#BA7517"}
    labels = {"r_progress":"Progress (tanh)","r_align":"Align×speed",
               "r_dist":"Dist penalty","r_time":"Time penalty","r_action":"Action penalty"}
    keys   = ["r_progress","r_align","r_dist","r_time","r_action"]

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("SAC Episode — Reward Analysis", fontsize=14, fontweight="bold")
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # 1. step + cumulative
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(steps, df["r_total"],    color="#2c2c2c", lw=1.2, alpha=0.6, label="Step reward")
    ax1.plot(steps, df["cumulative"], color="#534AB7", lw=2,               label="Cumulative")
    ax1.axhline(0, color="#ccc", lw=0.8, ls="--")
    ax1.set_title("Step & cumulative reward"); ax1.set_xlabel("Step"); ax1.set_ylabel("Reward")
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.25)

    # 2. stacked area
    ax2 = fig.add_subplot(gs[1, :])
    pv = np.zeros(len(df)); nv = np.zeros(len(df))
    for k in keys:
        v = df[k].values; p = np.clip(v,0,None); n = np.clip(v,None,0)
        ax2.fill_between(steps, pv, pv+p, color=colors[k], alpha=0.75, label=labels[k])
        ax2.fill_between(steps, nv, nv+n, color=colors[k], alpha=0.75)
        pv += p; nv += n
    ax2.axhline(0, color="#999", lw=0.8, ls="--")
    ax2.set_title("Component breakdown (stacked)"); ax2.set_xlabel("Step"); ax2.set_ylabel("Reward")
    ax2.legend(fontsize=8, ncol=2); ax2.grid(True, alpha=0.25)

    # 3. state variables
    ax3 = fig.add_subplot(gs[2, 0]); ax3b = ax3.twinx()
    ax3.plot(steps,  df["dist"],          color="#E24B4A", lw=1.5, label="Dist (m)")
    ax3.plot(steps,  df["speed"],         color="#1D9E75", lw=1.5, label="Speed (m/s)")
    ax3b.plot(steps, df["yaw_error_deg"], color="#BA7517", lw=1.2, ls="--", label="Yaw err (°)")
    ax3.set_title("State variables"); ax3.set_xlabel("Step")
    ax3.set_ylabel("Dist / Speed"); ax3b.set_ylabel("Yaw error (°)", color="#BA7517")
    l1,lb1 = ax3.get_legend_handles_labels(); l2,lb2 = ax3b.get_legend_handles_labels()
    ax3.legend(l1+l2, lb1+lb2, fontsize=8); ax3.grid(True, alpha=0.25)

    # 4. individual components
    ax4 = fig.add_subplot(gs[2, 1])
    for k in keys:
        ax4.plot(steps, df[k], color=colors[k], lw=1.3, label=labels[k], alpha=0.85)
    ax4.axhline(0, color="#ccc", lw=0.8, ls="--")
    ax4.set_title("Individual components"); ax4.set_xlabel("Step"); ax4.set_ylabel("Reward")
    ax4.legend(fontsize=8); ax4.grid(True, alpha=0.25)

    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[analyze] Plot saved → {path}")
    plt.show()
# ── END ADDED ─────────────────────────────────────────────────────────────────


if __name__ == "__main__":

    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"PyTorch is currently using: {device}")

    # env setup
    '''
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    '''
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.seed, 0)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # ── ADDED: load checkpoint if provided ───────────────────────────────────
    if args.checkpoint:
        actor.load_state_dict(torch.load(args.checkpoint, map_location=device))
        actor.eval()
        print(f"[analyze] Loaded checkpoint: {args.checkpoint}")
    # ── END ADDED ─────────────────────────────────────────────────────────────

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )
    start_time = time.time()
    ep_count = 0
    best_return = -np.inf

    # ── ADDED: per-step logging ───────────────────────────────────────────────
    step_records  = []
    cumulative_r  = 0.0
    _prev_action  = np.zeros(envs.single_action_space.shape, dtype=np.float32)

    # Live CSV — open file now, write header, flush every step
    import csv
    os.makedirs(f"runs/{run_name}", exist_ok=True)
    _live_csv_path = f"runs/{run_name}/reward_log.csv"
    _live_csv_cols = ["step", "spawn_x", "spawn_y", "goal_x", "goal_y",
                      "dist", "speed", "yaw_error_deg",
                      "left_action", "right_action",
                      "r_dist", "r_progress", "r_align", "r_time", "r_action",
                      "r_total", "cumulative", "goal_reached"]
    _live_csv_file   = open(_live_csv_path, "w", newline="")
    _live_csv_writer = csv.DictWriter(_live_csv_file, fieldnames=_live_csv_cols, extrasaction="ignore")
    _live_csv_writer.writeheader()
    _live_csv_file.flush()
    print(f"[analyze] Live CSV → {_live_csv_path}")
    # ── END ADDED ─────────────────────────────────────────────────────────────

    

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)

    # ── ADDED: print start and goal ───────────────────────────────────────────
    _inner_env = envs.envs[0].unwrapped
    print(f"[episode] Spawn : {_inner_env._get_info(_inner_env._last_sensors)['Spawn_Location']}")
    print(f"[episode] Goal  : {_inner_env.goal_pos.tolist()}")
    # ── END ADDED ─────────────────────────────────────────────────────────────
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:     #If the learning is in its early phases then it samples random actions.
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # ── ADDED: capture per-step reward breakdown ──────────────────────────
        _action_diff = float(np.linalg.norm(actions[0] - _prev_action))
        _prev_action = actions[0].copy()
        _inner_env   = envs.envs[0].unwrapped          # raw SurfaceVesselEnv
        _bd          = _reward_breakdown(_inner_env, _inner_env._last_sensors, _action_diff)
        cumulative_r += float(rewards[0])
        _bd["step"]         = global_step
        _bd["left_action"]  = float(actions[0][0])
        _bd["right_action"] = float(actions[0][1])
        _bd["r_total"]      = float(rewards[0])        # true reward from env
        _bd["cumulative"]   = cumulative_r
        _bd["goal_reached"] = bool(terminations[0])
        step_records.append(_bd)
        # Live write + flush so CSV updates in real time
        _live_csv_writer.writerow({c: _bd.get(c, "") for c in _live_csv_cols})
        _live_csv_file.flush()
        # ── END ADDED ─────────────────────────────────────────────────────────

        #ep_count = 0

        '''
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    ep_count += 1
                    ep_return = info['episode']['r'][0]
                    ep_len = info['episode']['l'][0]
                    dist = info.get('dist_to_goal', 0.0)[0]
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    break
        '''
        if "episode" in infos:
            ep_count += 1
            ep_return = infos['episode']['r'][0]
            ep_len = infos['episode']['l'][0]
            dist = infos.get('dist_to_goal', [0.0])[0]
            
            if ep_return > best_return:
                best_return = ep_return
                torch.save(actor.state_dict(), f"runs/{run_name}/actor_best.pth")
            
            if ep_count % 100 == 0:
                torch.save(actor.state_dict(), f"runs/{run_name}/actor_ep{ep_count}.pth")
                print(f"Checkpoint saved at EP {ep_count}")
            
            print(f"EP {ep_count} | step={global_step} | return={ep_return:.2f} | len={ep_len} | dist={dist:.2f} | best={best_return:.2f}")
            writer.add_scalar("charts/episodic_return", ep_return, global_step)
            writer.add_scalar("charts/episodic_length", ep_len, global_step)
            writer.add_scalar("charts/dist_to_goal_final", dist, global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        '''
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        '''
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):   #Here if the episode is truncated then we extract the last obs 
            if trunc:
                final_obs = infos.get("final_observation", infos.get("final_obs", {}))
                if idx in final_obs or isinstance(final_obs, np.ndarray):
                    real_next_obs[idx] = final_obs[idx] if isinstance(final_obs, np.ndarray) else final_obs.get(idx, next_obs[idx])
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)   #We sample from the replay buffer
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

                if global_step % 50000 == 0 and global_step > 0:
                    torch.save(actor.state_dict(), f"runs/{run_name}/actor_{global_step}.pth")

            

    envs.close()
    writer.close()

    # Save the actor
    torch.save(actor.state_dict(), f"runs/{run_name}/actor.pth")
    print(f"Model saved to runs/{run_name}/actor.pth")

    # ── ADDED: close live CSV and generate plots ──────────────────────────────
    _live_csv_file.close()
    print(f"[analyze] CSV finalised → {_live_csv_path}")
    plot_path = f"runs/{run_name}/reward_plots.png"
    df = pd.DataFrame(step_records)
    _plot_results(df, plot_path)
    # ── END ADDED ─────────────────────────────────────────────────────────────