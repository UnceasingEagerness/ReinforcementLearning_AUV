#The most feasible thing to do to get reward is to go near the goal to minimise the distance penalty and keep circling in order to get the velocity bonus


import numpy as np
from collections import deque

class SurfaceVesselReward:
    def __init__(self):
        self.prev_dist = None
        self.action_history = deque(maxlen=20)

    def reset(self):
        """Reset the internal state at the start of each episode."""
        self.prev_dist = None
        self.action_history.clear()

    def get_reward(self, pos, vel, rpy, goal_pos, action_diff,goal_radius):
        """Calculates the reward based on the current state."""
        dist = float(np.linalg.norm(goal_pos[:2] - pos[:2]))
        speed = float(np.linalg.norm(vel[:2]))

        reward = 0.0

        # 1. Linear distance penalty
        reward -= 100 * dist    #Increasing this from 0.1 to 100 made it just go near the goal and cirle and never reach it

        # 2. Progress reward
        if self.prev_dist is not None:
            progress = self.prev_dist - dist
            safe_progress = np.clip(progress, -2.0, 2.0) 
            reward += 50.0 * safe_progress
        self.prev_dist = dist

        # 3. Heading alignment 
        goal_dir_yaw = np.arctan2(
            goal_pos[1] - pos[1],
            goal_pos[0] - pos[0]
        )
        
        vessel_yaw_rad = np.deg2rad(rpy[2]) 
        yaw_error = abs((goal_dir_yaw - vessel_yaw_rad + np.pi) % (2 * np.pi) - np.pi)
        
        forward_alignment = np.cos(yaw_error)

        speed_gate = np.clip(speed / 2.0, 0.0, 1.0)
        align_gate = 1.0 if forward_alignment > 0 else 0.0
        
        reward += 10.0 * forward_alignment * speed_gate * align_gate

        # 4. Existential time penalty
        reward -= 1.0

        # 5. Action smoothness
        reward -= 0.5 * action_diff

        if dist < goal_radius:
            return 1000.0 , True
        
        return reward, False
    
    def get_reward2(self, pos, vel, rpy, goal_pos, action_diff, goal_radius=1.0):
        """Calculates the reward based on the current state."""
        dist = float(np.linalg.norm(goal_pos[:2] - pos[:2]))
        speed = float(np.linalg.norm(vel[:2]))

        # 1. THE KILL SWITCH: Terminate on success immediately
        if dist < goal_radius:
            return 1000.0, True

        reward = 0.0

        # 2. Linear distance penalty (Scaled down from 100 to prevent gradient explosions)
        reward -= 1.0 * dist

        # 3. Progress Calculation
        progress = 0.0
        if self.prev_dist is not None:
            progress = self.prev_dist - dist
            safe_progress = np.clip(progress, -2.0, 2.0) 
            reward += 50.0 * safe_progress
        self.prev_dist = dist

        # 4. Heading alignment (TIED TO PROGRESS)
        goal_dir_yaw = np.arctan2(goal_pos[1] - pos[1], goal_pos[0] - pos[0])
        vessel_yaw_rad = np.deg2rad(rpy[2]) 
        yaw_error = abs((goal_dir_yaw - vessel_yaw_rad + np.pi) % (2 * np.pi) - np.pi)
        
        # Linear normalized alignment: 1.0 is perfect aim, 0.0 is facing exactly away
        forward_alignment = (np.pi - yaw_error) / np.pi
        
        speed_gate = np.clip(speed / 2.0, 0.0, 1.0)
        # Gate activates only if pointing forward (error < 90 degrees)
        align_gate = 1.0 if yaw_error < (np.pi / 2.0) else 0.0
        
        # ONLY reward speed and alignment if the vessel is actually getting closer
        if progress > 0:
            reward += 10.0 * forward_alignment * speed_gate * align_gate

        # 5. THE BRAKING ZONE: Penalize speed when very close to the goal
        if dist < 3.0:
            reward -= 5.0 * speed

        # 6. Existential & Smoothness penalties
        reward -= 1.0
        reward -= 0.5 * action_diff
        
        return reward, False
    
    def get_reward3(self, pos, vel, rpy, goal_pos, action, goal_radius=1.0):
        """Calculates the reward based on the current state."""
        dist = float(np.linalg.norm(goal_pos[:2] - pos[:2]))
        speed = float(np.linalg.norm(vel[:2]))

        # 1. THE KILL SWITCH: Terminate on success immediately
        if dist < goal_radius:
            return 1000.0, True

        reward = 0.0

        # 2. Linear distance penalty
        reward -= 10.0 * dist

        # 3. Progress Calculation
        progress = 0.0
        if self.prev_dist is not None:
            progress = self.prev_dist - dist
            safe_progress = np.clip(progress, -2.0, 2.0) 
            reward += 50.0 * safe_progress
        self.prev_dist = dist

        # 4. Heading alignment
        goal_dir_yaw = np.arctan2(goal_pos[1] - pos[1], goal_pos[0] - pos[0])
        vessel_yaw_rad = np.deg2rad(rpy[2]) 
        yaw_error = abs((goal_dir_yaw - vessel_yaw_rad + np.pi) % (2 * np.pi) - np.pi)
        
        forward_alignment = (np.pi - yaw_error) / np.pi
        speed_gate = np.clip(speed / 2.0, 0.0, 1.0)
        align_gate = 1.0 if yaw_error < (np.pi / 2.0) else 0.0
        
        if progress > 0:
            reward += 10.0 * forward_alignment * speed_gate * align_gate

        # --- NEW: Hard Directional Boundary ---
        # Apply a flat penalty if facing more than 90 degrees away from target
        if yaw_error > (np.pi / 2.0):
            reward -= 10.0

        # 5. THE BRAKING ZONE: Penalize speed when very close to the goal
        if dist < 3.0:
            reward -= 5.0 * speed

        # 6. Existential penalty
        reward -= 1.0

        # --- NEW: Advanced Action Smoothness Penalty ---
        self.action_history.append(action)
        if len(self.action_history) > 1:
            # Calculate standard deviation across the history buffer
            sigma_delta = float(np.mean(np.std(self.action_history, axis=0)))
            # Exponential decay penalty: 0 penalty when perfectly smooth, approaching -1 when erratic
            reward += (np.exp(-3.0 * sigma_delta) - 1.0)
        
        return reward, False
    
    def get_rewardS3(self, pos, vel, rpy, goal_pos, action, goal_radius=1.0):
        """Calculates the reward based on the current state."""
        dist = float(np.linalg.norm(goal_pos[:2] - pos[:2]))
        speed = float(np.linalg.norm(vel[:2]))

        # 1. THE KILL SWITCH: Terminate on success immediately
        if dist < goal_radius:
            return 1000.0, True

        reward = 0.0

        # 2. Linear distance penalty (Scaled back to -1.0 to prevent gradient explosions)
        reward -= 1.0 * dist

        # 3. Progress Calculation
        progress = 0.0
        if self.prev_dist is not None:
            progress = self.prev_dist - dist
            safe_progress = np.clip(progress, -2.0, 2.0) 
            reward += 50.0 * safe_progress
        self.prev_dist = dist

        # 4. Heading alignment
        goal_dir_yaw = np.arctan2(goal_pos[1] - pos[1], goal_pos[0] - pos[0])
        vessel_yaw_rad = np.deg2rad(rpy[2]) 
        yaw_error = abs((goal_dir_yaw - vessel_yaw_rad + np.pi) % (2 * np.pi) - np.pi)
        
        forward_alignment = (np.pi - yaw_error) / np.pi
        speed_gate = np.clip(speed / 2.0, 0.0, 1.0)
        align_gate = 1.0 if yaw_error < (np.pi / 2.0) else 0.0
        
        if progress > 0:
            reward += 10.0 * forward_alignment * speed_gate * align_gate

        # SMOOTHED: Directional Boundary 
        # Scales linearly from 0 penalty at 90 degrees to -10 penalty at 180 degrees
        if yaw_error > (np.pi / 2.0):
            # Calculate how far past 90 degrees we are (0.0 to 1.0 scale)
            error_ratio = (yaw_error - (np.pi / 2.0)) / (np.pi / 2.0)
            reward -= 10.0 * error_ratio

        # --- SMOOTHED: Braking Zone ---
        # Scales linearly from 0 at 3.0m, increasing to full penalty at 0.0m
        if dist < 3.0:
            brake_weight = (3.0 - dist) / 3.0
            reward -= 5.0 * speed * brake_weight

        # 6. Existential penalty
        reward -= 1.0

        # 7. Advanced Action Smoothness Penalty 
        self.action_history.append(action)
        if len(self.action_history) > 1:
            sigma_delta = float(np.mean(np.std(self.action_history, axis=0)))
            reward += (np.exp(-3.0 * sigma_delta) - 1.0)
        
        return reward, False
    

        