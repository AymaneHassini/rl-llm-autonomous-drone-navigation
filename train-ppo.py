import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
import math
import torch
from stable_baselines3 import PPO

TOTAL_TIMESTEPS = 600_000  
LEARNING_RATE = 3e-4
N_STEPS = 2048
BATCH_SIZE = 64

class DroneTrainEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        # Action: [Vx, Vy]
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        
        # Observation: 9 values
        # [Rel_Goal_X, Rel_Goal_Y, Vel_X, Vel_Y, Ray1, Ray2, Ray3, Ray4, Ray5]
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        
        self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        self.drone = None
        self.obstacle_ids = []
        self.goal_pos = np.zeros(3)
        self.prev_dist = 0
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        
        # 1. Spawn Drone
        self.drone = p.loadURDF("cube.urdf", [0, 0, 1.0], globalScaling=0.5)
        
        # 2. Randomize Goal , we go as low as 0.2m to ensure it handles very close waypoints aggressively.
        angle = np.random.uniform(-math.pi/2, math.pi/2)
        dist = np.random.uniform(0.2, 6.0) 
        self.goal_pos = np.array([math.cos(angle)*dist, math.sin(angle)*dist, 1.0])
        
        # 3. Spawn multiple obstacles (1 to 3)
        self.obstacle_ids = []
        num_obstacles = np.random.randint(1, 4)
        
        for _ in range(num_obstacles):
            # Pick a spot roughly between start and goal
            t = np.random.uniform(0.2, 0.8)
            obs_pos = self.goal_pos * t
            
            # Add noise
            obs_pos[0] += np.random.uniform(-1.0, 1.0)
            obs_pos[1] += np.random.uniform(-1.0, 1.0)
            obs_pos[2] = 0.5
            
            d_drone = np.linalg.norm(obs_pos[:2])
            d_goal = np.linalg.norm(obs_pos[:2] - self.goal_pos[:2])
            if d_drone > 0.8 and d_goal > 0.8:
                scale = np.random.uniform(0.7, 1.3)
                oid = p.loadURDF("cube.urdf", obs_pos, globalScaling=scale)
                self.obstacle_ids.append(oid)
        
        self.prev_dist = np.linalg.norm(self.goal_pos[:2])
        self.step_count = 0
        return self._get_obs(), {}

    def _get_obs(self):
        pos, orn = p.getBasePositionAndOrientation(self.drone)
        vel, _ = p.getBaseVelocity(self.drone)
        pos = np.array(pos)
        
        # 1. Relative Goal
        rel_goal = self.goal_pos[:2] - pos[:2]
        
        # 2. Current Velocity
        current_vel = np.array(vel[:2])
        
        # 3. Raycasts (3.5m length)
        yaw = p.getEulerFromQuaternion(orn)[2]
        angles_deg = [-45, -20, 0, 20, 45]
        ray_len = 3.5 
        ray_starts = []
        ray_ends = []
        for deg in angles_deg:
            global_angle = yaw + math.radians(deg)
            dx = math.cos(global_angle) * ray_len
            dy = math.sin(global_angle) * ray_len
            ray_starts.append(pos)
            ray_ends.append(pos + [dx, dy, 0])

        results = p.rayTestBatch(ray_starts, ray_ends)
        lidar_readings = [res[2] for res in results]            
        obs = np.concatenate([rel_goal, current_vel, lidar_readings])
        return obs.astype(np.float32)

    def step(self, action):
        self.step_count += 1   
        # Apply Action
        vx = action[0] * 3.0
        vy = action[1] * 3.0    
        pos, _ = p.getBasePositionAndOrientation(self.drone)
        p.resetBaseVelocity(self.drone, linearVelocity=[vx, vy, 0])
        p.resetBasePositionAndOrientation(self.drone, [pos[0], pos[1], 1.0], [0,0,0,1])       
        p.stepSimulation()       
        new_pos, _ = p.getBasePositionAndOrientation(self.drone)
        dist_to_goal = np.linalg.norm(np.array(new_pos[:2]) - self.goal_pos[:2])      

        # Rewards 
        reward = 0
        terminated = False
        truncated = False        
        # 1. Progress (Encourage movement at short range)
        reward += (self.prev_dist - dist_to_goal) * 15.0
        self.prev_dist = dist_to_goal       
        # 2. Time Penalty
        reward -= 0.05      
        # 3. Collision
        for oid in self.obstacle_ids:
            contact = p.getContactPoints(self.drone, oid)
            if len(contact) > 0:
                reward -= 100.0
                terminated = True
                break       
        # 4. Fear Penalty (Keep distance)
        obs = self._get_obs()
        rays = obs[-5:] 
        if np.min(rays) < 0.2: # If closer than 20% of ray length (0.7m approx)
            reward -= 0.5
            
        # 5. Success (Threshold 0.4m)
        if dist_to_goal < 0.4:
            reward += 100.0
            terminated = True
            
        if self.step_count > 500:
            truncated = True
            
        return obs, reward, terminated, truncated, {}

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device.upper()}")

    env = DroneTrainEnv()
    model = PPO("MlpPolicy", env, verbose=1, device=device, learning_rate=LEARNING_RATE)
    
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save("ppo_drone")
    print("Model saved")