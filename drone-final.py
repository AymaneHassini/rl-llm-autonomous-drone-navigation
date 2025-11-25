import time
import os
import numpy as np
import pybullet as p
import pybullet_data
import json
import math
import google.generativeai as genai
from stable_baselines3 import PPO
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "ppo_drone.zip")

# System cts
TARGET_ALTITUDE = 1.0        
RL_RAY_LENGTH = 3.5          
SIM_TIMESTEP = 1.0 / 240.0   
INTERP_MAX_STEP = 0.5        
GOAL_THRESHOLD = 0.4         
CRUISE_SPEED = 3.0           
MAX_DEVIATION_METERS = 1.0   # Trigger check if we drift by > 1.0m

if not API_KEY:
    print("Warning: No API key found in .env (GEMINI_API_KEY). LLM planning disabled.")
    model = None
else:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel("gemini-flash-latest") 

class Drone:
    def __init__(self, start_pos=(0, 0, TARGET_ALTITUDE)):
        self.urdf_path = "simple_drone.urdf"
        try:
            self.id = p.loadURDF("simple_drone.urdf", start_pos, globalScaling=1.0, useFixedBase=False)
        except:
            self.id = p.loadURDF("cube.urdf", start_pos, globalScaling=0.5, useFixedBase=False)
        print(f"Drone initialized at {start_pos} with ID {self.id}")

    def get_state(self):
        pos, orn = p.getBasePositionAndOrientation(self.id)
        vel, ang_vel = p.getBaseVelocity(self.id)
        return np.array(pos), np.array(orn), np.array(vel), np.array(ang_vel)

    def get_camera_image(self, width=224, height=224):
        pos, orn, _, _ = self.get_state()
        rot_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        forward_vec = rot_matrix @ np.array([1, 0, 0])
        up_vec      = rot_matrix @ np.array([0, 0, 1])
        view_matrix = p.computeViewMatrix(pos + np.array([0.2, 0, 0.1]), pos + 2.0 * forward_vec, up_vec)
        proj_matrix = p.computeProjectionMatrixFOV(90, 1.0, 0.1, 20.0)
        _, _, rgb, depth, _ = p.getCameraImage(width, height, view_matrix, proj_matrix)
        rgb = np.reshape(rgb, (height, width, 4))[:, :, :3]
        return rgb, depth

    def apply_velocity_controller(self, target_velocity, current_pos):
        error_z = TARGET_ALTITUDE - current_pos[2]
        climb_rate = 1.0 * error_z 
        vx, vy, _ = target_velocity
        
        # Deadband Nudge
        speed = np.linalg.norm([vx, vy])
        if 0.01 < speed < 0.5:
            scale = 0.5 / speed
            vx *= scale
            vy *= scale
        
        # Visualize final velocity in Yellow Line
        p.addUserDebugLine(current_pos, current_pos + np.array([vx, vy, 0]), [1, 1, 0], 3, 0.1)
        p.resetBaseVelocity(self.id, linearVelocity=[float(vx), float(vy), float(climb_rate)])

class World:
    def __init__(self):
        self.client = p.connect(p.GUI, options="--width=960 --height=720 --title=NeuroSymbolic_Drone_Final")
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")

        self.obstacles_ids = [
            p.loadURDF("cube.urdf", [2, 2, 0.5], globalScaling=1.0),
            p.loadURDF("cube.urdf", [-2, 3, 0.5], globalScaling=1.0),
            p.loadURDF("cube.urdf", [1, -3, 0.5], globalScaling=1.0),
            p.loadURDF("cube.urdf", [2.5, 0, 0.5], globalScaling=1.0) 
        ]
        for obs in self.obstacles_ids:
            p.changeVisualShape(obs, -1, rgbaColor=[1, 0, 0, 1])

    def get_obstacle_positions(self):
        return [np.array(p.getBasePositionAndOrientation(i)[0]) for i in self.obstacles_ids]

def real_llm_planner(start, goal, obstacles):
    if model is None: return [goal]
    print(f"\nGemini Planning... Start:{start[:2].round(1)} Goal:{goal[:2].round(1)}")
    
    obs_str = ", ".join([f"[{o[0]:.1f}, {o[1]:.1f}]" for o in obstacles])
    prompt = f"""
    I am a drone at {list(start[:2].round(2))}. Goal is {list(goal[:2].round(2))}.
    Obstacles at: [{obs_str}].
    Plan a path avoiding obstacles by at least 1.5 meters.
    Return JSON with key "waypoints" as list of [x, y].
    Include start and goal.
    RETURN ONLY JSON.
    """
    try:
        response = model.generate_content(prompt)
        text = response.text.replace("```json", "").replace("```", "")
        data = json.loads(text)
        wps_3d = [np.array([p[0], p[1], TARGET_ALTITUDE]) for p in data["waypoints"]]
        print(f"Gemini generated {len(wps_3d)} waypoints.")
        return interpolate_waypoints(wps_3d)
    except Exception as e:
        print(f"Gemini Error: {e}")
        return [goal]

def interpolate_waypoints(waypoints, max_step=INTERP_MAX_STEP):
    if not waypoints or len(waypoints) < 2: return waypoints
    new_wps = [waypoints[0]]
    for i in range(1, len(waypoints)):
        p0, p1 = waypoints[i-1], waypoints[i]
        dist = np.linalg.norm(p1[:2] - p0[:2])
        n_steps = int(dist / max_step)
        for k in range(1, n_steps + 1):
            new_wps.append(p0 + (p1 - p0) * (k / (n_steps + 1)))
        new_wps.append(p1)
    return new_wps

def is_segment_clear(p_start, p_end, obstacles, clearance=1.2, samples=10):
    """
    Checks if the line from p_start to p_end is clear of obstacles.
    """
    p_start = np.array(p_start[:2])
    p_end   = np.array(p_end[:2])

    for k in range(samples + 1):
        alpha = k / float(samples)
        point = (1 - alpha) * p_start + alpha * p_end
        
        for o in obstacles:
            o_xy = np.array(o[:2])
            if np.linalg.norm(point - o_xy) < clearance:
                return False 
    return True

class RLReflexAgent:
    def __init__(self, drone_id):
        self.drone_id = drone_id
        self.ray_len = RL_RAY_LENGTH
        try:
            self.model = PPO.load(MODEL_PATH)
            print("RL Model Loaded.")
        except:
            self.model = None
            print("ERROR: RL Model not found.")

    def get_hybrid_velocity(self, target_pos, base_velocity):
        if self.model is None: return base_velocity, False

        pos, orn = p.getBasePositionAndOrientation(self.drone_id)
        vel, _ = p.getBaseVelocity(self.drone_id)
        pos = np.array(pos)
        
        rel_goal = target_pos[:2] - pos[:2]
        current_vel = np.array(vel[:2])
        
        # Raycasts
        yaw = p.getEulerFromQuaternion(orn)[2]
        angles_deg = [-45, -20, 0, 20, 45]
        ray_starts = []
        ray_ends = []
        
        for deg in angles_deg:
            global_angle = yaw + math.radians(deg)
            dx = math.cos(global_angle) * self.ray_len
            dy = math.sin(global_angle) * self.ray_len
            ray_starts.append(pos)
            ray_ends.append(pos + np.array([dx, dy, 0]))
            p.addUserDebugLine(pos, pos + np.array([dx, dy, 0]), [1, 0, 0], 1, 0.1)

        results = p.rayTestBatch(ray_starts, ray_ends)
        
        lidar_fractions = []
        if results is None:
            lidar_fractions = [1.0] * 5
        else:
            for res in results:
                if res[0] == self.drone_id: lidar_fractions.append(1.0)
                else: lidar_fractions.append(res[2])
        
        # RL Prediction
        obs = np.concatenate([rel_goal, current_vel, lidar_fractions]).astype(np.float32)
        action, _ = self.model.predict(obs, deterministic=True)
        rl_vel = np.array([action[0] * 3.0, action[1] * 3.0, 0.0])
        
        # Blender
        min_fraction = np.min(lidar_fractions)
        min_dist_meters = min_fraction * self.ray_len
        
        SAFE_DIST_M = 1.5     
        CRITICAL_DIST_M = 0.5 
        
        if min_dist_meters > SAFE_DIST_M:
            return base_velocity, False # Safe
        elif min_dist_meters < CRITICAL_DIST_M:
            print(f"RL Reflex Shield active! Dist: {min_dist_meters:.2f}m")
            p.addUserDebugLine(pos, pos + rl_vel, [1, 0, 0], 3, 0.1)
            return rl_vel, True # Active
        else:
            # Blend
            alpha = (SAFE_DIST_M - min_dist_meters) / (SAFE_DIST_M - CRITICAL_DIST_M)
            p.addUserDebugLine(pos, pos + base_velocity, [0, 1, 0], 1, 0.1)
            p.addUserDebugLine(pos, pos + rl_vel, [1, 0, 0], 1, 0.1)
            final_vel = (1 - alpha) * base_velocity + alpha * rl_vel
            return final_vel, True # Active (Blending)

class WaypointFollower:
    def __init__(self, waypoints, rl_agent):
        self.waypoints = waypoints
        self.target_index = 0
        self.rl_agent = rl_agent
        self.segment_start = waypoints[0]

    def is_finished(self):
        return False 

    def get_command(self, current_pos):
        target_wp = self.waypoints[self.target_index]
        dist = np.linalg.norm(target_wp[:2] - current_pos[:2])

        direction = (target_wp[:2] - current_pos[:2])
        norm = np.linalg.norm(direction)
        if norm > 0: direction /= norm
        base_vel = np.array([direction[0]*CRUISE_SPEED, direction[1]*CRUISE_SPEED, 0.0])

        if dist < 0.5:
            if self.target_index < len(self.waypoints) - 1:
                print(f"Reached Waypoint {self.target_index}")
                self.segment_start = target_wp
                self.target_index += 1
        
        return self.rl_agent.get_hybrid_velocity(target_wp, base_vel)

    def get_cross_track_error(self, current_pos):
        if self.target_index >= len(self.waypoints):
            return 0.0

        p1 = np.array(self.segment_start[:2])
        p2 = np.array(self.waypoints[self.target_index][:2])
        p3 = np.array(current_pos[:2])

        line_vec = p2 - p1
        line_len = np.linalg.norm(line_vec)
        if line_len < 1e-3:
            return np.linalg.norm(p3 - p1)

        v1 = line_vec
        v2 = p1 - p3
        cross_mag = abs(v1[0] * v2[1] - v1[1] * v2[0])

        return cross_mag / line_len

if __name__ == "__main__":
    world = World()
    drone = Drone()
    
    goal_pos = np.array([4.0, 4.0, TARGET_ALTITUDE])
    goal_vis = p.loadURDF("sphere2.urdf", goal_pos, globalScaling=0.5, useFixedBase=True)
    p.changeVisualShape(goal_vis, -1, rgbaColor=[0, 1, 0, 0.8])
    p.setCollisionFilterGroupMask(goal_vis, -1, 0, 0)
    
    rl_agent = RLReflexAgent(drone.id)
    waypoint_follower = None
    
    surprise_spawned = False
    ambush_id = None

    try:
        while True:
            pos, _, vel, _ = drone.get_state()
            rgb, depth = drone.get_camera_image()
            
            # 1. Check Success
            dist_to_goal = np.linalg.norm(pos[:2] - goal_pos[:2])
            if dist_to_goal < GOAL_THRESHOLD:
                print("\n Mission Comlete! Goal Reached.")
                break

            # 2. Planning Logic
            if waypoint_follower is None:
                obs_pos = world.get_obstacle_positions()
                if ambush_id is not None:
                    ambush_pos_data, _ = p.getBasePositionAndOrientation(ambush_id)
                    obs_pos.append(np.array(ambush_pos_data))
                    print("Including Surprise Obstacle in Re-plan!")

                new_waypoints = real_llm_planner(pos, goal_pos, obs_pos)
                
                if new_waypoints:
                    waypoint_follower = WaypointFollower(new_waypoints, rl_agent)
                    p.removeAllUserDebugItems()
                    for i in range(len(new_waypoints)-1):
                        p.addUserDebugLine(new_waypoints[i], new_waypoints[i+1], [0,0,1], 2, 0)

            # 3. SURPRISE OBSTACLE
            if not surprise_spawned and dist_to_goal < 2.5:
                print("\n Surprise! Spawning dynamic obstacle!")
                speed = np.linalg.norm(vel[:2])
                if speed > 0.1:
                    dir_motion = vel[:2] / speed
                    dir_3d = np.array([dir_motion[0], dir_motion[1], 0])
                else:
                    dir_3d = (goal_pos - pos)
                    dir_3d[2] = 0
                    dir_3d /= np.linalg.norm(dir_3d)

                ambush_pos = pos + dir_3d * 1.2
                ambush_pos[2] = 0.75 

                ambush_id = p.loadURDF("cube.urdf", ambush_pos, globalScaling=1.5)
                p.changeVisualShape(ambush_id, -1, rgbaColor=[1, 0, 0, 1])
                surprise_spawned = True

            # 4. Execution
            if waypoint_follower:
                velocity_cmd, is_shield_active = waypoint_follower.get_command(pos)
            else:
                velocity_cmd = np.zeros(3)

            drone.apply_velocity_controller(velocity_cmd, pos)
            p.stepSimulation()
            time.sleep(SIM_TIMESTEP)

            # 5. Geometric WatchDog 
            if waypoint_follower:
                deviation = waypoint_follower.get_cross_track_error(pos)
                
                # If we are forced too far off the path
                if deviation > MAX_DEVIATION_METERS:
                    # Check if the path to the NEXT waypoint is actually blocked
                    cur_idx = waypoint_follower.target_index
                    if cur_idx < len(waypoint_follower.waypoints):
                        next_wp = waypoint_follower.waypoints[cur_idx]
                        
                        # Build obstacle list
                        current_obs = world.get_obstacle_positions()
                        if ambush_id is not None:
                            ambush_pos_data, _ = p.getBasePositionAndOrientation(ambush_id)
                            current_obs.append(np.array(ambush_pos_data))
                        
                        # Check validity
                        if not is_segment_clear(pos, next_wp, current_obs, clearance=1.2):
                            print(f"Off the course! ({deviation:.2f}m) & Blocked: Replan")
                            waypoint_follower = None # Trigger Replan
                        else:
                            # We are off course, but the path is clear, so let RL fly it.
                            pass
    except KeyboardInterrupt:
        print("Simulation Stopped.")
    finally:
        if p.isConnected():
            p.disconnect()