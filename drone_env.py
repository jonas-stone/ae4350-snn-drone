import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

class DroneNavEnv(gym.Env):
    """A 2D drone navigating to a fixed goal through random rectangular obstacles."""

    def __init__(self):
        super().__init__()  # run gym.Env's own setup first

        # --- action space: 0 = up, 1 = down, 2 = left, 3 = right ---
        self.action_space = spaces.Discrete(4)

        # --- observation: 8 ray distances + (dx, dy) direction to goal ---
        # all values normalized to [0, 1]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(10,), dtype=np.float32
        )

        # --- world geometry ---
        self.width = 10.0
        self.height = 10.0
        self.start = np.array([0.5, 0.5], dtype=np.float32)
        self.goal = np.array([9.5, 9.5], dtype=np.float32)
        self.drone_radius = 0.2
        self.goal_radius = 0.5   # how close counts as "reached" (used later)

        self.step_size = 0.2

        # --- obstacle generation settings ---
        self.n_obstacles = 8
        self.obs_min_size = 0.5
        self.obs_max_size = 2.0
        self.margin = 0.8

        # will hold the drone position and obstacle array once reset() runs
        self.drone_pos = None
        self.obstacles = None

        # --- heading (forward direction). Constant for now, rotation is a future upgrade ---
        self.initial_heading = 0.0   # radians; 0 = facing North (+y)

        # --- sensor rays (body-fixed, ray 0 = forward, increasing clockwise) ---
        self.n_rays = 8
        self.max_ray_length = np.hypot(self.width, self.height)
        self.ray_step = 0.05
        # angular offset of each ray from "forward", going clockwise: [0, 45, 90, ... 315] deg
        self.ray_offsets = np.linspace(0, 2 * np.pi, self.n_rays, endpoint=False)

        # --- episode / reward settings ---
        self.max_steps = 400          # timeout length
        self.goal_reward = 1.0
        self.collision_penalty = -1.0
        self.progress_coeff = 1.0
        self.time_penalty = 0.01


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.drone_pos = self.start.copy()
        self.heading = self.initial_heading
        self.obstacles = self._generate_obstacles()
        self.step_count = 0
        self.prev_dist = self._dist_to_goal()
        observation = self._get_observation()
        info = {}
        
        return observation, info

    def step(self, action):
        
        moves = {
            0: np.array([0.0,  1.0]),   # up
            1: np.array([0.0, -1.0]),   # down
            2: np.array([-1.0, 0.0]),   # left
            3: np.array([1.0,  0.0]),   # right
        }
        new_pos = self.drone_pos + moves[action] * self.step_size
        self.drone_pos = np.clip(new_pos, [0, 0], [self.width, self.height])

        observation = self._get_observation()

        self.step_count += 1
        new_dist = self._dist_to_goal()

        # dense shaping: progress toward goal, minus a small time cost
        reward = self.progress_coeff * (self.prev_dist - new_dist) - self.time_penalty
        self.prev_dist = new_dist

        collision = self._check_collision()

        if collision:
            reward += self.collision_penalty
            terminated = True
        elif self._dist_to_goal() <= self.goal_radius:
            reward += self.goal_reward
            terminated = True
        else:
            terminated = False


        if self.step_count >= self.max_steps:
            truncated = True   # episode cut off (timeout)
        else:
            truncated = False

        info = {}
        return observation, reward, terminated, truncated, info

    def render(self, save_path=None):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')

        for obs in self.obstacles:
            x, y, w, h = obs
            ax.add_patch(Rectangle((x, y), w, h, color='gray'))

        ax.add_patch(Circle(self.goal, self.goal_radius, color='green'))
        ax.plot(self.start[0], self.start[1], 'gs')
        ax.add_patch(Circle(self.drone_pos, self.drone_radius, color='blue'))

        # plot beams
        dirs = self._ray_directions()
        dists = self._get_ray_distances()
        for dir, dist in zip(dirs, dists):
            x_start, y_start = self.drone_pos
            x_end, y_end = self.drone_pos + dist * dir
            ax.plot([x_start, x_end], [y_start, y_end], color='red', linewidth=0.8)
        
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()
        #plt.show()
        plt.close(fig)








    def _generate_obstacles(self):
        obstacles = []
        while len(obstacles) < self.n_obstacles:
            valid = False
            while valid == False:
                w = self.np_random.uniform(self.obs_min_size, self.obs_max_size)
                h = self.np_random.uniform(self.obs_min_size, self.obs_max_size)
                x = self.np_random.uniform(0.0, self.width  - w)
                y = self.np_random.uniform(0.0, self.height - h)

                rect = [x, y, w, h]
                if self._rect_hits_point(rect, self.start, self.margin) or \
                self._rect_hits_point(rect, self.goal, self.margin):
                    valid = False
                else:
                    valid = True
            obstacles.append([x, y, w, h])
        return np.array(obstacles, dtype=np.float32)

    def _rect_hits_point(self, rect, point, margin):
        x, y, w, h = rect
        px, py = point
        inside_x = x - margin <= px <= x + w + margin
        inside_y = y - margin <= py <= y + h + margin
        return inside_x and inside_y

    def _cast_ray(self, direction):
        dist = 0.0
        while dist < self.max_ray_length:
            point = self.drone_pos + dist * direction
            if not (0 <= point[0] <= self.width and 0 <= point[1] <= self.height):
                return dist
            for obs in self.obstacles:
                if self._rect_hits_point(obs, point, 0.0):
                    return dist
            dist += self.ray_step

        return self.max_ray_length

    def _get_ray_distances(self):
        dirs = self._ray_directions()
        return np.array([self._cast_ray(d) for d in dirs], dtype=np.float32)

    def _ray_directions(self):
        bearings = self.heading + self.ray_offsets        # clockwise-from-North angles
        return np.stack([np.sin(bearings), np.cos(bearings)], axis=1)  # world directions

    def _check_collision(self):
        for obs in self.obstacles:
            if self._rect_hits_point(obs, self.drone_pos, self.drone_radius):
                return True
        return False

    def _dist_to_goal(self):
        return np.linalg.norm(self.drone_pos - self.goal)

    def _get_observation(self):
        rays = self._get_ray_distances() / self.max_ray_length

        offset = self.goal - self.drone_pos
        offset = offset / np.array([self.width, self.height])    # -> roughly [-1, 1]
        offset = (offset + 1.0) / 2.0                            # -> [0, 1]

        obs = np.concatenate([rays, offset]).astype(np.float32)  # (10,)
        return obs