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

        # --- obstacle generation settings ---
        self.n_obstacles = 8
        self.obs_min_size = 0.5
        self.obs_max_size = 2.0
        self.margin = 0.8

        # will hold the drone position and obstacle array once reset() runs
        self.drone_pos = None
        self.obstacles = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.drone_pos = self.start.copy()
        self.obstacles = self._generate_obstacles()
        observation = np.zeros(10, dtype=np.float32)  # placeholder until sensors exist
        info = {}
        return observation, info

    def step(self, action):
        # TODO: apply action, move drone, compute sensors, reward, done-flags
        observation = np.zeros(10, dtype=np.float32)  # placeholder
        reward = 0.0
        terminated = False   # episode ended naturally (goal reached / crashed)
        truncated = False    # episode cut off (e.g. timeout)
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

        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()
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