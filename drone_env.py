import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from collections import deque


class DroneNavEnv(gym.Env):
    """2d point drone navigating to a goal through a fixed obstacle layout."""

    def __init__(self):
        super().__init__()

        # actions: 0=up, 1=down, 2=left, 3=right
        self.action_space = spaces.Discrete(4)
        # observation: 8 ray distances + (dx, dy) to goal, all normalised to [0, 1]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32)

        # world geometry
        self.width = 10.0
        self.height = 10.0
        self.drone_radius = 0.0
        self.goal_radius = 0.5
        self.step_size = 0.2

        # fixed centre start; goal sampled fresh each episode, at least min_start_goal_dist away
        self.start = np.array([5.0, 5.0], dtype=np.float32)
        self.goal = None
        self.drone_pos = None
        self.start_goal_clearance = 0.5
        self.min_start_goal_dist = 4.0

        self.obstacles = self._generate_obstacles()

        # grid for the obstacle-aware (bfs path) distance reward; obstacles are fixed
        self.grid_res = 0.2
        self.grid_nx = int(self.width / self.grid_res)
        self.grid_ny = int(self.height / self.grid_res)
        self._blocked = self._build_blocked_mask()

        self.initial_heading = 0.0   # radians, 0 = facing north (+y); rotation is a future upgrade

        # 8 body-fixed rays, ray 0 = forward, increasing clockwise (every 45 degrees)
        self.n_rays = 8
        self.max_ray_length = np.hypot(self.width, self.height)
        self.ray_step = 0.05
        self.ray_offsets = np.linspace(0, 2 * np.pi, self.n_rays, endpoint=False)

        # episode / reward settings
        self.max_steps = 400
        self.goal_reward = 10.0
        self.collision_penalty = -1.0
        self.wall_penalty = -0.05
        self.progress_coeff = 1.0
        self.time_penalty = 0.01

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        while True:                                          # resample the goal until it is far enough
            self.goal = self._sample_free_point(self.start_goal_clearance)
            if np.linalg.norm(self.goal - self.start) >= self.min_start_goal_dist:
                break
        self.drone_pos = self.start.copy()
        self.heading = self.initial_heading
        self.step_count = 0
        self._path_field = self._build_path_field()
        self.prev_path_dist = self._path_dist_to_goal()
        return self._get_observation(), {}

    def step(self, action):
        moves = {
            0: np.array([0.0,  1.0]),
            1: np.array([0.0, -1.0]),
            2: np.array([-1.0, 0.0]),
            3: np.array([1.0,  0.0]),
        }
        new_pos = self.drone_pos + moves[action] * self.step_size
        clipped_pos = np.clip(new_pos, [0, 0], [self.width, self.height])
        hit_wall = not np.array_equal(new_pos, clipped_pos)   # move clamped by a border
        self.drone_pos = clipped_pos

        self.step_count += 1
        new_path_dist = self._path_dist_to_goal()
        reward = self.progress_coeff * (self.prev_path_dist - new_path_dist) - self.time_penalty
        self.prev_path_dist = new_path_dist
        if hit_wall:
            reward += self.wall_penalty

        info = {}
        if self._check_collision():
            reward += self.collision_penalty
            terminated = True
            info["reached_goal"] = False
        elif self._dist_to_goal() <= self.goal_radius:
            reward += self.goal_reward
            terminated = True
            info["reached_goal"] = True
        else:
            terminated = False

        truncated = self.step_count >= self.max_steps
        return self._get_observation(), reward, terminated, truncated, info

    def render(self, save_path=None):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(0, self.width); ax.set_ylim(0, self.height); ax.set_aspect('equal')
        for x, y, w, h in self.obstacles:
            ax.add_patch(Rectangle((x, y), w, h, color='gray'))
        ax.add_patch(Circle(self.goal, self.goal_radius, color='green'))
        ax.plot(self.start[0], self.start[1], 'gs')
        ax.add_patch(Circle(self.drone_pos, 0.15, color='blue'))
        for d, dist in zip(self._ray_directions(), self._get_ray_distances()):
            end = self.drone_pos + dist * d
            ax.plot([self.drone_pos[0], end[0]], [self.drone_pos[1], end[1]], color='red', linewidth=0.8)
        plt.savefig(save_path) if save_path is not None else plt.show()
        plt.close(fig)

    def _generate_obstacles(self):
        # hand-designed fixed layout; each row is [x, y, w, h] (bottom-left corner + size)
        obstacles = [
            [3.0, 3.5, 2.5, 1.2],
            [6.5, 1.0, 1.2, 3.0],
            [1.0, 6.0, 1.8, 1.5],
            [4.0, 6.5, 1.5, 2.5],
            [7.0, 6.0, 2.2, 1.3],
            [8.5, 2.5, 1.0, 2.0],
            [1.5, 2.0, 1.3, 1.3],
            [5.8, 5.0, 1.0, 1.0],
            [1.8, 8.0, 1.8, 0.8],
        ]
        return np.array(obstacles, dtype=np.float32)

    def _sample_free_point(self, clearance):
        while True:
            p = self.np_random.uniform(low=[0.0, 0.0], high=[self.width, self.height]).astype(np.float32)
            if not any(self._rect_hits_point(obs, p, clearance) for obs in self.obstacles):
                return p

    def _rect_hits_point(self, rect, point, margin):
        x, y, w, h = rect
        px, py = point
        return (x - margin <= px <= x + w + margin) and (y - margin <= py <= y + h + margin)

    def _get_ray_distances(self):
        # analytic ray/axis-aligned-box intersection (slab method), vectorised over all rays
        dirs = self._ray_directions()
        px, py = self.drone_pos[0], self.drone_pos[1]
        dx, dy = dirs[:, 0], dirs[:, 1]

        # distance to the arena walls (exit time of the ray from the [0,W]x[0,H] box)
        with np.errstate(divide="ignore", invalid="ignore"):
            tx_exit = np.where(dx != 0.0, np.maximum((0.0 - px) / dx, (self.width - px) / dx), np.inf)
            ty_exit = np.where(dy != 0.0, np.maximum((0.0 - py) / dy, (self.height - py) / dy), np.inf)
        t_wall = np.minimum(tx_exit, ty_exit)

        # distance to each obstacle (entry time into its box)
        ox1 = self.obstacles[:, 0]; oy1 = self.obstacles[:, 1]
        ox2 = ox1 + self.obstacles[:, 2]; oy2 = oy1 + self.obstacles[:, 3]
        dxc = dx[:, None]; dyc = dy[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            txa = (ox1 - px) / dxc; txb = (ox2 - px) / dxc
            tya = (oy1 - py) / dyc; tyb = (oy2 - py) / dyc
        t_near = np.maximum(np.minimum(txa, txb), np.minimum(tya, tyb))
        t_far = np.minimum(np.maximum(txa, txb), np.maximum(tya, tyb))
        hit = (t_near <= t_far) & (t_far >= 0.0)
        t_hit = np.where(hit, np.maximum(t_near, 0.0), np.inf)
        t_obstacle = t_hit.min(axis=1)

        dists = np.clip(np.minimum(t_wall, t_obstacle), 0.0, self.max_ray_length)
        return dists.astype(np.float32)

    def _ray_directions(self):
        bearings = self.heading + self.ray_offsets
        return np.stack([np.sin(bearings), np.cos(bearings)], axis=1)

    def _check_collision(self):
        return any(self._rect_hits_point(obs, self.drone_pos, self.drone_radius) for obs in self.obstacles)

    def _dist_to_goal(self):
        return np.linalg.norm(self.drone_pos - self.goal)

    def _build_blocked_mask(self):
        # runs once; cell (i,j) is blocked if its centre lies inside any obstacle
        blocked = np.zeros((self.grid_nx, self.grid_ny), dtype=bool)
        for i in range(self.grid_nx):
            cx = (i + 0.5) * self.grid_res
            for j in range(self.grid_ny):
                cy = (j + 0.5) * self.grid_res
                for obs in self.obstacles:
                    if self._rect_hits_point(obs, (cx, cy), 0.0):
                        blocked[i, j] = True
                        break
        return blocked

    def _build_path_field(self):
        # bfs out from the goal cell over free cells -> navigable distance to goal
        nx, ny = self.grid_nx, self.grid_ny
        dist = np.full((nx, ny), np.inf, dtype=np.float32)
        gi = min(max(int(self.goal[0] / self.grid_res), 0), nx - 1)
        gj = min(max(int(self.goal[1] / self.grid_res), 0), ny - 1)
        if self._blocked[gi, gj]:
            return dist
        dist[gi, gj] = 0.0
        q = deque([(gi, gj)])
        while q:
            i, j = q.popleft()
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < nx and 0 <= nj < ny and not self._blocked[ni, nj] and np.isinf(dist[ni, nj]):
                    dist[ni, nj] = dist[i, j] + self.grid_res
                    q.append((ni, nj))
        return dist

    def _path_dist_to_goal(self):
        i = min(max(int(self.drone_pos[0] / self.grid_res), 0), self.grid_nx - 1)
        j = min(max(int(self.drone_pos[1] / self.grid_res), 0), self.grid_ny - 1)
        d = self._path_field[i, j]
        if not np.isinf(d):
            return float(d)
        # blocked cell: fall back to the nearest free neighbour so entering an obstacle never lowers distance
        best = np.inf
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ni, nj = i + di, j + dj
                if 0 <= ni < self.grid_nx and 0 <= nj < self.grid_ny:
                    nd = self._path_field[ni, nj]
                    if not np.isinf(nd) and nd < best:
                        best = nd
        if np.isinf(best):
            return float(self._dist_to_goal())
        return float(best) + self.grid_res

    def _get_observation(self):
        rays = self._get_ray_distances() / self.max_ray_length
        offset = (self.goal - self.drone_pos) / np.array([self.width, self.height])   # ~[-1, 1]
        offset = (offset + 1.0) / 2.0                                                 # -> [0, 1]
        return np.concatenate([rays, offset]).astype(np.float32)
