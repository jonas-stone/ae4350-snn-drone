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
        self.drone_radius = 0.0   # point drone: sensing and collision are now consistent
        self.goal_radius = 0.5    # how close counts as "reached"
        self.step_size = 0.2

        # fixed start (bottom-left corner); goal is sampled fresh & distant each episode
        self.start = np.array([0.7, 0.7], dtype=np.float32)
        self.goal = None
        self.drone_pos = None
        self.start_goal_clearance = 0.5   # keep the goal this far from any obstacle
        self.min_start_goal_dist = 8.0    # goal must be at least this far from start

        # --- obstacles: a hand-designed, fixed layout ---
        self.obstacles = self._generate_obstacles()

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
        # obstacles and start are fixed; only the goal changes each episode
        while True:
            self.goal = self._sample_free_point(self.start_goal_clearance)
            if np.linalg.norm(self.goal - self.start) >= self.min_start_goal_dist:
                break
        self.drone_pos = self.start.copy()
        self.heading = self.initial_heading
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

        reward = self.progress_coeff * (self.prev_dist - new_dist) - self.time_penalty
        self.prev_dist = new_dist

        info = {}
        if self._check_collision():
            reward += self.collision_penalty
            terminated = True
            info["reached_goal"] = False
        elif new_dist <= self.goal_radius:
            reward += self.goal_reward
            terminated = True
            info["reached_goal"] = True
        else:
            terminated = False

        truncated = self.step_count >= self.max_steps

        observation = self._get_observation()
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
        ax.add_patch(Circle(self.drone_pos, 0.15, color='blue'))  # 0.15 = display size only

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
        # hand-designed, fixed layout: asymmetric mix of blocks and walls of
        # varied sizes, arranged to leave winding corridors to navigate.
        # each row is [x, y, w, h] (bottom-left corner + width + height)
        obstacles = [
            [3.0, 3.5, 2.5, 1.2],   # wide block, center-left
            [6.5, 1.0, 1.2, 3.0],   # tall block, lower-right
            [1.0, 6.0, 1.8, 1.5],   # block, upper-left
            [4.0, 6.5, 1.5, 2.5],   # tall block, upper-center
            [7.0, 6.0, 2.2, 1.3],   # wide block, upper-right
            [8.5, 2.5, 1.0, 2.0],   # block, right edge
            [1.5, 2.0, 1.3, 1.3],   # small block, lower-left
            [5.8, 5.0, 1.0, 1.0],   # small block, center
            [1.8, 8.0, 1.8, 0.8],   # low wall, upper-left
        ]
        return np.array(obstacles, dtype=np.float32)

    def _sample_free_point(self, clearance):
        # uniformly sample a point that is at least `clearance` away from every obstacle
        while True:
            p = self.np_random.uniform(
                low=[0.0, 0.0], high=[self.width, self.height]
            ).astype(np.float32)
            if not any(self._rect_hits_point(obs, p, clearance) for obs in self.obstacles):
                return p

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
        # Analytic ray/axis-aligned-box intersection (the "slab method"), fully vectorized.
        # For a ray p + t*d, the distance to a box is found by algebra, with no marching.
        dirs = self._ray_directions()               # (R, 2)
        px, py = self.drone_pos[0], self.drone_pos[1]
        dx = dirs[:, 0]                             # (R,)
        dy = dirs[:, 1]                             # (R,)

        # --- distance to the arena walls: the ray starts inside the box [0,W]x[0,H]
        #     and EXITS at the nearer of the two axis exit-times ---
        with np.errstate(divide="ignore", invalid="ignore"):
            # a ray parallel to an axis (dx or dy == 0) never exits via that axis's
            # walls -> exit time is +inf; substitute it instead of computing 0/0
            tx_exit = np.where(dx != 0.0,
                               np.maximum((0.0 - px) / dx, (self.width - px) / dx),
                               np.inf)                                       # (R,)
            ty_exit = np.where(dy != 0.0,
                               np.maximum((0.0 - py) / dy, (self.height - py) / dy),
                               np.inf)                                       # (R,)
        t_wall = np.minimum(tx_exit, ty_exit)       # (R,)

        # --- distance to each obstacle: the ENTRY time into box [x1,x2]x[y1,y2] ---
        ox1 = self.obstacles[:, 0]                  # (M,)
        oy1 = self.obstacles[:, 1]
        ox2 = ox1 + self.obstacles[:, 2]
        oy2 = oy1 + self.obstacles[:, 3]
        dxc = dx[:, None]                           # (R, 1) -> broadcasts with (M,) to (R, M)
        dyc = dy[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            txa = (ox1 - px) / dxc                  # (R, M)
            txb = (ox2 - px) / dxc
            tya = (oy1 - py) / dyc
            tyb = (oy2 - py) / dyc
        t_near = np.maximum(np.minimum(txa, txb), np.minimum(tya, tyb))  # (R, M) enter box
        t_far = np.minimum(np.maximum(txa, txb), np.maximum(tya, tyb))   # (R, M) leave box
        hit = (t_near <= t_far) & (t_far >= 0.0)    # boxes actually struck, ahead of us
        t_hit = np.where(hit, np.maximum(t_near, 0.0), np.inf)           # (R, M)
        t_obstacle = t_hit.min(axis=1)              # (R,) nearest obstacle per ray

        dists = np.minimum(t_wall, t_obstacle)
        dists = np.clip(dists, 0.0, self.max_ray_length)
        return dists.astype(np.float32)
        
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