from typing import Any, SupportsFloat
import numpy as np
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal
from minigrid.minigrid_env import MiniGridEnv
from gymnasium.core import ObsType, ActType
from gymnasium import spaces

# world height
WORLD_HEIGHT = 7

# world width
WORLD_WIDTH = 10

# wind strength for each column
WIND = np.array([0, 0, 0, 1, 1, 1, 2, 2, 1, 0])

# start state
START = (0, 3)

# goal state
GOAL = (7, 3)

# living reward
REWARD = -1

# possible actions
ACTION_UP = 0
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 1


class WindyGridworld(MiniGridEnv):
    metadata = {'render_fps': 4}

    def __init__(self, **kwargs):
        self.agent_start_pos = START
        self.agent_start_dir = 0  # right
        self.goal = GOAL
        self.reward = REWARD
        self.wind = WIND
        mission_space = MissionSpace(mission_func=self._gen_mission)

        super().__init__(
            mission_space=mission_space,
            width=WORLD_WIDTH,
            height=WORLD_HEIGHT,
            # Set this to True for maximum speed
            see_through_walls=True,
            highlight=False,
            **kwargs,
        )

        self.action_space = spaces.Discrete(4)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None,) -> (
            tuple)[ObsType, dict[str, Any]]:
        super().reset()
        return self.observe(), {}

    def wind_move(self) -> int:
        return self.wind[self.agent_pos[0]]

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.step_count += 1
        terminated = False
        truncated = False

        if action == ACTION_UP:
            self.agent_dir = 3
        elif action == ACTION_RIGHT:
            self.agent_dir = 0
        elif action == ACTION_DOWN:
            self.agent_dir = 1
        elif action == ACTION_LEFT:
            self.agent_dir = 2
        else:
            raise 'Invalid Action'

        # Get the position in front of the agent
        i, j = self.front_pos

        if 0 <= i < self.width and 0 <= j < self.height:
            fwd_cell = self.grid.get(i, j)
            if fwd_cell is None or fwd_cell.can_overlap():
                j = max(0, j - self.wind_move())
                self.agent_pos = (i, j)

        if self.agent_pos == self.goal:
            terminated = True

        if self.render_mode == "human":
            self.render()

        return self.observe(), self.reward, terminated, truncated, {}

    def observe(self) -> int:
        return self.agent_pos[0] + self.agent_pos[1] * self.width

    @staticmethod
    def _gen_mission():
        return "get to the green goal square"

    def _gen_grid(self, width=WORLD_WIDTH, height=WORLD_HEIGHT):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        # self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), *self.goal)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = str(WIND)
