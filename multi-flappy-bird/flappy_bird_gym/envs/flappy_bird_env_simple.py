#
# Copyright (c) 2020 Gabriel Nogueira (Talendar)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

""" Implementation of a Flappy Bird OpenAI Gym environment that yields simple
numerical information about the game's state as observations.
"""

from typing import Dict, Tuple, Optional, Union

import gym
import numpy as np
import pygame
from datetime import datetime

from flappy_bird_gym.envs.game_logic import FlappyBirdLogic
from flappy_bird_gym.envs.game_logic import PIPE_WIDTH, PIPE_HEIGHT
from flappy_bird_gym.envs.game_logic import PLAYER_WIDTH, PLAYER_HEIGHT
from flappy_bird_gym.envs.renderer import FlappyBirdRenderer


class FlappyBirdEnvSimple(gym.Env):
    """ Flappy Bird Gym environment that yields simple observations.

    The observations yielded by this environment are simple numerical
    information about the game's state. Specifically, the observations are:

        * Horizontal distance to the next pipe;
        * Difference between the player's y position and the next hole's y
          position.

    The reward received by the agent in each step is equal to the score obtained
    by the agent in that step. A score point is obtained every time the bird
    passes a pipe.

    Args:
        screen_size (Tuple[int, int]): The screen's width and height.
        normalize_obs (bool): If `True`, the observations will be normalized
            before being returned.
        pipe_gap (int): Space between a lower and an upper pipe.
        bird_color (str): Color of the flappy bird. The currently available
            colors are "yellow", "blue" and "red".
        pipe_color (str): Color of the pipes. The currently available colors are
            "green" and "red".
        background (Optional[str]): Type of background image. The currently
            available types are "day" and "night". If `None`, no background will
            be drawn.
    """

    metadata = {'render.modes': ['human']}

    def __init__(self,
                 screen_size: Tuple[int, int] = (288, 512),
                 normalize_obs: bool = True,
                 pipe_gap: int = 100,
                 bird_color: str = "yellow",
                 pipe_color: str = "green",
                 background: Optional[str] = "day") -> None:
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf,
                                                shape=(2,),
                                                dtype=np.float32)
        self._screen_size = screen_size
        self._normalize_obs = normalize_obs
        self._pipe_gap = pipe_gap

        self._game = None
        self._renderer = None

        self._bird_color = bird_color
        self._pipe_color = pipe_color
        self._bg_type = background

    def _get_observation(self):
        """ 
        Returns:
            A list containing for each bird:
                * Horizontal distance to the next pipe;
                * Difference between the player's y position and the next hole's y
                  position.
        """
        to_return = []

        for bird in self._game.birds:
            up_pipe = low_pipe = None
            h_dist = 0
            for up_pipe, low_pipe in zip(self._game.upper_pipes,
                                        self._game.lower_pipes):
                h_dist = (low_pipe["x"] + PIPE_WIDTH / 2
                        - (bird.player_x - PLAYER_WIDTH / 2))
                h_dist += 3  # extra distance to compensate for the buggy hit-box
                if h_dist >= 0:
                    break

            upper_pipe_y = up_pipe["y"] + PIPE_HEIGHT
            lower_pipe_y = low_pipe["y"]
            player_y = bird.player_y

            v_dist = (upper_pipe_y + lower_pipe_y) / 2 - (player_y
                                                        + PLAYER_HEIGHT/2)

            if self._normalize_obs:
                h_dist /= self._screen_size[0]
                v_dist /= self._screen_size[1]

            to_return.append(np.array([
                h_dist,
                v_dist,
            ]))

        return to_return

    def step(self,
             actions: Union[FlappyBirdLogic.Actions, int],
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        """ Given an action, updates the game state.

        Args:
            action (Union[FlappyBirdLogic.Actions, int]): The actions taken by
                the agent. Zero (0) means "do nothing" and one (1) means "flap".

        Returns:
            A tuple containing, respectively:

                * a list of observations for each bird (horizontal distance to the next pipe;
                  difference between the player's y position and the next hole's
                  y position);
                * a np.array of rewards for each bird (0 if dead for more than 1 period, -100 if just died, else 
                  the sum of: -1 if action==FLAP, +10 if scored a point, +a small time reward for staying alive);
                * a np.array of alive statuses for each bird (`True` if a bird is dead or `False` otherwise. If the 
                  whole array is filled with `True`s the game is over)
                * a np.array of scores for each bird
        """
        missing_actions = self._game.nr_of_birds - len(actions)  # positive if the user did not specify an action for all birds
        if missing_actions > 0:
            actions = np.concatenate((actions, np.zeros(missing_actions)))

        start = datetime.now()

        point_factor = 1
        rewards = np.array([-point_multiplier*bird.score for bird in self._game.birds], dtype=float)  #save prev scores first to be able to compute rewards later
        prev_alives = np.array([bird.alive for bird in self._game.birds])

        alives = self._game.update_state(actions) #alive info for all birds

        obs = self._get_observation()
        done = np.array([not alive for alive in alives])
        scores = np.array([bird.score for bird in self._game.birds])

        for i in range(self._game.nr_of_birds):
            if done[i]:
                rewards[i] = -10 if prev_alives[i] else 0
            else:
                rewards[i] += self._game.birds[i].score*point_factor - actions[i] #rewards = (new_points - prev_points)*10 - action
                rewards[i] += (datetime.now() - start).microseconds/1000

        return obs, rewards, done, scores

    def reset(self, nr_of_birds):
        """ Resets the environment (starts a new game). Takes a single argument: the number of birds to reset the environment with"""
        self._game = FlappyBirdLogic(screen_size=self._screen_size,
                                     pipe_gap_size=self._pipe_gap,
                                     nr_of_birds=nr_of_birds)
        if self._renderer is not None:
            self._renderer.game = self._game

        return self._get_observation()

    def render(self, mode='human') -> None:
        """ Renders the next frame. """
        if self._renderer is None:
            self._renderer = FlappyBirdRenderer(screen_size=self._screen_size,
                                                bird_color=self._bird_color,
                                                pipe_color=self._pipe_color,
                                                background=self._bg_type)
            self._renderer.game = self._game
            self._renderer.make_display()

        self._renderer.draw_surface(show_score=True)
        self._renderer.update_display()

    def close(self):
        """ Closes the environment. """
        if self._renderer is not None:
            pygame.display.quit()
            self._renderer = None
        super().close()
