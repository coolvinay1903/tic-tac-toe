#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import random
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
import logging


class T3Env(py_environment.PyEnvironment):
    def __init__(self, board_size=9, name="dummy", verbose=False):
        """
        The gameboard will have 9 moves, odd moves for 'o' and even for 'x'
        """
        # Maintain a board data structure separate from the x,o,classical arrays
        self._board_size = board_size
        self._name = name
        self._verbose = verbose
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(level=log_level)

        self._win_patterns = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8],
            [0, 4, 8],
            [2, 4, 6],
        ]
        self._current_player = None
        self._mark = None
        self._board = None
        self._episode_ended = None
        self._iters = None
        self._state = None
        self._mask = None
        self._winner = None
        self._reward = None
        self._reset_all()

        self._action_spec = tensor_spec.from_spec(
            array_spec.BoundedArraySpec(
                shape=(),
                dtype=np.int32,
                minimum=0,
                maximum=self._board_size - 1,
                name="action",
            )
        )
        self._state_spec = tensor_spec.from_spec(
            array_spec.BoundedArraySpec(
                shape=(self._board_size,), dtype=np.float32, name="state"
            )
        )
        self._mask_spec = tensor_spec.from_spec(
            array_spec.ArraySpec(shape=(self._board_size,), dtype=bool, name="mask")
        )
        self._reward_spec = tensor_spec.from_spec(
            array_spec.ArraySpec(shape=(), dtype=np.float32, name="reward")
        )
        self._discount_spec = tensor_spec.from_spec(
            array_spec.ArraySpec(shape=(), dtype=np.float32, name="discount")
        )

    def name(self):
        return self._name

    def _reset_all(self):
        self._current_player = "X"
        self._mark = 1
        self._board = [0 for _ in range(self._board_size)]
        self._mask = [True for _ in range(self._board_size)]
        self._episode_ended = False
        self._iters = 0
        self._winner = None
        self._reward = 0
        self._all_moves = []

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        spec = {
            "state": self._state_spec,
            "mask": self._mask_spec,
        }
        return spec

    def reward_spec(self):
        return self._reward_spec

    def discount_spec(self):
        return self._discount_spec

    def _check_for_end(self):
        for pattern in self._win_patterns:
            pat_sum = sum([self._board[v] for v in pattern])
            if pat_sum == 3:
                self._winner = "X"
                self._reward = 10
            if pat_sum == -3:
                self._winner = "O"
                self._reward = -10
        if self._iters == self._board_size-1:
            self._winner = None
            self._reward = 5
        return self._reward != 0

    def _check_and_end_episode(self):
        self._episode_ended = self._check_for_end()

    def _reset(self):
        self._reset_all()
        self._state = self._get_state()
        return ts.restart(self._state)

    def _update_player(self):
        if not self._episode_ended:
            if self._mark == 1:
                self._mark = -1
                self._current_player = "O"
            else:
                self._mark = 1
                self._current_player = "X"

    def _validate_and_play(self, move):
        if move not in range(0, self._board_size):
            logging.debug(f"move {move} not in [0, {self._board_size})")
            return False

        if not self._mask[move]:
            logging.debug(f"Board is already occupied at {move}")
            return False

        self._play(move)
        return True

    def _make_move(self, move):
        self._board[move] = self._mark
        self._mask[move] = False
        self._all_moves.append(f"I:{self._iters} P:{self._current_player} M:{move}")
        self._iters += 1

    def _play_random_move(self):
        if self._episode_ended:
            return False
        move = np.random.choice([i for i, v in enumerate(self._mask) if v])
        self._make_move(move)
        
        return True

    def _play(self, move):
        assert self._board[move] == 0, f"Tile {move} is not empty. Invalid action!"
        self._make_move(move)
        self._check_and_end_episode()
        self._update_player()

        if self._current_player == "O" and self._play_random_move():
            self._check_and_end_episode()
            self._update_player()

    def _get_reward(self):
        return self._reward

    def _get_state(self):
        observation = {
            "state": np.array(self._board, dtype=np.float32),
            "mask": np.array(self._mask, dtype=bool),
        }
        return observation

    def _step(self, action):
        if self._validate_and_play(action) and self._verbose:
            self.print_board()
        if self._episode_ended:
            reward = self._get_reward()
            self._state = self._get_state()
            return ts.termination(observation=self._state, reward=reward)

        # Episode is running
        reward = 0
        self._state = self._get_state()
        return ts.transition(observation=self._state, reward=reward, discount=0.9)

    def print_board(self):
        board = ["--" for i in range(self._board_size)]
        for i, v in enumerate(self._board):
            if v == 1:
                board[i] = "X "
            elif v == -1:
                board[i] = "O "
        logging.debug("_____________")
        logging.debug(" {} | {} | {}".format(board[0], board[1], board[2]))
        logging.debug("_____________")
        logging.debug(" {} | {} | {}".format(board[3], board[4], board[5]))
        logging.debug("_____________")
        logging.debug(" {} | {} | {}".format(board[6], board[7], board[8]))
        logging.debug("_____________")


if __name__ == "__main__":
    board_size = 9
    env = T3Env(board_size, verbose=True)
    start = time.time()
    time_step = env.reset()
    # random.seed(1)
    while not time_step.is_last():
        action = random.randint(0, board_size - 1)
        time_step = env.step(action)
    print(env._all_moves)
    print(f"Time taken for {env._iters} timesteps = ", time.time() - start)
    print(time_step)
