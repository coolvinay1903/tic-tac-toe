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
from Logger import create_logger
from T3Env import T3Env


class T3NashEnv(T3Env):
    def __init__(self,
                 _board_size=9,
                 logger=None,
                 name="T3NashEnv",
                 verbose=False):
        """
        The gameboard will have 9 moves, odd moves for 'o' and even for 'x'
        """
        super(T3NashEnv, self).__init__(_board_size=_board_size,
                                        logger=logger,
                                        name=name,
                                        verbose=verbose)

    def _check_for_end(self):
        """
        Play for a draw as either player winning wil not lead to Nash eq.
        reward as follows:
        draw: 10
        any player wins:  -10
        should intermediate reward be +1, so that we always play the longest game?
        """
        for pattern in self._win_patterns:
            pat_sum = sum([self._board[v] for v in pattern])
            if pat_sum == 3:
                self._winner = "X"
                self._reward = -10
            if pat_sum == -3:
                self._winner = "O"
                self._reward = -10
        if not self._winner and self._iters == self._board_size:
            self._winner = None
            self._reward = 10
        return self._reward != 0

    def _play(self, move):
        assert self._board[
            move] == 0, f"Tile {move} is not empty. Invalid action!"
        self._make_move(move)
        self._check_and_end_episode()
        self._update_player()


if __name__ == "__main__":
    board_size = 9
    env = T3NashEnv(board_size, verbose=True)
    start = time.time()
    time_step = env.reset()
    # random.seed(1)
    while not time_step.is_last():
        action = random.randint(0, board_size - 1)
        time_step = env.step(action)
    print(env._all_moves)
    print(f"Time taken for {env._iters} timesteps = ", time.time() - start)
    print(time_step)
