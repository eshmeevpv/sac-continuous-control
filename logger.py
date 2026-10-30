from collections import deque

import gymnasium as gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class TensorboardSummaries(gym.Wrapper):
    """Minimal local replacement for the coursework's external logging wrapper."""

    def __init__(self, env, name):
        super().__init__(env)
        self.writer = SummaryWriter(log_dir=f'logs/{name}')
        self._step = 0
        self._episode_return = 0.0
        self._episode_length = 0
        self._recent_returns = deque(maxlen=100)

    def reset(self, **kwargs):
        self._episode_return = 0.0
        self._episode_length = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step += 1
        self._episode_return += reward
        self._episode_length += 1
        if terminated or truncated:
            self._recent_returns.append(self._episode_return)
            self.writer.add_scalar('Episodes/total_reward', self._episode_return, self._step)
            self.writer.add_scalar('Episodes/episode_length', self._episode_length, self._step)
            self.writer.add_scalar('Episodes/reward_mean_100', np.mean(self._recent_returns), self._step)
        return obs, reward, terminated, truncated, info

    def close(self):
        self.writer.close()
        return super().close()
