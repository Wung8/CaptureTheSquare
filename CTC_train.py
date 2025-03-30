import torch
import cv2
import numpy as np

from capture_the_cube_temp import CTCEnvironment as env
from RL.Recurrent_PPO import Recurrent_PPO as PPO
from RL.vec_env_handler import ParallelEnvManager
from RL.type_aliases import LSTMStates


n_envs = 12
n_agents = 8

def test(self, n_steps=1_000, **kwargs):
    cumulative_reward = 0
    env = self.env(**kwargs)
    obs, info = env.reset()
    hidden_state_shape = list(self.hidden_state_shape)
    hidden_state_shape[1] = n_agents
    lstm_states = LSTMStates(
        (
            torch.zeros(hidden_state_shape, dtype=torch.float32),
            torch.zeros(hidden_state_shape, dtype=torch.float32),
        ),
        (
            torch.zeros(hidden_state_shape, dtype=torch.float32),
            torch.zeros(hidden_state_shape, dtype=torch.float32),
        )
    )
    episode_starts = [[0] for i in range(n_agents)]
    for step in range(n_steps):
        obs = np.array(obs, dtype=np.float32)
        with torch.no_grad():
            episode_starts = torch.tensor(episode_starts, dtype=torch.float32)
            action, log_probs, values, lstm_states = self.get_action(np.expand_dims(obs,0), lstm_states, episode_starts)
        new_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated[0]
        
        cumulative_reward += sum(reward)
        
        if done: break
        obs = new_obs
    return cumulative_reward

class vec_env_wrapper:
    def __init__(self, vec_env):
        self.vec_env = vec_env

    def step(self, actions):
        global a
        a = actions
        actions = actions.reshape(n_envs, -1, actions.shape[-1])
        obs, rewards, dones = [np.array(info) for info in self.vec_env.step(actions)]
        obs = np.reshape(obs, (-1,*obs.shape[2:]))
        rewards = rewards.flatten()
        dones = dones.flatten()
        return obs, rewards, dones

    def reset(self):
        obs = self.vec_env.reset()
        obs = np.array(obs, dtype=np.float32)
        obs = np.reshape(obs, (-1,*obs.shape[2:]))
        return obs

    def reshape_info(self, infos):
        obs, rewards, dones = infos
        obs = np.array(obs, dtype=np.float32).reshape(-1,*obs[2:])
        rewards = np.array(rewards, dtype=np.float32).reshape(-1)
        dones = np.array(dones, dtype=np.float32).reshape(-1)
        return obs, rewards, dones


if __name__ == "__main__":     
    vec_env = ParallelEnvManager(env, n_envs)
    env_manager = vec_env_wrapper(vec_env)

    agent = PPO(
        env=None,
        observation_space=(3,48,48),
        action_space=(3,3,2,4,5),
        n_steps=3000,
        batch_size=125,
        epochs=5,
        n_envs=n_envs*n_agents
    )
    agent.env = env
    agent.env_manager = env_manager
    agent.last_obs = agent.env_manager.reset()
    agent.test = test

    agent.learn(total_steps=5_000_000)
