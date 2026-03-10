import gymnasium as gym
import tarware
import numpy as np
from typing import Dict, Tuple

class RwareToOAIWrapper(gym.Env):
    """
    Wraps the Rware environment to be compatible with oai_agents scripts.
    """
    def __init__(self, env_id: str, **kwargs):
        super().__init__()
        self.env = gym.make(env_id, **kwargs)
        self.n_agents = self.env.num_agents
        
        # OAI agents often expects a single combined action space 
        # or handles agents individually. 
        self.action_space = self.env.action_space[0] 
        self.observation_space = self.env.observation_space[0]
        
        # Attributes often accessed by oai_agents
        self.args = kwargs.get('args', None)
        self.num_players = self.n_agents

        self.mdp = None # dummy since some oai_agents scripts look for an env.mdp attribute

    def reset(self, seed=None, options=None):
        obs_tuple = self.env.reset(seed=seed)
        # Return as a numpy array if the agent expects a single vector
        return np.array(obs_tuple), {}

    def step(self, actions):
        # Actions might come in as a single value (if one agent is training) 
        # or a list/tuple. We ensure it's a list for rware.
        if not isinstance(actions, (list, tuple, np.ndarray)):
            actions = [actions]
        
        n_obs, rewards, truncated, terminated, info = self.env.step(actions)
        
        # oai_agents usually expects scalar rewards or aggregated info
        # We sum rewards if training a team, or return the list if the algo handles it.
        reward = np.sum(rewards) 
        done = any(terminated) or any(truncated)
        
        return np.array(n_obs), reward, done, done, info

    def render(self, mode='human'):
        return self.env.render()

    def close(self):
        self.env.close()