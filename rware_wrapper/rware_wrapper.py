import gymnasium as gym
import tarware # Do not remove. Custom env need to be imported at least once to register with Gymnasium.
from tarware.definitions import RewardType
import numpy as np
from typing import Dict, Tuple, Any


class RwareToOAIWrapper(gym.Env):
    """
    Wraps the Rware environment (Multi-Agent) to be compatible with 
    Single-Agent Gymnasium API for SB3 and oai_agents.
    """
    def __init__(self, env_id: str, **kwargs):
        super().__init__()
        self.args = kwargs.pop('args', None)
        self.env = gym.make(env_id, **kwargs)
        self.n_agents = self.env.unwrapped.num_agents
        
        # SB3 needs a single space. We expose the space of the first agent.
        self.action_space = self.env.action_space[0] 
        self.observation_space = self.env.observation_space[0]
        
        # Metadata and helper attributes
        self.env_name = env_id
        self.layout_name = env_id
        self.num_players = self.n_agents
        self.mdp = None #dummy
        self.step_count = 0
        self.num_envs = 1
        self.teammates = []
        self.reset_p_idx = 0 # Used by evaluate()
        self.last_obs_list = []  # Stores full MA obs for teammate prediction

    def reset(self, seed: int = None, options: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict]:
        self.step_count = 0
        # Gymnasium reset returns (obs, info)
        # rware returns only a list of observations (one per agent)
        reset_data = self.env.reset(seed=seed, options=options)

        # Check if reset returned (obs, info) or just obs
        if isinstance(reset_data, tuple) and len(reset_data) == 2:
            obs_list, info = reset_data
        else:
            obs_list = reset_data
            info = {} # Provide empty info for Gymnasium compatibility

        self.last_obs_list = obs_list
        
        # Convert the specified agent's observation to a numpy array for SB3
        return np.array(obs_list[self.reset_p_idx], dtype=np.float32), info

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.step_count += 1
        
        # 1. Prepare actions for ALL agents
        # 'action' is what SB3/PPO chose for the main agent
        all_actions = [0] * self.n_agents

        # 2. Assign the Learner's action (from SB3)
        # SB3 might pass a numpy scalar; .item() ensures a standard int/float
        if isinstance(action, (np.ndarray, np.generic)):
            action = action.item()
        all_actions[self.reset_p_idx] = int(action)

        # 3. Query teammates for their actions based on their perspective
        # Teammates are provided as a list. We map them to non-learner slots.
        tm_count = 0
        for i in range(self.n_agents):
            if i == self.reset_p_idx:
                continue
            
            if tm_count < len(self.teammates):
                # OAIAgents expect the observation for their specific index
                tm_obs = self.last_obs_list[i]
                # Predict returns (action, state)
                tm_action, _ = self.teammates[tm_count].predict(tm_obs, deterministic=False)
                if isinstance(tm_action, (np.ndarray, np.generic)):
                    tm_action = tm_action.item()
                all_actions[i] = int(tm_action)
                tm_count += 1
        
        # 4. Step the actual Multi-Agent environment
        obs_list, reward_list, term_list, trunc_list, info = self.env.step(all_actions)

        self.last_obs_list = obs_list
        
        # 5. Aggregate results for the single-agent learner
        obs = np.array(obs_list[self.reset_p_idx], dtype=np.float32)

        unwrapped_env = self.env.unwrapped
        # TODO check if tarware exposes RewardType on unwrapped level
        if hasattr(unwrapped_env, 'reward_type') and unwrapped_env.reward_type == RewardType.GLOBAL:
            # In global mode, everyone has the same value; don't sum them.
            reward = float(reward_list[0]) 
        else:
            # In individual mode, sum them to get the total team performance.
            # In two-stage mode: two-stage rewards are a small individual reward for picking up a shelf and a larger global reward for delivering it. So summning them makes sense for a single-agent "team" learner since it's the total utilityy generated in that step.
            reward = float(np.sum(reward_list))

        # Logic for Gymnasium: SB3 expects scalar bools for terminated/truncated
        # We usually consider the episode done if ANY agent is done
        terminated = bool(any(term_list)) if isinstance(term_list, list) else bool(term_list)
        truncated = bool(any(trunc_list)) if isinstance(trunc_list, list) else bool(trunc_list)

        return obs, float(reward), terminated, truncated, info

    def render(self):
        return self.env.render()
    
    def set_teammates(self, teammates):
        """
        Sets the agents that will act for the other slots in the environment.
        Called in OAITrainer.evaluate().
        The trainer is trying to "inject" a pre-trained partner into the environment so it can see how well our current agent plays with them. 
        Since tarware is a Multi-Agent Environment, but Stable Baselines3 is a Single-Agent Learner, this wrapper acts as the Orchestrator:
        1. It takes one action from SB3.
        2. It takes N-1 actions from the Teammates.
        3. It bundles them together and sends them to the tarware core.        
        """
        # OAI Trainer sends a list of agents
        self.teammates = teammates

    def set_reset_p_idx(self, p_idx: int):
        """Called by OAITrainer during evaluation to swap player positions."""
        self.reset_p_idx = p_idx if p_idx is not None else 0

    def get_layout_name(self) -> str:
        return self.env_name
    
    def get_attr(self, attr_name: str, indices=None):
        """Returns the attribute for 'all' environments (only 1 here)."""
        if hasattr(self, attr_name):
            attr_val = getattr(self, attr_name)
            # Vectorized environments return a list (one value per env)
            return [attr_val]
        raise AttributeError(f"{self.__class__.__name__} has no attribute {attr_name}")

    def close(self):
        self.env.close()