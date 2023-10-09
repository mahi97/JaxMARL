import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
import chex
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Tuple, Union
import distrax
import smax
from smax.environments.multi_agent_env import MultiAgentEnv, State
import matplotlib.pyplot as plt
from functools import partial



class MPEWrapper(object):
    """Base class for all SMAX wrappers."""

    def __init__(self, env: MultiAgentEnv):
        self._env = env

    def __getattr__(self, name: str):
        return getattr(self._env, name)
    
    def _batchify_floats(self, x: dict):
        return jnp.stack([x[a] for a in self._env.agents])
    
    
@struct.dataclass
class MPELogEnvState:
    env_state: State
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    total_episodes: int


class MPELogWrapper(MPEWrapper):
    def __init__(self, env: MultiAgentEnv, replace_info: bool = False):
        super().__init__(env)
        self.replace_info = replace_info

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, State]:
        obs, env_state = self._env.reset(key)
        state = MPELogEnvState(
            env_state,
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
            total_episodes=jnp.zeros((self._env.num_agents,)),
        )
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: MPELogEnvState,
        action: Union[int, float],
    ) -> Tuple[chex.Array, MPELogEnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action
        )
        ep_done = done["__all__"]
        #batch_reward = self._batchify_floats(reward)
        #print('batch reward', batch_reward.shape)
        new_episode_return = state.episode_returns + jnp.sum(self._batchify_floats(reward))
        new_episode_length = state.episode_lengths + 1
        state = MPELogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - ep_done),
            episode_lengths=new_episode_length * (1 - ep_done),
            returned_episode_returns=state.returned_episode_returns * (1 - ep_done)
            + new_episode_return * ep_done,
            total_episodes=state.total_episodes + (ep_done).astype(jnp.int32),
        )
        if self.replace_info:
            info = {}
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode"] = jnp.full((self._env.num_agents,), ep_done)
        return obs, state, reward, done, info
    
class WorldStateWrapper(MPEWrapper):
    
    def __init__(self, env: MultiAgentEnv):
        super().__init__(env)
        
        self.state_size = self.world_state_size()
        
    
    @partial(jax.jit, static_argnums=0)
    def reset(self,
              key):
        obs, env_state = self._env.reset(key)
        obs["world_state"] = self.world_state(obs)
        return obs, env_state
    
    @partial(jax.jit, static_argnums=0)
    def step(self,
             key,
             state,
             action):
        obs, env_state, reward, done, info = self._env.step(
            key, state, action
        )
        obs["world_state"] = self.world_state(obs)
        #reward["world_reward"] = self.world_reward(reward)
        return obs, env_state, reward, done, info

    @partial(jax.jit, static_argnums=0)
    def world_state(self, obs):
        """ 
        For each agent: [agent obs, all other agent obs]
        """
        
        @partial(jax.vmap, in_axes=(0, None))
        def _roll_obs(aidx, all_obs):
            #r = self._env.num_agents - aidx
            robs = jnp.roll(all_obs, -aidx, axis=0)
            robs = robs.flatten()
            return robs
            
        all_obs = jnp.array([obs[agent] for agent in self._env.agents])
        #print('all obs', all_obs.shape)
        return all_obs
        #return _roll_obs(jnp.arange(self._env.num_agents), all_obs)
        return jnp.repeat(all_obs[None], self._env.num_agents, axis=0) # just concat, same for all
    
    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: State):
        avail_actions = jnp.ones((5,))
        return {agent: avail_actions for agent in self.agents}
        
    
    def world_state_size(self):
        spaces = [self._env.observation_space(agent) for agent in self._env.agents]
        return self._env.observation_space(self._env.agents[0]).shape[-1]
        
        #return sum([space.shape[-1] for space in spaces])