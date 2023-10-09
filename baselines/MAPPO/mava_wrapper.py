
import jax 
import jax.numpy as jnp
import chex
from flax import struct
from typing import TYPE_CHECKING, NamedTuple, Union

from smax.environments.multi_agent_env import MultiAgentEnv
from smax.environments.multi_agent_env import State as JaxMARLState 

from smax.environments.mpe.simple import State as MPEJaxMARLState

from jumanji.env import State as JumanjiState
from jumanji.types import restart, termination, transition

@struct.dataclass
class MPEJumanjiState:
    key: chex.PRNGKey
    p_pos: chex.Array  # [num_entities, [x, y]]
    p_vel: chex.Array  # [n, [x, y]]
    c: chex.Array  # communication state [num_agents, [dim_c]]
    done: chex.Array  # bool [num_agents, ]
    step: int  # current step
    goal: int = None
    
class Observation(NamedTuple):
    
    agents_view: chex.Array
    action_mask: chex.Array
    step_count: chex.Array


class MavaWrapper(object):
    
    def __init__(self,
                 env: MultiAgentEnv):
        
        self._env = env 
        
    def __getattr__(self, name: str):
        return getattr(self._env, name)
        
    def _batchify_obs(self, x: dict):
        max_dim = max([x[a].shape[-1] for a in self._env.agents])
        def pad(z, length):
            return jnp.concatenate([z, jnp.zeros(z.shape[:-1] + [length - z.shape[-1]])], -1)

        x = jnp.stack([x[a] if x[a].shape[-1] == max_dim else pad(x[a]) for a in self._env.agents])
        return x.reshape((self._env.num_agents, -1))
    
    def _batchify_rew(self, x: dict):
        return jnp.stack([x[a] for a in self._env.agents])
    
    def reset(self, key: chex.PRNGKey):
        
        key, _key = jax.random.split(key)
        obs, state = self._env.reset(_key)
        
        obs = self._batchify_obs(obs)
        act_masks = jnp.ones((self._env.num_agents, 5))
        
        obs = Observation(
            agents_view=obs,
            action_mask=act_masks,
            step_count=jnp.array(0, int),
        )
        
        state = MPEJumanjiState(
            key=key,
            p_pos=state.p_pos,
            p_vel=state.p_vel,
            c=state.c,
            done=state.done,
            step=state.step,
            goal=state.goal,
        )
        
        timestep = restart(obs)
        
        return state, timestep
        
    def step(self, state: JumanjiState, action: chex.Array):
        
        key = state.key
        key, _key = jax.random.split(key)
        
        # step
        state = MPEJaxMARLState(
            p_pos=state.p_pos,
            p_vel=state.p_vel,
            c=state.c,
            done=state.done,
            step=state.step,
            goal=state.goal,
        )

        actions = {a: action[i] for i, a in enumerate(self._env.agents)}
        
        obs, state, rewards, dones, infos = self._env.step_env(_key, state, actions)
        
        # convert
        obs = self._batchify_obs(obs)
        rew = self._batchify_rew(rewards)
        print('rew ',rew)
        act_masks = jnp.ones((self._env.num_agents, 5))
        
        next_observation = Observation(
            agents_view=obs,
            action_mask=act_masks,
            step_count=jnp.array(state.step, int),
        )
        
        timestep = jax.lax.cond(
            dones["__all__"],
            termination,
            transition,
            rew,
            next_observation,
        )
        
        state = MPEJumanjiState(
            key=key,
            p_pos=state.p_pos,
            p_vel=state.p_vel,
            c=state.c,
            done=state.done,
            step=state.step,
            goal=state.goal,
        )
        
        return state, timestep
    

if __name__ == "__main__":
    
    from smax import make 
    
    env = make("MPE_simple_spread_v3")
    
    mava = MavaWrapper(env)
    
    key = jax.random.PRNGKey(0)
    
    key, _key = jax.random.split(key)
    
    state, timestep = mava.reset(_key)
    
    act = jnp.ones((3, 5))
    
    state, timestep = mava.step(state, act)
    
    state, timestep = mava.step(state, act)
    
    state, timestep = mava.step(state, act)