import jax 
import jax.numpy as jnp
import smax
from smax.environments import MultiAgentEnv
from smax.environments.mpe.default_params import DISCRETE_ACT, CONTINUOUS_ACT
from gymnax.environments.spaces import Box
from functools import partial

def MPEEnv(args,):
    
    smax_env = "MPE_" + args.scenario_name + "_v3" # TODO account for v4 env
    
    env = smax.make(smax_env, action_type=DISCRETE_ACT)
    rng = jax.random.PRNGKey(0)
    
    if args.n_rollout_threads==1:
        return SingleOnPolicyMPE(rng, env)
    else:
        return VecOnPolicyMPE(rng, env, args.n_rollout_threads)

class SingleOnPolicyMPE:
    
    def __init__(self,
                 rng,
                 env: MultiAgentEnv,
                ):
        
        self._env = env
        self.n = len(self._env.agents)
        
        self._jit_reset = jax.jit(self._env.reset)
        self._jit_step = jax.jit(self._env.step)
        
        self.rng = rng
        self.jax_state = None
        
        self.shared_reward = True
        
        self.action_space = [self._env.action_space(a) for a in self._env.agents]
        self.observation_space = [self._env.observation_space(a) for a in self._env.agents]
        self.shared_obs_dim = self.observation_space[0].shape[0]*len(self.observation_space)
        self.share_observation_space = [Box(
            low=-jnp.inf, high=+jnp.inf, shape=(self.shared_obs_dim,), dtype=jnp.float32
        ) for _ in range(self.n)]
    
    def seed(self, seed):
        self.rng = jax.random.PRNGKey(seed)
        
    def reset(self):
        
        self.rng, _rng = jax.random.split(self.rng)
        obs, self.jax_state = self._jit_reset(_rng)
        
        return self._batchify(obs)
    
    def step(self, action_n):
        
        acts = self._unbatchify(action_n)
        
        self.rng, _rng = jax.random.split(self.rng)
        
        obs, self.jax_state, rew, done, info = self._jit_step(_rng, self.jax_state, acts)
        
        rew = self._batchify(rew)[:,None]*self.n
        
        info = {a: {'individual_reward': rew[a].squeeze()} for a in range(self.n)}
        
        return self._batchify(obs), rew, self._batchify(done), info
    
    def _batchify(self, x):
        return jnp.stack([x[a] for a in self._env.agents])

    def _unbatchify(self, x):
        return {a: x[i] for i, a in enumerate(self._env.agents)}
    
    def close(self):
        pass


class VecOnPolicyMPE(SingleOnPolicyMPE):
    
    def __init__(self,
                 rng,
                 env: MultiAgentEnv,
                 num_envs: int,
                ):
        
        super().__init__(rng, env)
        self.num_envs = num_envs
        
        self._vreset = jax.vmap(self._env.reset, in_axes=(0,))
        self._vstep = jax.vmap(self._env.step, in_axes=(0, 0, 0))
        
    def reset(self):
        
        self.rng, _rng = jax.random.split(self.rng)
        _rng = jax.random.split(_rng, self.num_envs)
        obs, self.jax_state = self._vreset(_rng)
        return self._batchify(obs).swapaxes(0,1)
        
    def step(self, action_n):
        #print('act', jnp.shape(action_n))
        #print('act', action_n)
        acts = jnp.expand_dims(jnp.argmax(action_n, axis=-1), -1)
        #print('argmax', acts.shape, acts)
        #print('shape', acts[:,:, None].shape)
        acts = self._unbatchify(acts.swapaxes(0,1))
        #print('acts', acts)
        # turn into 1d actions 
        #acts = jax.tree_map(lambda x: jnp.argwhere(x==1, size=1).swapaxes(0,1), acts)
        #print('acts', acts)
        
        self.rng, _rng = jax.random.split(self.rng)
        _rng = jax.random.split(_rng, self.num_envs)
        obs, self.jax_state, rew, done, info = self._vstep(_rng, self.jax_state, acts)
            
        rew = self._batchify(rew).swapaxes(0,1)[:,:,None]*self.n
        
        #info = {a: {'individual_reward': rew[:,a].squeeze()} for a in range(self.n)}
        #print('info', info)
        info = [[{'individual_reward': rew[i, a]} for a in range(self.n)] for i in range(self.num_envs)]
        
        return self._batchify(obs).swapaxes(0, 1), rew, self._batchify(done).swapaxes(0, 1), info
    
        
if __name__=="__main__":
    
    scenario_name = "simple_spread"
    
    smax_env = "MPE_" + scenario_name + "_v3" # TODO account for v4 env
    
    env = smax.make(smax_env)
    rng = jax.random.PRNGKey(0)
    
    num_envs = 2 
    if num_envs == 1:
        env = SingleOnPolicyMPE(rng, env)
    else:
        env = VecOnPolicyMPE(rng, env, num_envs)
    
    obs = env.reset()
    print(obs)
    print(jnp.shape(obs))
    print(env.action_space[0].__class__.__name__)