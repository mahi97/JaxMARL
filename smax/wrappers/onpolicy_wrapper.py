import jax 
import jax.numpy as jnp
import smax
from smax.environments import MultiAgentEnv

def MPEEnv(args,):
    
    smax_env = "MPE_" + args.scenario_name + "_v3" # TODO account for v4 env
    
    env = smax.make(smax_env)
    rng = jax.random.PRNGKey(0)
    
    return OnPolicyMPE(rng, env)

class OnPolicyMPE:
    
    def __init__(self, 
                 rng,
                 env: MultiAgentEnv,
                ):
        
        self._env = env 
        self.rng = rng
        self.jax_state = None
        
        self.shared_reward = True
        
        self.action_space = [self._env.action_space(a) for a in self._env.agents]
        self.observation_space = [self._env.observation_space(a) for a in self._env.agents]
    
    def seed(self, seed):
        self.rng = jax.random.PRNGKey(seed)
        
    def reset(self):
        
        self.rng, _rng = jax.random.split(self.rng)
        obs, self.jax_state = self._env.reset(_rng)
        
        return self._batchify(obs)
    
    def step(self, action_n):
        
        acts = self._unbatchify(action_n)
        
        self.rng, _rng = jax.random.split(self.rng)
        
        obs, self.jax_state, rew, done, info = self._env.step(_rng, self.jax_state, acts)
        
        return self._batchify(obs), self._batchify(rew), self._batchify(done), info
    
    def _batchify(self, x):
        return jnp.stack([x[a] for a in self._env.agents])

    def _unbatchify(self, x):
        return {a: x[i] for i, a in enumerate(self._env.agents)}

        
if __name__=="__main__":
    
    env = MPEEnv("simple_spread")
    obs = env.reset()
    print(obs)
    print(jnp.shape(obs))
    print(env.action_space[0].__class__.__name__)