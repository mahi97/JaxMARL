import hydra
from omegaconf import OmegaConf
import wandb 
import smax
from safetensors.flax import load_file
from flax.traverse_util import unflatten_dict
from typing import Sequence, NamedTuple, Any, Tuple, Union
from flax import struct
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.initializers import constant, orthogonal
import numpy as np 
import distrax
from smax.environments.mpe import MPEVisualizer
import matplotlib.pyplot as plt

def batchify(x: dict, agent_list, num_actors):
    max_dim = max([x[a].shape[-1] for a in agent_list])
    def pad(z, length):
        return jnp.concatenate([z, jnp.zeros(z.shape[:-1] + [length - z.shape[-1]])], -1)

    x = jnp.stack([x[a] if x[a].shape[-1] == max_dim else pad(x[a]) for a in agent_list])
    return x.reshape((num_actors, -1))

def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}

class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)

@hydra.main(version_base=None, config_path="config", config_name="spread_ippo_pretrained")
def main(config):
    config = OmegaConf.to_container(config)
    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        mode="disabled",
    )

    rng = jax.random.PRNGKey(config["SEED"])
    env = smax.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    filepath = "baselines/IPPO/checkpoints/ippo_mpe_params.safetensors"

    flat_dict = load_file(filepath)
    params = unflatten_dict(flat_dict, sep=',')
    

    network = ActorCritic(env.action_space(env.agents[0]).n, activation=config["ACTIVATION"])
    rng, _rng = jax.random.split(rng)
    init_x = jnp.zeros(env.observation_space(env.agents[0]).shape)
    network_params = network.init(_rng, init_x)
    
    rng, _rng = jax.random.split(rng)
    obs, env_state = env.reset(_rng)

    obs_batch = batchify(obs, env.agents, env.num_agents)

    env_states = [env_state]
    for i in range(100):
        rng, _rng1, _rng2 = jax.random.split(rng, 3)
        pi, v = network.apply(params, obs_batch)
        action = pi.sample(seed=_rng1)
        #print(action)
        actions = unbatchify(action, env.agents, 1, env.num_agents)
        #print(actions)
        obs, env_state, rew, done, infos = env.step(_rng2, env_state, actions)
        
        print('rew', rew)
        env_states.append(env_state)

    viz = MPEVisualizer(env, env_states)
    viz.animate('mpe.gif')
    

if __name__=="__main__":
    main()	