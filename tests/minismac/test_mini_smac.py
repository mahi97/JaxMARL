import jax.numpy as jnp
import jax
from smax import make
from smax.environments.mini_smac.mini_smac_env import State
import pytest


def create_env(key):
    env, params = make("MiniSMAC")
    obs, state = env.reset(key)
    return env, params, obs, state


def get_random_actions(key, env):
    key_a = jax.random.split(key, num=env.num_agents)
    return {
        agent: env.action_space(agent).sample(key_a[i])
        for i, agent in enumerate(env.agents)
    }


@pytest.mark.parametrize(
    ("action", "vec_diff", "do_jit"),
    [
        (0, jnp.array([0, 2.5]), False),
        (0, jnp.array([0, 2.5]), True),
        (1, jnp.array([2.5, 0]), False),
        (1, jnp.array([2.5, 0]), True),
        (2, jnp.array([0, -2.5]), False),
        (2, jnp.array([0, -2.5]), True),
        (3, jnp.array([-2.5, 0]), False),
        (3, jnp.array([-2.5, 0]), True),
    ],
)
def test_move_actions(action, vec_diff, do_jit):
    with jax.disable_jit(do_jit):
        key = jax.random.PRNGKey(0)
        key, key_reset = jax.random.split(key)
        env, params, _, state = create_env(key_reset)

        init_pos = jnp.array([16, 16])
        unit_positions = state.unit_positions.at[0].set(init_pos)
        state = state.replace(unit_positions=unit_positions)
        key, key_action = jax.random.split(key)
        actions = get_random_actions(key_action, env)
        actions["ally_0"] = action
        key, key_step = jax.random.split(key)
        _, state, _, _, _ = env.step(key_step, state, actions, params)
        assert jnp.allclose(state.unit_positions[0], init_pos + vec_diff)


@pytest.mark.parametrize(
    (
        "unit_1_idx",
        "unit_2_idx",
        "unit_1_pos",
        "unit_2_pos",
        "unit_1_health",
        "unit_2_health",
        "unit_1_reward",
        "unit_2_reward",
        "do_jit",
    ),
    [
        (
            0,
            5,
            jnp.array([1.0, 1.0]),
            jnp.array([1.0, 2.0]),
            0.84,
            0.84,
            0.032,
            0.032,
            False,
        ),
        (
            0,
            5,
            jnp.array([1.0, 1.0]),
            jnp.array([1.0, 2.0]),
            0.84,
            0.84,
            0.032,
            0.032,
            True,
        ),
        (
            0,
            5,
            jnp.array([1.0, 1.0]),
            jnp.array([30.0, 30.0]),
            1.0,
            1.0,
            0.0,
            0.0,
            False,
        ),
        (
            0,
            5,
            jnp.array([1.0, 1.0]),
            jnp.array([30.0, 30.0]),
            1.0,
            1.0,
            0.0,
            0.0,
            True,
        ),
    ],
)
def test_attack_actions(
    unit_1_idx,
    unit_2_idx,
    unit_1_pos,
    unit_2_pos,
    unit_1_health,
    unit_2_health,
    unit_1_reward,
    unit_2_reward,
    do_jit,
):
    with jax.disable_jit(do_jit):
        key = jax.random.PRNGKey(0)
        key, key_reset = jax.random.split(key)
        env, params, _, state = create_env(key_reset)

        unit_1_action = unit_2_idx - env.num_agents_per_team + 4
        unit_2_action = env.num_agents_per_team - unit_1_idx + 3

        unit_positions = state.unit_positions.at[unit_1_idx].set(unit_1_pos)
        unit_positions = unit_positions.at[unit_2_idx].set(unit_2_pos)
        state = state.replace(unit_positions=unit_positions)

        key, key_actions = jax.random.split(key)
        actions = get_random_actions(key_actions, env)
        actions[f"ally_{unit_1_idx}"] = unit_1_action
        actions[f"enemy_{unit_2_idx - env.num_agents_per_team}"] = unit_2_action

        key, key_step = jax.random.split(key)
        _, state, rewards, _, _ = env.step(key_step, state, actions, params)

        assert jnp.allclose(state.unit_health[unit_1_idx], unit_1_health)
        assert jnp.allclose(state.unit_health[unit_2_idx], unit_2_health)
        assert jnp.allclose(rewards[f"ally_{unit_1_idx}"], unit_1_reward)
        assert jnp.allclose(
            rewards[f"enemy_{unit_2_idx - env.num_agents_per_team}"], unit_2_reward
        )


@pytest.mark.parametrize(
    ("ally_health", "enemy_health", "done_unit", "reward_unit", "do_jit"),
    [
        (1.0, 0.1, "enemy_0", "ally_0", True),
        (1.0, 0.1, "enemy_0", "ally_0", False),
        (0.1, 1.0, "ally_0", "enemy_0", True),
        (0.1, 1.0, "ally_0", "enemy_0", False),
    ],
)
def test_episode_end(ally_health, enemy_health, done_unit, reward_unit, do_jit):
    with jax.disable_jit(do_jit):
        key = jax.random.PRNGKey(0)
        key, key_reset = jax.random.split(key)
        env, params, _, state = create_env(key_reset)

        unit_1_idx = 0
        unit_2_idx = env.num_agents_per_team

        unit_1_health = ally_health
        unit_2_health = enemy_health

        unit_1_action = unit_2_idx - env.num_agents_per_team + 4
        unit_2_action = env.num_agents_per_team - unit_1_idx + 3

        unit_positions = state.unit_positions.at[unit_1_idx].set(jnp.array([1.0, 1.0]))
        unit_positions = unit_positions.at[unit_2_idx].set(jnp.array([1.0, 2.0]))
        unit_alive = jnp.zeros((env.num_agents,), dtype=jnp.bool_)
        unit_alive = unit_alive.at[unit_1_idx].set(1)
        unit_alive = unit_alive.at[unit_2_idx].set(1)

        unit_health = jnp.zeros((env.num_agents,))
        unit_health = unit_health.at[unit_1_idx].set(unit_1_health)
        unit_health = unit_health.at[unit_2_idx].set(unit_2_health)

        state = state.replace(
            unit_positions=unit_positions,
            unit_alive=unit_alive,
            unit_health=unit_health,
        )

        key, key_actions = jax.random.split(key)
        actions = get_random_actions(key_actions, env)
        actions[f"ally_{unit_1_idx}"] = unit_1_action
        actions[f"enemy_{unit_2_idx - env.num_agents_per_team}"] = unit_2_action

        key, key_step = jax.random.split(key)
        _, state, rewards, dones, _ = env.step(key_step, state, actions, params)
        assert dones[done_unit]
        assert dones["__all__"]
        assert jnp.allclose(rewards[reward_unit], params.won_battle_bonus + 0.02)


@pytest.mark.parametrize(("do_jit"), [True, False])
def test_episode_time_limit(do_jit):
    with jax.disable_jit(do_jit):
        key = jax.random.PRNGKey(0)
        key, key_reset = jax.random.split(key)
        env, params, _, state = create_env(key_reset)
        state = state.replace(time=params.max_steps)

        key, key_actions = jax.random.split(key)
        actions = get_random_actions(key_actions, env)

        key, key_step = jax.random.split(key)

        _, state, _, dones, _ = env.step(key_step, state, actions, params)

        assert dones["__all__"]


@pytest.mark.parametrize(("do_jit"), [False, True])
def test_obs_function(do_jit):
    with jax.disable_jit(do_jit):
        key = jax.random.PRNGKey(0)
        key, key_reset = jax.random.split(key)
        env, params, obs, state = create_env(key_reset)
        first_enemy_idx = (env.num_agents_per_team - 1) * len(env.unit_features)
        assert jnp.allclose(
            obs["ally_0"][0 : len(env.unit_features)],
            jnp.array([1.0, -0.5755913, 0.43648314]),
        )
        assert jnp.allclose(
            obs["ally_0"][first_enemy_idx : first_enemy_idx + len(env.unit_features)],
            jnp.zeros((3,)),
        )
        assert jnp.allclose(
            obs["enemy_0"][0 : len(env.unit_features)],
            jnp.array([1.0, 0.00752163, 0.5390887]),
        )
        assert jnp.allclose(
            obs["enemy_0"][first_enemy_idx : first_enemy_idx + len(env.unit_features)],
            jnp.zeros((3,)),
        )
        # test a dead agent sees nothing
        unit_alive = state.unit_alive.at[0].set(0)
        unit_health = state.unit_health.at[0].set(0)
        # test that the right unit corresponds to the right agent
        unit_positions = state.unit_positions.at[0].set(jnp.array([1.0, 1.0]))
        unit_positions = unit_positions.at[env.num_agents_per_team].set(
            jnp.array([1.0, 2.0])
        )
        unit_positions = unit_positions.at[env.num_agents_per_team + 1].set(
            jnp.array([1.5, 3.0])
        )
        state = state.replace(
            unit_alive=unit_alive,
            unit_health=unit_health,
            unit_positions=unit_positions,
        )
        key, key_actions = jax.random.split(key)
        actions = get_random_actions(key_actions, env)
        actions["enemy_0"] = 0
        actions["enemy_1"] = 0

        key, key_step = jax.random.split(key)

        obs, state, _, _, _ = env.step(key_step, state, actions, params)
        assert jnp.allclose(obs["ally_0"], jnp.zeros((env.obs_size,)))
        assert jnp.allclose(
            obs["enemy_0"][first_enemy_idx : first_enemy_idx + len(env.unit_features)],
            jnp.zeros((3,)),
        )
        last_ally_idx = first_enemy_idx - len(env.unit_features)
        assert jnp.allclose(
            obs["enemy_0"][last_ally_idx : last_ally_idx + len(env.unit_features)],
            jnp.array([1.0, 0.125, 0.25]),
        )


@pytest.mark.parametrize("do_jit", [True, False])
def test_world_state(do_jit):
    with jax.disable_jit(do_jit):
        key = jax.random.PRNGKey(0)
        key, key_reset = jax.random.split(key)
        env, params, _, state = create_env(key_reset)
        start_positions = jnp.concatenate(
            [
                jnp.stack([jnp.array([8.0, 16.0])] * env.num_agents_per_team),
                jnp.stack([jnp.array([24.0, 16.0])] * env.num_agents_per_team),
            ]
        )
        state = state.replace(unit_positions=start_positions)
        unit_1_idx = 0
        unit_2_idx = env.num_agents_per_team

        unit_alive = jnp.zeros((env.num_agents,), dtype=jnp.bool_)
        unit_alive = unit_alive.at[unit_1_idx].set(1)
        unit_alive = unit_alive.at[unit_2_idx].set(1)

        unit_health = jnp.zeros((env.num_agents,))
        unit_health = unit_health.at[unit_1_idx].set(0.5)
        unit_health = unit_health.at[unit_2_idx].set(0.5)

        state = state.replace(unit_alive=unit_alive, unit_health=unit_health)

        unit_1_action = 0
        unit_2_action = 2

        key, key_actions = jax.random.split(key)
        actions = get_random_actions(key_actions, env)
        actions[f"ally_{unit_1_idx}"] = unit_1_action
        actions[f"enemy_{unit_2_idx - env.num_agents_per_team}"] = unit_2_action

        key, key_step = jax.random.split(key)
        obs, state, _, _, _ = env.step(key_step, state, actions, params)

        world_state = jnp.zeros((env.state_size,))
        world_state = world_state.at[0 : len(env.unit_features)].set(
            jnp.array([0.5, 8.0, 18.5])
        )
        idx = env.num_agents_per_team * len(env.unit_features)
        end_idx = idx + len(env.unit_features)
        world_state = world_state.at[idx:end_idx].set(jnp.array([0.5, 24.0, 13.5]))
        idx = env.num_agents * len(env.unit_features) + env.num_agents_per_team
        end_idx = idx + env.num_agents_per_team
        world_state = world_state.at[idx:end_idx].set(
            jnp.ones((env.num_agents_per_team,))
        )
        assert jnp.allclose(obs["world_state"], world_state)