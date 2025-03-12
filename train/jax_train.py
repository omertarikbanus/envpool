import os
os.environ["JAX_ENABLE_X64"] = "1"  # Enable float64 precision in JAX

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import optax
import envpool

# 1) Remove discrete one-hot usage from PPO loss
# 2) Fix train_step signature

class ActorCritic(nn.Module):
    action_dim: int
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        actor_logits = nn.Dense(self.action_dim)(x)  # Treat this as mean of Gaussian
        critic_value = nn.Dense(1)(x)
        return actor_logits, jnp.squeeze(critic_value, axis=-1)

def ppo_loss(params, actor_critic, states, actions, old_log_probs, returns, advantages):
    logits, values = actor_critic.apply(params, states)

    # Continuous Gaussian policy
    mu = logits
    log_std = jnp.zeros_like(mu)  # or make it a learned parameter
    std = jnp.exp(log_std)

    # log prob of the chosen continuous actions
    gauss_log_probs = -0.5 * (
        ((actions - mu) / (std + 1e-8))**2
        + 2*log_std
        + jnp.log(2.0 * jnp.pi)
    )
    selected_log_probs = jnp.sum(gauss_log_probs, axis=-1)

    ratio = jnp.exp(selected_log_probs - old_log_probs)
    clipped_ratio = jnp.clip(ratio, 0.8, 1.2)

    policy_loss = -jnp.mean(jnp.minimum(ratio * advantages, clipped_ratio * advantages))

    # Critic loss
    value_loss = jnp.mean((returns - values)**2)

    # Entropy bonus (approximate for Gaussian)
    entropy = 0.5 * jnp.mean(
        1.0 + jnp.log(2.0 * jnp.pi * (std**2 + 1e-8))
    )
    entropy_bonus = 0.01 * entropy

    total_loss = policy_loss + 0.5 * value_loss - entropy_bonus
    return total_loss

def train_step(params, opt_state, actor_critic, optimizer, batch):
    """One PPO training step."""
    def loss_fn(p):
        return ppo_loss(
            p, actor_critic,
            batch["states"], batch["actions"],
            batch["old_log_probs"], batch["returns"], batch["advantages"]
        )
    grads = jax.grad(loss_fn)(params)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state


# Example usage
def create_model(key, obs_shape, action_dim):
    model = ActorCritic(action_dim=action_dim)
    variables = model.init(key, jnp.ones(obs_shape))  # Initialize
    return model, variables

# Example EnvWrapper
class JAXEnvWrapper:
    def __init__(self, env_name="Humanoid-v4", num_envs=4):
        self.env = envpool.make_gym(env_name, num_envs=num_envs)
        self.num_envs = num_envs
        self.action_shape = self.env.action_space.shape[0]

    def reset(self, key):
        obs, _ = self.env.reset()
        return jnp.array(obs)

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        obs, rew, term, trunc, info = self.env.step(action)
        return (
            np.asarray(obs, dtype=np.float64),
            np.asarray(rew, dtype=np.float64),
            np.asarray(term, dtype=np.float64),
            np.asarray(trunc, dtype=np.float64),
            info,
        )

def ppo_training_loop():
    key = jax.random.PRNGKey(0)
    env = JAXEnvWrapper(num_envs=4)
    obs = env.reset(key)
    
    # Create model & optimizer
    model, params = create_model(key, obs.shape, env.action_shape)
    optimizer = optax.adam(learning_rate=3e-4)
    opt_state = optimizer.init(params)

    old_log_probs = jnp.zeros((env.num_envs,))  # store from previous step for ratio calc

    for step in range(1000):
        key, subkey = jax.random.split(key)
        # sample continuous actions
        action = jax.random.uniform(
            subkey, (env.num_envs, env.action_shape), minval=-1.0, maxval=1.0
        )

        # step env
        obs_next, rew, term, trunc, info = env.step(action)
        
        # Dummy returns/advantages for demonstration
        returns = rew
        advantages = returns - 0.0

        batch = {
            "states": obs, 
            "actions": action, 
            "old_log_probs": old_log_probs, 
            "returns": returns,
            "advantages": advantages,
        }

        # update policy
        params, opt_state = train_step(params, opt_state, model, optimizer, batch)

        # compute new log_probs for next ratio
        logits, _ = model.apply(params, obs)
        mu = logits
        log_std = jnp.zeros_like(mu)
        std = jnp.exp(log_std)
        gauss_log_probs = -0.5 * (
            ((action - mu)/(std+1e-8))**2 + 2*log_std + jnp.log(2*jnp.pi)
        )
        old_log_probs = jnp.sum(gauss_log_probs, axis=-1)
        
        # move to next
        obs = obs_next

        if step % 100 == 0:
            print(f"Step {step} reward={jnp.mean(rew)}")

ppo_training_loop()
