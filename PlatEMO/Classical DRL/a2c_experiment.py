# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 14:25:41 2025
Train A2C 30 times on multiple environments (each with custom timesteps & episodes)
and save results with average and std deviation.
"""

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes
import numpy as np
import pandas as pd

# === Environment configurations ===
env_configs = {
    "HalfCheetah-v4": {"max_episodes": 1000, "timesteps": 000000},
    #"Hopper-v4": {"max_episodes": 2000, "timesteps": 2000000},
    #"FrozenLake-v1": {"max_episodes": 1000, "timesteps": 100000},
    #"LunarLander-v3": {"max_episodes": 5000, "timesteps": 5000000},
    #"InvertedDoublePendulum-v4": {"max_episodes": 2000, "timesteps": 2000000},
    #"MountainCarContinuous-v0": {"max_episodes": 2000, "timesteps": 2000000},
    "Reacher-v4": {"max_episodes": 2000, "timesteps": 100000}
}

n_runs = 30
results = []

# === Loop over environments ===
for env_id, config in env_configs.items():
    max_episodes = config["max_episodes"]
    timesteps_per_run = config["timesteps"]

    print(f"\n=== Environment: {env_id} ===")
    print(f"Max Episodes: {max_episodes}, Timesteps per Run: {timesteps_per_run}")
    all_rewards = []

    for run in range(n_runs):
        print(f"--- Training run {run+1}/{n_runs} ---")

        # Create fresh environment
        vec_env = make_vec_env(env_id, n_envs=1)
        callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=max_episodes, verbose=0)

        # Create and train A2C model
        model = A2C("MlpPolicy", vec_env, policy_kwargs={'net_arch': [32, 32]}, verbose=0, device="cpu")
        model.learn(total_timesteps=timesteps_per_run, callback=callback_max_episodes)

        # Evaluate after training (final deterministic rollout)
        obs = vec_env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = vec_env.step(action)
            episode_reward += reward

        episode_reward = float(episode_reward)
        all_rewards.append(episode_reward)
        print(f"Reward for run {run+1}: {episode_reward:.4f}")

    # Compute statistics for environment
    avg_reward = float(np.mean(all_rewards))
    std_reward = float(np.std(all_rewards))
    print(f"\n→ {env_id} | Mean: {avg_reward:.4f}, Std: {std_reward:.4f}")

    # Save individual and summary results
    for i, r in enumerate(all_rewards, start=1):
        results.append({
            "Environment": env_id,
            "Run": i,
            "Reward": round(float(r), 4)
        })

    results.append({
        "Environment": env_id,
        "Run": "Average",
        "Reward": round(avg_reward, 4)
    })
    results.append({
        "Environment": env_id,
        "Run": "StdDev",
        "Reward": round(std_reward, 4)
    })

# === Save to CSV ===
df = pd.DataFrame(results)
df.to_csv("3_a2c_env_rewards.csv", index=False)
print("\n✅ All results saved to 'a2c_env_rewards.csv'")
