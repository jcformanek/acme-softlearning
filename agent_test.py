# python3
# Written by Claude Formanek, 7th of June 2021.
# Code borrowed from stable-baselines3
# https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/sac/
# as well as Acme https://github.com/deepmind/acme

"""Tests for the SAC agent."""

from typing import Dict, Sequence

from absl.testing import absltest
import acme
from acme import specs, wrappers, types
from acme.testing import fakes
from acme.tf import networks
import numpy as np
import sonnet as snt
import tensorflow as tf
import gym
from tensorflow import keras

import agent
import utils


def make_networks(
    action_spec: types.NestedSpec,
    policy_layer_sizes: Sequence[int] = (512, 256),
    critic_layer_sizes: Sequence[int] = (512, 256),
) -> Dict[str, snt.Module]:
    """Creates networks used by the agent."""

    num_dimensions = np.prod(action_spec.shape, dtype=int)
    policy_layer_sizes = list(policy_layer_sizes)
    critic_layer_sizes = list(critic_layer_sizes) + [1]

    policy_network = snt.Sequential(
        [
            networks.AtariTorso(),
            networks.LayerNormMLP(policy_layer_sizes), 
            networks.MultivariateNormalDiagHead(num_dimensions),
            networks.TanhToSpec(action_spec)
        ]
    )

    # The multiplexer concatenates the (maybe transformed) observations/actions.
    critic_network_1 = networks.CriticMultiplexer(
        observation_network=networks.AtariTorso(),
        critic_network=networks.LayerNormMLP(critic_layer_sizes)
    )
    critic_network_2 = networks.CriticMultiplexer(
        observation_network=networks.AtariTorso(),
        critic_network=networks.LayerNormMLP(critic_layer_sizes)
    )

    return {
        'policy': policy_network,
        'critic_1': critic_network_1,
        'critic_2': critic_network_2,
    }

def train_sac(env_name):
    # Create environment.
    environment = gym.make(env_name)
    # Make sure the environment obeys the dm_env.Environment interface.
    environment = utils.BetterGymWrapper(environment)
    # Clip the action returned by the agent to the environment spec.
    environment = wrappers.CanonicalSpecWrapper(environment, clip=True)
    environment = wrappers.SinglePrecisionWrapper(environment)

    spec = specs.make_environment_spec(environment)

    # Create the networks to optimize (online) and target networks.
    agent_networks = make_networks(spec.actions)

    agent_logger = utils.TFSummaryLogger("logs", "agent")
    env_loop_logger = utils.TFSummaryLogger("logs", "env")

    # Construct the agent.
    sac_agent = agent.SAC(
        environment_spec=spec,
        policy_network=agent_networks['policy'],
        critic_network_1=agent_networks['critic_1'],
        critic_network_2=agent_networks['critic_2'],
        logger=agent_logger,
        samples_per_insert=32,
        alpha=0.2,
        min_replay_size=1000,
        batch_size=256
    )

    # Run the environment loop
    loop = acme.EnvironmentLoop(environment, sac_agent, logger=env_loop_logger)
    loop.run(num_episodes=1000)

if __name__ == '__main__':
  train_sac("CarRacing-v0")