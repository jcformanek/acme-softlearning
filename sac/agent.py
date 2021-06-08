# python3
# Written by Claude Formanek, 7th of June 2021.
# Code borrowed from stable-baselines3
# https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/sac/

"""SAC agent implementation"""

import copy
from typing import Optional

from acme import specs, types, datasets
from acme.adders import reverb as adders
from acme.agents import agent
from acme.utils import loggers, counting
from acme.tf import utils as tf2_utils
from acme.agents.tf import actors
import reverb
import sonnet as snt

from sac import learning

class SAC(agent.Agent):
    """SAC Agent.

    This implements a single-process SAC agent.
    """

    def __init__(self,
        environment_spec: specs.EnvironmentSpec,
        policy_network: snt.Module,
        critic_network_1: snt.Module,
        critic_network_2: snt.Module,
        discount: float = 0.99,
        batch_size: int = 256,
        prefetch_size: int = 4,
        alpha=0.1,
        target_update_period: int = 100,
        min_replay_size: int = 1000,
        max_replay_size: int = 1000000,
        samples_per_insert: float = 32.0,
        n_step: int = 5,
        clipping: bool = True,
        logger: Optional[loggers.Logger] = None,
        counter: Optional[counting.Counter] = None,
        checkpoint: bool = True,
        replay_table_name: str = adders.DEFAULT_PRIORITY_TABLE):

        """Initialize the agent.
        
        Args:
            environment_spec: description of the actions, observations, etc.
            policy_network: the online (optimized) policy.
            critic_network: the online critic.
            observation_network: optional network to transform the observations before
                they are fed into any network.
            discount: discount to use for TD updates.
            batch_size: batch size for updates.
            prefetch_size: size to prefetch from replay.
            target_update_period: number of learner steps to perform before updating
                the target networks.
            min_replay_size: minimum replay size before updating.
            max_replay_size: maximum replay size.
            samples_per_insert: number of samples to take from replay for every insert
                that is made.
            n_step: number of steps to squash into a single transition.
            sigma: standard deviation of zero-mean, Gaussian exploration noise.
            clipping: whether to clip gradients by global norm.
            logger: logger object to be used by learner.
            counter: counter object used to keep track of steps.
            checkpoint: boolean indicating whether to checkpoint the learner.
            replay_table_name: string indicating what name to give the replay table.
        """

        # Create a replay server to add data to. This uses no limiter behavior
        # in order to allow the Agent interface to handle it.
        replay_table = reverb.Table(
            name=replay_table_name,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=max_replay_size,
            rate_limiter=reverb.rate_limiters.MinSize(1),
            signature=adders.NStepTransitionAdder.signature(environment_spec))
        self._server = reverb.Server([replay_table], port=None)

        # The adder is used to insert observations into replay.
        address = f'localhost:{self._server.port}'
        adder = adders.NStepTransitionAdder(
            priority_fns={replay_table_name: lambda x: 1.},
            client=reverb.Client(address),
            n_step=n_step,
            discount=discount)

        # The dataset provides an interface to sample from replay.
        dataset = datasets.make_reverb_dataset(
            table=replay_table_name,
            server_address=address,
            batch_size=batch_size,
            prefetch_size=prefetch_size)

        # Get observation and action specs.
        act_spec = environment_spec.actions
        obs_spec = environment_spec.observations

        # Create target networks.
        target_critic_network_1 = copy.deepcopy(critic_network_1)
        target_critic_network_2 = copy.deepcopy(critic_network_2)

        # Create variables.
        tf2_utils.create_variables(policy_network, [obs_spec])
        tf2_utils.create_variables(critic_network_1, [obs_spec, act_spec])
        tf2_utils.create_variables(critic_network_2, [obs_spec, act_spec])
        tf2_utils.create_variables(target_critic_network_1, [obs_spec, act_spec])
        tf2_utils.create_variables(target_critic_network_2, [obs_spec, act_spec])

        # Create the actor which defines how we take actions.
        actor = actors.FeedForwardActor(policy_network, adder=adder)

        # Create optimizers.
        policy_optimizer = snt.optimizers.Adam(learning_rate=3e-4)
        critic_optimizer_1 = snt.optimizers.Adam(learning_rate=3e-4)
        critic_optimizer_2 = snt.optimizers.Adam(learning_rate=3e-4)

        # The learner updates the parameters (and initializes them).
        learner = learning.SACLearner(
            policy_network=policy_network,
            critic_network_1=critic_network_1,
            critic_network_2=critic_network_2,
            target_critic_network_1=target_critic_network_1,
            target_critic_network_2=target_critic_network_2,
            policy_optimizer=policy_optimizer,
            critic_optimizer_1=critic_optimizer_1,
            critic_optimizer_2=critic_optimizer_2,
            clipping=clipping,
            alpha=alpha,
            discount=discount,
            target_update_period=target_update_period,
            dataset=dataset,
            counter=counter,
            logger=logger,
            checkpoint=checkpoint,
        )

        super().__init__(
        actor=actor,
        learner=learner,
        min_observations=max(batch_size, min_replay_size),
        observations_per_step=float(batch_size) / samples_per_insert)