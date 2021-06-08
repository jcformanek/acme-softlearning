# python3
# Written by Claude Formanek, 7th of June 2021.
# Code borrowed from stable-baselines3
# https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/sac/

"""SAC learner implementation"""

import time
from typing import List, Optional

import acme
from acme import types
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils
from acme.utils import counting
from acme.utils import loggers
import numpy as np
import sonnet as snt
import tensorflow as tf

class SACLearner(acme.Learner):
    """DDPG learner.
    This is the learning component of a DDPG agent. IE it takes a dataset as input
    and implements update functionality to learn from this dataset.
    """

    def __init__(
        self,
        policy_network: snt.Module,
        critic_network_1: snt.Module,
        critic_network_2: snt.Module,
        target_critic_network_1: snt.Module,
        target_critic_network_2: snt.Module,
        discount: float,
        target_update_period: int,
        dataset: tf.data.Dataset,
        alpha: float = 0,
        tau: float = 0.005,
        policy_optimizer: Optional[snt.Optimizer] = None,
        critic_optimizer_1: Optional[snt.Optimizer] = None,
        critic_optimizer_2: Optional[snt.Optimizer] = None,
        clipping: bool = True,
        counter: Optional[counting.Counter] = None,
        logger: Optional[loggers.Logger] = None,
        checkpoint: bool = True,
    ):

        """Initializes the learner.

        Args:
            policy_network: the online (optimized) policy.
            critic_network: the online critic.
            target_critic_network: the target critic.
            discount: discount to use for TD updates.
            target_update_period: number of learner steps to perform before updating
                the target networks.
            dataset: dataset to learn from, whether fixed or from a replay buffer
                (see `acme.datasets.reverb.make_dataset` documentation).
            observation_network: an optional online network to process observations
                before the policy and the critic.
            target_observation_network: the target observation network.
            policy_optimizer: the optimizer to be applied to the DPG (policy) loss.
            critic_optimizer: the optimizer to be applied to the critic loss.
            clipping: whether to clip gradients by global norm.
            counter: counter object used to keep track of steps.
            logger: logger object to be used by learner.
            checkpoint: boolean indicating whether to checkpoint the learner.
        """

        # Store online and target networks.
        self._policy_network = policy_network
        self._critic_network_1 = critic_network_1
        self._critic_network_2 = critic_network_2
        self._target_critic_network_1 = target_critic_network_1
        self._target_critic_network_2 = target_critic_network_2

        # General learner book-keeping and loggers.
        self._counter = counter or counting.Counter()
        self._logger = logger or loggers.make_default_logger('learner')

        # Other learner parameters.
        self._discount = discount
        self._alpha = alpha
        self._tau = tau
        self._clipping = clipping

        # Necessary to track when to update target networks.
        self._num_steps = tf.Variable(0, dtype=tf.int32)
        self._target_update_period = target_update_period

        # Create an iterator to go through the dataset.
        self._iterator = iter(dataset)

        # Create optimizers if they aren't given.
        self._critic_optimizer_1 = critic_optimizer_1 or snt.optimizers.Adam(1e-4)
        self._critic_optimizer_2 = critic_optimizer_2 or snt.optimizers.Adam(1e-4)
        self._policy_optimizer = policy_optimizer or snt.optimizers.Adam(1e-4)

        # Expose the variables.
        policy_network_to_expose = self._policy_network

        self._variables = {
            'critic_1': target_critic_network_1.variables,
            'critic_2': target_critic_network_2.variables,
            'policy': policy_network_to_expose.variables,
        }

        # Setup checkpointing.
        self._checkpointer = tf2_savers.Checkpointer(
            time_delta_minutes=5,
            objects_to_save={
                'counter': self._counter,
                'policy': self._policy_network,
                'critic_1': self._critic_network_1,
                'critic_2': self._critic_network_2,
                'target_critic_1': self._target_critic_network_1,
                'target_critic_2': self._target_critic_network_2,
                'policy_optimizer': self._policy_optimizer,
                'critic_optimizer_1': self._critic_optimizer_1,
                'critic_optimizer_2': self._critic_optimizer_2,
                'num_steps': self._num_steps,
            },
            enable_checkpointing=checkpoint,
        )

        # Do not record timestamps until after the first learning step is done.
        # This is to avoid including the time it takes for actors to come online and
        # fill the replay buffer.
        self._timestamp = None

    @tf.function
    def _step(self):

        # Update target network.
        online_variables = (
            *self._critic_network_1.variables,
            *self._critic_network_2.variables,
        )
        target_variables = (
            *self._target_critic_network_1.variables,
            *self._target_critic_network_2.variables,
        )

        # Make online -> target network update ops.
        for src, dest in zip(online_variables, target_variables):
            dest.assign(self._tau * src + (1 - self._tau) * dest)

        self._num_steps.assign_add(1)

        # Get data from replay (dropping extras if any). Note there is no
        # extra data here because we do not insert any into Reverb.
        inputs = next(self._iterator)
        transitions: types.Transition = inputs.data

        # Cast the additional discount to match the environment discount dtype.
        discount = tf.cast(self._discount, dtype=transitions.discount.dtype)

        with tf.GradientTape(persistent=True) as tape:
            o_tm1 = transitions.observation
            o_t = transitions.next_observation

            # Sample action fresh from current policy.
            pi = self._policy_network(o_tm1)
            actions = pi.sample()
            log_probs = pi.log_prob(actions)

            # Sample next action fresh from current policy.
            pi = self._policy_network(o_t)
            next_actions = pi.sample()
            next_log_probs = pi.log_prob(next_actions)

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            q_tm1_critic_1 = tf.squeeze(self._critic_network_1(o_tm1, transitions.action))
            q_tm1_critic_2 = tf.squeeze(self._critic_network_2(o_tm1, transitions.action))

            q_t = tf.reduce_min(
                tf.concat(
                    [
                        self._target_critic_network_1(o_t, next_actions),
                        self._target_critic_network_2(o_t, next_actions)
                    ],
                    axis=1
                ), 
                axis=1, 
                keepdims=True
            )
            # Reshape [B]
            q_t = tf.squeeze(q_t)

             # Compute target with entropy term.
            q_target = transitions.reward + discount * transitions.discount * (q_t - self._alpha * next_log_probs)
            q_target = tf.stop_gradient(q_target)

            # Compute critic losses
            critic_loss_1 = 0.5 * tf.reduce_mean(tf.square(q_target - q_tm1_critic_1))
            critic_loss_2 = 0.5 * tf.reduce_mean(tf.square(q_target - q_tm1_critic_2))

            # Actor loss
            q_values_pi = tf.reduce_min(
                tf.concat(
                    [
                        self._critic_network_2(o_tm1, actions),
                        self._critic_network_1(o_tm1, actions)
                    ],
                    axis=1
                ),
                axis=1,
                keepdims=True
            )

            # Reshape [B]
            q_values_pi = tf.squeeze(q_values_pi)

            policy_loss = tf.reduce_mean(self._alpha * log_probs - q_values_pi)

        # Get trainable variables.
        policy_variables = self._policy_network.trainable_variables
        critic_variables_1 = self._critic_network_1.trainable_variables
        critic_variables_2 = self._critic_network_2.trainable_variables

        # Compute gradients.
        policy_gradients = tape.gradient(policy_loss, policy_variables)
        critic_gradients_1 = tape.gradient(critic_loss_1, critic_variables_1)
        critic_gradients_2 = tape.gradient(critic_loss_2, critic_variables_2)

        # Delete the tape manually because of the persistent=True flag.
        del tape

        # Maybe clip gradients.
        if self._clipping:
            policy_gradients = tf.clip_by_global_norm(policy_gradients, 40.)[0]
            critic_gradients_1 = tf.clip_by_global_norm(critic_gradients_1, 40.)[0]
            critic_gradients_2 = tf.clip_by_global_norm(critic_gradients_2, 40.)[0]

        # Apply gradients.
        self._policy_optimizer.apply(policy_gradients, policy_variables)
        self._critic_optimizer_1.apply(critic_gradients_1, critic_variables_1)
        self._critic_optimizer_2.apply(critic_gradients_2, critic_variables_2)

        # Losses to track.
        return {
            'critic_loss_1': critic_loss_1,
            'critic_loss_2': critic_loss_2,
            'policy_loss': policy_loss,
        }

    def step(self):
        # Run the learning step.
        fetches = self._step()

        # Compute elapsed time.
        timestamp = time.time()
        elapsed_time = timestamp - self._timestamp if self._timestamp else 0
        self._timestamp = timestamp

        # Update our counts and record it.
        counts = self._counter.increment(steps=1, walltime=elapsed_time)
        fetches.update(counts)

        # Checkpoint and attempt to write the logs.
        self._checkpointer.save()
        self._logger.write(fetches)

    def get_variables(self, names: List[str]) -> List[List[np.ndarray]]:
        return [tf2_utils.to_numpy(self._variables[name]) for name in names]