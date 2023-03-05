#!/usr/bin/env python

from tf_agents.networks import network
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step
import numpy as np
from tf_agents.networks import normal_projection_network
from tf_agents.networks import categorical_projection_network
from tf_agents.utils import nest_utils
from tf_agents.specs import array_spec
from tf_agents.networks import utils
import tensorflow as tf

from T3Env import T3Env


class ActorNet(network.Network):
    def __init__(
        self,
        input_tensor_spec,
        output_tensor_spec,
        fc_layer_params,
        name="ActorNet",
    ):
        super(ActorNet, self).__init__(
            input_tensor_spec=input_tensor_spec, state_spec=(), name=name
        )
        num_actions = output_tensor_spec.maximum - output_tensor_spec.minimum + 1
        # Define a helper function to create Dense layers configured with the right
        # activation and kernel initializer.
        def __dense_layer(num_units):
            return tf.keras.layers.Dense(
                num_units,
                activation=tf.keras.activations.relu,
                kernel_initializer=tf.keras.initializers.VarianceScaling(
                    scale=2.0, mode="fan_in", distribution="truncated_normal"
                ),
            )

        # QNetwork consists of a sequence of Dense layers followed by a dense layer
        # with `num_actions` units to generate one q_value per available action as
        # its output.
        self._sub_layers = [__dense_layer(num_units) for num_units in fc_layer_params]
        self._sub_layers.append(
            tf.keras.layers.Dense(
                num_actions,
                activation=None,
                kernel_initializer=tf.keras.initializers.RandomUniform(
                    minval=-0.03, maxval=0.03
                ),
                bias_initializer=tf.keras.initializers.Constant(-0.2),
            )
        )

    def call(self, observations, step_type, network_state=None, training=False):

        del step_type
        outer_rank = nest_utils.get_outer_rank(observations, self.input_tensor_spec)
        batch_squash = utils.BatchSquash(outer_rank)
        observations = tf.nest.map_structure(batch_squash.flatten, observations)

        state = observations

        for layer in self._sub_layers:
            state = layer(state, training=training)

        state = tf.nest.map_structure(batch_squash.unflatten, state)
        output = state
        return output, network_state


if __name__ == "__main__":
    env = T3Env(9, verbose=True)
    fc_layer_params = [10, 5]
    q_net = ActorNet(
        input_tensor_spec=env.observation_spec()['state'],
        output_tensor_spec=env.action_spec(),
        fc_layer_params=fc_layer_params,
    )
    q_net.create_variables()
    i = env.reset().observation['state']
    a, _ = q_net(i, None, ())

    env.verbose = True
    for i in range(5):
        action = np.argmax(a)
        print(a)
        print("action = ", action)
        ts = env.step(action)
        a, _ = q_net(ts.observation['state'], None, ())
