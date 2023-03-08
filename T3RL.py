#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
from builtins import iter
import argparse
from absl import app
import os
import time
from tf_agents.environments import tf_py_environment
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.metrics import tf_metrics
from tf_agents.eval import metric_utils
from tf_agents.agents.dqn import dqn_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.policies import policy_saver
import tensorflow as tf
from tensorflow.python.eager.context import eager_mode
from policy_network import ActorNet
from T3Env import T3Env
from T3NashEnv import T3NashEnv

warnings.filterwarnings("ignore", category=DeprecationWarning)
from Logger import create_logger


def get_logger(name):
    return create_logger(name)


def create_networks(tf_env):
    fc_layer_params = (10, 5)
    policy_net = ActorNet(
        tf_env.observation_spec()["state"],
        tf_env.action_spec(),
        fc_layer_params=fc_layer_params,
    )
    return policy_net


def create_envs(board_size, nash, logger):
    env = T3NashEnv if nash else T3Env
    train_py_env = env(board_size, name="train", logger=logger)
    eval_py_env = env(board_size, name="eval", logger=logger, verbose=True)

    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    return train_env, eval_env


def train_eval_t3(args):

    nash = args["nash"]
    train = args["train"]
    eval_ = args["evaluate"]
    num_parallel_envs = 1
    num_train_epochs = 25
    num_iterations = 10000 if train else 0
    summary_interval = num_train_epochs
    checkpoint_interval = num_train_epochs * 10
    num_eval_episodes = 1 if train else 5  # @param
    eval_interval = num_parallel_envs * num_train_epochs * 2
    log_interval = num_train_epochs  # @param
    gamma = 0.99
    use_tf_functions = True
    normalize_rewards = True
    profile_training = False
    batched_training = True
    replay_buffer_capacity = 30000
    batch_size = 32

    logger = get_logger("Nash") if nash else get_logger("RandomPolicy")

    if batched_training:
        num_training_batches = 1
    else:
        num_training_batches = num_parallel_envs
        checkpoint_interval *= num_parallel_envs

    # Create directories for plots, logs, graphs, tensorboard data
    root_dir = os.path.expanduser(os.getcwd())
    tag = "self_play" if nash else "random_other_player"
    log_dir = os.path.join(root_dir, f"logs_{tag}")
    train_dir = os.path.join(log_dir, "train")
    eval_dir = os.path.join(log_dir, "eval")
    saved_model_dir = os.path.join(log_dir, "policy_saved_model")

    # create env here
    train_env, eval_env = create_envs(9, nash, logger)

    if train:
        logger.info("Training mode: ")
        logger.info(f"Num training episodes: {num_iterations}")
        if nash:
            logger.info("\tSelf play: 1")
        else:
            logger.info("\tRandom policy for other player: 1")

    if eval_:
        logger.info("Evaluation mode: ")
        if nash:
            logger.info("\tSelf play: 1")
        else:
            logger.info("\tRandom policy for other player: 1")

    with eager_mode():

        # initialize globaql step
        global_step = tf.compat.v1.train.get_or_create_global_step()

        # Initlialize metrics logger
        summaries_flush_secs = 10
        train_summary_writer = tf.compat.v2.summary.create_file_writer(
            train_dir, flush_millis=summaries_flush_secs * 1000
        )
        train_summary_writer.set_as_default()

        eval_summary_writer = tf.compat.v2.summary.create_file_writer(
            eval_dir, flush_millis=summaries_flush_secs * 1000
        )
        eval_metrics = [
            tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
            tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes),
        ]

        environment_steps_metric = tf_metrics.EnvironmentSteps()
        step_metrics = [
            tf_metrics.NumberOfEpisodes(),
            environment_steps_metric,
        ]

        train_metrics = step_metrics + [
            tf_metrics.AverageReturnMetric(batch_size=train_env.batch_size),
            tf_metrics.AverageEpisodeLengthMetric(batch_size=train_env.batch_size),
        ]

        policy_net = create_networks(train_env)

        def observation_and_action_constraint_splitter(observation):
            return observation["state"], observation["mask"]

        tf_agent = dqn_agent.DqnAgent(
            train_env.time_step_spec(),
            train_env.action_spec(),
            q_network=policy_net,
            target_update_period=10,
            observation_and_action_constraint_splitter=observation_and_action_constraint_splitter,
            optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
            debug_summaries=True,
            epsilon_greedy=0.1,
            train_step_counter=global_step,
            gradient_clipping=True,
            gamma=gamma,
        )

        tf_agent.initialize()

        environment_steps_metric = tf_metrics.EnvironmentSteps()
        step_metrics = [
            tf_metrics.NumberOfEpisodes(),
            environment_steps_metric,
        ]
        env_batch_size = train_env.batch_size
        train_metrics = step_metrics + [
            tf_metrics.AverageReturnMetric(batch_size=env_batch_size),
            tf_metrics.AverageEpisodeLengthMetric(batch_size=env_batch_size),
        ]

        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            tf_agent.collect_data_spec,
            batch_size=env_batch_size,
            max_length=replay_buffer_capacity,
        )

        eval_policy = tf_agent.policy
        train_checkpointer = common.Checkpointer(
            ckpt_dir=train_dir,
            agent=tf_agent,
            global_step=global_step,
            metrics=metric_utils.MetricsGroup(train_metrics, "train_metrics"),
        )

        policy_checkpointer = common.Checkpointer(
            ckpt_dir=os.path.join(train_dir, "policy"),
            policy=eval_policy,
            global_step=global_step,
        )
        saved_model = policy_saver.PolicySaver(eval_policy, train_step=global_step)
        train_checkpointer.initialize_or_restore()

        replay_observer = [replay_buffer.add_batch] + train_metrics
        collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
            train_env, tf_agent.collect_policy, replay_observer, num_episodes=1
        )

        def evaluate():
            metric_utils.eager_compute(
                eval_metrics,
                eval_env,
                tf_agent.policy,
                num_eval_episodes,
                global_step,
                eval_summary_writer,
                "Metrics",
            )

        if use_tf_functions:
            collect_driver.run = common.function(collect_driver.run, autograph=True)
            tf_agent.train = common.function(tf_agent.train, autograph=True)

        dataset = replay_buffer.as_dataset(
            sample_batch_size=batch_size, num_steps=2
        ).prefetch(3)
        iterator = iter(dataset)

        collect_time = 0
        train_time = 0
        timed_at_step = global_step.numpy()

        for _ in range(num_iterations):

            start_time = time.time()
            collect_driver.run()
            collect_time += time.time() - start_time

            start_time = time.time()
            trajectories, _ = next(iterator)
            train_loss, _ = tf_agent.train(experience=trajectories)
            train_time += time.time() - start_time
            # step = tf_agent.train_step_counter.numpy()
            step = global_step.numpy()

            for train_metric in train_metrics:
                train_metric.tf_summaries(
                    train_step=global_step, step_metrics=step_metrics
                )

            if step % log_interval == 0:
                logger.info("step = %d, loss = %f", step, train_loss)
                steps_per_sec = (step - timed_at_step) / (collect_time + train_time)
                logger.info("%.3f steps/sec", steps_per_sec)
                logger.info(
                    "collect_time = {}, train_time = {}".format(
                        collect_time, train_time
                    )
                )

                tf.compat.v2.summary.scalar(
                    name="global_steps_per_sec",
                    data=steps_per_sec,
                    step=global_step,
                )

                timed_at_step = step
                collect_time = 0
                train_time = 0

            if step % eval_interval == 0 and step > 0:
                print("Entering Evaluation phase")
                evaluate()

            if step % checkpoint_interval == 0:
                train_checkpointer.save(global_step=step)
                policy_checkpointer.save(global_step=step)
                saved_model_path = os.path.join(
                    saved_model_dir, "policy_" + ("%d" % step).zfill(9)
                )
                saved_model.save(saved_model_path)

        # one final evlauation
        evaluate()
    if eval_:
        evaluate()


def parse_args():

    parser = argparse.ArgumentParser(
        description="Process input arguments for RL tic-tac-toe.", add_help=True
    )
    parser.add_argument("-t", "--train", action="store_true", help="Train the RL agent")
    parser.add_argument(
        "-e", "--evaluate", action="store_true", help="Evaluate the RL agent"
    )
    parser.add_argument(
        "-s", "--self_play", action="store_true", help="Train the agent for self-play"
    )
    args = parser.parse_args()
    train = args.train
    evaluate = args.evaluate
    nash = args.self_play
    return {
        "train": train,
        "evaluate": evaluate,
        "nash": nash,
    }


def main(args):
    train_eval_t3(args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
