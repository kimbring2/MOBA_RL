# coding=utf-8
# Copyright 2019 The SEED Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# python3
"""V-trace based SEED learner."""

import collections
import math
import os
import time
import observation
import networks
import gym
from gym.spaces import Dict, Discrete, Box, Tuple

import argparse
from absl import flags
from absl import logging

import grpc
import utils
import vtrace
from parametric_distribution import get_parametric_distribution_for_action_space

import tensorflow as tf

parser = argparse.ArgumentParser(description='Derk implementation')
parser.add_argument('--workspace_path', type=str, help='root directory')
arguments = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def compute_loss(logger, parametric_action_distribution, agent, 
                   agent_state, prev_actions, env_outputs, agent_outputs):
  learner_outputs, _ = agent(prev_actions, env_outputs, agent_state, unroll=True, is_training=True)

  # Use last baseline value (from the value function) to bootstrap.
  bootstrap_value = learner_outputs.baseline[-1]

  # At this point, we have unroll length + 1 steps. The last step is only used
  # as bootstrap value, so it's removed.
  agent_outputs = tf.nest.map_structure(lambda t: t[:-1], agent_outputs)
  rewards, done, _, _, _ = tf.nest.map_structure(lambda t: t[1:], env_outputs)
  learner_outputs = tf.nest.map_structure(lambda t: t[:-1], learner_outputs)

  max_abs_reward = 0.0
  if max_abs_reward:
    rewards = tf.clip_by_value(rewards, -max_abs_reward, max_abs_reward)

  discounting = 0.99
  discounts = tf.cast(~done, tf.float32) * discounting

  #tf.print("learner_outputs.policy_logits: ", learner_outputs.policy_logits)
  #tf.print("agent_outputs.action: ", agent_outputs.action)
  target_action_log_probs = parametric_action_distribution.log_prob(learner_outputs.policy_logits, agent_outputs.action)
  behaviour_action_log_probs = parametric_action_distribution.log_prob(agent_outputs.policy_logits, agent_outputs.action)

  # Compute V-trace returns and weights.
  lambda_ = 1.0
  vtrace_returns = vtrace.from_importance_weights(
      target_action_log_probs=target_action_log_probs, behaviour_action_log_probs=behaviour_action_log_probs,
      discounts=discounts, rewards=rewards, values=learner_outputs.baseline, bootstrap_value=bootstrap_value,
      lambda_=lambda_)

  # Policy loss based on Policy Gradients
  policy_loss = -tf.reduce_mean(target_action_log_probs * tf.stop_gradient(vtrace_returns.pg_advantages))

  # Value function loss
  baseline_cost = 0.5
  v_error = vtrace_returns.vs - learner_outputs.baseline
  v_loss = baseline_cost * 0.5 * tf.reduce_mean(tf.square(v_error))

  # Entropy reward
  entropy = tf.reduce_mean(parametric_action_distribution.entropy(learner_outputs.policy_logits))
  entropy_loss = tf.stop_gradient(agent.entropy_cost()) * -entropy

  # KL(old_policy|new_policy) loss
  kl = behaviour_action_log_probs - target_action_log_probs
  kl_cost = 0.0
  kl_loss = kl_cost * tf.reduce_mean(kl)

  # Entropy cost adjustment (Langrange multiplier style)
  target_entropy = None
  if target_entropy:
    entropy_adjustment_loss = agent.entropy_cost() * tf.stop_gradient(tf.reduce_mean(entropy) - target_entropy)
  else:
    entropy_adjustment_loss = 0. * agent.entropy_cost()  # to avoid None in grad

  total_loss = (policy_loss + v_loss + entropy_loss + kl_loss + entropy_adjustment_loss)

  # value function
  session = logger.log_session()
  logger.log(session, 'V/value function', tf.reduce_mean(learner_outputs.baseline))
  logger.log(session, 'V/L2 error', tf.sqrt(tf.reduce_mean(tf.square(v_error))))

  # losses
  logger.log(session, 'losses/policy', policy_loss)
  logger.log(session, 'losses/V', v_loss)
  logger.log(session, 'losses/entropy', entropy_loss)
  logger.log(session, 'losses/kl', kl_loss)
  logger.log(session, 'losses/total', total_loss)

  # policy
  dist = parametric_action_distribution.create_dist(learner_outputs.policy_logits)
  if hasattr(dist, 'scale'):
    logger.log(session, 'policy/std', tf.reduce_mean(dist.scale))

  logger.log(session, 'policy/max_action_abs(before_tanh)', tf.reduce_max(tf.abs(agent_outputs.action)))
  logger.log(session, 'policy/entropy', entropy)
  logger.log(session, 'policy/entropy_cost', agent.entropy_cost())
  logger.log(session, 'policy/kl(old|new)', tf.reduce_mean(kl))

  return total_loss, session


Unroll = collections.namedtuple('Unroll', 'agent_state prev_actions env_outputs agent_outputs')
EnvOutput = collections.namedtuple('EnvOutput', 'reward done observation abandoned episode_step')


def learner_loop():
  """Main learner loop.

  Args:
    create_env_fn: Callable that must return a newly created environment. The
      callable takes the task ID as argument - an arbitrary task ID of 0 will be
      passed by the learner. The returned environment should follow GYM's API.
      It is only used for infering tensor shapes. This environment will not be
      used to generate experience.
    create_agent_fn: Function that must create a new tf.Module with the neural
      network that outputs actions and new agent state given the environment
      observations and previous agent state. See dmlab.agents.ImpalaDeep for an
      example. The factory function takes as input the environment action and
      observation spaces and a parametric distribution over actions.
    create_optimizer_fn: Function that takes the final iteration as argument
      and must return a tf.keras.optimizers.Optimizer and a
      tf.keras.optimizers.schedules.LearningRateSchedule.
  """
  logging.info('Starting learner loop')
  #validate_config()

  num_training_tpus = 1
  settings = utils.init_learner_multi_host(num_training_tpus)
  strategy, hosts, training_strategy, encode, decode = settings

  env_output_specs = EnvOutput(
      tf.TensorSpec([], tf.float32, 'reward'),
      tf.TensorSpec([], tf.bool, 'done'),
      tf.TensorSpec([3,64], tf.float32, 'observation'),
      tf.TensorSpec([], tf.bool, 'abandoned'),
      tf.TensorSpec([], tf.int32, 'episode_step'),
  )

  action_specs = tf.TensorSpec([], tf.int64, 'action')
  agent_input_specs = (action_specs, env_output_specs)

  # Initialize agent and variables.
  parametric_action_distribution = get_parametric_distribution_for_action_space(Discrete(15))
  agent = networks.DrDerk(parametric_action_distribution)
  initial_agent_state = agent.initial_state(1)
  agent_state_specs = tf.nest.map_structure(lambda t: tf.TensorSpec(t.shape[1:], t.dtype), initial_agent_state)
  unroll_specs = [None]  # Lazy initialization.
  input_ = tf.nest.map_structure(lambda s: tf.zeros([1] + list(s.shape), s.dtype), agent_input_specs)
  input_ = encode(input_)

  with strategy.scope():
    @tf.function
    def create_variables(*args):
      return agent.get_action(*decode(args))

    initial_agent_output, _ = create_variables(*input_, initial_agent_state)
    if not hasattr(agent, 'entropy_cost'):
      entropy_cost_adjustment_speed = 10.
      mul = entropy_cost_adjustment_speed

      # Without the constraint, the param gradient may get rounded to 0 for very small values.
      entropy_cost = 0.00025
      agent.entropy_cost_param = tf.Variable(tf.math.log(entropy_cost) / mul,
                                                  constraint=lambda v: tf.clip_by_value(v, -20 / mul, 20 / mul),
                                                  trainable=True, 
                                                  dtype=tf.float32)

      agent.entropy_cost = lambda: tf.exp(mul * agent.entropy_cost_param)

    # Create optimizer.
    batch_size = 1
    unroll_length = 10
    num_action_repeats = 1
    total_environment_frames = int(1e9)
    iter_frame_ratio = (batch_size * unroll_length * num_action_repeats)
    final_iteration = int(math.ceil(total_environment_frames / iter_frame_ratio))

    def create_optimizer(unused_final_iteration):
      learning_rate = 0.00001
      learning_rate_fn = lambda iteration: learning_rate
      optimizer = tf.keras.optimizers.Adam(learning_rate)
      return optimizer, learning_rate_fn

    create_optimizer_fn = create_optimizer
    optimizer, learning_rate_fn = create_optimizer_fn(final_iteration)

    iterations = optimizer.iterations
    optimizer._create_hypers()  
    optimizer._create_slots(agent.trainable_variables)  

    # ON_READ causes the replicated variable to act as independent variables for
    # each replica.
    temp_grads = [
        tf.Variable(tf.zeros_like(v), trainable=False, synchronization=tf.VariableSynchronization.ON_READ)
        for v in agent.trainable_variables
    ]

  @tf.function
  def minimize(iterator):
    data = next(iterator)

    def compute_gradients(args):
      args = tf.nest.pack_sequence_as(unroll_specs[0], decode(args, data))
      with tf.GradientTape() as tape:
        loss, logs = compute_loss(logger, parametric_action_distribution, agent, *args)

      grads = tape.gradient(loss, agent.trainable_variables)
      for t, g in zip(temp_grads, grads):
        t.assign(g)

      return loss, logs

    loss, logs = training_strategy.run(compute_gradients, (data,))
    loss = training_strategy.experimental_local_results(loss)[0]

    def apply_gradients(_):
      optimizer.apply_gradients(zip(temp_grads, agent.trainable_variables))

    strategy.run(apply_gradients, (loss,))
    getattr(agent, 'end_of_training_step_callback', lambda: logging.info('end_of_training_step_callback not found'))()
    logger.step_end(logs, training_strategy, iter_frame_ratio)


  agent_output_specs = tf.nest.map_structure(lambda t: tf.TensorSpec(t.shape[1:], t.dtype), initial_agent_output)

  init_checkpoint = None
  workspace_path = arguments.workspace_path
  logdir = workspace_path + "/tboard/agent1"

  # Setup checkpointing and restore checkpoint.
  ckpt = tf.train.Checkpoint(agent=agent, optimizer=optimizer)
  if init_checkpoint is not None:
    tf.print('Loading initial checkpoint from %s...' % FLAGS.init_checkpoint)
    #ckpt.restore(FLAGS.init_checkpoint).assert_consumed()

  manager = tf.train.CheckpointManager(ckpt, logdir, max_to_keep=1, keep_checkpoint_every_n_hours=6)
  last_ckpt_time = 0  # Force checkpointing of the initial model.
  if manager.latest_checkpoint:
    logging.info('Restoring checkpoint: %s', manager.latest_checkpoint)
    #ckpt.restore(manager.latest_checkpoint).assert_consumed()
    last_ckpt_time = time.time()
  
  # Logging.
  summary_writer = tf.summary.create_file_writer(logdir, flush_millis=20000, max_queue=1000)
  logger = utils.ProgressLogger(summary_writer=summary_writer, starting_step=iterations * iter_frame_ratio)
  
  servers = []
  unroll_queues = []
  info_specs = (
      tf.TensorSpec([], tf.int64, 'episode_num_frames'),
      tf.TensorSpec([], tf.float32, 'episode_returns'),
      tf.TensorSpec([], tf.float32, 'episode_raw_returns'),
  )
  info_queue = utils.StructuredFIFOQueue(-1, info_specs)

  num_envs = 150
  inference_batch_size = 50
  def create_host(i, host, inference_devices):
    with tf.device(host):
      server_address = "localhost:8686"
      server = grpc.Server([server_address])

      store = utils.UnrollStore(num_envs, unroll_length, (action_specs, env_output_specs, agent_output_specs))

      env_run_ids = utils.Aggregator(num_envs, tf.TensorSpec([], tf.int64, 'run_ids'))
      env_infos = utils.Aggregator(num_envs, info_specs, 'env_infos')

      # First agent state in an unroll.
      first_agent_states = utils.Aggregator(num_envs, agent_state_specs, 'first_agent_states')

      # Current agent state and action.
      agent_states = utils.Aggregator(num_envs, agent_state_specs, 'agent_states')
      actions = utils.Aggregator(num_envs, action_specs, 'actions')

      unroll_specs[0] = Unroll(agent_state_specs, *store.unroll_specs)
      unroll_queue = utils.StructuredFIFOQueue(1, unroll_specs[0])

      def add_batch_size(ts):
        return tf.TensorSpec([inference_batch_size] + list(ts.shape), ts.dtype, ts.name)

      inference_specs = (
          tf.TensorSpec([], tf.int32, 'env_id'),
          tf.TensorSpec([], tf.int64, 'run_id'),
          env_output_specs,
          tf.TensorSpec([], tf.float32, 'raw_reward'),
      )

      inference_specs = tf.nest.map_structure(add_batch_size, inference_specs)
      def create_inference_fn(inference_device):
        @tf.function(input_signature=inference_specs)
        def inference(env_ids, run_ids, env_outputs, raw_rewards):
          # Reset the environments that had their first run or crashed.
          previous_run_ids = env_run_ids.read(env_ids)
          env_run_ids.replace(env_ids, run_ids)
          reset_indices = tf.where(tf.not_equal(previous_run_ids, run_ids))[:, 0]

          envs_needing_reset = tf.gather(env_ids, reset_indices)
          if tf.not_equal(tf.shape(envs_needing_reset)[0], 0):
            tf.print('Environment ids needing reset:', envs_needing_reset)

          store.reset(envs_needing_reset)

          env_infos.reset(envs_needing_reset)
          initial_agent_states = agent.initial_state(tf.shape(envs_needing_reset)[0])
          first_agent_states.replace(envs_needing_reset, initial_agent_states)
          agent_states.replace(envs_needing_reset, initial_agent_states)
          actions.reset(envs_needing_reset)

          #tf.debugging.assert_non_positive(tf.cast(env_outputs.abandoned, tf.int32), 'Abandoned done states are not supported in VTRACE.')

          # Update steps and return.
          env_infos.add(env_ids, (0, env_outputs.reward, raw_rewards))
          done_ids = tf.gather(env_ids, tf.where(env_outputs.done)[:, 0])
          if i == 0:
            info_queue.enqueue_many(env_infos.read(done_ids))

          env_infos.reset(done_ids)
          env_infos.add(env_ids, (num_action_repeats, 0., 0.))

          # Inference.
          prev_actions = actions.read(env_ids)
          #tf.print('env_outputs.shape:', env_outputs.shape)
          input_ = encode((prev_actions, env_outputs))
          prev_agent_states = agent_states.read(env_ids)

          with tf.device(inference_device):
            @tf.function
            def agent_inference(*args):
              return agent(*decode(args), is_training=False)

            agent_outputs, curr_agent_states = agent_inference(*input_, prev_agent_states)

          # Append the latest outputs to the unroll and insert completed unrolls in queue.
          completed_ids, unrolls = store.append(env_ids, (prev_actions, env_outputs, agent_outputs))

          # Unroll = collections.namedtuple('Unroll', 'agent_state prev_actions env_outputs agent_outputs')
          unrolls = Unroll(first_agent_states.read(completed_ids), *unrolls)
          unroll_queue.enqueue_many(unrolls)
          first_agent_states.replace(completed_ids, agent_states.read(completed_ids))

          # Update current state.
          agent_states.replace(env_ids, curr_agent_states)
          actions.replace(env_ids, agent_outputs.action)

          # Return environment actions to environments.
          return agent_outputs.action

        return inference


      with strategy.scope():
        server.bind([create_inference_fn(d) for d in inference_devices])

      server.start()
      unroll_queues.append(unroll_queue)
      servers.append(server)

  # hosts:  [('/cpu', ['/device:CPU:0'])]
  for i, (host, inference_devices) in enumerate(hosts):
    create_host(i, host, inference_devices)


  def dequeue(ctx):
    # Create batch (time major).
    dequeue_outputs = tf.nest.map_structure(
        lambda *args: tf.stack(args), 
        *[unroll_queues[ctx.input_pipeline_id].dequeue() for i in range(ctx.get_per_replica_batch_size(batch_size))]
      )

    dequeue_outputs = dequeue_outputs._replace(
        prev_actions=utils.make_time_major(dequeue_outputs.prev_actions),
        env_outputs=utils.make_time_major(dequeue_outputs.env_outputs),
        agent_outputs=utils.make_time_major(dequeue_outputs.agent_outputs)
      )

    dequeue_outputs = dequeue_outputs._replace(env_outputs=encode(dequeue_outputs.env_outputs))

    # tf.data.Dataset treats list leafs as tensors, so we need to flatten and repack.
    return tf.nest.flatten(dequeue_outputs)


  def dataset_fn(ctx):
    dataset = tf.data.Dataset.from_tensors(0).repeat(None)
    def _dequeue(_):
      return dequeue(ctx)

    return dataset.map(_dequeue, num_parallel_calls=ctx.num_replicas_in_sync // len(hosts))


  dataset = training_strategy.experimental_distribute_datasets_from_function(dataset_fn)
  it = iter(dataset)

  def additional_logs():
    tf.summary.scalar('learning_rate', learning_rate_fn(iterations))
    n_episodes = info_queue.size()
    log_episode_frequency = 1
    n_episodes -= n_episodes % log_episode_frequency
    if tf.not_equal(n_episodes, 0):
      episode_stats = info_queue.dequeue_many(n_episodes)
      episode_keys = [
          'episode_num_frames', 'episode_return', 'episode_raw_return'
      ]
      for key, values in zip(episode_keys, episode_stats):
        for value in tf.split(values, values.shape[0] // log_episode_frequency):
          tf.summary.scalar(key, tf.reduce_mean(value))

      for (frames, ep_return, raw_return) in zip(*episode_stats):
        logging.info('Return: %f Raw return: %f Frames: %i', ep_return, raw_return, frames)


  # Execute learning.
  while iterations < final_iteration:
    # Save checkpoint.
    current_time = time.time()
    save_checkpoint_secs = 1800

    tf.print(current_time)
    if current_time - last_ckpt_time >= save_checkpoint_secs:
      manager.save()
      # Apart from checkpointing, we also save the full model (including
      # the graph). This way we can load it after the code/parameters changed.
      tf.saved_model.save(agent, os.path.join(logdir, 'saved_model'))
      last_ckpt_time = current_time

    minimize(it)

  logger.shutdown()
  manager.save()
  tf.saved_model.save(agent, os.path.join(logdir, 'saved_model'))
  for server in servers:
    server.shutdown()

  for unroll_queue in unroll_queues:
    unroll_queue.close()

learner_loop()