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
import numpy as np
import observation
import networks
import gym
from gym.spaces import Dict, Discrete, Box, Tuple

from absl import flags
from absl import logging

import argparse
import grpc
import utils
import vtrace
from parametric_distribution import get_parametric_distribution_for_action_space

import tensorflow as tf

parser = argparse.ArgumentParser(description='OpenAI Five implementation')
parser.add_argument('--env_number', type=int, default=1, help='Number of environment')
parser.add_argument('--train', type=bool, default=True, help='Whether training model or not')
arguments = parser.parse_args()

'''
gpus = tf.config.experimental.list_physical_devices('GPU')
print("gpus: ", gpus)
if len(gpus) != 0:
  tf.config.experimental.set_virtual_device_configuration(gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3000)])
'''

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

TICKS_PER_OBSERVATION = 15 
TICKS_PER_SECOND = 30
MAX_MOVE_SPEED = 550
MAX_MOVE_IN_OBS = (MAX_MOVE_SPEED / TICKS_PER_SECOND) * TICKS_PER_OBSERVATION
N_MOVE_ENUMS = 9
MOVE_ENUMS = np.arange(N_MOVE_ENUMS, dtype=np.float32) - int(N_MOVE_ENUMS / 2)
MOVE_ENUMS *= MAX_MOVE_IN_OBS / (N_MOVE_ENUMS - 1) * 2
OBSERVATIONS_PER_SECOND = TICKS_PER_SECOND / TICKS_PER_OBSERVATION
MAX_UNITS = 5 + 5 + 16 + 16 + 1 + 1

def compute_loss(logger, enum_parametric_action_distribution, 
                 x_parametric_action_distribution, y_parametric_action_distribution, 
                 target_unit_parametric_action_distribution, ability_parametric_action_distribution,
                 item_parametric_action_distribution, agent, agent_state, prev_actions, env_outputs, agent_outputs):
  learner_outputs, _ = agent(prev_actions, env_outputs, agent_state, unroll=True, is_training=True)

  # Use last baseline value (from the value function) to bootstrap.
  bootstrap_value = learner_outputs.baseline[-1]

  # At this point, we have unroll length + 1 steps. The last step is only used
  # as bootstrap value, so it's removed.
  agent_outputs = tf.nest.map_structure(lambda t: t[:-1], agent_outputs)

  # reward done env allied_heroes enemy_heroes allied_nonheroes enemy_nonheroes allied_towers enemy_towers abandoned episode_step
  rewards, done, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = tf.nest.map_structure(lambda t: t[1:], env_outputs)
  learner_outputs = tf.nest.map_structure(lambda t: t[:-1], learner_outputs)

  max_abs_reward = 0.0
  if max_abs_reward:
    rewards = tf.clip_by_value(rewards, -max_abs_reward, max_abs_reward)

  discounting = 0.99
  discounts = tf.cast(~done, tf.float32) * discounting

  enum_target_action_log_probs = enum_parametric_action_distribution.log_prob(learner_outputs.enum_logits, agent_outputs.enum)
  enum_behaviour_action_log_probs = enum_parametric_action_distribution.log_prob(agent_outputs.enum_logits, agent_outputs.enum)

  x_target_action_log_probs = x_parametric_action_distribution.log_prob(learner_outputs.x_logits, agent_outputs.x)
  x_behaviour_action_log_probs = x_parametric_action_distribution.log_prob(agent_outputs.x_logits, agent_outputs.x)

  y_target_action_log_probs = y_parametric_action_distribution.log_prob(learner_outputs.y_logits, agent_outputs.y)
  y_behaviour_action_log_probs = y_parametric_action_distribution.log_prob(agent_outputs.y_logits, agent_outputs.y)

  target_unit_target_action_log_probs = target_unit_parametric_action_distribution.log_prob(learner_outputs.target_unit_logits, agent_outputs.target_unit)
  target_unit_behaviour_action_log_probs = target_unit_parametric_action_distribution.log_prob(agent_outputs.target_unit_logits, agent_outputs.target_unit)

  ability_target_action_log_probs = ability_parametric_action_distribution.log_prob(learner_outputs.ability_logits, agent_outputs.ability)
  ability_behaviour_action_log_probs = ability_parametric_action_distribution.log_prob(agent_outputs.ability_logits, agent_outputs.ability)

  item_target_action_log_probs = item_parametric_action_distribution.log_prob(learner_outputs.item_logits, agent_outputs.item)
  item_behaviour_action_log_probs = item_parametric_action_distribution.log_prob(agent_outputs.item_logits, agent_outputs.item)

  # Compute V-trace returns and weights.
  lambda_ = 1.0
  enum_vtrace_returns = vtrace.from_importance_weights(
      target_action_log_probs=enum_target_action_log_probs, behaviour_action_log_probs=enum_behaviour_action_log_probs,
      discounts=discounts, rewards=rewards, values=learner_outputs.baseline, bootstrap_value=bootstrap_value,
      lambda_=lambda_)

  x_vtrace_returns = vtrace.from_importance_weights(
      target_action_log_probs=x_target_action_log_probs, behaviour_action_log_probs=x_behaviour_action_log_probs,
      discounts=discounts, rewards=rewards, values=learner_outputs.baseline, bootstrap_value=bootstrap_value,
      lambda_=lambda_)

  y_vtrace_returns = vtrace.from_importance_weights(
      target_action_log_probs=y_target_action_log_probs, behaviour_action_log_probs=y_behaviour_action_log_probs,
      discounts=discounts, rewards=rewards, values=learner_outputs.baseline, bootstrap_value=bootstrap_value,
      lambda_=lambda_)

  target_unit_vtrace_returns = vtrace.from_importance_weights(
      target_action_log_probs=target_unit_target_action_log_probs, behaviour_action_log_probs=target_unit_behaviour_action_log_probs,
      discounts=discounts, rewards=rewards, values=learner_outputs.baseline, bootstrap_value=bootstrap_value,
      lambda_=lambda_)

  ability_vtrace_returns = vtrace.from_importance_weights(
      target_action_log_probs=ability_target_action_log_probs, behaviour_action_log_probs=ability_behaviour_action_log_probs,
      discounts=discounts, rewards=rewards, values=learner_outputs.baseline, bootstrap_value=bootstrap_value,
      lambda_=lambda_)

  item_vtrace_returns = vtrace.from_importance_weights(
      target_action_log_probs=item_target_action_log_probs, behaviour_action_log_probs=item_behaviour_action_log_probs,
      discounts=discounts, rewards=rewards, values=learner_outputs.baseline, bootstrap_value=bootstrap_value,
      lambda_=lambda_)

  # Policy loss based on Policy Gradients
  enum_policy_loss = -tf.reduce_mean(enum_target_action_log_probs * tf.stop_gradient(enum_vtrace_returns.pg_advantages))
  x_policy_loss = -tf.reduce_mean(x_target_action_log_probs * tf.stop_gradient(x_vtrace_returns.pg_advantages))
  y_policy_loss = -tf.reduce_mean(y_target_action_log_probs * tf.stop_gradient(y_vtrace_returns.pg_advantages))
  target_unit_policy_loss = -tf.reduce_mean(target_unit_target_action_log_probs * tf.stop_gradient(target_unit_vtrace_returns.pg_advantages))
  ability_policy_loss = -tf.reduce_mean(ability_target_action_log_probs * tf.stop_gradient(ability_vtrace_returns.pg_advantages))
  item_policy_loss = -tf.reduce_mean(item_target_action_log_probs * tf.stop_gradient(item_vtrace_returns.pg_advantages))

  # Value function loss
  baseline_cost = 0.5
  enum_v_error = enum_vtrace_returns.vs - learner_outputs.baseline
  enum_v_loss = baseline_cost * 0.5 * tf.reduce_mean(tf.square(enum_v_error))

  x_v_error = x_vtrace_returns.vs - learner_outputs.baseline
  x_v_loss = baseline_cost * 0.5 * tf.reduce_mean(tf.square(x_v_error))

  y_v_error = y_vtrace_returns.vs - learner_outputs.baseline
  y_v_loss = baseline_cost * 0.5 * tf.reduce_mean(tf.square(y_v_error))

  target_unit_v_error = target_unit_vtrace_returns.vs - learner_outputs.baseline
  target_unit_v_loss = baseline_cost * 0.5 * tf.reduce_mean(tf.square(target_unit_v_error))

  ability_v_error = ability_vtrace_returns.vs - learner_outputs.baseline
  ability_v_loss = baseline_cost * 0.5 * tf.reduce_mean(tf.square(ability_v_error))

  item_v_error = item_vtrace_returns.vs - learner_outputs.baseline
  item_v_loss = baseline_cost * 0.5 * tf.reduce_mean(tf.square(item_v_error))

  # Entropy reward
  enum_entropy = tf.reduce_mean(enum_parametric_action_distribution.entropy(learner_outputs.enum_logits))
  enum_entropy_loss = tf.stop_gradient(agent.entropy_cost()) * -enum_entropy

  x_entropy = tf.reduce_mean(x_parametric_action_distribution.entropy(learner_outputs.x_logits))
  x_entropy_loss = tf.stop_gradient(agent.entropy_cost()) * -x_entropy

  y_entropy = tf.reduce_mean(y_parametric_action_distribution.entropy(learner_outputs.y_logits))
  y_entropy_loss = tf.stop_gradient(agent.entropy_cost()) * -y_entropy

  target_unit_entropy = tf.reduce_mean(target_unit_parametric_action_distribution.entropy(learner_outputs.target_unit_logits))
  target_unit_entropy_loss = tf.stop_gradient(agent.entropy_cost()) * -target_unit_entropy

  ability_entropy = tf.reduce_mean(ability_parametric_action_distribution.entropy(learner_outputs.ability_logits))
  ability_entropy_loss = tf.stop_gradient(agent.entropy_cost()) * -ability_entropy

  item_entropy = tf.reduce_mean(item_parametric_action_distribution.entropy(learner_outputs.item_logits))
  item_entropy_loss = tf.stop_gradient(agent.entropy_cost()) * -item_entropy

  # KL(old_policy|new_policy) loss
  kl = enum_behaviour_action_log_probs - enum_target_action_log_probs
  kl_cost = 0.0
  kl_loss = kl_cost * tf.reduce_mean(kl)

  # Entropy cost adjustment (Langrange multiplier style)
  target_entropy = None
  if target_entropy:
    entropy_adjustment_loss = agent.entropy_cost() * tf.stop_gradient(tf.reduce_mean(entropy) - target_entropy)
  else:
    entropy_adjustment_loss = 0. * agent.entropy_cost()  # to avoid None in grad

  enum_total_loss = (enum_policy_loss + enum_v_loss + enum_entropy_loss + kl_loss + entropy_adjustment_loss)
  x_total_loss = (x_policy_loss + x_v_loss + x_entropy_loss + kl_loss + entropy_adjustment_loss)
  y_total_loss = (y_policy_loss + y_v_loss + y_entropy_loss + kl_loss + entropy_adjustment_loss)
  target_unit_total_loss = (target_unit_policy_loss + target_unit_v_loss + target_unit_entropy_loss + kl_loss + entropy_adjustment_loss)
  ability_total_loss = (ability_policy_loss + ability_v_loss + ability_entropy_loss + kl_loss + entropy_adjustment_loss)
  item_total_loss = (item_policy_loss + item_v_loss + item_entropy_loss + kl_loss + entropy_adjustment_loss)

  total_loss = enum_total_loss + x_total_loss + y_total_loss + target_unit_total_loss + ability_total_loss + item_total_loss

  # value function
  session = logger.log_session()
  #logger.log(session, 'V/value function', tf.reduce_mean(learner_outputs.baseline))
  #logger.log(session, 'V/L2 error', tf.sqrt(tf.reduce_mean(tf.square(v_error))))

  # losses
  #logger.log(session, 'losses/policy', policy_loss)
  #logger.log(session, 'losses/V', v_loss)
  #ogger.log(session, 'losses/entropy', entropy_loss)
  #logger.log(session, 'losses/kl', kl_loss)
  #logger.log(session, 'losses/total', total_loss)

  # policy
  #dist = enum_parametric_action_distribution.create_dist(learner_outputs.enum_logits)
  #if hasattr(dist, 'scale'):
  #  logger.log(session, 'policy/std', tf.reduce_mean(dist.scale))

  #logger.log(session, 'policy/max_action_abs(before_tanh)', tf.reduce_max(tf.abs(agent_outputs.enum)))
  #logger.log(session, 'policy/entropy', entropy)
  #logger.log(session, 'policy/entropy_cost', agent.entropy_cost())
  #logger.log(session, 'policy/kl(old|new)', tf.reduce_mean(kl))

  return total_loss, session


Unroll = collections.namedtuple('Unroll', 'agent_state prev_actions env_outputs agent_outputs')
EnvOutput = collections.namedtuple('EnvOutput', 'reward done env allied_heroes enemy_heroes allied_nonheroes enemy_nonheroes \
                                    allied_towers enemy_towers enum_mask x_mask y_mask target_unit_mask ability_mask \
                                    item_mask abandoned episode_step')
Action = collections.namedtuple('Action', 'enum_logits x_logits y_logits target_unit_logits ability_logits item_logits')


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

  num_training_tpus = 0
  settings = utils.init_learner_multi_host(num_training_tpus)
  strategy, hosts, training_strategy, encode, decode = settings
  
  env_output_specs = EnvOutput(
      tf.TensorSpec([], tf.float32, 'reward'),
      tf.TensorSpec([], tf.bool, 'done'),
      tf.TensorSpec([3], tf.float32, 'env_observation'),
      tf.TensorSpec([5,29], tf.float32, 'allied_heroes_observation'),
      tf.TensorSpec([5,29], tf.float32, 'enemy_heroes_observation'),
      tf.TensorSpec([16,29], tf.float32, 'allied_nonheroes_observation'),
      tf.TensorSpec([16,29], tf.float32, 'enemy_nonheroes_observation'),
      tf.TensorSpec([1,29], tf.float32, 'allied_towers_observation'),
      tf.TensorSpec([1,29], tf.float32, 'enemy_towers_observation'),
      tf.TensorSpec([5], tf.float32, 'enum_mask'),
      tf.TensorSpec([9], tf.float32, 'x_mask'),
      tf.TensorSpec([9], tf.float32, 'y_mask'),
      tf.TensorSpec([44], tf.float32, 'target_unit_mask'),
      tf.TensorSpec([3], tf.float32, 'ability_mask'),
      tf.TensorSpec([6], tf.float32, 'item_mask'),
      tf.TensorSpec([], tf.bool, 'abandoned'),
      tf.TensorSpec([], tf.int32, 'episode_step'),
  )
  
  action_specs = Action(
      tf.TensorSpec([], tf.int32, 'enum'),
      tf.TensorSpec([], tf.int32, 'x'),
      tf.TensorSpec([], tf.int32, 'y'),
      tf.TensorSpec([], tf.int32, 'target_unit'),
      tf.TensorSpec([], tf.int32, 'ability'),
      tf.TensorSpec([], tf.int32, 'item')
  )
  
  agent_input_specs = (action_specs, env_output_specs)

  # Initialize agent and variables.
  enum_parametric_action_distribution = get_parametric_distribution_for_action_space(Discrete(5))
  x_parametric_action_distribution = get_parametric_distribution_for_action_space(Discrete(N_MOVE_ENUMS))
  y_parametric_action_distribution = get_parametric_distribution_for_action_space(Discrete(N_MOVE_ENUMS))
  target_unit_parametric_action_distribution = get_parametric_distribution_for_action_space(Discrete(MAX_UNITS))
  ability_parametric_action_distribution = get_parametric_distribution_for_action_space(Discrete(3))
  item_parametric_action_distribution = get_parametric_distribution_for_action_space(Discrete(6))

  agent = networks.Dota(enum_parametric_action_distribution, x_parametric_action_distribution, 
                        y_parametric_action_distribution, target_unit_parametric_action_distribution,
                        ability_parametric_action_distribution, item_parametric_action_distribution)
  initial_agent_state = agent.initial_state(1)
  agent_state_specs = tf.nest.map_structure(lambda t: tf.TensorSpec(t.shape[1:], t.dtype), initial_agent_state)
  unroll_specs = [None]  # Lazy initialization.
  input_ = tf.nest.map_structure(lambda s: tf.zeros([1] + list(s.shape), s.dtype), agent_input_specs)
  input_ = encode(input_)

  with strategy.scope():
    @tf.function
    def create_variables(*args):
      return agent.get_action(*decode(args))

    #print("input_: ", input_)
    #print("initial_agent_state: ", initial_agent_state)
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
    batch_size = 4
    unroll_length = 50
    num_action_repeats = 1
    total_environment_frames = int(1e9)
    iter_frame_ratio = (batch_size * unroll_length * num_action_repeats)
    final_iteration = int(math.ceil(total_environment_frames / iter_frame_ratio))

    def create_optimizer(unused_final_iteration):
      learning_rate = 0.0001
      learning_rate_fn = lambda iteration: learning_rate
      optimizer = tf.keras.optimizers.Adam(learning_rate, epsilon=1e-04)

      return optimizer, learning_rate_fn

    create_optimizer_fn = create_optimizer
    optimizer, learning_rate_fn = create_optimizer_fn(final_iteration)

    iterations = optimizer.iterations
    optimizer._create_hypers()  
    optimizer._create_slots(agent.trainable_variables)  

    # ON_READ causes the replicated variable to act as independent variables for each replica.
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
        loss, logs = compute_loss(logger, enum_parametric_action_distribution, 
                                  x_parametric_action_distribution, 
                                  y_parametric_action_distribution, target_unit_parametric_action_distribution,
                                  ability_parametric_action_distribution, item_parametric_action_distribution,
                                  agent, *args)

      grads = tape.gradient(loss, agent.trainable_variables)
      grads_norm = tf.linalg.global_norm(grads)
      #print("grads_norm: ", grads_norm)
      grads, _ = tf.clip_by_global_norm(grads, 1.0)
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


  #print("initial_agent_output: " + str(initial_agent_output))
  agent_output_specs = tf.nest.map_structure(lambda t: tf.TensorSpec(t.shape[1:], t.dtype), initial_agent_output)

  init_checkpoint = None
  logdir = "model_omninight"

  # Setup checkpointing and restore checkpoint.
  ckpt = tf.train.Checkpoint(agent=agent, optimizer=optimizer)
  if init_checkpoint is not None:
    tf.print('Loading initial checkpoint from %s...' % FLAGS.init_checkpoint)
    ckpt.restore(FLAGS.init_checkpoint).assert_consumed()

  manager = tf.train.CheckpointManager(ckpt, logdir, max_to_keep=1, keep_checkpoint_every_n_hours=6)
  last_ckpt_time = 0  # Force checkpointing of the initial model.
  if manager.latest_checkpoint:
    logging.info('Restoring checkpoint: %s', manager.latest_checkpoint)
    ckpt.restore(manager.latest_checkpoint).assert_consumed()
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
  info_queue = utils.StructuredFIFOQueue(100, info_specs)

  num_envs = arguments.env_number
  inference_batch_size = 1
  def create_host(i, host, inference_devices):
    with tf.device(host):
      server_address = "localhost:8688"
      server = grpc.Server([server_address])

      store = utils.UnrollStore(num_envs, unroll_length, (action_specs, env_output_specs, agent_output_specs))

      env_run_ids = utils.Aggregator(num_envs, tf.TensorSpec([], tf.int64, 'run_ids'))
      env_infos = utils.Aggregator(num_envs, info_specs, 'env_infos')

      # First agent state in an unroll.
      first_agent_states = utils.Aggregator(num_envs, agent_state_specs, 'first_agent_states')

      # Current agent state and action.
      agent_states = utils.Aggregator(num_envs, agent_state_specs, 'agent_states')

      enum_actions = utils.Aggregator(num_envs, action_specs[0], 'actions_enum')
      x_actions = utils.Aggregator(num_envs, action_specs[1], 'actions_x')
      y_actions = utils.Aggregator(num_envs, action_specs[2], 'actions_y')
      target_unit_actions = utils.Aggregator(num_envs, action_specs[3], 'actions_target_unit')
      ability_actions = utils.Aggregator(num_envs, action_specs[4], 'actions_ability')
      item_actions = utils.Aggregator(num_envs, action_specs[5], 'actions_item')

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
          enum_actions.reset(envs_needing_reset)
          x_actions.reset(envs_needing_reset)
          y_actions.reset(envs_needing_reset)
          target_unit_actions.reset(envs_needing_reset)
          ability_actions.reset(envs_needing_reset)
          item_actions.reset(envs_needing_reset)

          #tf.debugging.assert_non_positive(tf.cast(env_outputs.abandoned, tf.int32), 'Abandoned done states are not supported in VTRACE.')

          # Update steps and return.
          env_infos.add(env_ids, (0, env_outputs.reward, raw_rewards))

          done_ids = tf.gather(env_ids, tf.where(env_outputs.done)[:, 0])

          if i == 0:
            info_queue.enqueue_many(env_infos.read(done_ids))

          env_infos.reset(done_ids)
          env_infos.add(env_ids, (num_action_repeats, 0., 0.))

          # Inference.
          prev_enum_actions = enum_actions.read(env_ids)
          prev_x_actions = x_actions.read(env_ids)
          prev_y_actions = y_actions.read(env_ids)
          prev_target_unit_actions = target_unit_actions.read(env_ids)
          prev_ability_actions = ability_actions.read(env_ids)
          prev_item_actions = item_actions.read(env_ids)

          input_ = encode(((prev_enum_actions, prev_x_actions, prev_y_actions, prev_target_unit_actions, prev_ability_actions,
                            prev_item_actions), env_outputs))
          prev_agent_states = agent_states.read(env_ids)

          with tf.device(inference_device):
            @tf.function
            def agent_inference(*args):
              return agent(*decode(args), is_training=False)

            agent_outputs, curr_agent_states = agent_inference(*input_, prev_agent_states)

          # Append the latest outputs to the unroll and insert completed unrolls in queue.
          completed_ids, unrolls = store.append(env_ids, ((prev_enum_actions, prev_x_actions, prev_y_actions, 
            prev_target_unit_actions, prev_ability_actions, prev_item_actions), env_outputs, agent_outputs))

          unrolls = Unroll(first_agent_states.read(completed_ids), *unrolls)
          unroll_queue.enqueue_many(unrolls)
          first_agent_states.replace(completed_ids, agent_states.read(completed_ids))

          # Update current state.
          agent_states.replace(env_ids, curr_agent_states)
          enum_actions.replace(env_ids, agent_outputs.enum)
          x_actions.replace(env_ids, agent_outputs.x)
          y_actions.replace(env_ids, agent_outputs.y)
          target_unit_actions.replace(env_ids, agent_outputs.target_unit)
          ability_actions.replace(env_ids, agent_outputs.ability)
          item_actions.replace(env_ids, agent_outputs.item)

          # Return environment actions to environments.
          return (agent_outputs.enum, agent_outputs.x, agent_outputs.y, agent_outputs.target_unit, agent_outputs.ability, agent_outputs.item)

        return inference


      with strategy.scope():
        server.bind([create_inference_fn(d) for d in inference_devices])

      server.start()
      unroll_queues.append(unroll_queue)
      servers.append(server)

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


  #logger.start(additional_logs)

  # Execute learning.
  while iterations < final_iteration:
    #print("iterations: " + str(iterations))

    # Save checkpoint.
    current_time = time.time()

    save_checkpoint_secs = 1800
    if current_time - last_ckpt_time >= save_checkpoint_secs:
      if arguments.train == True:
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
