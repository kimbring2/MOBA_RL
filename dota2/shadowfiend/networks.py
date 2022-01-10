import collections
import utils
import observation
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import utils
from tensorflow.keras.layers import Input, Dense, Lambda, Add, Conv2D, Flatten, LSTM, Reshape, ConvLSTM2D, BatchNormalization, Conv3D
from tensorflow_probability.python.distributions import kullback_leibler
from itertools import repeat

tfd = tfp.distributions

TICKS_PER_OBSERVATION = 15 
TICKS_PER_SECOND = 30
MAX_MOVE_SPEED = 550
MAX_MOVE_IN_OBS = (MAX_MOVE_SPEED / TICKS_PER_SECOND) * TICKS_PER_OBSERVATION
N_MOVE_ENUMS = 9
MOVE_ENUMS = np.arange(N_MOVE_ENUMS, dtype=np.float32) - int(N_MOVE_ENUMS / 2)
MOVE_ENUMS *= MAX_MOVE_IN_OBS / (N_MOVE_ENUMS - 1) * 2
OBSERVATIONS_PER_SECOND = TICKS_PER_SECOND / TICKS_PER_OBSERVATION
MAX_UNITS = 5 + 5 + 16 + 16 + 1 + 1
ACTION_OUTPUT_COUNTS = {'enum': 5, 'x': 9, 'y': 9, 'target_unit': MAX_UNITS, 'ability': 4, 'item': 6}
OUTPUT_KEYS = ACTION_OUTPUT_COUNTS.keys()
INPUT_KEYS = ['env', 'allied_heroes', 'enemy_heroes', 'allied_nonheroes', 'enemy_nonheroes',
              'allied_towers', 'enemy_towers']


AgentOutput = collections.namedtuple('AgentOutput', 'enum enum_logits x x_logits y y_logits target_unit target_unit_logits ability \
                                                     ability_logits item item_logits baseline')


class Dota(tf.Module):
  """Agent with ResNet, but without LSTM and additional inputs.

  Four blocks instead of three in ImpalaAtariDeep.
  """
  def __init__(self, enum_parametric_action_distribution, x_parametric_action_distribution, 
               y_parametric_action_distribution, target_unit_parametric_action_distribution,
               ability_parametric_action_distribution, item_parametric_action_distribution):
    super(Dota, self).__init__(name='dota')

    # Parameters and layers for unroll.
    self._enum_parametric_action_distribution = enum_parametric_action_distribution
    self._x_parametric_action_distribution = x_parametric_action_distribution
    self._y_parametric_action_distribution = y_parametric_action_distribution
    self._target_unit_parametric_action_distribution = target_unit_parametric_action_distribution
    self._ability_parametric_action_distribution = ability_parametric_action_distribution
    self._item_parametric_action_distribution = item_parametric_action_distribution

    # Parameters and layers for _torso.
    self.affine_env = tf.keras.layers.Dense(128, activation='relu')

    self.affine_unit_basic_stats = tf.keras.layers.Dense(128, activation='relu')

    self.affine_unit_ah = tf.keras.layers.Dense(128, activation='relu')
    self.affine_unit_eh = tf.keras.layers.Dense(128, activation='relu')
    self.affine_unit_anh = tf.keras.layers.Dense(128, activation='relu')
    self.affine_unit_enh = tf.keras.layers.Dense(128, activation='relu')
    self.affine_unit_ath = tf.keras.layers.Dense(128, activation='relu')
    self.affine_unit_eth = tf.keras.layers.Dense(128, activation='relu')

    self.affine_pre_rnn = tf.keras.layers.Dense(MAX_UNITS, activation='relu')

    self._core = tf.keras.layers.LSTMCell(128)

    self.affine_unit_attention = tf.keras.layers.Dense(128, name='target_unit_policy_logits',
                                                       kernel_initializer='lecun_normal')

    # Layers for _head.
    self.affine_head_enum = tf.keras.layers.Dense(self._enum_parametric_action_distribution.param_size, 
                                                  name='enum_policy_logits',
                                                  kernel_initializer='lecun_normal')
    self.affine_move_x = tf.keras.layers.Dense(self._x_parametric_action_distribution.param_size, name='x_policy_logits',
                                               kernel_initializer='lecun_normal')
    self.affine_move_y = tf.keras.layers.Dense(self._y_parametric_action_distribution.param_size, name='y_policy_logits',
                                               kernel_initializer='lecun_normal')
    self.affine_head_ability = tf.keras.layers.Dense(self._ability_parametric_action_distribution.param_size, 
                                                     name='ability_policy_logits',
                                                     kernel_initializer='lecun_normal')
    self.affine_head_item = tf.keras.layers.Dense(self._item_parametric_action_distribution.param_size, 
                                                     name='item_policy_logits',
                                                     kernel_initializer='lecun_normal')

    self._baseline = tf.keras.layers.Dense(1, name='baseline', kernel_initializer='lecun_normal')

  def initial_state(self, batch_size):
    return self._core.get_initial_state(batch_size=batch_size, dtype=tf.float32)

  def _torso(self, unused_prev_action, env_output):
    #_, _, env, allied_heroes, enemy_heroes, allied_nonheroes, enemy_nonheroes, allied_towers, enemy_towers, _, _ = env_output
    env = env_output[2]
    allied_heroes = env_output[3]
    enemy_heroes = env_output[4]
    allied_nonheroes = env_output[5]
    enemy_nonheroes = env_output[6]
    allied_towers = env_output[7]
    enemy_towers = env_output[8]

    enum_mask = env_output[9]
    x_mask = env_output[10]
    y_mask = env_output[11]
    target_unit_mask = env_output[12]
    ability_mask = env_output[13]
    item_mask = env_output[14]

    env = self.affine_env(env)
    
    ah_basic = self.affine_unit_basic_stats(allied_heroes)
    ah_embedding = self.affine_unit_ah(ah_basic)
    ah_embedding_max = tf.math.reduce_max(ah_embedding, 1)
    
    eh_basic = self.affine_unit_basic_stats(enemy_heroes)
    eh_embedding = self.affine_unit_eh(eh_basic)
    eh_embedding_max = tf.math.reduce_max(eh_embedding, 1)
    
    anh_basic = self.affine_unit_basic_stats(allied_nonheroes)
    anh_embedding = self.affine_unit_anh(anh_basic)
    anh_embedding_max = tf.math.reduce_max(anh_embedding, 1)
    
    enh_basic = self.affine_unit_basic_stats(enemy_nonheroes)
    enh_embedding = self.affine_unit_enh(enh_basic)
    enh_embedding_max = tf.math.reduce_max(enh_embedding, 1)
    
    ath_basic = self.affine_unit_basic_stats(allied_towers)
    ath_embedding = self.affine_unit_ath(ath_basic)
    ath_embedding_max = tf.math.reduce_max(ath_embedding, 1)
    
    eth_basic = self.affine_unit_basic_stats(enemy_towers)
    eth_embedding = self.affine_unit_eth(eth_basic)
    eth_embedding_max = tf.math.reduce_max(eth_embedding, 1)

    unit_embedding = tf.concat([ah_embedding, eh_embedding, anh_embedding, enh_embedding, ath_embedding,
                                eth_embedding], axis=1)                         
    unit_embedding = tf.transpose(unit_embedding, perm=[0, 2, 1])

    x = tf.concat((env, ah_embedding_max, eh_embedding_max, anh_embedding_max, enh_embedding_max,
                   ath_embedding_max, eth_embedding_max), axis=1)
    x = self.affine_pre_rnn(x)

    return unit_embedding, x, enum_mask, x_mask, y_mask, target_unit_mask, ability_mask, item_mask

  def _head(self, torso_output):
    unit_embedding, x, enum_mask, x_mask, y_mask, target_unit_mask, ability_mask, item_mask = torso_output
    batch_size = unit_embedding.shape[0]

    unit_attention = self.affine_unit_attention(x) 
    unit_attention = tf.expand_dims(unit_attention, 1)

    action_scores_enum = self.affine_head_enum(x)
    action_scores_x = self.affine_move_x(x)
    action_scores_y = self.affine_move_y(x)
    action_target_unit = tf.linalg.matmul(unit_attention, unit_embedding)   
    action_target_unit = tf.squeeze(action_target_unit, 1)  
    action_ability = self.affine_head_ability(x)
    action_item = self.affine_head_item(x)

    baseline = tf.squeeze(self._baseline(x), axis=-1)

    enum_action_list = []
    x_action_list = []
    y_action_list = []
    target_unit_action_list = []
    ability_action_list = []
    item_action_list = []

    enum_logits_list = []
    x_logits_list = []
    y_logits_list = []
    target_unit_logits_list = []
    ability_logits_list = []
    item_logits_list = []
    for e_l, x_l, y_l, t_l, a_l, i_l, e_m, x_m, y_m, t_m, a_m, i_m in zip(tf.unstack(action_scores_enum), 
                                         tf.unstack(action_scores_x), tf.unstack(action_scores_y), 
                                         tf.unstack(action_target_unit), tf.unstack(action_ability), tf.unstack(action_item), 
                                         tf.unstack(enum_mask), tf.unstack(x_mask),
                                         tf.unstack(y_mask), tf.unstack(target_unit_mask), 
                                         tf.unstack(ability_mask), tf.unstack(item_mask)):
      heads_logits = {'enum': tf.expand_dims(tf.expand_dims(e_l, 0), 0),
                      'x': tf.expand_dims(tf.expand_dims(x_l, 0), 0),
                      'y': tf.expand_dims(tf.expand_dims(y_l, 0), 0),
                      'target_unit': tf.expand_dims(tf.expand_dims(t_l, 0), 0),
                      'ability': tf.expand_dims(tf.expand_dims(a_l, 0), 0),
                      'item': tf.expand_dims(tf.expand_dims(i_l, 0), 0)
                     }
      action_masks = {'enum': tf.expand_dims(tf.expand_dims(e_m, 0), 0),
                      'x': tf.expand_dims(tf.expand_dims(x_m, 0), 0),
                      'y': tf.expand_dims(tf.expand_dims(y_m, 0), 0),
                      'target_unit': tf.expand_dims(tf.expand_dims(t_m, 0), 0),
                      'ability': tf.expand_dims(tf.expand_dims(a_m, 0), 0),
                      'item': tf.expand_dims(tf.expand_dims(i_m, 0), 0)
                     }

      action_dict = {'enum': -1, 'x': -1, 'y': -1, 'target_unit': -1, 'ability': -1, 'item': -1}
      masked_heads_logits = {'enum': heads_logits['enum'], 'x': heads_logits['x'], 
                             'y': heads_logits['y'], 'target_unit': heads_logits['target_unit'], 
                             'ability': heads_logits['ability'], 'item': heads_logits['item']}
      masked_heads_logits['enum'] = tf.convert_to_tensor([[list(repeat(-1.0, heads_logits['enum'].shape[2]))]], 
                                                           dtype=tf.float32)
      masked_heads_logits['x'] = tf.convert_to_tensor([[list(repeat(-1.0, heads_logits['x'].shape[2]))]], 
                                                        dtype=tf.float32)
      masked_heads_logits['y'] = tf.convert_to_tensor([[list(repeat(-1.0, heads_logits['y'].shape[2]))]], 
                                                        dtype=tf.float32)
      masked_heads_logits['target_unit'] = tf.convert_to_tensor([[list(repeat(-1.0, heads_logits['target_unit'].shape[2]))]], 
                                                                  dtype=tf.float32)
      masked_heads_logits['ability'] = tf.convert_to_tensor([[list(repeat(-1.0, heads_logits['ability'].shape[2]))]], 
                                                              dtype=tf.float32)
      masked_heads_logits['item'] = tf.convert_to_tensor([[list(repeat(-1.0, heads_logits['item'].shape[2]))]], 
                                                              dtype=tf.float32)


      #tf.print("masked_heads_logits b: ", masked_heads_logits)
      action_dict, masked_heads_logits = utils.select_actions(action_dict, heads_logits, 
                                                              action_masks, masked_heads_logits)
      #tf.print("masked_heads_logits a: ", masked_heads_logits)
      #tf.print("")
      #print("masked_heads_logits: ", masked_heads_logits)
      enum_action_list.append(action_dict['enum'])
      x_action_list.append(action_dict['x'])
      y_action_list.append(action_dict['y'])
      target_unit_action_list.append(action_dict['target_unit'])
      ability_action_list.append(action_dict['ability'])
      item_action_list.append(action_dict['item'])

      enum_logits_list.append(masked_heads_logits['enum'][0][0])
      x_logits_list.append(masked_heads_logits['x'][0][0])
      y_logits_list.append(masked_heads_logits['y'][0][0])
      target_unit_logits_list.append(masked_heads_logits['target_unit'][0][0])
      ability_logits_list.append(masked_heads_logits['ability'][0][0])
      item_logits_list.append(masked_heads_logits['item'][0][0])

    enum_action_list = tf.stack(enum_action_list)
    x_action_list = tf.stack(x_action_list)
    y_action_list = tf.stack(y_action_list)
    target_unit_action_list = tf.stack(target_unit_action_list)
    ability_action_list = tf.stack(ability_action_list)
    item_action_list = tf.stack(item_action_list)

    enum_logits_list = tf.stack(enum_logits_list)
    x_logits_list = tf.stack(x_logits_list)
    y_logits_list = tf.stack(y_logits_list)
    target_unit_logits_list = tf.stack(target_unit_logits_list)
    ability_logits_list = tf.stack(ability_logits_list)
    item_logits_list = tf.stack(item_logits_list)

    return AgentOutput(enum_action_list, enum_logits_list, x_action_list, x_logits_list, y_action_list, 
                       y_logits_list, target_unit_action_list, target_unit_logits_list, ability_action_list, 
                       ability_logits_list, item_action_list, item_logits_list, baseline)


  # Not clear why, but if "@tf.function" declarator is placed directly onto
  # __call__, training fails with "uninitialized variable *baseline".
  # when running on multiple learning tpu cores.
  @tf.function
  def get_action(self, *args, **kwargs):
    return self.__call__(*args, **kwargs)

  def __call__(self, prev_actions, env_outputs, core_state, unroll=False, is_training=False):
    if not unroll:
      # Add time dimension.
      prev_actions, env_outputs = tf.nest.map_structure(lambda t: tf.expand_dims(t, 0), (prev_actions, env_outputs))

    outputs, core_state = self._unroll(prev_actions, env_outputs, core_state)

    if not unroll:
      # Remove time dimension.
      outputs = tf.nest.map_structure(lambda t: tf.squeeze(t, 0), outputs)

    return outputs, core_state

  def _unroll(self, prev_actions, env_outputs, core_state):
    unused_reward, done, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = env_outputs

    torso_outputs = utils.batch_apply(self._torso, (prev_actions, env_outputs))

    unit_embedding, x, enum_mask, x_mask, y_mask, target_unit_mask, ability_mask, item_mask = torso_outputs

    initial_core_state = self._core.get_initial_state(batch_size=tf.shape(prev_actions[0])[1], dtype=tf.float32)
    core_output_list = []
    for input_, d in zip(tf.unstack(x), tf.unstack(done)):
      # If the episode ended, the core state should be reset before the next.
      core_state = tf.nest.map_structure(
          lambda x, y, d=d: tf.where(  
              tf.reshape(d, [d.shape[0]] + [1] * (x.shape.rank - 1)), x, y),
          initial_core_state, core_state)

      core_output, core_state = self._core(input_, core_state)
      core_output_list.append(core_output)

    core_outputs = tf.stack(core_output_list)

    return utils.batch_apply(self._head, ((unit_embedding, core_outputs, enum_mask, x_mask, y_mask, target_unit_mask, 
                                           ability_mask, item_mask),)), core_state