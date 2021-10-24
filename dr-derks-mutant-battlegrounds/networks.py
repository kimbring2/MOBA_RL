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

"""SEED agent using Keras."""

import collections
import utils
import observation
import tensorflow as tf


AgentOutput = collections.namedtuple('AgentOutput', 'action policy_logits baseline')


class _Stack(tf.Module):
  """Stack of pooling and convolutional blocks with residual connections."""
  def __init__(self, num_ch, num_blocks):
    super(_Stack, self).__init__(name='stack')
    self._conv = tf.keras.layers.Conv2D(num_ch, 3, strides=1, padding='same', kernel_initializer='lecun_normal')
    self._max_pool = tf.keras.layers.MaxPool2D(pool_size=3, padding='same', strides=2)

    self._res_convs0 = [
        tf.keras.layers.Conv2D(num_ch, 3, strides=1, padding='same', name='res_%d/conv2d_0' % i, kernel_initializer='lecun_normal')
        for i in range(num_blocks)
    ]
    self._res_convs1 = [
        tf.keras.layers.Conv2D(num_ch, 3, strides=1, padding='same', name='res_%d/conv2d_1' % i, kernel_initializer='lecun_normal')
        for i in range(num_blocks)
    ]

  def __call__(self, conv_out):
    #tf.print('conv_out.shape:', conv_out.shape)

    # Downscale.
    conv_out = self._conv(conv_out)
    conv_out = self._max_pool(conv_out)

    # Residual block(s).
    for (res_conv0, res_conv1) in zip(self._res_convs0, self._res_convs1):
      block_input = conv_out
      conv_out = tf.nn.relu(conv_out)
      conv_out = res_conv0(conv_out)
      conv_out = tf.nn.relu(conv_out)
      conv_out = res_conv1(conv_out)
      conv_out += block_input

    return conv_out


def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead) 
  but it must be broadcastable for addition.

  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable 
          to (..., seq_len_q, seq_len_k). Defaults to None.

  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)  

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)

    self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.dropout = tf.keras.layers.Dropout(0.1)
    
    #v = tf.Variable(tf.random.truncated_normal([10, 40]))
    # W = tf.Variable(tf.random_uniform([16,4],0,0.01))
    #self.w = tf.Variable(tf.random.truncated_normal([64, 1, 100]), 0.0, 0.1)
    
    self.dense = tf.keras.layers.Dense(d_model)

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'd_model': self.d_model,
        'num_heads': self.num_heads,
    })
    return config
    
  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, v, k, q, mask, training):
    batch_size = tf.shape(q)[0]
    
    v_original = v
    
    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)

    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

    output = self.dense(output) 
    
    return output, attention_weights


class DrDerk(tf.Module):
  """Agent with ResNet, but without LSTM and additional inputs.

  Four blocks instead of three in ImpalaAtariDeep.
  """
  def __init__(self, parametric_action_distribution):
    super(DrDerk, self).__init__(name='drderk')

    # Parameters and layers for unroll.
    self._parametric_action_distribution = parametric_action_distribution

    # Parameters and layers for _torso.
    #self._stacks = [_Stack(num_ch, num_blocks) for num_ch, num_blocks in [(16, 2), (32, 2), (32, 2), (32, 2)]]

    self._attention = MultiHeadAttention(65, 5)
    self._layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self._dropout = tf.keras.layers.Dropout(0.1)

    self._to_linear = tf.keras.layers.Dense(256, kernel_initializer='lecun_normal')

    # Layers for _head.
    self._policy_logits = tf.keras.layers.Dense(self._parametric_action_distribution.param_size, name='policy_logits',
                                                     kernel_initializer='lecun_normal')
    self._baseline = tf.keras.layers.Dense(1, name='baseline', kernel_initializer='lecun_normal')

    self._entity_size = 3
    self._locs = []
    for i in range(0, 3):
        self._locs.append(i / float(self._entity_size))
        
    self._locs = tf.expand_dims(self._locs, 0)
    self._locs = tf.expand_dims(self._locs, 2)

  def initial_state(self, batch_size):
    return ()

  def _torso(self, unused_prev_action, env_output, is_training=False):
    _, _, frame, _, _ = env_output
    batch_size = tf.shape(frame)[0]

    #conv_out = frame
    #for stack in self._stacks:
      #conv_out = stack(conv_out)
    
    locs = tf.tile(self._locs, [batch_size, 1, 1])
    frame_features_locs = tf.concat([frame, locs], 2)
           
    attention_output, _ = self._attention(frame_features_locs, frame_features_locs, frame_features_locs, None)
    attention_output = self._dropout(attention_output, training=is_training)
    attention_output = self._layernorm(frame_features_locs + attention_output)
    max_pool_1d = tf.math.reduce_max(attention_output, 1)
    
    attention_out = max_pool_1d
    #attention_out = tf.nn.relu(attention_out)
    attention_out = tf.keras.layers.Flatten()(attention_out)
    out = self._to_linear(attention_out)

    return tf.nn.relu(out)

  def _head(self, core_output):
    policy_logits = self._policy_logits(core_output)
    baseline = tf.squeeze(self._baseline(core_output), axis=-1)

    # Sample an action from the policy.
    new_action = self._parametric_action_distribution.sample(policy_logits)

    return AgentOutput(new_action, policy_logits, baseline)

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

    outputs, core_state = self._unroll(prev_actions, env_outputs, core_state, is_training)

    if not unroll:
      # Remove time dimension.
      outputs = tf.nest.map_structure(lambda t: tf.squeeze(t, 0), outputs)

    return outputs, core_state

  def _unroll(self, prev_actions, env_outputs, core_state, is_training):
    torso_outputs = utils.batch_apply(self._torso, (prev_actions, env_outputs))

    return utils.batch_apply(self._head, (torso_outputs,)), core_state
