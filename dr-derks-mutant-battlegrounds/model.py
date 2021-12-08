import numpy as np

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense, Conv2D, Flatten, TimeDistributed, LSTM, Reshape, Dropout, LayerNormalization
import tensorflow.keras.backend as K
from tensorflow.python.keras.engine.base_layer import InputSpec
mapping = dict()


def register(name):
    def _thunk(func):
        mapping[name] = func
        return func
    return _thunk


def get_network_builder(name):
    """
    If you want to register your own network outside models.py, you just need:

    Usage Example:
    -------------
    from ForgER.model import register
    @register("your_network_name")
    def your_network_define(**net_kwargs):
        ...
        return network_fn

    """
    if callable(name):
        return name
    elif name in mapping:
        return mapping[name]
    else:
        raise ValueError('Registered networks:', ', '.join(mapping.keys()))


class ActorCriticModel(tf.keras.Model):
    def __init__(self, units, action_dim, reg=1e-6):
        super(ActorCriticModel, self).__init__()
        reg = {'kernel_regularizer': l2(reg), 'bias_regularizer': l2(reg)}

        kernel_init = tf.keras.initializers.VarianceScaling(scale=2.)
        self.h_layers = Sequential([Dense(num, 'relu', use_bias=True, kernel_initializer=kernel_init,
                                          **reg) for num in units[:-1]])
        self.a_head = Dense(units[-1]/2, 'relu', use_bias=True, kernel_initializer=kernel_init, **reg)
        self.v_head = Dense(units[-1]/2, 'relu', use_bias=True, kernel_initializer=kernel_init, **reg)
        self.a_head1 = Dense(action_dim, use_bias=True, kernel_initializer=kernel_init, **reg)
        self.v_head1 = Dense(1, use_bias=True, kernel_initializer=kernel_init, **reg)

    #@tf.function
    def call(self, inputs):
        print("inputs.shape: ", inputs.shape)

        print('Building model')
        features = self.h_layers(inputs)

        policy_logits = self.a_head(features)
        policy_logits = self.a_head1(policy_logits)
        #advantage = advantage - tf.reduce_mean(advantage, axis=-1, keepdims=True)

        baseline = self.v_head(features)
        baseline = self.v_head1(baseline)
        #out = value + advantage
        
        return policy_logits, baseline


@register("derks_dq")
def make_model(name, reg=1e-5, action_dim=16):
    #pov = tf.keras.Input(shape=obs_space.shape)
    pov = tf.keras.Input(shape=(64))
    normalized_pov = pov
    policy_logits, baseline = ActorCriticModel([256], action_dim, reg=reg)(normalized_pov)
    model = tf.keras.Model(inputs={'pov': pov}, outputs={'policy_logits': policy_logits, 'baseline': baseline}, name=name)
    return model
