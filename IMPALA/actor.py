import grpc
import tensorflow as tf
import numpy as np
import collections
from absl import logging

import os
import time
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class nullcontext(object):  
  def __init__(self, *args, **kwds):
    del args  # unused
    del kwds  # unused

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    pass

client = grpc.Client("localhost:8687")
timer_cls = nullcontext
elapsed_inference_s_timer = timer_cls('actor/elapsed_inference_s', 1000)
EnvOutput = collections.namedtuple('EnvOutput', 'reward done env allied_heroes enemy_heroes allied_nonheroes enemy_nonheroes \
                                    allied_towers enemy_towers enum_mask x_mask y_mask target_unit_mask ability_mask \
                                    abandoned episode_step')
for i in range(0, 1000):
	run_id = np.random.randint(low=0, high=np.iinfo(np.int64).max, size=3, dtype=np.int64)

	for j in range(0,500):
		try:
			#print("np.array([1.0, 1.0]).shape: " + str(np.array([1.0, 1.0]).shape))
			#print("np.zeros((2,1,64), dtype=np.int32).shape: " + str(np.zeros((2,1,64), dtype=np.float32).shape))

			'''
			EnvOutput = collections.namedtuple('EnvOutput', 'reward done env allied_heroes enemy_heroes allied_nonheroes 
												enemy_nonheroes allied_towers enemy_towers abandoned episode_step')
			'''
			env_output = EnvOutput(np.array([0.1], dtype=np.float32), np.array([False]), 
								   np.ones((1,3), dtype=np.float32) * random.uniform(0, 1.0), 
								   np.ones((1,5,12), dtype=np.float32) * random.uniform(0, 1.0), 
								   np.ones((1,5,12), dtype=np.float32) * random.uniform(0, 1.0), 
								   np.ones((1,16,12), dtype=np.float32)* random.uniform(0, 1.0),
								   np.ones((1,16,12), dtype=np.float32) * random.uniform(0, 1.0), 
								   np.ones((1,11,12), dtype=np.float32) * random.uniform(0, 1.0),
								   np.ones((1,11,12), dtype=np.float32) * random.uniform(0, 1.0), 
								   np.ones((1,4), dtype=np.float32),
								   np.ones((1,9), dtype=np.float32),
								   np.ones((1,9), dtype=np.float32),
								   np.ones((1,64), dtype=np.float32),
								   np.ones((1,3), dtype=np.float32), 
								   np.array([False]), 
								   np.array([j], dtype=np.int32))
			
			#env_output = EnvOutput(np.array([1.0], dtype=np.float32), np.array([False]), 
			#					   np.ones((1,3), dtype=np.float32) * 0.1, np.array([False]), np.array([0], dtype=np.int32))
			# client.inference(env_id, run_id, env_output, raw_reward)
			action = client.inference(np.array([0], dtype=np.int32), np.array([run_id[0]], dtype=np.int64), env_output, 
									  np.array([1.0], dtype=np.float32))

			print("action: " + str(action))
			print("j: ", j)
			time.sleep(0.1)
		except (tf.errors.UnavailableError, tf.errors.CancelledError):
			logging.info('Inference call failed. This is normal at the end of training.')