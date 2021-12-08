import os
import random
import timeit
import yaml
from collections import deque
import numpy as np
import tensorflow as tf
import abc
import collections

from discretization import SmartDiscrete, get_dtype_dict
from utils import get_entity, get_action
from model import get_network_builder

import grpc

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

EnvOutput = collections.namedtuple('EnvOutput', 'reward done observation abandoned episode_step')

class DerkPlayer:
    """
    Player which controls all agents
    Each arena has 3 agents which the player must control.
    """

    def __init__(self, n_agents, action_space):
        """
        Parameters:
         - n_agents: TOTAL number of agents being controlled (= #arenas * #agents per arena)
        """
        self.n_agents = n_agents

        self.discrete_maker = SmartDiscrete(ignore_keys=[])

        self.action_dim = 15

        self.n_agents = n_agents
        self.n_arenas = int(n_agents / 3)
        self.action_space = action_space

        self.previous_ordi = None
        self.step = 0
        self.reward_sum = 0
        self.iteration_num = 0

        self.run_id = np.random.randint(low=0, high=np.iinfo(np.int64).max, size=self.n_arenas, dtype=np.int64)

        self.client = grpc.Client("localhost:8687")

        self.writer = tf.summary.create_file_writer("./tboard/agent1/")

    def signal_env_reset(self, obs):
        """
        env.reset() was called
        """
        self.step = 0

        with self.writer.as_default():
            tf.summary.scalar("reward", self.reward_sum, step=self.iteration_num)

        self.reward_sum = 0

    def take_action(self, env_step_ret):
        """
        Parameters:
         - env_step_ret: whatever env.step() returned (obs_n, rew_n, done_n, info_n)

        Returns: action for each agent for each arena

        Actions:
         - MoveX: A number between -1 and 1
         - Rotate: A number between -1 and 1
         - ChaseFocus: A number between 0 and 1
         - CastingSlot:
                        0 = don’t cast
                    1 - 3 = cast corresponding ability
         - ChangeFocus:
                        0 = keep current focus
                        1 = focus home statue
                    2 - 3 = focus teammates
                        4 = focus enemy statue
                    5 - 7 = focus enemy
        """

        if self.step == 0:
            self.previous_ordi = env_step_ret

        _obs_n, _rew_n, _done_n, _info_n = env_step_ret

        self.reward_sum += sum(_rew_n) / self.n_arenas

        _previous_obs_n_agent_1, _previous_rew_n_agent_1, _previous_done_n_agent_1 = [], [], []
        _previous_obs_n_agent_2, _previous_rew_n_agent_2, _previous_done_n_agent_2 = [], [], []
        _previous_obs_n_agent_3, _previous_rew_n_agent_3, _previous_done_n_agent_3 = [], [], []

        _obs_n_agent_1, _rew_n_agent_1, _done_n_agent_1 = [], [], []
        _obs_n_agent_2, _rew_n_agent_2, _done_n_agent_2 = [], [], []
        _obs_n_agent_3, _rew_n_agent_3, _done_n_agent_3 = [], [], []

        action = []
        actions_index_1 = []
        actions_index_2 = []
        actions_index_3 = []

        # self.previous_ordi[0].shape:  (300, 64)
        # _rew_n.shape:  (300,)
        for i in range(self.n_arenas):
            _previous_obs_n = self.previous_ordi[0]
            _previous_rew_n = self.previous_ordi[1]
            _previous_done_n = self.previous_ordi[2]

            _previous_obs_n_arena = _previous_obs_n[3*i:3*(i+1)]
            _previous_rew_n_arena = _previous_rew_n[3*i:3*(i+1)]
            _previous_done_n_arena = _previous_done_n

            _previous_obs_n_agent_1.append(_previous_obs_n_arena[0])
            _previous_rew_n_agent_1.append(_previous_rew_n_arena[0])
            _previous_done_n_agent_1.append(_previous_done_n_arena)

            _previous_obs_n_agent_2.append(_previous_obs_n_arena[1])
            _previous_rew_n_agent_2.append(_previous_rew_n_arena[1])
            _previous_done_n_agent_2.append(_previous_done_n_arena)

            _previous_obs_n_agent_3.append(_previous_obs_n_arena[2])
            _previous_rew_n_agent_3.append(_previous_rew_n_arena[2])
            _previous_done_n_agent_3.append(_previous_done_n_arena)

            _obs_n_arena = _obs_n[3*i:3*(i+1)]
            _rew_n_arena = _rew_n[3*i:3*(i+1)]
            _done_n_arena =_done_n[0]

            #print("_obs_n_arena.shape: ", _obs_n_arena.shape)

            #_obs_n_agent_1.append(_obs_n_arena[0])
            _obs_n_agent_1.append(np.concatenate((_obs_n_arena[0], _obs_n_arena[1], _obs_n_arena[2]), axis=0))
            _rew_n_agent_1.append(_rew_n_arena[0])
            _done_n_agent_1.append(_done_n_arena)

            #_obs_n_agent_2.append(_obs_n_arena[1])
            _obs_n_agent_2.append(np.concatenate((_obs_n_arena[1], _obs_n_arena[2], _obs_n_arena[0]), axis=0))
            _rew_n_agent_2.append(_rew_n_arena[1])
            _done_n_agent_2.append(_done_n_arena)

            #_obs_n_agent_3.append(_obs_n_arena[2])
            _obs_n_agent_3.append(np.concatenate((_obs_n_arena[2], _obs_n_arena[0], _obs_n_arena[1]), axis=0))
            _rew_n_agent_3.append(_rew_n_arena[2])
            _done_n_agent_3.append(_done_n_arena)

        reward_list_1 = []
        done_list_1 = []
        observation_list_1 = []
        abandoned_list_1 = []
        episode_step_list_1 = []

        reward_list_2 = []
        done_list_2 = []
        observation_list_2 = []
        abandoned_list_2 = []
        episode_step_list_2 = []

        reward_list_3 = []
        done_list_3 = []
        observation_list_3 = []
        abandoned_list_3 = []
        episode_step_list_3 = []

        reward_sum_list = []
        for i in range(self.n_arenas):
            reward_list_1.append(_rew_n_agent_1[i] / 1000.0)
            reward_list_2.append(_rew_n_agent_2[i] / 1000.0)
            reward_list_3.append(_rew_n_agent_3[i] / 1000.0)

            done_list_1.append(bool(_done_n_agent_1[i]))
            done_list_2.append(bool(_done_n_agent_2[i]))
            done_list_3.append(bool(_done_n_agent_3[i]))

            observation_list_1.append(np.reshape(_obs_n_agent_1[i], [3, 64]))
            observation_list_2.append(np.reshape(_obs_n_agent_2[i], [3, 64]))
            observation_list_3.append(np.reshape(_obs_n_agent_3[i], [3, 64]))

            abandoned_list_1.append(False)
            abandoned_list_2.append(False)
            abandoned_list_3.append(False)

            episode_step_list_1.append(self.step)
            episode_step_list_2.append(self.step)
            episode_step_list_3.append(self.step)

            reward_sum_list.append(self.reward_sum)

        reward_1 = np.array(reward_list_1, dtype=np.float32)
        done_1 = np.array(done_list_1)
        observation_1 = np.array(observation_list_1)
        abandoned_1 = np.array(abandoned_list_1)
        episode_step_1 = np.array(episode_step_list_1, dtype=np.int32)

        reward_2 = np.array(reward_list_2, dtype=np.float32)
        done_2 = np.array(done_list_2)
        observation_2 = np.array(observation_list_2)
        abandoned_2 = np.array(abandoned_list_2)
        episode_step_2 = np.array(episode_step_list_2, dtype=np.int32)

        reward_3 = np.array(reward_list_3, dtype=np.float32)
        done_3 = np.array(done_list_3)
        observation_3 = np.array(observation_list_3)
        abandoned_3 = np.array(abandoned_list_3)
        episode_step_3 = np.array(episode_step_list_3, dtype=np.int32)

        reward_sum_array = np.array(reward_sum_list, dtype=np.float32)

        # observation_1.shape: (100, 64)

        env_output_1 = EnvOutput(reward_1, done_1, observation_1, abandoned_1, episode_step_1)
        env_output_2 = EnvOutput(reward_2, done_2, observation_2, abandoned_2, episode_step_2)
        env_output_3 = EnvOutput(reward_3, done_3, observation_3, abandoned_3, episode_step_3)
        # client.inference(env_id, run_id, env_output, raw_reward)
        action_agent_1 = self.client.inference(np.array(range(0,50), dtype=np.int32), 
                                                        np.array(self.run_id, dtype=np.int64), 
                                                        env_output_1, 
                                                        reward_1)
        action_agent_2 = self.client.inference(np.array(range(50,100), dtype=np.int32), 
                                                        np.array(self.run_id, dtype=np.int64), 
                                                        env_output_2, 
                                                        reward_2)
        action_agent_3 = self.client.inference(np.array(range(100,150), dtype=np.int32), 
                                                        np.array(self.run_id, dtype=np.int64), 
                                                        env_output_3, 
                                                        reward_3)
        #print("action_index_1: " + str(action_index_1))
        for i in range(self.n_arenas):
            action_index_1 = int(action_agent_1[i])
            action_1 = self.discrete_maker.get_action_dict_by_key(action_index_1)
            #print("action_1: " + str(action_1))
            action.append(action_1)

            action_index_2 = int(action_agent_2[i])
            action_2 = self.discrete_maker.get_action_dict_by_key(action_index_2)
            #action_2 = [0.0, 0.0, 0.0, 0, 0]
            action.append(action_2)

            action_index_3 = int(action_agent_3[i])
            action_3 = self.discrete_maker.get_action_dict_by_key(action_index_3)
            #action_3 = [0.0, 0.0, 0.0, 0, 0]
            action.append(action_3)
        
        self.previous_ordi = env_step_ret
        self.iteration_num += 1
        self.step += 1

        return action
