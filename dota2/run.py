from grpclib.client import Channel
from dotaservice.protos.DotaService_grpc import DotaServiceStub
from dotaservice.protos.DotaService_pb2 import Actions
from dotaservice.protos.DotaService_pb2 import GameConfig

from dotaservice.protos.dota_shared_enums_pb2 import DOTA_GAMEMODE_1V1MID
from dotaservice.protos.DotaService_pb2 import HostMode
from dotaservice.protos.DotaService_pb2 import ObserveConfig
from dotaservice.protos.DotaService_pb2 import TEAM_DIRE, TEAM_RADIANT, Hero, HeroPick, HeroControlMode
from dotaservice.protos.DotaService_pb2 import NPC_DOTA_HERO_NEVERMORE, NPC_DOTA_HERO_SNIPER, NPC_DOTA_HERO_OMNIKNIGHT
from dotaservice.protos.DotaService_pb2 import HERO_CONTROL_MODE_IDLE, HERO_CONTROL_MODE_DEFAULT, HERO_CONTROL_MODE_CONTROLLED
from dotaservice.protos.DotaService_pb2 import Status
from dotaservice.protos.dota_gcmessages_common_bot_script_pb2 import CMsgBotWorldState

import tensorflow_probability as tfp
tfd = tfp.distributions

import os
import time
import asyncio
import argparse
import math
import numpy as np
import collections
from absl import logging

import grpc
import tensorflow as tf

import shadowfiend.utils as utils1
import shadowfiend.networks as networks1

import omninight.utils as utils2
import omninight.networks as networks2

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

client1 = grpc.Client("localhost:8687")
client2 = grpc.Client("localhost:8688")
EnvOutput = collections.namedtuple('EnvOutput', 'reward done env allied_heroes enemy_heroes allied_nonheroes enemy_nonheroes \
                                    allied_towers enemy_towers enum_mask x_mask y_mask target_unit_mask ability_mask item_mask \
                                    abandoned episode_step')

parser = argparse.ArgumentParser(description='OpenAI Five implementation')
parser.add_argument('--render', type=bool, default=False, help='render with GUI')
parser.add_argument('--id', type=int, default=0, help='id for environment')
parser.add_argument('--ip', type=str, default='127.0.0.1', help='ip of dotaservice')
arguments = parser.parse_args()

render = arguments.render

# Connect to the DotaService.
env = DotaServiceStub(Channel(arguments.ip, 13337 + arguments.id))
'''
if render == True:
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if len(gpus) != 0:
    tf.config.experimental.set_virtual_device_configuration(gpus[0],
              [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8000)])
else:
'''

if arguments.id == 0:
  writer = tf.summary.create_file_writer("./tensorboard")


TICKS_PER_OBSERVATION = 15
HOST_TIMESCALE = 10

if render == True:
  HOST_MODE = HostMode.Value('HOST_MODE_GUI')
else:
  HOST_MODE = HostMode.Value('HOST_MODE_DEDICATED')

modes = [HERO_CONTROL_MODE_CONTROLLED, HERO_CONTROL_MODE_DEFAULT]
hero_picks = [
        HeroPick(team_id=TEAM_RADIANT, hero_id=NPC_DOTA_HERO_NEVERMORE, control_mode=modes[0]),
        HeroPick(team_id=TEAM_RADIANT, hero_id=NPC_DOTA_HERO_OMNIKNIGHT, control_mode=modes[0]),
        HeroPick(team_id=TEAM_RADIANT, hero_id=NPC_DOTA_HERO_SNIPER, control_mode=HERO_CONTROL_MODE_IDLE),
        HeroPick(team_id=TEAM_RADIANT, hero_id=NPC_DOTA_HERO_SNIPER, control_mode=HERO_CONTROL_MODE_IDLE),
        HeroPick(team_id=TEAM_RADIANT, hero_id=NPC_DOTA_HERO_SNIPER, control_mode=HERO_CONTROL_MODE_IDLE),
        HeroPick(team_id=TEAM_DIRE, hero_id=NPC_DOTA_HERO_NEVERMORE, control_mode=modes[1]),
        HeroPick(team_id=TEAM_DIRE, hero_id=NPC_DOTA_HERO_OMNIKNIGHT, control_mode=modes[1]),
        HeroPick(team_id=TEAM_DIRE, hero_id=NPC_DOTA_HERO_SNIPER, control_mode=HERO_CONTROL_MODE_IDLE),
        HeroPick(team_id=TEAM_DIRE, hero_id=NPC_DOTA_HERO_SNIPER, control_mode=HERO_CONTROL_MODE_IDLE),
        HeroPick(team_id=TEAM_DIRE, hero_id=NPC_DOTA_HERO_SNIPER, control_mode=HERO_CONTROL_MODE_IDLE),
    ]


async def reset(env):
  # Get the initial observation.
  response = await asyncio.wait_for(env.reset(GameConfig(
        ticks_per_observation=TICKS_PER_OBSERVATION,
        host_timescale=HOST_TIMESCALE,
        host_mode=HOST_MODE,
        game_mode=DOTA_GAMEMODE_1V1MID,
        hero_picks=hero_picks,
    )), timeout=120)

  return response
 

item_name_list = ['item_branches', 'item_clarity', 'item_ward_observer', 'item_tango', 'item_gauntlets', 
                  'item_magic_stick', 'item_magic_wand', 'item_circlet', 'item_recipe_bracer', 'item_boots',
                  'item_faerie_fire', 'item_flask', 'item_enchanted_mango', 'item_ward_sentry', 'item_bracer',
                  'item_recipe_magic_wand', 'item_recipe_wraith_band', 'item_slippers'
                 ]

shadowfiend_item_route = [
          {'gold': 0, 'item': ['item_ward_observer', 'item_magic_stick', 'item_tango', 'item_tango']},
          {'gold': 500, 'item': ['item_boots']},
          {'gold': 250, 'item': ['item_branches', 'item_branches', 'item_recipe_magic_wand']},
          {'gold': 250, 'item': ['item_clarity', 'item_clarity', 'item_faerie_fire', 'item_faerie_fire']},
          {'gold': 505, 'item': ['item_circlet', 'item_slippers', 'item_recipe_wraith_band']}
        ]

'''
"Ability1": "nevermore_shadowraze1",
"Ability2": "nevermore_shadowraze2",
"Ability3": "nevermore_shadowraze3",
"Ability4": "nevermore_necromastery",
"Ability5": "nevermore_dark_lord",
"Ability6": "nevermore_requiem",
"Ability10": "special_bonus_spell_amplify_8",
"Ability11": "special_bonus_attack_speed_25",
"Ability12": "special_bonus_unique_nevermore_3",
"Ability13": "special_bonus_unique_nevermore_2",
"Ability14": "special_bonus_unique_nevermore_1",
"Ability15": "special_bonus_unique_nevermore_6",
"Ability16": "special_bonus_unique_nevermore_5",
"Ability17": "special_bonus_unique_nevermore_4",
'''
shadowfiend_ability_route = [
        'nevermore_shadowraze1', 'nevermore_necromastery', 'nevermore_shadowraze1', 'nevermore_necromastery',
        'nevermore_shadowraze1', 'nevermore_necromastery', 'nevermore_shadowraze1', 'nevermore_necromastery',
        'nevermore_dark_lord', 'special_bonus_spell_amplify_8', 'nevermore_requiem', 'nevermore_dark_lord',
        'nevermore_dark_lord', 'nevermore_dark_lord', 'special_bonus_unique_nevermore_3', 'nevermore_requiem',
        'nevermore_requiem', 'special_bonus_unique_nevermore_1', 'special_bonus_unique_nevermore_5'
        ]


omniknight_item_route = [
          {'gold': 0, 'item': ['item_ward_sentry', 'item_circlet', 'item_clarity', 'item_clarity']},
          {'gold': 500, 'item': ['item_boots']},
          {'gold': 350, 'item': ['item_gauntlets', 'item_recipe_bracer']},
          {'gold': 250, 'item': ['item_clarity', 'item_clarity', 'item_faerie_fire', 'item_faerie_fire']},
          {'gold': 450, 'item': ['item_magic_wand']}
        ]

'''
"Ability1": "omniknight_purification",
"Ability2": "omniknight_repel",
"Ability3": "omniknight_degen_aura",
"Ability4": "omniknight_hammer_of_purity",
"Ability5": "generic_hidden",
"Ability6": "omniknight_guardian_angel"
"Ability10": "special_bonus_unique_omniknight_5",
"Ability11": "special_bonus_attack_base_damage_50",
"Ability12": "special_bonus_unique_omniknight_6",
"Ability13": "special_bonus_movement_speed_30",
"Ability14": "special_bonus_unique_omniknight_2",
"Ability15": "special_bonus_unique_omniknight_3",
"Ability16": "special_bonus_unique_omniknight_1",
"Ability17": "special_bonus_unique_omniknight_7",
'''
omniknight_ability_route = [
        'omniknight_purification', 'omniknight_repel', 'omniknight_purification', 'omniknight_degen_aura',
        'omniknight_repel', 'omniknight_guardian_angel', 'omniknight_purification', 'omniknight_degen_aura',
        'omniknight_repel', 'special_bonus_attack_base_damage_50', 'omniknight_guardian_angel', 'omniknight_purification',
        'omniknight_degen_aura', 'omniknight_purification', "special_bonus_unique_omniknight_2", "omniknight_guardian_angel",
        'omniknight_repel', "special_bonus_unique_omniknight_3", "special_bonus_unique_omniknight_7"
        ]


drowranger_item_route = [
          {'gold': 0, 'item': ['item_ward_observer', 'item_slippers', 'item_tango', 'item_clarity']},
          {'gold': 500, 'item': ['item_boots']},
          {'gold': 370, 'item': ['item_circlet', 'item_recipe_wraith_band']},
          {'gold': 250, 'item': ['item_clarity', 'item_clarity', 'item_faerie_fire', 'item_faerie_fire']},
          {'gold': 450, 'item': ['item_magic_wand']}
        ]

'''
"Ability1": "drow_ranger_frost_arrows",
"Ability2": "drow_ranger_wave_of_silence",
"Ability3": "drow_ranger_multishot",
"Ability4": "generic_hidden",
"Ability5": "generic_hidden",
"Ability6": "drow_ranger_marksmanship",
"Ability10": "special_bonus_unique_drow_ranger_5",
"Ability11": "special_bonus_unique_drow_ranger_2",
"Ability12": "special_bonus_unique_drow_ranger_4",
"Ability13": "special_bonus_unique_drow_ranger_6",
"Ability14": "special_bonus_unique_drow_ranger_1",
"Ability15": "special_bonus_unique_drow_ranger_7",
"Ability16": "special_bonus_unique_drow_ranger_3",
"Ability17": "special_bonus_unique_drow_ranger_8",
'''
drowranger_ability_route = [
        'drow_ranger_frost_arrows', 'drow_ranger_multishot', 'drow_ranger_frost_arrows', 'drow_ranger_wave_of_silence',
        'drow_ranger_multishot', 'drow_ranger_marksmanship', 'drow_ranger_frost_arrows', 'drow_ranger_wave_of_silence',
        'drow_ranger_multishot', 'special_bonus_unique_drow_ranger_1', 'drow_ranger_frost_arrows', 'drow_ranger_multishot',
        'drow_ranger_wave_of_silence', 'drow_ranger_frost_arrows', "special_bonus_unique_drow_ranger_3", "drow_ranger_marksmanship",
        'drow_ranger_multishot', "special_bonus_unique_drow_ranger_4", "special_bonus_unique_drow_ranger_6"
        ]


def network_init():
  '''
  tf.TensorSpec([], tf.float32, 'reward'),
  tf.TensorSpec([], tf.bool, 'done'),
  tf.TensorSpec([3], tf.float32, 'env_observation'),
  tf.TensorSpec([5,32], tf.float32, 'allied_heroes_observation'),
  tf.TensorSpec([5,32], tf.float32, 'enemy_heroes_observation'),
  tf.TensorSpec([16,32], tf.float32, 'allied_nonheroes_observation'),
  tf.TensorSpec([16,32], tf.float32, 'enemy_nonheroes_observation'),
  tf.TensorSpec([1,32], tf.float32, 'allied_towers_observation'),
  tf.TensorSpec([1,32], tf.float32, 'enemy_towers_observation'),
  tf.TensorSpec([5], tf.float32, 'enum_mask'),
  tf.TensorSpec([9], tf.float32, 'x_mask'),
  tf.TensorSpec([9], tf.float32, 'y_mask'),
  tf.TensorSpec([44], tf.float32, 'target_unit_mask'),
  tf.TensorSpec([4], tf.float32, 'ability_mask'),
  tf.TensorSpec([6], tf.float32, 'item_mask'),
  tf.TensorSpec([], tf.bool, 'abandoned'),
  tf.TensorSpec([], tf.int32, 'episode_step')
  '''
  for step in range(0, 300):
    print("init step: ", step)
    env_output = EnvOutput(np.array([0], dtype=np.float32), 
                            np.array([False]), 
                            np.zeros((1,3), np.float32), 
                            np.zeros((1,5,32), np.float32), 
                            np.zeros((1,5,32), np.float32), 
                            np.zeros((1,16,32), np.float32),
                            np.zeros((1,16,32), np.float32), 
                            np.zeros((1,1,32), np.float32),
                            np.zeros((1,1,32), np.float32), 
                            np.zeros((1,5), np.float32),
                            np.zeros((1,9), np.float32),
                            np.zeros((1,9), np.float32),
                            np.zeros((1,44), np.float32),
                            np.zeros((1,4), np.float32), 
                            np.zeros((1,6), np.float32), 
                            np.array([False]), 
                            np.array([0], dtype=np.int32))

    action1 = client1.inference(np.array([arguments.id], dtype=np.int32), np.array([run_id1[0]], dtype=np.int64), 
                                env_output, np.array([0], dtype=np.float32))

    env_output = EnvOutput(np.array([0], dtype=np.float32), 
                            np.array([False]), 
                            np.zeros((1,3), np.float32), 
                            np.zeros((1,5,32), np.float32), 
                            np.zeros((1,5,32), np.float32), 
                            np.zeros((1,16,32), np.float32),
                            np.zeros((1,16,32), np.float32), 
                            np.zeros((1,1,32), np.float32),
                            np.zeros((1,1,32), np.float32), 
                            np.zeros((1,5), np.float32),
                            np.zeros((1,9), np.float32),
                            np.zeros((1,9), np.float32),
                            np.zeros((1,44), np.float32),
                            np.zeros((1,3), np.float32), 
                            np.zeros((1,6), np.float32), 
                            np.array([False]), 
                            np.array([0], dtype=np.int32))

    action2 = client2.inference(np.array([arguments.id], dtype=np.int32), np.array([run_id1[0]], dtype=np.int64), 
                                env_output, np.array([0], dtype=np.float32))



total_step = 0
max_dota_time = 600
run_id1 = np.random.randint(low=0, high=np.iinfo(np.int64).max, size=3, dtype=np.int64)
run_id2 = np.random.randint(low=0, high=np.iinfo(np.int64).max, size=3, dtype=np.int64)
async def step(env):
  global total_step

  reward_sum1 = 0.0
  reward_sum2 = 0.0
  reward1 = 0.0
  reward2 = 0.0
  step = 0
  done = False
  prev_level1 = 0
  prev_level2 = 0
  try:
    print("reset")
    response = await asyncio.wait_for(env.reset(GameConfig(
          ticks_per_observation=TICKS_PER_OBSERVATION,
          host_timescale=HOST_TIMESCALE,
          host_mode=HOST_MODE,
          game_mode=DOTA_GAMEMODE_1V1MID,
          hero_picks=hero_picks,
    )), timeout=120)
  except:
    return 

  teleport_temp_flag_1 = False
  teleport_temp_flag_2 = False

  prev_obs = response.world_state_radiant
  item_flag1 = False
  item_flag2 = False
  item_index1 = 0
  item_index2 = 0
  item_route_index1 = 0
  item_route_index2 = 0
  courier_stash_flag1 = False
  courier_stash_flag2 = False
  courier_delivery_flag1 = False
  courier_delivery_flag2 = False
  teleport_flag_1 = False
  teleport_buy_flag_1 = False
  teleport_flag_2 = False
  teleport_buy_flag_2 = False
  while True:
    if done == True:
      print("done == True")
      if arguments.id == 0:
        if response.status == Status.Value('RADIANT_WIN'):
          #print('RADIANT_WIN')
          with writer.as_default():
            tf.summary.scalar("reward_sum1", reward_sum1, step=total_step)
            tf.summary.scalar("reward_sum2", reward_sum2, step=total_step)
        elif response.status == Status.Value('DIRE_WIN'):
          #print('DIRE_WIN')
          with writer.as_default():
            tf.summary.scalar("reward_sum1", reward_sum1, step=total_step)
            tf.summary.scalar("reward_sum2", reward_sum2, step=total_step)

      break

    try:
      response = await env.observe(ObserveConfig(team_id=TEAM_RADIANT))
    except:
      #print("observe break")
      continue

    obs = response.world_state
    dota_time = obs.dota_time

    if arguments.id == 0:
      print("total_step: ", total_step)

    if response.status == Status.Value('RADIANT_WIN'):
      print('RADIANT_WIN')
      done = True
    elif response.status == Status.Value('DIRE_WIN'):
      print('DIRE_WIN')
      done = True
    elif response.status == Status.Value('RESOURCE_EXHAUSTED'):
      print('RESOURCE_EXHAUSTED')
      done = True
    else:
      done = False

    dota_time_norm = obs.dota_time / 1200.  # Normalize by 20 minutes
    creepwave_sin = math.sin(obs.dota_time * (2. * math.pi) / 60)
    team_id = TEAM_RADIANT
    team_float = -.2 if team_id == TEAM_DIRE else .2
    
    env_state = np.array([dota_time_norm, creepwave_sin, team_float])

    hero_unit1 = utils1.get_unit(obs, team_id=TEAM_RADIANT, player_id=0)
    hero_unit2 = utils2.get_unit(obs, team_id=TEAM_RADIANT, player_id=1)
    
    #print("item_flag1: ", item_flag1)
    #print("item_route_index1: ", item_route_index1)
    #print("item_index1: ", item_index1)
    #print("len(shadowfiend_item_route): ", len(shadowfiend_item_route))
    #print("shadowfiend_item_route[item_route_index1 - 1]['gold']: ", shadowfiend_item_route[item_route_index1 - 1]['gold'])
    #print("len(shadowfiend_item_route[item_route_index1 - 1]['item']): ", len(shadowfiend_item_route[item_route_index1 - 1]['item']))
    #print("courier_stash_flag1: ", courier_stash_flag1)
    #print("courier_delivery_flag1: ", courier_delivery_flag1)
    #print("teleport_flag_1: ", teleport_flag_1)
    #print("teleport_buy_flag_1: ", teleport_buy_flag_1)
    if hero_unit1 != None:
      gold1 = hero_unit1.unreliable_gold + hero_unit1.reliable_gold
      mana1 = hero_unit1.mana
      location1 = hero_unit1.location

    if hero_unit2 != None:
      gold2 = hero_unit2.unreliable_gold + hero_unit2.reliable_gold
      mana2 = hero_unit2.mana
      location2 = hero_unit2.location

    action_pb_item_and_ability1 = None
    action_pb_item_and_ability2 = None

    if int(dota_time) % 140 == 0 and dota_time >= 0:
      if teleport_temp_flag_1 == False:
        teleport_temp_flag_1 = True

      if teleport_temp_flag_2 == False:
        teleport_temp_flag_2 = True

    if hero_unit1 != None:
      if mana1 < 150 and mana1 >= 75 and teleport_temp_flag_1 == True and gold1 >= 100:
        teleport_flag_1 = True
        teleport_temp_flag_1 = False
      elif hero_unit1.level > prev_level1:
        prev_level1 = hero_unit1.level
        action_pb_item_and_ability1 = utils1.train_ability(hero_unit1, shadowfiend_ability_route[hero_unit1.level - 1], 0)
      elif item_route_index1 < len(shadowfiend_item_route) and item_flag1 == False and courier_stash_flag1 == False \
        and courier_delivery_flag1 == False:
        if gold1 >= shadowfiend_item_route[item_route_index1]['gold']:
          if item_flag1 == False:
            item_flag1 = True

          item_route_index1 += 1
      elif item_flag1 == True:
        if item_index1 <= len(shadowfiend_item_route[item_route_index1 - 1]['item']) - 1:
          action_pb_item_and_ability1 = utils1.buy_item(shadowfiend_item_route[item_route_index1 - 1]['item'][item_index1], 0)
          item_index1 += 1
        else:
          courier_stash_flag1 = True

          item_flag1 = False
          item_index1 = 0
      elif courier_stash_flag1 == True:
        action_pb_item_and_ability1 = CMsgBotWorldState.Action()
        action_pb_item_and_ability1.player = 0
        action_pb_item_and_ability1.actionDelay = 0 
        action_pb_item_and_ability1.actionType = CMsgBotWorldState.Action.Type.Value('ACTION_COURIER')

        action_pb_item_and_ability1.courier.unit = 0 
        action_pb_item_and_ability1.courier.courier = 0
        action_pb_item_and_ability1.courier.action = 3

        courier_stash_flag1 = False
        courier_delivery_flag1 = True
      elif courier_delivery_flag1 == True:
        action_pb_item_and_ability1 = CMsgBotWorldState.Action()
        action_pb_item_and_ability1.player = 0
        action_pb_item_and_ability1.actionDelay = 0 
        action_pb_item_and_ability1.actionType = CMsgBotWorldState.Action.Type.Value('ACTION_COURIER')

        action_pb_item_and_ability1.courier.unit = 0 
        action_pb_item_and_ability1.courier.courier = 0
        action_pb_item_and_ability1.courier.action = 6
        courier_delivery_flag1 = False

    #print("item_flag2: ", item_flag2)
    if hero_unit2 != None:
      if mana2 < 150 and mana2 >= 75 and teleport_temp_flag_2 == True and gold2 >= 100:
        teleport_flag_2 = True
        teleport_temp_flag_2 = False
        #pass
      elif hero_unit2.level > prev_level2:
        prev_level2 = hero_unit2.level
        action_pb_item_and_ability2 = utils2.train_ability(hero_unit2, omniknight_ability_route[hero_unit2.level - 1], 1)
      elif item_route_index2 < len(omniknight_item_route) and item_flag2 == False and courier_stash_flag2 == False \
        and courier_delivery_flag2 == False:
        if gold2 >= omniknight_item_route[item_route_index2]['gold']:
          if item_flag2 == False:
            item_flag2 = True

          item_route_index2 += 1
      elif item_flag2 == True:
        if item_index2 <= len(omniknight_item_route[item_route_index2 - 1]['item']) - 1:
          action_pb_item_and_ability2 = utils2.buy_item(omniknight_item_route[item_route_index2 - 1]['item'][item_index2], 1)
          item_index2 += 1
        else:
          courier_stash_flag2 = True

          item_flag2 = False
          item_index2 = 0
      elif courier_stash_flag2 == True:
        action_pb_item_and_ability2 = CMsgBotWorldState.Action()
        action_pb_item_and_ability2.player = 1
        action_pb_item_and_ability2.actionDelay = 0 
        action_pb_item_and_ability2.actionType = CMsgBotWorldState.Action.Type.Value('ACTION_COURIER')

        action_pb_item_and_ability2.courier.unit = 0 
        action_pb_item_and_ability2.courier.courier = 0
        action_pb_item_and_ability2.courier.action = 3

        courier_stash_flag2 = False
        courier_delivery_flag2 = True
      elif courier_delivery_flag2 == True:
        action_pb_item_and_ability2 = CMsgBotWorldState.Action()
        action_pb_item_and_ability2.player = 1
        action_pb_item_and_ability2.actionDelay = 0 
        action_pb_item_and_ability2.actionType = CMsgBotWorldState.Action.Type.Value('ACTION_COURIER')

        action_pb_item_and_ability2.courier.unit = 0 
        action_pb_item_and_ability2.courier.courier = 0
        action_pb_item_and_ability2.courier.action = 6
        courier_delivery_flag2 = False

    actions = []
    
    if hero_unit1 != None and hero_unit1.is_alive:
      #print("hero_unit.is_alive: ", hero_unit.is_alive)
      ah, eh, anh, enh, ac, ec, at, et = utils1.unit_separation(obs, TEAM_RADIANT)

      # Process units into Tensors & Handles
      allied_heroes1, allied_hero_handles1 = utils1.unit_matrix(unit_list=ah, hero_unit=hero_unit1, only_self=False, max_units=5)
        
      enemy_heroes1, enemy_hero_handles1 = utils1.unit_matrix(unit_list=eh, hero_unit=hero_unit1, max_units=5)

      allied_nonheroes1, allied_nonhero_handles1 = utils1.unit_matrix(unit_list=[*anh, *ac], hero_unit=hero_unit1, max_units=16)

      enemy_nonheroes1, enemy_nonhero_handles1 = utils1.unit_matrix(unit_list=[*enh, *ec], hero_unit=hero_unit1, max_units=16)

      allied_towers1, allied_tower_handles1 = utils1.unit_matrix(unit_list=at, hero_unit=hero_unit1, max_units=1)

      enemy_towers1, enemy_tower_handles1 = utils1.unit_matrix(unit_list=et, hero_unit=hero_unit1, max_units=1)

      unit_handles1 = np.concatenate([allied_hero_handles1, enemy_hero_handles1, allied_nonhero_handles1, 
                                     enemy_nonhero_handles1, allied_tower_handles1, enemy_tower_handles1])
      #print("unit_handles1: ", unit_handles1)
      action_masks1 = utils1.action_masks(player_unit=hero_unit1, unit_handles=unit_handles1)

      #print("action_masks: ", action_masks)
      env_output1 = EnvOutput(np.array([reward1], dtype=np.float32), 
                              np.array([done]), 
                              np.array([env_state], dtype=np.float32), 
                              np.array([allied_heroes1], dtype=np.float32), 
                              np.array([enemy_heroes1], dtype=np.float32), 
                              np.array([allied_nonheroes1], dtype=np.float32),
                              np.array([enemy_nonheroes1], dtype=np.float32), 
                              np.array([allied_towers1], dtype=np.float32),
                              np.array([enemy_towers1], dtype=np.float32), 
                              np.array(action_masks1['enum'][0], dtype=np.float32),
                              np.array(action_masks1['x'][0], dtype=np.float32),
                              np.array(action_masks1['y'][0], dtype=np.float32),
                              np.array(action_masks1['target_unit'][0], dtype=np.float32),
                              np.array(action_masks1['ability'][0], dtype=np.float32), 
                              np.array(action_masks1['item'][0], dtype=np.float32), 
                              np.array([False]), 
                              np.array([step], dtype=np.int32))

      action1 = client1.inference(np.array([arguments.id], dtype=np.int32), np.array([run_id1[0]], dtype=np.int64), 
                                  env_output1, np.array([reward1], dtype=np.float32))
      action_dict1 = {'enum': action1[0], 'x': action1[1], 'y': action1[2], 'target_unit': action1[3], 'ability': action1[4], 'item': action1[5]}
      

      if teleport_flag_1 == True:
        action_pb1 = utils1.use_tp(0)
        teleport_buy_flag_1 = True
        teleport_flag_1 = False
      elif teleport_buy_flag_1 == True and location1.x - -6700 < 200 and location1.y - -6700:
        # action_pb1.castLocation.location.x = -6700 action_pb1.castLocation.location.y = -6700
        action_pb1.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_PURCHASE_ITEM')

        i = CMsgBotWorldState.Action.PurchaseItem()
        i.item = 2
        i.item_name = 'item_tpscroll'

        action_pb1.purchaseItem.CopyFrom(i) 

        teleport_buy_flag_1 = False
      elif action_pb_item_and_ability1 == None:
        if response.world_state.dota_time > -70.0:
          action_pb1 = utils1.action_to_pb(0, TEAM_RADIANT, action_dict1, response.world_state, unit_handles1)
        else:
          action_pb1 = utils1.none_action(0)
      elif courier_stash_flag1 == True:
        action_pb1 = action_pb_item_and_ability1
      elif courier_delivery_flag1 == True:
        action_pb1 = action_pb_item_and_ability1
      else:
        action_pb1 = action_pb_item_and_ability1

      #print("action_pb1: ", action_pb1)

      hero_location1 = hero_unit1.location
      
      actions.append(action_pb1)

      try:
        reward1 = utils1.get_reward(prev_obs, TEAM_RADIANT, obs, 0)
        reward1 = sum(reward1.values())
      except:
        reward1 = 0

    else:
      #print("hero_unit1 none")

      action_pb1 = CMsgBotWorldState.Action()
      action_pb1.actionDelay = 0 
      action_pb1.player = 0  
      action_pb1.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_NONE')

      actions.append(action_pb1)
    
      reward1 = 0


    if hero_unit2 != None and hero_unit2.is_alive:
      #print("hero_unit2.is_alive: ", hero_unit2.is_alive)
      ah, eh, anh, enh, ac, ec, at, et = utils2.unit_separation(obs, TEAM_RADIANT)

      # Process units into Tensors & Handles
      allied_heroes2, allied_hero_handles2 = utils2.unit_matrix(unit_list=ah, hero_unit=hero_unit2, only_self=False, max_units=5)
        
      enemy_heroes2, enemy_hero_handles2 = utils2.unit_matrix(unit_list=eh, hero_unit=hero_unit2, max_units=5)

      allied_nonheroes2, allied_nonhero_handles2 = utils2.unit_matrix(unit_list=[*anh, *ac], hero_unit=hero_unit2, max_units=16)

      enemy_nonheroes2, enemy_nonhero_handles2 = utils2.unit_matrix(unit_list=[*enh, *ec], hero_unit=hero_unit2, max_units=16)

      allied_towers2, allied_tower_handles2 = utils2.unit_matrix(unit_list=at, hero_unit=hero_unit2, max_units=1)

      enemy_towers2, enemy_tower_handles2 = utils2.unit_matrix(unit_list=et, hero_unit=hero_unit2, max_units=1)

      unit_handles2 = np.concatenate([allied_hero_handles2, enemy_hero_handles2, allied_nonhero_handles2, 
                                      enemy_nonhero_handles2, allied_tower_handles2, enemy_tower_handles2])
      #print("unit_handles2: ", unit_handles2)
      action_masks2 = utils2.action_masks(player_unit=hero_unit2, unit_handles=unit_handles2)

      env_output2 = EnvOutput(np.array([reward2], dtype=np.float32), 
                              np.array([done]), 
                              np.array([env_state], dtype=np.float32), 
                              np.array([allied_heroes2], dtype=np.float32), 
                              np.array([enemy_heroes2], dtype=np.float32), 
                              np.array([allied_nonheroes2], dtype=np.float32),
                              np.array([enemy_nonheroes2], dtype=np.float32), 
                              np.array([allied_towers2], dtype=np.float32),
                              np.array([enemy_towers2], dtype=np.float32), 
                              np.array(action_masks2['enum'][0], dtype=np.float32),
                              np.array(action_masks2['x'][0], dtype=np.float32),
                              np.array(action_masks2['y'][0], dtype=np.float32),
                              np.array(action_masks2['target_unit'][0], dtype=np.float32),
                              np.array(action_masks2['ability'][0], dtype=np.float32), 
                              np.array(action_masks2['item'][0], dtype=np.float32), 
                              np.array([False]), 
                              np.array([step], dtype=np.int32))

      action2 = client2.inference(np.array([arguments.id], dtype=np.int32), np.array([run_id2[0]], dtype=np.int64), 
                                  env_output2, np.array([reward2], dtype=np.float32))
      action_dict2 = {'enum': action2[0], 'x': action2[1], 'y': action2[2], 'target_unit': action2[3], 'ability': action2[4], 'item': action2[5]}


      if teleport_flag_2 == True:
        action_pb2 = utils2.use_tp(1)
        teleport_buy_flag_2 = True
        teleport_flag_2 = False
      elif teleport_buy_flag_2 == True and location2.x - -6700 < 200 and location2.y - -6700:
        action_pb2.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_PURCHASE_ITEM')

        i = CMsgBotWorldState.Action.PurchaseItem()
        i.item = 2
        i.item_name = 'item_tpscroll'

        action_pb2.purchaseItem.CopyFrom(i) 

        teleport_buy_flag_2 = False
      elif action_pb_item_and_ability2 == None:
        if response.world_state.dota_time > -70.0:
          action_pb2 = utils2.action_to_pb(1, TEAM_RADIANT, action_dict2, response.world_state, unit_handles2)
        else:
          action_pb2 = utils2.none_action(1)
      elif courier_stash_flag2 == True:
        action_pb2 = action_pb_item_and_ability2
      elif courier_delivery_flag2 == True:
        action_pb2 = action_pb_item_and_ability2
      else:
        action_pb2 = action_pb_item_and_ability2

      #print("action_pb2: ", action_pb2)

      hero_location2 = hero_unit2.location
      
      actions.append(action_pb2)
      try:
        reward2 = utils2.get_reward(prev_obs, TEAM_RADIANT, obs, 1)
        reward2 = sum(reward2.values())
      except:
        reward2 = 0
    else:
      #print("hero_unit2 none")

      action_pb2 = CMsgBotWorldState.Action()
      action_pb2.actionDelay = 0 
      action_pb2.player = 1  
      action_pb2.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_NONE')

      actions.append(action_pb2)
    
      reward2 = 0

    if response.status == Status.Value('RADIANT_WIN'):
      #print('RADIANT_WIN')
      reward1 += 5.0
      reward2 += 5.0
    elif response.status == Status.Value('DIRE_WIN'):
      #print('DIRE_WIN')
      reward1 -= 5.0
      reward2 -= 5.0


    actions_pb = CMsgBotWorldState.Actions(actions=actions)
    _ = await env.act(Actions(actions=actions_pb, team_id=TEAM_RADIANT))

    if arguments.id == 0:
      #print("reward1: ", reward1)
      #print("reward2: ", reward2)
      #print("response.status: ", response.status)
      pass
    else:
      #print("dota_time: ", dota_time)
      #print("response.status: ", response.status)
      pass

    step += 1
    prev_obs = obs
    total_step += 1
    reward_sum1 += reward1
    reward_sum2 += reward2


async def main():
  #try:
  #  print("reset")
    #await reset(env)

  #except:
  #  print("except")
  
  while True:
    await step(env)

    
if __name__ == '__main__':
  network_init()

  loop = asyncio.get_event_loop()
  coro = main()
  loop.run_until_complete(coro)