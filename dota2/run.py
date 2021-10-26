from grpclib.client import Channel
from dotaservice.protos.DotaService_grpc import DotaServiceStub
from dotaservice.protos.DotaService_pb2 import Actions
from dotaservice.protos.DotaService_pb2 import GameConfig

from dotaservice.protos.dota_shared_enums_pb2 import DOTA_GAMEMODE_1V1MID
from dotaservice.protos.DotaService_pb2 import HostMode
from dotaservice.protos.DotaService_pb2 import ObserveConfig
from dotaservice.protos.DotaService_pb2 import TEAM_DIRE, TEAM_RADIANT, Hero, HeroPick, HeroControlMode
from dotaservice.protos.DotaService_pb2 import NPC_DOTA_HERO_NEVERMORE, NPC_DOTA_HERO_SNIPER
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

import utils
import networks

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

client = grpc.Client("localhost:8687")
EnvOutput = collections.namedtuple('EnvOutput', 'reward done env allied_heroes enemy_heroes allied_nonheroes enemy_nonheroes \
                                    allied_towers enemy_towers enum_mask x_mask y_mask target_unit_mask ability_mask \
                                    abandoned episode_step')

parser = argparse.ArgumentParser(description='OpenAI Five implementation')
parser.add_argument('--render', type=bool, default=False, help='render with GUI')
parser.add_argument('--id', type=int, default=0, help='id for environment')
arguments = parser.parse_args()

# Connect to the DotaService.
env = DotaServiceStub(Channel('127.0.0.1', 13337 + arguments.id, loop=asyncio.get_event_loop()))


if arguments.id == 0:
  writer = tf.summary.create_file_writer("./tensorboard")


TICKS_PER_OBSERVATION = 15
HOST_TIMESCALE = 10

render = arguments.render
if render == True:
  HOST_MODE = HostMode.Value('HOST_MODE_GUI')
else:
  HOST_MODE = HostMode.Value('HOST_MODE_DEDICATED')

modes = [HERO_CONTROL_MODE_CONTROLLED, HERO_CONTROL_MODE_DEFAULT]
hero_picks = [
        HeroPick(team_id=TEAM_RADIANT, hero_id=NPC_DOTA_HERO_NEVERMORE, control_mode=HERO_CONTROL_MODE_CONTROLLED),
        HeroPick(team_id=TEAM_RADIANT, hero_id=NPC_DOTA_HERO_SNIPER, control_mode=HERO_CONTROL_MODE_CONTROLLED),
        HeroPick(team_id=TEAM_RADIANT, hero_id=NPC_DOTA_HERO_SNIPER, control_mode=HERO_CONTROL_MODE_IDLE),
        HeroPick(team_id=TEAM_RADIANT, hero_id=NPC_DOTA_HERO_SNIPER, control_mode=HERO_CONTROL_MODE_IDLE),
        HeroPick(team_id=TEAM_RADIANT, hero_id=NPC_DOTA_HERO_SNIPER, control_mode=HERO_CONTROL_MODE_IDLE),
        HeroPick(team_id=TEAM_DIRE, hero_id=NPC_DOTA_HERO_NEVERMORE, control_mode=modes[1]),
        HeroPick(team_id=TEAM_DIRE, hero_id=NPC_DOTA_HERO_SNIPER, control_mode=HERO_CONTROL_MODE_IDLE),
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
 
total_step = 0
max_dota_time = 600
run_id = np.random.randint(low=0, high=np.iinfo(np.int64).max, size=3, dtype=np.int64)
async def step(env, unit_id):
  global total_step

  reward_sum = 0.0
  reward = 0.0
  step = 0
  prev_level = 0

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

  prev_obs = response.world_state_radiant
  while True:
    try:
      response = await env.observe(ObserveConfig(team_id=TEAM_RADIANT))
    except:
      print("observe break")
      break

    obs = response.world_state

    dota_time = obs.dota_time
    #print("dota_time: ", dota_time)
    if dota_time == 0.0:
      print("dota_time break")
      break

    if dota_time > 600:
      if arguments.id == 0:
        with writer.as_default():
          tf.summary.scalar("reward", reward_sum, step=total_step)

      print("dota_time is over 600")
      break

    dota_time_norm = obs.dota_time / 1200.  # Normalize by 20 minutes
    creepwave_sin = math.sin(obs.dota_time * (2. * math.pi) / 60)
    team_id = TEAM_RADIANT
    team_float = -.2 if team_id == TEAM_DIRE else .2
      
    env_state = np.array([dota_time_norm, creepwave_sin, team_float])
    
    hero_unit = utils.get_unit(obs, player_id=unit_id)
    if hero_unit.level > prev_level:
      prev_level = hero_unit.level
      action_pb_train_ability = utils.train_ability(hero_unit)
    else:
      action_pb_train_ability = None

    #print("hero_unit: ", hero_unit)
    if hero_unit != None and hero_unit.is_alive:
      #print("hero_unit.is_alive: ", hero_unit.is_alive)
      ah, eh, anh, enh, ac, ec, at, et = utils.unit_separation(obs, TEAM_RADIANT)
      #print("at: ", at)
      #for allied_tower in at:
      #print("allied_tower.location.x: ", allied_tower.location.x)
        #print("allied_tower.location.y: ", allied_tower.location.y)
        #print("")

      # Process units into Tensors & Handles
      allied_heroes, allied_hero_handles = utils.unit_matrix(
          unit_list=ah,
          hero_unit=hero_unit,
          max_units=1,
      )
        
      enemy_heroes, enemy_hero_handles = utils.unit_matrix(
         unit_list=eh,
          hero_unit=hero_unit,
         max_units=5,
      )

      allied_nonheroes, allied_nonhero_handles = utils.unit_matrix(
         unit_list=[*anh, *ac],
         hero_unit=hero_unit,
         max_units=16,
      )

      enemy_nonheroes, enemy_nonhero_handles = utils.unit_matrix(
         unit_list=[*enh, *ec],
         hero_unit=hero_unit,
         max_units=16,
      )

      allied_towers, allied_tower_handles = utils.unit_matrix(
         unit_list=at,
         hero_unit=hero_unit,
         max_units=1,
      )

      enemy_towers, enemy_tower_handles = utils.unit_matrix(
         unit_list=et,
         hero_unit=hero_unit,
         max_units=1,
      )

      unit_handles = np.concatenate([allied_hero_handles, enemy_hero_handles, allied_nonhero_handles, 
                                     enemy_nonhero_handles, allied_tower_handles, enemy_tower_handles])

      action_masks = utils.action_masks(player_unit=hero_unit, unit_handles=unit_handles)
      #print("action_masks: ", action_masks)
      env_output = EnvOutput(np.array([reward], dtype=np.float32), 
                             np.array([False]), 
                             np.array([env_state], dtype=np.float32), 
                             np.array([allied_heroes], dtype=np.float32), 
                             np.array([enemy_heroes], dtype=np.float32), 
                             np.array([allied_nonheroes], dtype=np.float32),
                             np.array([enemy_nonheroes], dtype=np.float32), 
                             np.array([allied_towers], dtype=np.float32),
                             np.array([enemy_towers], dtype=np.float32), 
                             np.array(action_masks['enum'][0], dtype=np.float32),
                             np.array(action_masks['x'][0], dtype=np.float32),
                             np.array(action_masks['y'][0], dtype=np.float32),
                             np.array(action_masks['target_unit'][0], dtype=np.float32),
                             np.array(action_masks['ability'][0], dtype=np.float32), 
                             np.array([False]), 
                             np.array([step], dtype=np.int32))

      action = client.inference(np.array([arguments.id], dtype=np.int32), np.array([run_id[0]], dtype=np.int64), 
                                env_output, np.array([reward], dtype=np.float32))

      action_dict = {'enum': action[0], 'x': action[1], 'y': action[2], 'target_unit': action[3], 'ability': action[4]}
      if action_pb_train_ability == None:
        if response.world_state.dota_time > 0.:
          action_pb = utils.action_to_pb(unit_id, action_dict, response.world_state, unit_handles)
        else:
          action_pb = CMsgBotWorldState.Action()
          action_pb.actionDelay = 0 
          action_pb.player = 0  
          action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_NONE')
      else:
        #print("action_pb_train_ability")
        action_pb = action_pb_train_ability
        #action_pb = CMsgBotWorldState.Action()
        action_pb.actionDelay = 0 
        #action_pb.player = 0  
        #action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_NONE')

      hero_location = hero_unit.location

      actions = []
      actions.append(action_pb)

      actions_pb = CMsgBotWorldState.Actions(actions=actions)
    
      _ = await env.act(Actions(actions=actions_pb, team_id=TEAM_RADIANT))

      try:
        reward = utils.get_reward(prev_obs, obs, unit_id)
        reward = sum(reward.values())
      except:
        reward = 0
    else:
      print("hero_unit none")

      action_pb = CMsgBotWorldState.Action()
      action_pb.actionDelay = 0 
      action_pb.player = 0  
      action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_NONE')

      actions = []
      actions.append(action_pb)
      actions_pb = CMsgBotWorldState.Actions(actions=actions)
    
      _ = await env.act(Actions(actions=actions_pb, team_id=TEAM_RADIANT))
    
      reward = 0

    if arguments.id == 0:
      print("reward: ", reward)
      #print("dota_time: ", dota_time)
    else:
      print("dota_time: ", dota_time)

    step += 1
    reward_sum += reward
    prev_obs = obs
    total_step += 1


async def main():
  #try:
  #  print("reset")
    #await reset(env)

  #except:
  #  print("except")
  
  while True:
    await step(env, 0)

    
if __name__ == '__main__':
  loop = asyncio.get_event_loop()
  coro = main()
  loop.run_until_complete(coro)
