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

import time
import asyncio
import argparse

parser = argparse.ArgumentParser(description='OpenAI Five implementation')
parser.add_argument('--render', type=bool, default=False, help='render with GUI')
arguments = parser.parse_args()

# Connect to the DotaService.
env = DotaServiceStub(Channel('127.0.0.1', 13337))

TICKS_PER_OBSERVATION = 15
HOST_TIMESCALE = 10

render = arguments.render
if render == True:
  HOST_MODE = HostMode.Value('HOST_MODE_GUI')
else:
  HOST_MODE = HostMode.Value('HOST_MODE_DEDICATED')

modes = [HERO_CONTROL_MODE_CONTROLLED, HERO_CONTROL_MODE_DEFAULT]
hero_picks = [
        HeroPick(team_id=TEAM_RADIANT, hero_id=NPC_DOTA_HERO_NEVERMORE, control_mode=modes[0]),
        HeroPick(team_id=TEAM_RADIANT, hero_id=NPC_DOTA_HERO_SNIPER, control_mode=HERO_CONTROL_MODE_IDLE),
        HeroPick(team_id=TEAM_RADIANT, hero_id=NPC_DOTA_HERO_SNIPER, control_mode=HERO_CONTROL_MODE_IDLE),
        HeroPick(team_id=TEAM_RADIANT, hero_id=NPC_DOTA_HERO_SNIPER, control_mode=HERO_CONTROL_MODE_IDLE),
        HeroPick(team_id=TEAM_RADIANT, hero_id=NPC_DOTA_HERO_SNIPER, control_mode=HERO_CONTROL_MODE_IDLE),
        HeroPick(team_id=TEAM_DIRE, hero_id=NPC_DOTA_HERO_NEVERMORE, control_mode=modes[1]),
        HeroPick(team_id=TEAM_DIRE, hero_id=NPC_DOTA_HERO_SNIPER, control_mode=HERO_CONTROL_MODE_IDLE),
        HeroPick(team_id=TEAM_DIRE, hero_id=NPC_DOTA_HERO_SNIPER, control_mode=HERO_CONTROL_MODE_IDLE),
        HeroPick(team_id=TEAM_DIRE, hero_id=NPC_DOTA_HERO_SNIPER, control_mode=HERO_CONTROL_MODE_IDLE),
        HeroPick(team_id=TEAM_DIRE, hero_id=NPC_DOTA_HERO_SNIPER, control_mode=HERO_CONTROL_MODE_IDLE),
    ]


async def reset():
  # Get the initial observation.
  response = await asyncio.wait_for(env.reset(GameConfig(
        ticks_per_observation=TICKS_PER_OBSERVATION,
        host_timescale=HOST_TIMESCALE,
        host_mode=HOST_MODE,
        game_mode=DOTA_GAMEMODE_1V1MID,
        hero_picks=hero_picks,
    )), timeout=120)

  print("response: ", response)


async def step():
  response = await asyncio.wait_for(env.observe(ObserveConfig(team_id=TEAM_RADIANT)), timeout=120)
  #print('response.world_state: ', response.world_state)
  print('response.world_state.dota_time: ', response.world_state.dota_time)

  hero_unit = None
  for unit in response.world_state.units:
    #print("unit: ", unit)
    if unit.unit_type == CMsgBotWorldState.UnitType.Value('HERO') and unit.player_id == 0:
      hero_unit = unit
      #print("unit: ", unit)

  mid_tower = None
  for unit in response.world_state.units:
    if unit.unit_type == CMsgBotWorldState.UnitType.Value('TOWER') \
            and unit.team_id == TEAM_RADIANT and 'tower1_mid' in unit.name:
            mid_tower = unit
            
  mid_tower_location = mid_tower.location
  #print("mid_tower_location: ", mid_tower_location)

  action_pb = CMsgBotWorldState.Action()
  action_pb.actionDelay = 0  # action_dict['delay'] * DELAY_ENUM_TO_STEP
  action_pb.player = 0  # action_dict['delay'] * DELAY_ENUM_TO_STEP
  action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_MOVE_DIRECTLY')

  hero_location = hero_unit.location
  #print("hero_location: ", hero_location)

  m = CMsgBotWorldState.Action.MoveToLocation()
  m.location.x = mid_tower_location.x
  m.location.y = mid_tower_location.y
  m.location.z = 0

  action_pb.moveDirectly.CopyFrom(m) 

  actions = []
  for i in range(0, 1):
    actions.append(action_pb)

  actions_pb = CMsgBotWorldState.Actions(actions=actions)
    
  response = await asyncio.wait_for(env.act(Actions(actions=actions_pb, team_id=TEAM_RADIANT)), timeout=120)
  #print('response_home.world_state.dota_time: ', response_home.world_state.dota_time)


async def main():
  try:
    print("reset")
    await reset()
  except:
    print("except")

  while True:
    print("step")
    await step()
    
    # Sample an action from the action protobuf
    #action = Actions.MoveToLocation(x=0, y=0, z=0)
    # Take an action, returning the resulting observation.
    #observation = env.step(action)
    
if __name__ == '__main__':
  loop = asyncio.get_event_loop()
  coro = main()
  loop.run_until_complete(coro)
