from grpclib.client import Channel
from dotaservice.protos.DotaService_grpc import DotaServiceStub
from dotaservice.protos.DotaService_pb2 import Actions
from dotaservice.protos.DotaService_pb2 import GameConfig

from dotaservice.protos.dota_shared_enums_pb2 import DOTA_GAMEMODE_1V1MID, DOTA_GAMEMODE_TUTORIAL
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


skill_flag = False
item_flag = False
move_flag = True
tree_flag = False
async def step():
  global skill_flag
  global item_flag
  global move_flag
  global tree_flag

  response = await asyncio.wait_for(env.observe(ObserveConfig(team_id=TEAM_RADIANT)), timeout=120)
  #print('response.world_state: ', response.world_state)
  dota_time = response.world_state.dota_time
  print('response.world_state.dota_time: ', response.world_state.dota_time)

  hero_unit = None
  enermy_hero = None
  for unit in response.world_state.units:
    #print("unit.name: ", unit.name)
    #print("unit.team_id: ", unit.team_id)
    #print("")
    if unit.unit_type == CMsgBotWorldState.UnitType.Value('HERO') and unit.name == "npc_dota_hero_nevermore" \
      and unit.team_id == 2:
      hero_unit = unit
      #print("unit: ", unit)

    if unit.unit_type == CMsgBotWorldState.UnitType.Value('HERO') and unit.name == "npc_dota_hero_nevermore" \
      and unit.team_id == 3:
      enermy_hero = unit

  #print("enermy_hero: ", enermy_hero)
  #print("enermy_hero: ", enermy_hero)

  mid_tower = None
  for unit in response.world_state.units:
    if unit.unit_type == CMsgBotWorldState.UnitType.Value('TOWER') \
            and unit.team_id == TEAM_RADIANT and 'tower1_mid' in unit.name:
            mid_tower = unit
            
  mid_tower_location = mid_tower.location
  #print("mid_tower_location: ", mid_tower_location)

  #action_pb.actionType = CMsgBotWorldState.Action.Type.Value('ACTION_CHAT')
  
  hero_location = hero_unit.location
  hero_item = hero_unit.items
  #print("hero_item: ", hero_item)

  if move_flag == False:
    if (abs(mid_tower_location.x + 800 - hero_location.x) >= 500) or (abs(mid_tower_location.y + 800 - hero_location.y) >= 500):
      move_flag = True

  m = CMsgBotWorldState.Action.MoveToLocation()
  m.location.x = mid_tower_location.x + 800
  m.location.y = mid_tower_location.y + 800
  m.location.z = 0
  
  c = CMsgBotWorldState.Action.Chat()
  c.message = "test"
  c.to_allchat = 1

  i = CMsgBotWorldState.Action.PurchaseItem()
  i.item = 2
  i.item_name = "item_tango"

  t = CMsgBotWorldState.Action.CastTree()
  t.abilitySlot = 0
  t.tree = 50

  #action_pb.chat.CopyFrom(t) 
  action_pb = CMsgBotWorldState.Action()
  if dota_time > -80.0:
    if item_flag == False:
      action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_PURCHASE_ITEM')
      action_pb.player = 0
      action_pb.purchaseItem.CopyFrom(i) 
      item_flag = True
    elif tree_flag == False and item_flag == True:
      action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_CAST_TARGET_TREE')
      action_pb.player = 0
      action_pb.castTree.CopyFrom(t) 
      tree_flag = True
    elif skill_flag == False:
      action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_TRAIN_ABILITY')
      action_pb.player = 0
      action_pb.trainAbility.ability = "nevermore_shadowraze1"
      skill_flag = True
    elif move_flag == False:
      action_pb.actionDelay = 0  # action_dict['delay'] * DELAY_ENUM_TO_STEP
      action_pb.player = 0  # action_dict['delay'] * DELAY_ENUM_TO_STEP
      action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_MOVE_DIRECTLY')
      action_pb.moveDirectly.CopyFrom(m)
      move_flag = False
    elif enermy_hero:
      #print("enermy_hero.handle: ", enermy_hero.handle)
      #action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_CAST_TARGET_TREE')
      action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_CAST_NO_TARGET')
      action_pb.player = 0 
      action_pb.cast.abilitySlot = 2
      #action_pb.castTarget.target = enermy_hero.handle
    else:
      action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_NONE')
  else:
    action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_NONE')
  
  print("action_pb: ", action_pb)
  #action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_PURCHASE_ITEM')
  #action_pb.player = 0 
  #action_pb.purchaseItem.CopyFrom(i) 
  #item_flag = True

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
