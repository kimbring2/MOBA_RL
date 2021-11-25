from grpclib.client import Channel
from dotaservice.protos.DotaService_grpc import DotaServiceStub
from dotaservice.protos.DotaService_pb2 import Actions
from dotaservice.protos.DotaService_pb2 import GameConfig

from dotaservice.protos.dota_shared_enums_pb2 import DOTA_GAMEMODE_1V1MID, DOTA_GAMEMODE_TUTORIAL, DOTA_GAMEMODE_CUSTOM
from dotaservice.protos.DotaService_pb2 import HostMode
from dotaservice.protos.DotaService_pb2 import ObserveConfig
from dotaservice.protos.DotaService_pb2 import TEAM_DIRE, TEAM_RADIANT, Hero, HeroPick, HeroControlMode
from dotaservice.protos.DotaService_pb2 import NPC_DOTA_HERO_NEVERMORE, NPC_DOTA_HERO_SNIPER
from dotaservice.protos.DotaService_pb2 import HERO_CONTROL_MODE_IDLE, HERO_CONTROL_MODE_DEFAULT, HERO_CONTROL_MODE_CONTROLLED
from dotaservice.protos.DotaService_pb2 import Status
from dotaservice.protos.dota_gcmessages_common_bot_script_pb2 import CMsgBotWorldState

import time, math
import asyncio
import argparse
import numpy as np

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


def get_vector_lenth(p):
    return math.sqrt(math.pow(p.x, 2) + math.pow(p.y, 2))


def get_vector_dot(p1, p2):
    return p1.x * p2.x + p1.y * p2.y


def get_lane_distance(p1):
    start_position = CMsgBotWorldState.Vector(x=-4450, y=-3800, z=256)
    stop_position = CMsgBotWorldState.Vector(x=-600, y=-480, z=256)

    self_tower_position = CMsgBotWorldState.Vector(x=-1544, y=-1408, z=256)
    enemy_tower_position = CMsgBotWorldState.Vector(x=524, y=652, z=256)

    #print("start_position.x: ", start_position.x)
    #print("start_position.y: ", start_position.y)

    x = start_position.x - stop_position.x
    y = start_position.y - stop_position.y
    lane_vector = CMsgBotWorldState.Vector(x=x, y=y, z=256)
    x = p1.x - stop_position.x
    y = p1.y - stop_position.y
    p_vector = CMsgBotWorldState.Vector(x=x, y=y, z=256)

    cos_to_lane_axis = get_vector_dot(lane_vector, p_vector) / (
        get_vector_lenth(lane_vector) * get_vector_lenth(p_vector))
    d = get_vector_lenth(p_vector)

    return abs(d * cos_to_lane_axis)


def cal_distance(p1, p2):
    if p1 is None or p2 is None:
        return -1

    return np.sqrt(np.power(p1.x - p2.x, 2) + np.power(p1.y - p2.y, 2))


skill_learn_flag = False
skill_use_flag = False
item_buy_flag = True
item_use_flag = False
move_flag = False
tree_flag = False
ward_flag = True
stick_flag = False
courier_stash_flag = False
courier_transfer_flag = False
teleport_flag = True
async def step():
  global skill_learn_flag
  global skill_use_flag
  global item_buy_flag
  global item_use_flag
  global move_flag
  global tree_flag
  global ward_flag
  global stick_flag
  global courier_stash_flag
  global courier_transfer_flag
  global teleport_flag

  creeps = []

  response = await asyncio.wait_for(env.observe(ObserveConfig(team_id=TEAM_RADIANT)), timeout=120)
  #print('response.world_state: ', response.world_state)
  dota_time = response.world_state.dota_time
  print('response.world_state.dota_time: ', response.world_state.dota_time)
  #print('response.world_state.glyph_cooldown: ', response.world_state.glyph_cooldown)
  #print('response.world_state.damage_events: ', response.world_state.damage_events)

  hero_unit = None
  enermy_hero = None
  for unit in response.world_state.units:
    #print("unit.unit_type: ", unit.unit_type)
    #print("unit.handle: ", unit.handle)
    #print("CMsgBotWorldState.UnitType.Value('LANE_CREEP'): ", CMsgBotWorldState.UnitType.Value('LANE_CREEP'))
    #print("unit.team_id: ", unit.team_id)
    #print("")
    if unit.unit_type == CMsgBotWorldState.UnitType.Value('HERO') and unit.name == "npc_dota_hero_nevermore":
      if unit.team_id == TEAM_RADIANT:
        hero_unit = unit
      elif unit.team_id == TEAM_DIRE:
        enermy_hero = unit

  mid_tower = None
  for unit in response.world_state.units:
    if unit.unit_type == CMsgBotWorldState.UnitType.Value('TOWER') and unit.name == "npc_dota_goodguys_tower1_mid":
      if unit.team_id == TEAM_RADIANT:
        #print("unit.name: ", unit.name)
        mid_tower = unit
  
  mid_tower_location = mid_tower.location
  dis_2tower = cal_distance(hero_unit.location, mid_tower_location)
  #print("dis_2tower: ", dis_2tower)

  #action_pb.actionType = CMsgBotWorldState.Action.Type.Value('ACTION_CHAT')
  
  hero_location = hero_unit.location
  hero_item = hero_unit.items

  self_to_lane_distance = get_lane_distance(hero_unit.location)
  #print("self_to_lane_distance: ", self_to_lane_distance)

  for unit in response.world_state.units:
    if unit.unit_type == CMsgBotWorldState.UnitType.Value('LANE_CREEP'):
      if unit.team_id == TEAM_RADIANT:
        unit_distance = get_lane_distance(unit.location)
        creeps.append((unit_distance, unit))

        print("self_to_lane_distance: ", self_to_lane_distance)
        print("unit_distance: ", unit_distance)

  #print("hero_location: ", hero_location)
  #print("hero_unit.name: ", hero_unit.name)
  #print("hero_unit.is_alive: ", hero_unit.is_alive)
  #print("hero_unit.team_id: ", hero_unit.team_id)
  '''
  for ability in hero_unit.abilities:
    #print("ability: ", ability)
    print("ability.ability_id: ", ability.ability_id)
    print("ability.slot: ", ability.slot)
    print("ability.is_activated: ", ability.is_activated)
    print("ability.level: ", ability.level)
    print("ability.cooldown_remaining: ", ability.cooldown_remaining)
    print("")
  '''
  
  #print("hero_unit.action_type: ", hero_unit.action_type)
  #print("hero_unit.anim_activity: ", hero_unit.anim_activity)
  #print("hero_unit.health_max: ", hero_unit.health_max)
  #print("hero_unit.xp_needed_to_level: ", hero_unit.xp_needed_to_level)
  #print("hero_unit.reliable_gold: ", hero_unit.reliable_gold)
  #print("hero_unit.unreliable_gold: ", hero_unit.unreliable_gold)
  #print("hero_unit.location: ", hero_unit.location)
  #print("hero_unit.facing: ", hero_unit.facing)
  print("hero_unit.incoming_tracking_projectiles: ", hero_unit.incoming_tracking_projectiles)
  #print("hero_unit.current_movement_speed: ", hero_unit.current_movement_speed)
  #print("hero_unit.base_movement_speed: ", hero_unit.base_movement_speed)
  #print("hero_unit.anim_cycle: ", hero_unit.anim_cycle)
  #print("hero_unit.unit_type: ", hero_unit.unit_type)
  #print("hero_unit.unit_type: ", hero_unit.unit_type)
  #print("hero_unit.modifiers: ", hero_unit.modifiers)
  #print("hero_unit.strength: ", hero_unit.strength)
  #print("hero_unit.agility: ", hero_unit.agility)
  #print("hero_unit.intelligence: ", hero_unit.intelligence)
  #print("hero_unit.primary_attribute: ", hero_unit.primary_attribute)

  #print("hero_item: ", hero_item)
  for item in hero_unit.items:
    #print("item: ", item)
    #print("item.ability_id: ", item.ability_id)
    #print("item.charges: ", item.charges)
    #rint("item.is_activated: ", item.is_activated)
    #print("item.slot: ", item.slot)
    #print("item.cooldown_remaining: ", item.cooldown_remaining)

    if item.ability_id == 34:
      if item.is_activated == True:
        #print("item.ability_id: ", item.ability_id)
        #print("item.is_activated: ", item.is_activated)
        #print("item.slot: ", item.slot)
        #print("item.cooldown_remaining: ", item.cooldown_remaining)

        stick_flag = True

  if (abs(mid_tower_location.x + 600 - hero_location.x) >= 500) or (abs(mid_tower_location.y + 600 - hero_location.y) >= 500):
    if item_buy_flag == False:
      move_flag = True
    #courier_flag = True
  else:
    if courier_stash_flag == False and courier_transfer_flag == False:
      item_use_flag = True
      #courier_flag = True
      #item_buy_flag = True
      #teleport_flag = True

  m = CMsgBotWorldState.Action.MoveToLocation()
  # x=-4450, y=-3800, z=256
  # x=-600, y=-480, z=256
  #m.location.x = mid_tower_location.x + 600
  #m.location.y = mid_tower_location.y + 600
  m.location.x = -600
  m.location.y = -480
  m.location.z = 0
  
  c = CMsgBotWorldState.Action.Chat()
  c.message = "test"
  c.to_allchat = 1

  i = CMsgBotWorldState.Action.PurchaseItem()
  i.item = 2
  #i.item_name = "item_tango"
  #i.item_name = "item_magic_stick"
  i.item_name = "item_ward_sentry"

  t = CMsgBotWorldState.Action.CastTree()
  t.abilitySlot = 0
  t.tree = 50

  #print("item_buy_flag: ", item_buy_flag)
  #print("item_use_flag: ", item_use_flag)
  #print("tree_flag: ", tree_flag)
  #print("stick_flag: ", stick_flag)
  #print("skill_learn_flag: ", skill_learn_flag)
  #print("move_flag: ", move_flag)
  #print("skill_use_flag: ", skill_use_flag)
  #print("courier_stash_flag: ", courier_stash_flag)
  #print("teleport_flag: ", teleport_flag)

  #action_pb.chat.CopyFrom(t) 
  action_pb = CMsgBotWorldState.Action()
  if dota_time > -80.0:
    if item_buy_flag == True:
      action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_PURCHASE_ITEM')
      action_pb.player = 0
      action_pb.purchaseItem.CopyFrom(i) 
      item_buy_flag = False
      #courier_stash_flag = True
    elif item_use_flag == True:
      if tree_flag == True:
        action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_CAST_TARGET_TREE')
        action_pb.player = 0
        action_pb.castTree.abilitySlot = 0
        action_pb.castTree.tree = 50
        #action_pb.castTree.CopyFrom(t)
        tree_flag = False
      elif stick_flag == True:
        action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_CAST_NO_TARGET')
        action_pb.player = 0 
        action_pb.cast.abilitySlot = -1
        stick_flag = False
      elif ward_flag == True:
        action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_CAST_POSITION')
        action_pb.player = 0

        action_pb.castLocation.abilitySlot = -1
        action_pb.castLocation.location.x = mid_tower_location.x + 600
        action_pb.castLocation.location.y = mid_tower_location.y + 600
        action_pb.castLocation.location.z = 0
        ward_flag = False
      else:
        action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_NONE')
    elif skill_learn_flag == True:
      action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_TRAIN_ABILITY')
      action_pb.player = 0
      action_pb.trainAbility.ability = "nevermore_shadowraze1"
      skill_flag = False
    elif move_flag == True:
      action_pb.actionDelay = 0  # action_dict['delay'] * DELAY_ENUM_TO_STEP
      action_pb.player = 0  # action_dict['delay'] * DELAY_ENUM_TO_STEP
      action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_MOVE_DIRECTLY')
      action_pb.moveDirectly.CopyFrom(m)
      move_flag = False
    elif skill_use_flag == True:
      #print("enermy_hero.handle: ", enermy_hero.handle)
      #action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_CAST_TARGET_TREE')
      action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_CAST_NO_TARGET')
      action_pb.player = 0 
      action_pb.cast.abilitySlot = 2
      #action_pb.castTarget.target = enermy_hero.handle
      skill_use_flag = False
    elif courier_stash_flag == True:
      action_pb.actionDelay = 0 
      action_pb.player = 0 
      action_pb.actionType = CMsgBotWorldState.Action.Type.Value('ACTION_COURIER')

      action_pb.courier.unit = 0 
      action_pb.courier.courier = 0
      action_pb.courier.action = 3
      courier_stash_flag = False
      courier_transfer_flag = True
    elif courier_transfer_flag == True:
      action_pb.actionDelay = 0 
      action_pb.player = 0 
      action_pb.actionType = CMsgBotWorldState.Action.Type.Value('ACTION_COURIER')

      action_pb.courier.unit = 0 
      action_pb.courier.courier = 0
      action_pb.courier.action = 6
      #courier_transfer_flag = False
    elif teleport_flag == True and dota_time > 20.0:
      action_pb.actionDelay = 0 
      action_pb.player = 0 
      action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_CAST_POSITION')

      action_pb.castLocation.abilitySlot = -16
      action_pb.castLocation.location.x = -6700
      action_pb.castLocation.location.y = -6700
      action_pb.castLocation.location.z = 0
    else:
      action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_NONE')
  else:
    action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_NONE')
  
  #print("action_pb: ", action_pb)
  print("")
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
  print("")


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
