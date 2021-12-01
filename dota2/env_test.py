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

eps = np.finfo(np.float32).eps.item()

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

  #print('get_vector_lenth(lane_vector): ', get_vector_lenth(lane_vector))
  #print('p_vector: ', p_vector)
  #print('get_vector_lenth(p_vector): ', get_vector_lenth(p_vector))
  cos_to_lane_axis = get_vector_dot(lane_vector, p_vector) / (
      get_vector_lenth(lane_vector) * get_vector_lenth(p_vector) + eps)
  d = get_vector_lenth(p_vector)

  return abs(d * cos_to_lane_axis)


def cal_distance(p1, p2):
    if p1 is None or p2 is None:
        return -1

    return np.sqrt(np.power(p1.x - p2.x, 2) + np.power(p1.y - p2.y, 2))


def get_item_number(unit, item_id):
  res = 0
  items = unit.items

  for item in items:
    if item.ability_id == item_id:
        res += item.charges

  return res


def get_item_slot(unit, item_id, in_equipment=False):
  items = unit.items
  for item in items:
      if item.ability_id == item_id: # and item['slot'] <= 5:
          if not in_equipment:
              return item.slot
          elif in_equipment and item.slot <= 5:
              return item.slot

  return None


def get_item_type(unit, item_slot):
  # 
  item_dict = {
    ITEM_FLASK_ID : 2,
    ITEM_FAERIE_FIRE_ID : 8,
    ITEM_TANGO_ID : 6,
    ITEM_MANGO_ID : 7,
    ITEM_CLARITY_ID : 1,
    ITEM_BRANCH_ID : 0,
    ITEM_BOTTLE_ID : 3,
    ITEM_WARD_ID : 4,
    ITEM_WARD_SENTRY_ID : 5,
    ITEM_MAGIC_STICK_ID : 9,
    ITEM_MAGIC_WAND_ID : 10,
    ITEM_CIRCLET : 11,
    ITEM_BRACER : 12,
    ITEM_WRAITH : 13,
    ITEM_LESSER_CRIT : 14,
    ITEM_WARD_DISPENSER_ID : 15,
    ITEM_BOOTS : 16
  }

  items = unit.items
  for item in items:
    if item.slot == item_slot:
      return item.slot

  return None


def get_ability_level(unit, ability_id):
  abilities = unit.abilities
  for ability in abilities:
    if ability.ability_id == ability_id: # and item['slot'] <= 5:
      return ability.level

  return None


def in_facing_distance(hero, u, distance, r=250, normalization=False):
    x = math.cos(math.radians(hero.facing)) * distance + \
        hero.location.x
    y = math.sin(math.radians(hero.facing)) * distance + \
        hero.location.y

    location = CMsgBotWorldState.Vector(x=x, y=y, z=512.0)

    d = cal_distance(u.location, location) # 技能中心点距离单位的位置
    if d < r and u.team_id != hero.team_id:
        if normalization:
            return (r - d) / r # 在中心点时为1，靠近边缘趋近与0，出范围为0
        else:
            return 1
    else:
        return 0


ITEM_BRANCH_ID = 16
ITEM_CLARITY_ID = 38
ITEM_FLASK_ID = 39
ITEM_BOTTLE_ID = 41
ITEM_WARD_ID = 42
ITEM_WARD_SENTRY_ID = 43
ITEM_TANGO_ID = 44
ITEM_MANGO_ID = 216
ITEM_FAERIE_FIRE_ID = 237
ITEM_MAGIC_STICK_ID = 34
ITEM_MAGIC_WAND_ID = 36
ITEM_CIRCLET = 20
ITEM_BRACER = 73
ITEM_WRAITH = 75
ITEM_WARD_DISPENSER_ID = 218
ITEM_BOOTS = 29

ITEM_ID_LIST = [ITEM_BRANCH_ID, ITEM_CLARITY_ID, ITEM_FLASK_ID, ITEM_BOTTLE_ID, ITEM_WARD_ID, ITEM_WARD_SENTRY_ID,
                ITEM_TANGO_ID, ITEM_MANGO_ID, ITEM_FAERIE_FIRE_ID, ITEM_MAGIC_STICK_ID, ITEM_MAGIC_WAND_ID,
                ITEM_CIRCLET, ITEM_BRACER, ITEM_WRAITH, ITEM_WARD_DISPENSER_ID, ITEM_BOOTS]
def get_item_matrix(unit):
  item_dict = {
    ITEM_BRANCH_ID : 0,
    ITEM_CLARITY_ID : 1,
    ITEM_FLASK_ID : 2,
    ITEM_BOTTLE_ID : 3,
    ITEM_WARD_ID : 4,
    ITEM_WARD_SENTRY_ID : 5,
    ITEM_TANGO_ID : 6,
    ITEM_MANGO_ID : 7,
    ITEM_FAERIE_FIRE_ID : 8,
    ITEM_MAGIC_STICK_ID : 9,
    ITEM_MAGIC_WAND_ID : 10,
    ITEM_CIRCLET : 11,
    ITEM_BRACER : 12,
    ITEM_WRAITH : 13,
    ITEM_WARD_DISPENSER_ID : 14,
    ITEM_BOOTS : 15
  }

  item_matrix = np.zeros(len(item_dict))
  for item_id in ITEM_ID_LIST:
    item_num = get_item_number(unit, item_id)
    item_matrix[item_dict[item_id]] = item_num

  return item_matrix


ABILITY_SHADOWRAZE1_ID = 5059
ABILITY_SHADOWRAZE2_ID = 5060
ABILITY_SHADOWRAZE3_ID = 5061
ABILITY_NECROMASTERY_ID = 5062
ABILITY_DARKLOAD_ID = 5063
ABILITY_REQUIEM_ID = 5064

ABILITY_ID_LIST = [ABILITY_SHADOWRAZE1_ID, ABILITY_SHADOWRAZE2_ID, ABILITY_SHADOWRAZE3_ID, ABILITY_NECROMASTERY_ID,
                   ABILITY_DARKLOAD_ID, ABILITY_REQUIEM_ID]
def get_ability_matrix(unit):
  # 0 : No Target
  # 1 : Passive
  # 2 : Aura
  # 3 : Etc
  ability_dict = {
    ABILITY_SHADOWRAZE1_ID : 0,
    ABILITY_SHADOWRAZE2_ID : 1,
    ABILITY_SHADOWRAZE3_ID : 2,
    ABILITY_NECROMASTERY_ID : 3,
    ABILITY_DARKLOAD_ID : 4,
    ABILITY_REQUIEM_ID : 5
  }

  ability_matrix = np.zeros(len(ability_dict))
  for ability_id in ABILITY_ID_LIST:
    ability_level = get_ability_level(unit, ability_id)
    ability_matrix[ability_dict[ability_id]] = ability_level

  return ability_matrix


# 'magic_wand': ['item_branches', 'item_branches', 'item_recipe_magic_wand']
# 'bracer': ['item_circlet', 'item_gauntlets', 'item_recipe_bracer']
# 'wraith_band': ['item_circlet', 'item_slippers', 'item_recipe_wraith_band']
init_item = [
              'item_circlet', 'item_slippers', 'item_recipe_wraith_band', 'item_enchanted_mango', 'item_enchanted_mango',
              'item_enchanted_mango', 'item_enchanted_mango'
            ]

modifier_name = {
        "modifier_nevermore_necromastery": 1,
        "modifier_nevermore_shadowraze_debuff": 2,
        "modifier_flask_healing": 3,
        "modifier_nevermore_requiem_fear": 4,
        "modifier_tango_heal": 5,
        "modifier_item_faerie_fire": 6,
        "modifier_item_enchanted_mango": 7,
        "modifier_fountain_aura_buff": 8,
        "modifier_tower_aura_bonus": 9,
        "modifier_item_ironwood_branch": 10,
        "modifier_item_circlet": 11,
        "modifier_item_slippers": 12,
        "modifier_item_wraith_band": 13
    }

routes = [
        'nevermore_necromastery', 'nevermore_shadowraze1', 'nevermore_shadowraze1', 'nevermore_necromastery',
        'nevermore_shadowraze1', 'nevermore_necromastery', 'nevermore_shadowraze1', 'nevermore_necromastery',
        'nevermore_dark_lord', 'special_bonus_spell_amplify_8', 'nevermore_requiem', 'nevermore_dark_lord',
        'nevermore_dark_lord', 'nevermore_dark_lord', 'special_bonus_unique_nevermore_3', 'nevermore_requiem',
        'nevermore_requiem', 'special_bonus_unique_nevermore_1', 'special_bonus_unique_nevermore_5'
    ]

blank_slots = [0, 1, 2, 3, 4, 5]

skill_learn_flag = True
skill_use_flag = False
item_buy_flag = 0
item_use_flag = True
move_flag = False
tree_flag = False
mango_flag = False
ward_flag = False
stick_flag = False
faerie_fire_flag = False
flask_flag = True
branches_flag = False
courier_stash_flag = False
courier_transfer_flag = False
teleport_flag = False
async def step():
  global skill_learn_flag
  global skill_use_flag
  global item_buy_flag
  global item_use_flag
  global move_flag
  global tree_flag
  global mango_flag
  global ward_flag
  global stick_flag
  global faerie_fire_flag
  global branches_flag
  global flask_flag
  global courier_stash_flag
  global courier_transfer_flag
  global teleport_flag

  creeps = []
  self_creep_min = None
  enemey_creep_min = None
  self_creep_min_distance = 9999
  enemey_creep_min_distance = 9999

  current_items = {}

  response = await asyncio.wait_for(env.observe(ObserveConfig(team_id=TEAM_RADIANT)), timeout=120)
  #print('response.world_state: ', response.world_state)
  dota_time = response.world_state.dota_time
  print('response.world_state.dota_time: ', response.world_state.dota_time)
  #print('response.world_state.glyph_cooldown: ', response.world_state.glyph_cooldown)
  #print('response.world_state.damage_events: ', response.world_state.damage_events)

  hero_unit = None
  enermy_hero = None

  self_tower = None
  enemy_tower = None

  hero_courier = None
  for unit in response.world_state.units:
    #print("unit.unit_type: ", unit.unit_type)
    #print("unit.handle: ", unit.handle)
    #print("CMsgBotWorldState.UnitType.Value('LANE_CREEP'): ", CMsgBotWorldState.UnitType.Value('LANE_CREEP'))
    #print("unit.team_id: ", unit.team_id)
    #print("")
    if unit.unit_type == CMsgBotWorldState.UnitType.Value('HERO') and unit.name == "npc_dota_hero_nevermore":
      if unit.team_id == TEAM_RADIANT and unit.player_id == 0:
        hero_unit = unit
      elif unit.team_id == TEAM_DIRE:
        enermy_hero = unit
    elif unit.unit_type == CMsgBotWorldState.UnitType.Value('TOWER'):
      #print("unit.name: ", unit.name)
      if unit.team_id == TEAM_RADIANT and unit.name == "npc_dota_goodguys_tower1_mid":
        self_tower = unit
      elif unit.team_id == TEAM_DIRE and unit.name == "npc_dota_badguys_tower1_mid":
        enemy_tower = unit
    elif unit.unit_type == CMsgBotWorldState.UnitType.Value('COURIER'):
      #print("unit.team_id: ", unit.team_id)
      #print("unit.player_id: ", unit.player_id)
      if unit.team_id == TEAM_RADIANT and unit.player_id == 0:
        hero_courier = unit
        #print("unit: ", unit)

  #print("self_tower_position: ", self_tower_position)
  #print("enemy_tower_position: ", enemy_tower_position)
  
  hero_modifiers = hero_unit.modifiers
  #print("hero_modifiers: ", hero_modifiers)
  for hero_modifier in hero_modifiers:
    #print("hero_modifier.name: ", hero_modifier.name)
    hero_modifier_index = modifier_name[hero_modifier.name]
    #print("hero_modifier_index: ", hero_modifier_index)

  dis_2tower = cal_distance(hero_unit.location, self_tower.location)
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
        dist = cal_distance(unit.location, enemy_tower.location)
        if dist < self_creep_min_distance:
          self_creep_min = unit
          self_creep_min_distance = dist
      elif unit.team_id == TEAM_DIRE:
        dist = cal_distance(unit.location, self_tower.location)
        if dist < enemey_creep_min_distance:
          enemey_creep_min = unit
          enemey_creep_min_distance = dist

  flask_item_num = get_item_number(hero_unit, ITEM_FLASK_ID)
  print("flask_item_num: ", flask_item_num)

  mango_item_num = get_item_number(hero_unit, ITEM_MANGO_ID)
  #print("mango_item_num: ", mango_item_num)

  faerie_fire_item_num = get_item_number(hero_unit, ITEM_FAERIE_FIRE_ID)
  #print("faerie_fire_item_num: ", faerie_fire_item_num)

  flask_item_slot = get_item_slot(hero_unit, ITEM_FLASK_ID)
  print("flask_item_slot: ", flask_item_slot)

  mango_item_slot = get_item_slot(hero_unit, ITEM_MANGO_ID)
  #print("mango_item_slot: ", mango_item_slot)

  branches_item_slot = get_item_slot(hero_unit, ITEM_BRANCH_ID)
  #print("branches_item_slot: ", branches_item_slot)

  bottle_item_slot = get_item_slot(hero_unit, ITEM_BOTTLE_ID)
  #print("bottle_item_slot: ", bottle_item_slot)

  faerie_fire_item_slot = get_item_slot(hero_unit, ITEM_FAERIE_FIRE_ID)
  #print("faerie_fire_item_slot: ", faerie_fire_item_slot)

  item_matrix = get_item_matrix(hero_unit)
  #print("item_matrix: ", item_matrix)

  ability_matrix = get_ability_matrix(hero_unit)
  #print("ability_matrix: ", ability_matrix)

  if faerie_fire_item_num != 0:
    #item_use_flag = True
    #faerie_fire_flag = True
    pass
  #  pass
  #if mango_item_num != 0:
  #  item_use_flag = True
  #  mango_flag = True

  print("hero_location: ", hero_location)
  #print("enemey_creep_min_distance: ", enemey_creep_min_distance)
  #if enemey_creep_min_distance < 2000:
  #  skill_use_flag = True

  if enermy_hero != None:
    enemey_hero_distance = in_facing_distance(enermy_hero, hero_unit, 750, r=250, normalization=False)
    #print("enemey_hero_distance: ", enemey_hero_distance)

    if enemey_hero_distance == 1:
      skill_use_flag = True

  #print("item_use_flag: ", item_use_flag)
  #print("faerie_fire_flag: ", faerie_fire_flag)
  #print("hero_unit.name: ", hero_unit.name)
  #print("hero_unit.is_alive: ", hero_unit.is_alive)
  #print("hero_unit.team_id: ", hero_unit.team_id)
  
  for ability in hero_unit.abilities:
    #ability_name_type = get_ability_name_type(ability)
    #print("ability_name_type: ", ability_name_type)

    #print("ability: ", ability)
    #print("ability.ability_id: ", ability.ability_id)
    #print("ability.slot: ", ability.slot)
    #print("ability.is_activated: ", ability.is_activated)
    #print("ability.level: ", ability.level)
    #print("ability.cooldown_remaining: ", ability.cooldown_remaining)
    #print("")
    pass
  
  
  #print("hero_unit.action_type: ", hero_unit.action_type)
  #print("hero_unit.anim_activity: ", hero_unit.anim_activity)
  #print("hero_unit.health_max: ", hero_unit.health_max)
  #print("hero_unit.xp_needed_to_level: ", hero_unit.xp_needed_to_level)
  #print("hero_unit.reliable_gold: ", hero_unit.reliable_gold)
  #print("hero_unit.unreliable_gold: ", hero_unit.unreliable_gold)
  #print("hero_unit.location: ", hero_unit.location)
  #print("hero_unit.facing: ", hero_unit.facing)
  #print("hero_unit.incoming_tracking_projectiles: ", hero_unit.incoming_tracking_projectiles)
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
        #print("item.charges: ", item.charges)
        #print("item.is_activated: ", item.is_activated)
        #print("item.slot: ", item.slot)
        #print("item.cooldown_remaining: ", item.cooldown_remaining)

        #stick_flag = True
        pass

  if (abs(self_tower.location.x + 600 - hero_location.x) >= 500) or (abs(self_tower.location.y + 600 - hero_location.y) >= 500):
    if item_buy_flag == False:
      move_flag = True
      #courier_flag = True
      pass
  else:
    if courier_stash_flag == False and courier_transfer_flag == False:
      #item_use_flag = True
      #courier_flag = True
      #item_buy_flag = True
      #teleport_flag = True
      pass

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
  i.item_name = init_item[item_buy_flag]
  #i.item_name = "item_magic_stick"
  #i.item_name = "item_ward_sentry"

  t = CMsgBotWorldState.Action.CastTree()
  t.abilitySlot = 0
  t.tree = 50

  print("item_buy_flag: ", item_buy_flag)
  #print("item_use_flag: ", item_use_flag)
  #print("tree_flag: ", tree_flag)
  #print("mango_flag: ", mango_flag)
  #print("stick_flag: ", stick_flag)
  #print("skill_learn_flag: ", skill_learn_flag)
  #print("move_flag: ", move_flag)
  #print("skill_use_flag: ", skill_use_flag)
  #print("courier_stash_flag: ", courier_stash_flag)
  #print("teleport_flag: ", teleport_flag)

  #action_pb.chat.CopyFrom(t) 
  action_pb = CMsgBotWorldState.Action()
  if dota_time > -80.0:
    if item_buy_flag != len(init_item) - 1:
      action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_PURCHASE_ITEM')
      action_pb.player = 0
      action_pb.purchaseItem.CopyFrom(i) 
      item_buy_flag += 1
      #courier_stash_flag = True
    elif item_use_flag == True:
      if tree_flag == True:
        action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_CAST_TARGET_TREE')
        action_pb.player = 0
        action_pb.castTree.abilitySlot = -1
        action_pb.castTree.tree = 50
        #action_pb.castTree.CopyFrom(t)
        tree_flag = False
      elif stick_flag == True:
        action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_CAST_NO_TARGET')
        action_pb.player = 0 
        action_pb.cast.abilitySlot = -1
        stick_flag = False
      elif mango_flag == True:
        action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_CAST_NO_TARGET')
        action_pb.player = 0 
        #print("mango_item_slot: ", mango_item_slot)
        #action_pb.cast.abilitySlot = -(mango_item_slot+1)
        action_pb.cast.abilitySlot = -(mango_item_slot+1)
        mango_flag = False
      elif ward_flag == True:
        action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_CAST_POSITION')
        action_pb.player = 0
        action_pb.castLocation.abilitySlot = -1
        action_pb.castLocation.location.x = self_tower.position.x + 600
        action_pb.castLocation.location.y = self_tower.position.y + 600
        action_pb.castLocation.location.z = 0
        ward_flag = False
      elif faerie_fire_flag == True:
        action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_CAST_NO_TARGET')
        action_pb.player = 0 
        action_pb.cast.abilitySlot = -(faerie_fire_item_slot+1)
        faerie_fire_flag = False
      elif flask_flag == True:
        action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_CAST_POSITION')
        action_pb.player = 0 
        action_pb.castLocation.abilitySlot = -(flask_item_slot+1)
        action_pb.castLocation.location.x = hero_unit.location.x
        action_pb.castLocation.location.y = hero_unit.location.y
        action_pb.castLocation.location.z = 0
        flask_flag = False
      elif branches_flag == True:
        # {{action.castLocation.abilitySlot}, {action.castLocation.location.x, action.castLocation.location.y, 0.0}, {0}}
        action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_CAST_POSITION')
        action_pb.player = 0 
        action_pb.castLocation.abilitySlot = -(branches_item_slot+1)
        action_pb.castLocation.location.x = hero_unit.location.x + 300
        action_pb.castLocation.location.y = hero_unit.location.y + 900
        action_pb.castLocation.location.z = 0
        branches_flag = False
      else:
        action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_NONE')
    elif skill_learn_flag == True:
      action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_TRAIN_ABILITY')
      action_pb.player = 0
      action_pb.trainAbility.ability = "nevermore_shadowraze1"
      skill_learn_flag = False
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
  
  print("action_pb: ", action_pb)
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