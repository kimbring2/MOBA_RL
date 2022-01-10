from grpclib.client import Channel
from dotaservice.protos.DotaService_grpc import DotaServiceStub
from dotaservice.protos.DotaService_pb2 import Actions
from dotaservice.protos.DotaService_pb2 import GameConfig

from dotaservice.protos.dota_shared_enums_pb2 import DOTA_GAMEMODE_1V1MID, DOTA_GAMEMODE_TUTORIAL, DOTA_GAMEMODE_CUSTOM
from dotaservice.protos.DotaService_pb2 import HostMode
from dotaservice.protos.DotaService_pb2 import ObserveConfig
from dotaservice.protos.DotaService_pb2 import TEAM_DIRE, TEAM_RADIANT, Hero, HeroPick, HeroControlMode
from dotaservice.protos.DotaService_pb2 import NPC_DOTA_HERO_NEVERMORE, NPC_DOTA_HERO_OMNIKNIGHT, NPC_DOTA_HERO_DROW_RANGER, NPC_DOTA_HERO_SNIPER
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
#env = DotaServiceStub(Channel('192.168.1.150', 13337))

TICKS_PER_OBSERVATION = 15
HOST_TIMESCALE = 10

eps = np.finfo(np.float32).eps.item()

render = arguments.render
if render == True:
  HOST_MODE = HostMode.Value('HOST_MODE_GUI')
else:
  HOST_MODE = HostMode.Value('HOST_MODE_DEDICATED')


# NPC_DOTA_HERO_NEVERMORE, NPC_DOTA_HERO_OMNIKNIGHT, NPC_DOTA_HERO_DROW_RANGER

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


async def reset():
  # Get the initial observation.
  response = await asyncio.wait_for(env.reset(GameConfig(
        ticks_per_observation=TICKS_PER_OBSERVATION,
        host_timescale=HOST_TIMESCALE,
        host_mode=HOST_MODE,
        game_mode=DOTA_GAMEMODE_CUSTOM,
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


def get_item_charge(unit, item_id):
  items = unit.items
  for item in items:
    if item.ability_id == item_id: # and item['slot'] <= 5:
      return item.charges

  return None


def get_item_id(unit, item_slot):
  items = unit.items
  for item in items:
    if item.slot == item_slot: # and item['slot'] <= 5:
      return item.ability_id

  return None


item_name_list = ['item_branches', 'item_clarity', 'item_ward_observer', 'item_tango', 'item_gauntlets', 
                  'item_magic_stick', 'item_magic_wand', 'item_circlet', 'item_recipe_bracer', 'item_boots',
                  'item_faerie_fire', 'item_flask', 'item_enchanted_mango', 'item_ward_sentry', 'item_bracer',
                  'item_recipe_magic_wand', 'item_recipe_wraith_band', 'item_slippers', 'item_wraith_band'
                  ]

def get_item_type(unit, item_slot):
  # 0: non target
  # 1: unit target 
  # 2: position target
  # 3: tree target
  # 4: equipment
  item_dict = {
    ITEM_BRANCH_ID : 2,
    ITEM_CLARITY_ID : 0,
    ITEM_WARD_ID : 2,
    ITEM_TANGO_ID : 3,
    ITEM_GAUNTLETS_ID : 4,
    ITEM_MAGIC_STICK_ID : 0,
    ITEM_MAGIC_WAND_ID : 0,
    ITEM_CIRCLET : 4,
    ITEM_BRACER : 4,
    ITEM_BOOTS : 4,
    ITEM_FLASK_ID : 1,
    ITEM_MANGO_ID : 0,
    ITEM_FAERIE_FIRE_ID : 0,
    ITEM_RECIPE_BRACER : 4,
    ITEM_WARD_SENTRY_ID : 2,
    ITEM_RECIPE_MAGIC_STICK_ID : 4,
    ITEM_SLIPPERS : 4,
    ITEM_WRAITH_BAND : 4,
    ITEM_RECIPE_WRAITH_BAND : 4
  }

  items = unit.items
  for item in items:
    if item.slot == item_slot:
      item_type = get_item_id(unit, item.slot)
      return item_dict[item_type]

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
ITEM_WARD_ID = 42
ITEM_WARD_SENTRY_ID = 43
ITEM_TANGO_ID = 44
ITEM_GAUNTLETS_ID = 13
ITEM_MAGIC_STICK_ID = 34
ITEM_RECIPE_MAGIC_STICK_ID = 35
ITEM_MAGIC_WAND_ID = 36
ITEM_MANGO_ID = 216
ITEM_CIRCLET = 20
ITEM_RECIPE_BRACER = 72
ITEM_BRACER = 73
ITEM_BOOTS = 29
ITEM_FAERIE_FIRE_ID = 237
ITEM_SLIPPERS = 14
ITEM_WRAITH_BAND = 75
ITEM_RECIPE_WRAITH_BAND = 74

ITEM_ID_LIST = [ITEM_BRANCH_ID, ITEM_CLARITY_ID, ITEM_WARD_ID, ITEM_TANGO_ID, ITEM_GAUNTLETS_ID,
                ITEM_MAGIC_STICK_ID, ITEM_MAGIC_WAND_ID, ITEM_CIRCLET, ITEM_RECIPE_BRACER, ITEM_BRACER, ITEM_BOOTS,
                ITEM_RECIPE_BRACER, ITEM_MANGO_ID, ITEM_WARD_SENTRY_ID, ITEM_FLASK_ID, ITEM_RECIPE_MAGIC_STICK_ID, ITEM_SLIPPERS,
                ITEM_WRAITH_BAND, ITEM_RECIPE_WRAITH_BAND]
def get_item_matrix(unit):
  item_dict = {
    ITEM_BRANCH_ID : 0,
    ITEM_CLARITY_ID : 1,
    ITEM_WARD_ID : 2,
    ITEM_TANGO_ID : 3,
    ITEM_GAUNTLETS_ID: 4,
    ITEM_MAGIC_STICK_ID : 5,
    ITEM_MAGIC_WAND_ID : 6,
    ITEM_CIRCLET : 7,
    ITEM_RECIPE_BRACER : 8,
    ITEM_BRACER : 9,
    ITEM_BOOTS : 10,
    ITEM_FAERIE_FIRE_ID : 11,
    ITEM_FLASK_ID : 12,
    ITEM_MANGO_ID : 13,
    ITEM_WARD_SENTRY_ID : 14,
    ITEM_RECIPE_MAGIC_STICK_ID : 15,
    ITEM_SLIPPERS : 16,
    ITEM_WRAITH_BAND: 17,
    ITEM_RECIPE_WRAITH_BAND: 18
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
#item_name_list = ['item_branches', 'item_clarity', 'item_ward_observer', 'item_tango', 'item_gauntlets', 
#                  'item_magic_stick', 'item_magic_wand', 'item_circlet', 'item_recipe_bracer',  'item_boots',
#                  'item_recipe_wraith_band', 'item_slippers', 'item_wraith_band', 'item_tpscroll']
'''
init_item_1 = [
               'item_tango', 'item_tango', 'item_faerie_fire', 'item_clarity', 'item_circlet'
              ]

init_item_2 = [
               'item_ward_sentry', 'item_ward_sentry', 'item_branches', 'item_flask', 'item_gauntlets'
              ]
'''

init_item_1 = [
               'item_tpscroll', 'item_slippers', 'item_circlet'
              ]

init_item_2 = [
               'item_ward_sentry', 'item_flask', 'item_flask'
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
        "modifier_item_circlet": 10,
        "modifier_item_boots_of_speed": 11,
        "modifier_item_observer_ward": 12,
        "modifier_item_gauntlets": 13,
        "modifier_item_magic_stick": 14,
        "modifier_nevermore_shadowraze_counter": 15,
        "modifier_item_ironwood_branch": 16,
        "modifier_omniknight_repel": 17,
        "modifier_truesight": 18,
        'modifier_item_slippers': 19,
        'modifier_item_wraith_band': 20,
        'modifier_drow_ranger_frost_arrows': 21

    }
routes = [
        'nevermore_necromastery', 'nevermore_shadowraze1', 'nevermore_shadowraze1', 'nevermore_necromastery',
        'nevermore_shadowraze1', 'nevermore_necromastery', 'nevermore_shadowraze1', 'nevermore_necromastery',
        'nevermore_dark_lord', 'special_bonus_spell_amplify_8', 'nevermore_requiem', 'nevermore_dark_lord',
        'nevermore_dark_lord', 'nevermore_dark_lord', 'special_bonus_unique_nevermore_3', 'nevermore_requiem',
        'nevermore_requiem', 'special_bonus_unique_nevermore_1', 'special_bonus_unique_nevermore_5',
    ]

blank_slots = [0, 1, 2, 3, 4, 5]


skill_learn_flag1 = True
skill_learn_flag2 = True
skill_use_flag1 = False
skill_use_flag2 = True
item_buy_flag1 = False
item_buy_flag2 = 0
item_buy_index1 = 0
item_buy_index2 = len(init_item_2)
item_use_flag1 = False
item_use_flag2 = False
move_flag1 = False
move_flag2 = False
attack_flag1 = False
clarity_flag = False
tango_flag = False
mango_flag = False
flask_flag = False
faerie_fire_flag = False
ward_observer_flag = False
ward_sentry_flag = False
stick_flag = False
courier_stash_flag = False
courier_transfer_flag = False
teleport_flag = False
toggle_flag = False
async def step():
  global skill_learn_flag1
  global skill_learn_flag2
  global skill_use_flag1
  global skill_use_flag2
  global item_buy_flag1
  global item_buy_flag2
  global item_buy_index1
  global item_buy_index2
  global item_use_flag1
  global item_use_flag2
  global move_flag1
  global move_flag2
  global attack_flag1
  global clarity_flag
  global tango_flag
  global mango_flag
  global flask_flag
  global faerie_fire_flag
  global ward_observer_flag
  global ward_sentry_flag
  global stick_flag
  global courier_stash_flag
  global courier_transfer_flag
  global teleport_flag
  global toggle_flag

  creeps = []
  self_creep_min = None
  enemey_creep_min = None
  self_creep_min_distance = 9999
  enemey_creep_min_distance = 9999

  current_items = {}
  #time.sleep(5)

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

  while True:
    try:
      response = await env.observe(ObserveConfig(team_id=TEAM_RADIANT))
    except:
      print("observe break")
      continue
      #return -1

    #print('response.world_state: ', response.world_state)
    dota_time = response.world_state.dota_time
    print('response.world_state.dota_time: ', response.world_state.dota_time)

    #print('response.world_state.glyph_cooldown: ', response.world_state.glyph_cooldown)
    #print('response.world_state.damage_events: ', response.world_state.damage_events)
    if response.status == Status.Value('RADIANT_WIN'):
      print('RADIANT_WIN')
      break
    elif response.status == Status.Value('DIRE_WIN'):
      print('DIRE_WIN')
      break
    elif response.status == Status.Value('RESOURCE_EXHAUSTED'):
      print('RESOURCE_EXHAUSTED')
      continue
      #break

    hero1_unit = None
    hero2_unit = None
    enermy_hero = None

    self_tower = None
    enemy_tower = None

    hero_courier = None
    for unit in response.world_state.units:
      #print("unit.unit_type: ", unit.unit_type)
      #print("unit.handle: ", unit.handle)
      #print("CMsgBotWorldState.UnitType.Value('LANE_CREEP'): ", CMsgBotWorldState.UnitType.Value('LANE_CREEP'))
      #print("")
      if unit.unit_type == CMsgBotWorldState.UnitType.Value('HERO'):
        if unit.team_id == TEAM_RADIANT and unit.player_id == 0:
          #print("unit.player_id: ", unit.player_id)
          hero1_unit = unit
        elif unit.team_id == TEAM_RADIANT and unit.player_id == 1:
          #print("unit.player_id: ", unit.player_id)
          hero2_unit = unit
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

    #for player in response.world_state.players:
    #  print("player: ", player)

    #print("self_tower_position: ", self_tower_position)
    #print("enemy_tower_position: ", enemy_tower_position)
    
    hero1_modifiers = hero1_unit.modifiers
    #print("hero_modifiers: ", hero_modifiers)
    for hero1_modifier in hero1_modifiers:
      #print("hero_modifier.name: ", hero_modifier.name)
      hero1_modifier_index = modifier_name[hero1_modifier.name]
      #print("hero_modifier_index: ", hero_modifier_index)

    #dis_2tower = cal_distance(hero1_unit.location, self_tower.location)
    #print("hero1_unit: ", hero1_unit)

    #action_pb.actionType = CMsgBotWorldState.Action.Type.Value('ACTION_CHAT')
    
    hero1_location = hero1_unit.location
    hero2_location = hero2_unit.location

    hero1_item = hero1_unit.items

    #self_to_lane_distance = get_lane_distance(hero1_unit.location)
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

    flask_item_num = get_item_number(hero1_unit, ITEM_FLASK_ID)
    #print("flask_item_num: ", flask_item_num)

    mango_item_num = get_item_number(hero1_unit, ITEM_MANGO_ID)
    #print("mango_item_num: ", mango_item_num)

    faerie_fire_item_num = get_item_number(hero1_unit, ITEM_FAERIE_FIRE_ID)
    #print("faerie_fire_item_num: ", faerie_fire_item_num)

    ward_observer_item_num = get_item_number(hero1_unit, ITEM_WARD_ID)
    #print("ward_observer_item_num: ", ward_observer_item_num)

    ward_sentry_item_num = get_item_number(hero2_unit, ITEM_WARD_SENTRY_ID)
    #print("ward_sentry_item_num: ", ward_sentry_item_num)

    clarity_item_num = get_item_number(hero1_unit, ITEM_CLARITY_ID)
    #print("clarity_item_num: ", clarity_item_num)

    tango_item_num = get_item_number(hero1_unit, ITEM_TANGO_ID)
    #print("tango_item_num: ", tango_item_num)

    stick_item_num = get_item_number(hero1_unit, ITEM_MAGIC_STICK_ID)
    #print("stick_item_num: ", stick_item_num)


    flask_item_slot = get_item_slot(hero1_unit, ITEM_FLASK_ID)
    #print("flask_item_slot: ", flask_item_slot)

    ward_observer_item_slot = get_item_slot(hero1_unit, ITEM_WARD_ID)
    #print("ward_observer_item_slot: ", ward_observer_item_slot)

    ward_sentry_item_slot = get_item_slot(hero2_unit, ITEM_WARD_SENTRY_ID)
    #print("ward_sentry_item_slot: ", ward_sentry_item_slot)

    clarity_item_slot = get_item_slot(hero1_unit, ITEM_CLARITY_ID)
    #print("clarity_item_slot: ", clarity_item_slot)

    tango_item_slot = get_item_slot(hero1_unit, ITEM_TANGO_ID)
    #print("tango_item_slot: ", tango_item_slot)

    faerie_fire_item_slot = get_item_slot(hero1_unit, ITEM_FAERIE_FIRE_ID)
    #print("faerie_fire_item_num: ", faerie_fire_item_num)

    mango_item_slot = get_item_slot(hero1_unit, ITEM_MANGO_ID)
    #print("mango_item_slot: ", mango_item_slot)

    stick_item_slot = get_item_slot(hero1_unit, ITEM_MAGIC_STICK_ID)
    #print("stick_item_slot: ", stick_item_slot)

    item_matrix = get_item_matrix(hero1_unit)
    #print("item_matrix: ", item_matrix)

    #ability_matrix = get_ability_matrix(hero1_unit)
    #print("ability_matrix: ", ability_matrix)

    #print("hero1_location: ", hero1_location)
    #print("hero2_location: ", hero2_location)

    #print("hero1_unit.health_max: ", hero1_unit.health_max)
    #print("hero1_unit.last_hits: ", hero1_unit.last_hits)

    if hero1_unit.health < 500:
      skill_use_flag2 = True
    #print("enemey_creep_min_distance: ", enemey_creep_min_distance)
    #if enemey_creep_min_distance < 2000:
    #  skill_use_flag = True

    if enermy_hero != None:
      enemey_hero_distance = in_facing_distance(enermy_hero, hero1_unit, 750, r=250, normalization=False)
      #print("enemey_hero_distance: ", enemey_hero_distance)
      if enemey_hero_distance == 1:
        skill_use_flag1 = True

    #print("enemey_creep_min: ", enemey_creep_min)
    if enemey_creep_min != None:
      enemey_creep_distance = in_facing_distance(enemey_creep_min, hero1_unit, 400, r=250, normalization=False)
      #print("enemey_creep_distance: ", enemey_creep_distance)
      if enemey_creep_distance == 1:
        #skill_use_flag1 = True
        attack_flag1 = True

    for ability in hero1_unit.abilities:
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

    for ability in hero2_unit.abilities:
      #ability_name_type = get_ability_name_type(ability)
      #print("ability_name_type: ", ability_name_type)

      print("ability: ", ability)
      print("ability.ability_id: ", ability.ability_id)
      print("ability.slot: ", ability.slot)
      #print("ability.is_activated: ", ability.is_activated)
      print("ability.level: ", ability.level)
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
    for item in hero1_unit.items:
      #print("item: ", item)
      #print("item.ability_id: ", item.ability_id)
      #print("item.charges: ", item.charges)
      #print("item.is_activated: ", item.is_activated)
      #print("item.slot: ", item.slot)
      #print("item.cooldown_remaining: ", item.cooldown_remaining)

      if item.slot <= 6:
        #print("item.slot: ", item.slot)
        #item_id = get_item_id(hero_unit, item.slot)
        #print("item_id: ", item_id)
        item_type = get_item_type(hero1_unit, item.slot)
        #print("item_type: ", item_type)

    if (abs(-900 - hero1_location.x) >= 500) or (abs(-870 - hero1_location.y) >= 500):
      #if item_buy_index1 == len(init_item_1):
      #move_flag1 = True
      #courier_flag = True
      pass
    else:
      if courier_stash_flag == False and courier_transfer_flag == False:
        #item_buy_flag1 = True
        #stick_flag = True
        #item_use_flag = True
        #courier_flag = True
        #item_buy_flag = True
        #teleport_flag = True
        teleport_flag = True
        pass

    if (abs(self_tower.location.x + 300 - hero2_location.x) >= 1000) or (abs(self_tower.location.y + 300 - hero2_location.y) >= 1000):
      if item_buy_flag2 == len(init_item_2):
        #move_flag2 = True
        pass
    
    c = CMsgBotWorldState.Action.Chat()
    c.message = "test"
    c.to_allchat = 1

    #print("teleport_flag: ", teleport_flag)
    #print("attack_flag1: ", attack_flag1)
    #print("move_flag1: ", move_flag1)
    #print("move_flag2: ", move_flag2)
    #print("item_use_flag: ", item_use_flag)
    #print("item_buy_flag1: ", item_buy_flag1)
    #print("item_buy_index1: ", item_buy_index1)
    #print("item_buy_flag2: ", item_buy_flag2)
    print("toggle_flag: ", toggle_flag)

    #action_pb.chat.CopyFrom(t) 
    action_pb1 = CMsgBotWorldState.Action()
    action_pb2 = CMsgBotWorldState.Action()
    #print("skill_use_flag1: ", skill_use_flag1)
    if dota_time > -85.0:
      if item_buy_flag1 == True and item_buy_index1 < len(init_item_1):
        action_pb1.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_PURCHASE_ITEM')

        i = CMsgBotWorldState.Action.PurchaseItem()
        i.item = 1
        i.item_name = init_item_1[item_buy_index1]
        #print("i.item_name: ", i.item_name)

        action_pb1.purchaseItem.CopyFrom(i) 

        item_buy_index1 += 1
        #item_use_flag = True
        courier_stash_flag = True
      elif attack_flag1 == True:
        action_pb1.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_ATTACK_TARGET')
        m = CMsgBotWorldState.Action.AttackTarget()
        if enemey_creep_min != None:
          m.target = enemey_creep_min.handle
        else:
          m.target = -1

        m.once = False
        action_pb1.attackTarget.CopyFrom(m)
      elif item_use_flag1 == True:
        # 0: non target
        # 1: unit target
        # 2: position target
        # 3: tree target
        if clarity_flag == True and clarity_item_num >= 1:
          # 0: non target
          action_pb1.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_CAST_TARGET')
          action_pb1.castTarget.abilitySlot = -(clarity_item_slot + 1)
          action_pb1.castTarget.target = hero1_unit.handle
          clarity_flag = False
        elif tango_flag == True and tango_item_num >= 1:
          # 3: tree target
          action_pb1.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_CAST_TARGET_TREE')
          action_pb1.castTree.abilitySlot = -(tango_item_slot + 1)
          action_pb1.castTree.tree = 10
          tango_flag = False
        elif mango_flag == True and mango_item_num >= 1:
          action_pb1.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_CAST_NO_TARGET')
          action_pb1.cast.abilitySlot = -(mango_item_slot+1)
          mango_flag = False
        elif stick_flag == True and stick_item_num >= 1:
          # 0: non target
          action_pb1.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_CAST_NO_TARGET')
          action_pb1.cast.abilitySlot = -(stick_item_slot + 1)
          stick_flag = False
        elif ward_observer_flag == True and ward_observer_item_num >= 1:
          # 2: position target
          action_pb1.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_CAST_POSITION')
          action_pb1.castLocation.abilitySlot = -(ward_observer_item_slot + 1)
          action_pb1.castLocation.location.x = hero1_location.x + 600
          action_pb1.castLocation.location.y = hero1_location.y + 600
          action_pb1.castLocation.location.z = 0
          ward_observer_flag = False
        elif flask_flag == True and flask_item_num >= 1:
          action_pb1.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_CAST_TARGET')
          action_pb1.castTarget.target = hero1_unit.handle
          action_pb1.castTarget.abilitySlot = -(flask_item_slot + 1)
          faerie_fire_flag = False
        elif faerie_fire_flag == True:
          action_pb1.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_CAST_NO_TARGET')
          action_pb1.cast.abilitySlot = -(faerie_fire_item_slot+1)
          faerie_fire_flag = False
        else:
          action_pb1.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_NONE')
      elif skill_learn_flag1 == True:
        action_pb1.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_TRAIN_ABILITY')
        #action_pb1.trainAbility.ability = "nevermore_shadowraze1"
        action_pb1.trainAbility.ability = "drow_ranger_frost_arrows"

        skill_learn_flag1 = False
      elif move_flag1 == True:
        action_pb1.actionDelay = 0
        action_pb1.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_MOVE_DIRECTLY')

        m = CMsgBotWorldState.Action.MoveToLocation()
        m.location.x = -900 + 200
        m.location.y = -780 + 200
        m.location.z = 0

        action_pb1.moveDirectly.CopyFrom(m)

        move_flag1 = False
      elif skill_use_flag1 == True:
        #print("enermy_hero.handle: ", enermy_hero.handle)
        action_pb1.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_CAST_NO_TARGET')
        action_pb1.cast.abilitySlot = 2

        skill_use_flag1 = False
      elif courier_stash_flag == True:
        action_pb1.actionDelay = 0 
        action_pb1.actionType = CMsgBotWorldState.Action.Type.Value('ACTION_COURIER')

        action_pb1.courier.unit = 0 
        action_pb1.courier.courier = 0
        action_pb1.courier.action = 3
        courier_stash_flag = False
        courier_transfer_flag = True
      elif courier_transfer_flag == True:
        action_pb1.actionDelay = 0 
        action_pb1.actionType = CMsgBotWorldState.Action.Type.Value('ACTION_COURIER')

        action_pb1.courier.unit = 0 
        action_pb1.courier.courier = 0
        action_pb1.courier.action = 6
        courier_transfer_flag = False
      elif teleport_flag == True:
        action_pb1.actionDelay = 0 
        action_pb1.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_CAST_POSITION')

        action_pb1.castLocation.abilitySlot = -16
        action_pb1.castLocation.location.x = -6700
        action_pb1.castLocation.location.y = -6700
        action_pb1.castLocation.location.z = 0

        teleport_flag = False
      elif toggle_flag == True:
        action_pb1.actionDelay = 0 
        action_pb1.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_CAST_TOGGLE')

        action_pb1.castToggle.abilitySlot = 0

        #toggle_flag = False
      else:
        action_pb1.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_NONE')
    else:
      action_pb1.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_NONE')
    

    if dota_time > -85.0:
      if item_buy_flag2 != len(init_item_2):
        action_pb2.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_PURCHASE_ITEM')

        i2 = CMsgBotWorldState.Action.PurchaseItem()
        i2.item = 2
        i2.item_name = init_item_2[item_buy_flag2]

        action_pb2.purchaseItem.CopyFrom(i2) 

        item_buy_flag2 += 1
      elif item_use_flag2 == True:
        if ward_sentry_flag == True and ward_sentry_item_num >= 1:
            # 2: position target
            action_pb2.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_CAST_POSITION')
            action_pb2.castLocation.abilitySlot = -(ward_sentry_item_slot + 1)
            action_pb2.castLocation.location.x = hero2_location.x + 600
            action_pb2.castLocation.location.y = hero2_location.y + 600
            action_pb2.castLocation.location.z = 0
            ward_sentry_flag = False
        else:
          action_pb2.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_NONE')
      elif skill_learn_flag2 == True:
        action_pb2.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_TRAIN_ABILITY')
        #action_pb2.trainAbility.ability = "omniknight_purification"
        action_pb2.trainAbility.ability = "drow_ranger_wave_of_silence"
        skill_learn_flag2 = False
      elif skill_use_flag2 == True:
        #action_pb2.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_CAST_TARGET')
        #action_pb2.castTarget.abilitySlot = 1
        #action_pb2.castTarget.target = hero1_unit.handle

        action_pb2.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_CAST_POSITION')
        action_pb2.castLocation.abilitySlot = 1
        action_pb2.castLocation.location.x = hero2_location.x + 600
        action_pb2.castLocation.location.y = hero2_location.y + 600
        action_pb2.castLocation.location.z = 0

        skill_use_flag2 = False
      elif move_flag2 == True:
        action_pb2.actionDelay = 0
        action_pb2.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_MOVE_DIRECTLY')

        m = CMsgBotWorldState.Action.MoveToLocation()
        m.location.x = -900 + 300
        m.location.y = -780 + 300
        m.location.z = 0

        action_pb2.moveDirectly.CopyFrom(m)

        move_flag2 = False
      else:
        action_pb2.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_NONE')
    else:
      action_pb2.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_NONE')
    '''
    action_pb1 = CMsgBotWorldState.Action()
    action_pb2 = CMsgBotWorldState.Action()

    action_pb1.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_NONE')
    action_pb2.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_NONE')
    '''
    print("action_pb1: ", action_pb1)
    #print("action_pb2: ", action_pb2)
    print("")
    
    actions = []
    for i in range(0, 1):
      action_pb1.player = 0
      actions.append(action_pb1)

    for i in range(0, 1):
      action_pb2.player = 1
      actions.append(action_pb2)
    
    actions_pb = CMsgBotWorldState.Actions(actions=actions)
    response = await env.act(Actions(actions=actions_pb, team_id=TEAM_RADIANT))
    #print('response_home.world_state.dota_time: ', response_home.world_state.dota_time)

    print("")


async def main():
  #try:
  #  print("reset")
  #  await reset()
  #except:
  #  print("except")

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