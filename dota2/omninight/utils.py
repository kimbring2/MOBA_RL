import numpy as np
import math
import random
import collections
import threading
import time
import timeit
from absl import logging
import tensorflow as tf
from itertools import repeat

from dotaservice.protos.dota_gcmessages_common_bot_script_pb2 import CMsgBotWorldState
from dotaservice.protos.DotaService_pb2 import TEAM_DIRE, TEAM_RADIANT, Hero, HeroPick, HeroControlMode

import tensorflow_probability as tfp
tfd = tfp.distributions

MAP_HALF_WIDTH = 7000.
OPPOSITE_TEAM = {TEAM_DIRE: TEAM_RADIANT, TEAM_RADIANT: TEAM_DIRE}

TICKS_PER_OBSERVATION = 15 
TICKS_PER_SECOND = 30
MAX_MOVE_SPEED = 550
MAX_MOVE_IN_OBS = (MAX_MOVE_SPEED / TICKS_PER_SECOND) * TICKS_PER_OBSERVATION
N_MOVE_ENUMS = 9
MOVE_ENUMS = np.arange(N_MOVE_ENUMS, dtype=np.float32) - int(N_MOVE_ENUMS / 2)
MOVE_ENUMS *= MAX_MOVE_IN_OBS / (N_MOVE_ENUMS - 1) * 2
OBSERVATIONS_PER_SECOND = TICKS_PER_SECOND / TICKS_PER_OBSERVATION
MAX_UNITS = 5 + 5 + 16 + 16 + 1 + 1
ACTION_OUTPUT_COUNTS = {'enum': 5, 'x': 9, 'y': 9, 'target_unit': MAX_UNITS, 'ability': 3, 'item': 6}
OUTPUT_KEYS = ACTION_OUTPUT_COUNTS.keys()
INPUT_KEYS = ['env', 'allied_heroes', 'enemy_heroes', 'allied_nonheroes', 'enemy_nonheroes',
              'allied_towers', 'enemy_towers']
REWARD_KEYS = ['enemy', 'win', 'xp', 'hp', 'kills', 'death', 'lh', 'denies', 'tower_hp', 'mana']

xp_to_reach_level = {
    1: 0,
    2: 230,
    3: 600,
    4: 1080,
    5: 1680,
    6: 2300,
    7: 2940,
    8: 3600,
    9: 4280,
    10: 5080,
    11: 5900,
    12: 6740,
    13: 7640,
    14: 8865,
    15: 10115,
    16: 11390,
    17: 12690,
    18: 14015,
    19: 15415,
    20: 16905,
    21: 18405,
    22: 20155,
    23: 22155,
    24: 24405,
    25: 26905
}

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
ITEM_ID_LIST = [ITEM_BRANCH_ID, ITEM_CLARITY_ID, ITEM_WARD_ID, ITEM_TANGO_ID, ITEM_GAUNTLETS_ID,
                ITEM_MAGIC_STICK_ID, ITEM_MAGIC_WAND_ID, ITEM_CIRCLET, ITEM_RECIPE_BRACER, ITEM_BRACER, ITEM_BOOTS,
                ITEM_RECIPE_BRACER, ITEM_MANGO_ID, ITEM_WARD_SENTRY_ID, ITEM_FLASK_ID, ITEM_RECIPE_MAGIC_STICK_ID]


def get_player(state, player_id):
    for player in state.players:
        if player.player_id == player_id:
            return player
            
    raise ValueError("hero {} not found in state:\n{}".format(player_id, state))


def get_unit(state, player_id):
    for unit in state.units:
        if unit.unit_type == CMsgBotWorldState.UnitType.Value('HERO') \
            and unit.player_id == player_id:
            return unit
    
    return None  
    #raise ValueError("unit {} not found in state:\n{}".format(player_id, state))


def get_mid_tower(state, team_id):
    for unit in state.units:
        if unit.unit_type == CMsgBotWorldState.UnitType.Value('TOWER') \
            and unit.team_id == team_id \
            and 'tower1_mid' in unit.name:
            return unit
            
    raise ValueError("tower not found in state:\n{}".format(state))


def is_unit_attacking_unit(unit_attacker, unit_target):
    # Check for a direct attack.
    if unit_attacker.attack_target_handle == unit_target.handle:
        return 1.
        
    # Go over the incoming projectiles from this unit.
    for projectile in unit_target.incoming_tracking_projectiles:
        if projectile.caster_handle == unit_attacker.handle and projectile.is_attack:
            return 1.
            
    # Otherwise, the unit is not attacking the target, and there are no incoming projectiles.
    return 0.


def is_invulnerable(unit):
    for mod in unit.modifiers:
        if mod.name == "modifier_invulnerable":
            return True
            
    return False
    
    
def unit_separation(state, team_id):
    # Break apart the full unit-list into specific categories for allied and
    # enemy unit groups of various types so we don't have to repeatedly iterate
    # the full unit-list again.
    allied_heroes, allied_nonheroes, allied_creep, allied_towers = [], [], [], []
    enemy_heroes, enemy_nonheroes, enemy_creep, enemy_towers = [], [], [], []
    for unit in state.units:
        # check if allied or enemy unit
        if unit.team_id == team_id:
            if unit.unit_type == CMsgBotWorldState.UnitType.Value('HERO'):
                allied_heroes.append(unit)
            elif unit.unit_type == CMsgBotWorldState.UnitType.Value('CREEP_HERO'):
                allied_nonheroes.append(unit)
            elif unit.unit_type == CMsgBotWorldState.UnitType.Value('LANE_CREEP'):
                allied_creep.append(unit)
            elif unit.unit_type == CMsgBotWorldState.UnitType.Value('TOWER'):
                if unit.name[-5:] == "1_mid":  # Only consider the mid tower for now.
                  allied_towers.append(unit)
        else:
            if unit.unit_type == CMsgBotWorldState.UnitType.Value('HERO'):
                enemy_heroes.append(unit)
            elif unit.unit_type == CMsgBotWorldState.UnitType.Value('CREEP_HERO'):
                enemy_nonheroes.append(unit)
            elif unit.unit_type == CMsgBotWorldState.UnitType.Value('LANE_CREEP'):
                enemy_creep.append(unit)
            elif unit.unit_type == CMsgBotWorldState.UnitType.Value('TOWER'):
                if unit.name[-5:] == "1_mid":  # Only consider the mid tower for now.
                  enemy_towers.append(unit)

    return allied_heroes, enemy_heroes, allied_nonheroes, enemy_nonheroes, \
        allied_creep, enemy_creep, allied_towers, enemy_towers
        

def is_unit_attacking_unit(unit_attacker, unit_target):
    # Check for a direct attack.
    if unit_attacker.attack_target_handle == unit_target.handle:
        return 1.
        
    # Go over the incoming projectiles from this unit.
    for projectile in unit_target.incoming_tracking_projectiles:
        if projectile.caster_handle == unit_attacker.handle and projectile.is_attack:
            return 1.
            
    # Otherwise, the unit is not attacking the target, and there are no incoming projectiles.
    return 0.


def is_invulnerable(unit):
    for mod in unit.modifiers:
        if mod.name == "modifier_invulnerable":
            return True
            
    return False


def get_vector_lenth( p):
    return math.sqrt(math.pow(p.x, 2) + math.pow(p.y, 2))


def get_vector_dot(p1, p2):
    return p1.x * p2.x + p1.y * p2.y


def get_lane_distance(p1):
    start_position = CMsgBotWorldState.Vector(x=3700, y=3100, z=256)
    stop_position = CMsgBotWorldState.Vector(x=-240, y=-110, z=256)

    self_tower_position = CMsgBotWorldState.Vector(x=524, y=652, z=256)
    enemy_tower_position = CMsgBotWorldState.Vector(x=-1544, y=-1408, z=256)

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


def get_ability_level(unit, ability_id):
  abilities = unit.abilities
  for ability in abilities:
    if ability.ability_id == ability_id: # and item['slot'] <= 5:
      return ability.level

  return None


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


def unit_matrix(unit_list, hero_unit, only_self=False, max_units=16):
    # We are always inserting an 'zero' unit to make sure the policy doesn't barf
    # We can't just pad this, because we will otherwise lose track of corresponding chosen
    # actions relating to output indices. Even if we would, batching multiple sequences together
    # would then be another error prone nightmare.
    handles = np.full(max_units, -1)
    m = np.zeros((max_units, 29))
    i = 0
    for unit in unit_list:
        if unit.is_alive:
            if only_self:
                if unit != hero_unit:
                    continue
                     
            if i >= max_units:
                break
            
            rel_hp = 1.0 - (unit.health / unit.health_max)
            rel_mana = 0.0
            if unit.mana_max > 0:
                rel_mana = 1.0 - (unit.mana / unit.mana_max)
                  
            loc_x = unit.location.x / MAP_HALF_WIDTH
            loc_y = unit.location.y / MAP_HALF_WIDTH
            loc_z = (unit.location.z / 512.) - 0.5
            distance_x = (hero_unit.location.x - unit.location.x)
            distance_y = (hero_unit.location.y - unit.location.y)
            distance = math.sqrt(distance_x**2 + distance_y**2)
            norm_distance = (distance / MAP_HALF_WIDTH) - 0.5

            # Get the direction where the unit is facing.
            facing_sin = math.sin(unit.facing * (2 * math.pi) / 360)	
            facing_cos = math.cos(unit.facing * (2 * math.pi) / 360)

            # Calculates normalized boolean value [-0.5 or 0.5] of if unit is within 
            # attack range of hero.
            in_attack_range = float(distance <= hero_unit.attack_range) - 0.5

            # Calculates normalized boolean value [-0.5 or 0.5] of if that unit
            # is currently targeting me with right-click attacks.
            is_attacking_me = float(is_unit_attacking_unit(unit, hero_unit)) - 0.5
            me_attacking_unit = float(is_unit_attacking_unit(hero_unit, unit)) - 0.5

            in_ability_phase = -0.5
            for a in unit.abilities:
                if a.is_in_ability_phase or a.is_channeling:
                    in_ability_phase = 0.5
                    break

            lane_distance = float(get_lane_distance(unit.location) / 9999)

            item_matrix = get_item_matrix(unit) / 5.0
            #if unit.unit_type == CMsgBotWorldState.UnitType.Value('HERO'):
            #    print("item_matrix: ", item_matrix)

            matrix = np.array([
                    rel_hp, loc_x, loc_y, loc_z, norm_distance, facing_sin, facing_cos,
                    in_attack_range, is_attacking_me, me_attacking_unit, rel_mana, in_ability_phase, lane_distance
                ])

            m[i] = np.concatenate((matrix, item_matrix))
            #print("m[i]: ", m[i])

            # Because we are currently only attacking, check if these units are valid
            # HACK: Make a nice interface for this, per enum used?
            if unit.is_invulnerable or unit.is_attack_immune:
                handles[i] = -1
            elif unit.team_id == hero_unit.team_id and unit.unit_type == CMsgBotWorldState.UnitType.Value('TOWER'):
                # Its own tower:
                handles[i] = -1
            else:
                handles[i] = unit.handle

            i += 1
              
    return m, handles


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
    ITEM_RECIPE_MAGIC_STICK_ID : 15
  }

  item_matrix = np.zeros(len(item_dict))
  for item_id in ITEM_ID_LIST:
    item_num = get_item_number(unit, item_id)
    item_matrix[item_dict[item_id]] = item_num

  return item_matrix


def get_item_id(unit, item_slot):
  items = unit.items
  for item in items:
    if item.slot == item_slot: # and item['slot'] <= 5:
      return item.ability_id

  return None


def get_item_slot(unit, item_id, in_equipment=False):
  items = unit.items
  for item in items:
    if item.ability_id == item_id: # and item['slot'] <= 5:
      if not in_equipment:
        return item.slot
      elif in_equipment and item.slot <= 5:
        return item.slot

  return None


item_name_list = ['item_branches', 'item_clarity', 'item_ward_observer', 'item_tango', 'item_gauntlets', 
                  'item_magic_stick', 'item_magic_wand', 'item_circlet', 'item_recipe_bracer', 'item_boots',
                  'item_faerie_fire', 'item_flask', 'item_enchanted_mango', 'item_ward_sentry', 'item_bracer',
                  'item_recipe_magic_wand'
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
    ITEM_RECIPE_MAGIC_STICK_ID : 4
  }

  items = unit.items
  for item in items:
    if item.slot == item_slot:
      item_type = get_item_id(unit, item.slot)
      return item_dict[item_type]

  return None

def ability_available(ability):
  return ability.is_activated and ability.level > 0 and ability.cooldown_remaining == 0 \
      and ability.is_fully_castable


def item_available(item):
  return item.is_activated and item.level > 0 and item.cooldown_remaining == 0 \
      and item.is_fully_castable


def action_masks(player_unit, unit_handles):
  """Mask the head with possible actions."""
  if not player_unit.is_alive:
    # Dead player means it can only do the NoOp.
    masks = {key: np.zeros((1, 1, val)) for key, val in ACTION_OUTPUT_COUNTS.items()}
    masks['enum'][0, 0, 0] = 1

    return masks

  masks = {key: np.ones((1, 1, val)) for key, val in ACTION_OUTPUT_COUNTS.items()}
  for ability in player_unit.abilities:
    if ability.slot != 0 and ability.slot != 1 and ability.slot != 5:
      continue

    # Note: `is_fully_castable` implies there is mana for it.
    # Note: `is_in_ability_phase` means it is currently doing an ability.
    if not ability_available(ability):
      # Can't use ability
      if ability.slot == 0 or ability.slot == 1:
        masks['ability'][0, 0, ability.slot] = 0
      elif ability.slot == 5:
        masks['ability'][0, 0, 2] = 0

  if not masks['ability'].any():
    # No abilities possible, so we cannot choose to use any abilities.
    masks['enum'][0, 0, 3] = 0

  for item in player_unit.items:
    if item.slot >= 6:
      continue

    # Note: `is_fully_castable` implies there is mana for it.
    # Note: `is_in_ability_phase` means it is currently doing an ability.
    if not item_available(item):
      # Can't use ability
      masks['item'][0, 0, item.slot] = 0

  if not masks['item'].any():
    # No abilities possible, so we cannot choose to use any abilities.
    masks['enum'][0, 0, 4] = 0

  valid_units = unit_handles != -1
  valid_units[0] = 0 # The 'self' hero can never be targetted.
  if not valid_units.any():
    # All units invalid, so we cannot choose the high-level attack head:
    masks['enum'][0, 0, 2] = 0

  masks['target_unit'][0, 0] = valid_units

  return masks


def action_to_pb(unit_id, action_dict, state, unit_handles):
  if action_dict['target_unit'][0] > 44:
    print("action_dict['target_unit']: ", action_dict['target_unit'])

  # TODO(tzaman): Recrease the scope of this function. Make it a converter only.
  hero_unit = get_unit(state, player_id=0)
  action_pb = CMsgBotWorldState.Action()
  action_pb.actionDelay = 0  # action_dict['delay'] * DELAY_ENUM_TO_STEP
  action_pb.player = unit_id
  action_enum = action_dict['enum']

  hero_location = hero_unit.location
  if action_enum == 0:
    action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_NONE')
  elif action_enum == 1:
    action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_MOVE_DIRECTLY')
    m = CMsgBotWorldState.Action.MoveToLocation()
    hero_location = hero_unit.location

    m.location.x = hero_location.x + MOVE_ENUMS[action_dict['x']]
    m.location.y = hero_location.y + MOVE_ENUMS[action_dict['y']]
    m.location.z = 0
    action_pb.moveDirectly.CopyFrom(m)
  elif action_enum == 2:
    action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_ATTACK_TARGET')
    m = CMsgBotWorldState.Action.AttackTarget()
    if 'target_unit' in action_dict:
      m.target = unit_handles[action_dict['target_unit']]
    else:
      m.target = -1

    m.once = False
    action_pb.attackTarget.CopyFrom(m)
  elif action_enum == 3:
    if action_dict['ability'] == 0:
      action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_CAST_TARGET')
      action_pb.castTarget.abilitySlot = action_dict['ability']

      # purification
      if 'target_unit' in action_dict:
        action_pb.castTarget.target = unit_handles[action_dict['target_unit']]
      else:
        action_pb.castTarget.target = -1
    elif action_dict['ability'] == 1:
      action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_CAST_TARGET')
      action_pb.castTarget.abilitySlot = action_dict['ability']

      # repel
      if 'target_unit' in action_dict:
        action_pb.castTarget.target = unit_handles[action_dict['target_unit']]
      else:
        action_pb.castTarget.target = -1
    elif action_dict['ability'] == 2:
      action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_CAST_NO_TARGET')
      action_pb.cast.abilitySlot = action_dict['ability']

  elif action_enum == 4:
    item_type = get_item_type(hero_unit, action_dict['item'])

    # 0: non target
    # 1: unit target 
    # 2: position target
    # 3: tree target
    # 4: equipment
    if item_type == 0:
      action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_CAST_NO_TARGET')
      action_pb.cast.abilitySlot = -(action_dict['item'] + 1)
    elif item_type == 1:
      action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_CAST_TARGET')
      action_pb.castTarget.abilitySlot = -(action_dict['item'] + 1)

      if 'target_unit' in action_dict:
        action_pb.castTarget.target = unit_handles[action_dict['target_unit']]
      else:
        action_pb.castTarget.target = -1

    elif item_type == 2:
      action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_CAST_POSITION')
      action_pb.castLocation.abilitySlot = -(action_dict['item'] + 1)
      action_pb.castLocation.location.x = hero_location.x + MOVE_ENUMS[action_dict['x']]
      action_pb.castLocation.location.y = hero_location.y + MOVE_ENUMS[action_dict['y']]
      action_pb.castLocation.location.z = 0
    elif item_type == 3:
      action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_CAST_TARGET_TREE')
      t = CMsgBotWorldState.Action.CastTree()
      t.abilitySlot = -(action_dict['item'] + 1)
      t.tree = 50
      action_pb.castTree.CopyFrom(t) 
    else:
      action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_NONE')
  else:
    raise ValueError("unknown action {}".format(action_enum))
    action_pb.player = TEAM_RADIANT
    
  return action_pb


@tf.function
def sample_action(logits, mask):
  class MaskedCategorical():
    def __init__(self, log_probs, mask):
      self.log_probs = log_probs

    def sample(self):
      return tfd.Categorical(probs=self.log_probs[-1]).sample()

  def masked_softmax(logits, mask, dim=1):
    """Returns log-probabilities."""
    mask = tf.cast(mask, 'int32')
    exp = tf.math.exp(logits)
    masked_exp = exp
    masked_exp *= tf.cast(tf.not_equal(mask, 0), 'float32')
    masked_sumexp = tf.math.reduce_sum(masked_exp, axis=dim, keepdims=True)
    logsumexp = tf.math.log(masked_sumexp)
    log_probs = logits - logsumexp

    masked_logits = logits * tf.cast(tf.not_equal(mask, 0), 'float32')
    masked_log_logits = tf.keras.layers.Softmax()(masked_logits)

    return masked_log_logits, tf.expand_dims(masked_logits, 0)

  # TODO(tzaman): Have the sampler kind be user-configurable.
  log_probs, masked_logits = masked_softmax(logits=logits, mask=mask)
  sample = MaskedCategorical(log_probs=log_probs, mask=mask).sample()

  return sample, masked_logits


@tf.function
def sample_x_actions(logits, mask):
  class MaskedCategorical():
    def __init__(self, log_probs, mask):
      self.log_probs = log_probs

    def sample(self):
      return tfd.Categorical(probs=self.log_probs[-1]).sample()

  def masked_softmax(logits, mask, dim=1):
    """Returns log-probabilities."""
    mask = tf.cast(mask, 'int32')
    exp = tf.math.exp(logits)
    masked_exp = exp
    masked_exp *= tf.cast(tf.not_equal(mask, 0), 'float32')
    masked_sumexp = tf.math.reduce_sum(masked_exp, axis=dim, keepdims=True)
    logsumexp = tf.math.log(masked_sumexp)
    log_probs = logits - logsumexp
    masked_logits = logits * tf.cast(tf.not_equal(mask, 0), 'float32')

    masked_log_logits = tf.keras.layers.Softmax()(masked_logits)

    return masked_log_logits, tf.expand_dims(masked_logits, 0)

  # TODO(tzaman): Have the sampler kind be user-configurable.
  log_probs, masked_logits = masked_softmax(logits=logits, mask=mask)
  sample = MaskedCategorical(log_probs=log_probs, mask=mask).sample()

  return sample, masked_logits


@tf.function
def sample_y_actions(logits, mask):
  class MaskedCategorical():
    def __init__(self, log_probs, mask):
      self.log_probs = log_probs

    def sample(self):
      return tfd.Categorical(probs=self.log_probs[-1]).sample()

  def masked_softmax(logits, mask, dim=1):
    """Returns log-probabilities."""
    mask = tf.cast(mask, 'int32')
    exp = tf.math.exp(logits)
    masked_exp = exp
    masked_exp *= tf.cast(tf.not_equal(mask, 0), 'float32')
    masked_sumexp = tf.math.reduce_sum(masked_exp, axis=dim, keepdims=True)
    logsumexp = tf.math.log(masked_sumexp)
    log_probs = logits - logsumexp
    masked_logits = logits * tf.cast(tf.not_equal(mask, 0), 'float32')

    masked_log_logits = tf.keras.layers.Softmax()(masked_logits)

    return masked_log_logits, tf.expand_dims(masked_logits, 0)

  log_probs, masked_logits = masked_softmax(logits=logits, mask=mask)
  sample = MaskedCategorical(log_probs=log_probs, mask=mask).sample()

  return sample, masked_logits


@tf.function
def sample_target_unit_actions(logits, mask):
  class MaskedCategorical():
    def __init__(self, log_probs, mask):
      self.log_probs = log_probs

    def sample(self):
      return tfd.Categorical(probs=self.log_probs[-1]).sample()

  def masked_softmax(logits, mask, dim=1):
    """Returns log-probabilities."""
    mask = tf.cast(mask, 'int32')
    exp = tf.math.exp(logits)
    masked_exp = exp
    masked_exp *= tf.cast(tf.not_equal(mask, 0), 'float32')
    masked_sumexp = tf.math.reduce_sum(masked_exp, axis=dim, keepdims=True)
    logsumexp = tf.math.log(masked_sumexp)
    log_probs = logits - logsumexp
    masked_logits = logits * tf.cast(tf.not_equal(mask, 0), 'float32')

    masked_log_logits = tf.keras.layers.Softmax()(masked_logits)

    return masked_log_logits, tf.expand_dims(masked_logits, 0)

  log_probs, masked_logits = masked_softmax(logits=logits, mask=mask)
  sample = MaskedCategorical(log_probs=log_probs, mask=mask).sample()

  return sample, masked_logits


@tf.function
def sample_ability_actions(logits, mask):
  class MaskedCategorical():
    def __init__(self, log_probs, mask):
      self.log_probs = log_probs

    def sample(self):
      return tfd.Categorical(probs=self.log_probs[-1]).sample()

  def masked_softmax(logits, mask, dim=1):
    """Returns log-probabilities."""
    mask = tf.cast(mask, 'int32')
    exp = tf.math.exp(logits)
    masked_exp = exp
    masked_exp *= tf.cast(tf.not_equal(mask, 0), 'float32')
    masked_sumexp = tf.math.reduce_sum(masked_exp, axis=dim, keepdims=True)
    logsumexp = tf.math.log(masked_sumexp)
    log_probs = logits - logsumexp
    masked_logits = logits * tf.cast(tf.not_equal(mask, 0), 'float32')

    masked_log_logits = tf.keras.layers.Softmax()(masked_logits)

    return masked_log_logits, tf.expand_dims(masked_logits, 0)

  log_probs, masked_logits = masked_softmax(logits=logits, mask=mask)
  sample = MaskedCategorical(log_probs=log_probs, mask=mask).sample()

  return sample, masked_logits


@tf.function
def sample_item_actions(logits, mask):
  class MaskedCategorical():
    def __init__(self, log_probs, mask):
      self.log_probs = log_probs

    def sample(self):
      return tfd.Categorical(probs=self.log_probs[-1]).sample()

  def masked_softmax(logits, mask, dim=1):
    """Returns log-probabilities."""
    mask = tf.cast(mask, 'int32')
    exp = tf.math.exp(logits)
    masked_exp = exp
    masked_exp *= tf.cast(tf.not_equal(mask, 0), 'float32')
    masked_sumexp = tf.math.reduce_sum(masked_exp, axis=dim, keepdims=True)
    logsumexp = tf.math.log(masked_sumexp)
    log_probs = logits - logsumexp
    masked_logits = logits * tf.cast(tf.not_equal(mask, 0), 'float32')

    masked_log_logits = tf.keras.layers.Softmax()(masked_logits)

    return masked_log_logits, tf.expand_dims(masked_logits, 0)

  log_probs, masked_logits = masked_softmax(logits=logits, mask=mask)
  sample = MaskedCategorical(log_probs=log_probs, mask=mask).sample()

  return sample, masked_logits


def select_actions(action_dict, heads_logits, action_masks, masked_heads_logits):
  """From all heads, select actions."""
  # First select the high-level action.
  action_dict['enum'], enum_masked_probs = sample_action(heads_logits['enum'][0], action_masks['enum'][0])
  masked_heads_logits['enum'] = enum_masked_probs

  if action_dict['enum'] == 0:  # Nothing
    pass
  elif action_dict['enum'] == 1:  # Move
    action_dict['x'], x_masked_probs = sample_x_actions(heads_logits['x'][0], action_masks['x'][0])
    action_dict['y'], y_masked_probs = sample_y_actions(heads_logits['y'][0], action_masks['y'][0])

    masked_heads_logits['x'] = x_masked_probs
    masked_heads_logits['y'] = y_masked_probs
  elif action_dict['enum'] == 2:  # Attack
    action_dict['target_unit'], target_unit_masked_probs = sample_target_unit_actions(heads_logits['target_unit'][0], 
                                                                                      action_masks['target_unit'][0])
    masked_heads_logits['target_unit'] = target_unit_masked_probs
  elif action_dict['enum'] == 3:  # Ability
    action_dict['ability'], ability_masked_probs = sample_ability_actions(heads_logits['ability'][0], 
                                                                          action_masks['ability'][0])
    masked_heads_logits['ability'] = ability_masked_probs
  elif action_dict['enum'] == 4:  # Item
    action_dict['item'], item_masked_probs = sample_item_actions(heads_logits['item'][0], 
                                                                 action_masks['item'][0])
    masked_heads_logits['item'] = item_masked_probs
  else:
    ValueError("Invalid Action Selection.")
  
  return action_dict, masked_heads_logits


def train_ability(hero_unit, ability_name, unit_id):
  # Just try to level up the first ability.
  action_pb = CMsgBotWorldState.Action()
  action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_TRAIN_ABILITY')
  action_pb.player = unit_id
  action_pb.trainAbility.ability = ability_name
  action_pb.actionDelay = 0 

  return action_pb


def buy_item(item_name, unit_id):
  # Just try to level up the first ability.
  action_pb = CMsgBotWorldState.Action()

  i = CMsgBotWorldState.Action.PurchaseItem()
  i.item = 2
  i.item_name = item_name

  action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_PURCHASE_ITEM')
  action_pb.player = unit_id
  action_pb.purchaseItem.CopyFrom(i) 

  return action_pb


def none_action(unit_id):
  action_pb = CMsgBotWorldState.Action()
  action_pb.actionDelay = 0 
  action_pb.player = unit_id  
  action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_NONE')

  return action_pb


def get_total_xp(level, xp_needed_to_level):
    if level == 25:
        return xp_to_reach_level[level]

    xp_required_for_next_level = xp_to_reach_level[level + 1] - xp_to_reach_level[level]
    missing_xp_for_next_level = (xp_required_for_next_level - xp_needed_to_level)

    return xp_to_reach_level[level] + missing_xp_for_next_level


def get_reward(prev_obs, obs, player_id):
    """Get the reward."""
    unit_init = get_unit(prev_obs, player_id=player_id)
    unit = get_unit(obs, player_id=player_id)
    player_init = get_player(prev_obs, player_id=player_id)
    player = get_player(obs, player_id=player_id)

    mid_tower_init = get_mid_tower(prev_obs, team_id=player.team_id)
    mid_tower = get_mid_tower(obs, team_id=player.team_id)

    # TODO(tzaman): make a nice reward container?
    reward = {key: 0. for key in REWARD_KEYS}

    # XP Reward
    xp_init = get_total_xp(level=unit_init.level, xp_needed_to_level=unit_init.xp_needed_to_level)
    xp = get_total_xp(level=unit.level, xp_needed_to_level=unit.xp_needed_to_level)
    reward['xp'] = (xp - xp_init) * 0.001  # One creep is around 40 xp; 40*0.001=0.04

    # HP and death reward
    if unit_init.is_alive and unit.is_alive:
        hp_rel = unit.health / unit.health_max
        low_hp_factor = 1. + (1 - hp_rel)**2  # rel=0 -> 3; rel=0 -> 2; rel=0.5->1.25; rel=1 -> 1.
        hp_rel_init = unit_init.health / unit_init.health_max
        reward['hp'] = (hp_rel - hp_rel_init) * low_hp_factor * 0.3
        # NOTE: Fully depleting hp costs: (0 - 1) * (1+(1-0)^2) * 0.2 = - 0.4

        mana_rel = unit.mana / unit.mana_max
        low_mana_factor = 1. + (1 - mana_rel)**2  # rel=0 -> 3; rel=0 -> 2; rel=0.5->1.25; rel=1 -> 1.
        mana_rel_init = unit_init.mana / unit_init.mana_max
        reward['mana'] = (mana_rel - mana_rel_init) * low_mana_factor * 0.03
        # NOTE: Fully depleting mana costs: (0 - 1) * (1+(1-0)^2) * 0.1 = - 0.2

    # Kill and death rewards
    reward['kills'] = (player.kills - player_init.kills) * 1.0
    reward['death'] = (player.deaths - player_init.deaths) * -1.0

    # Last-hit reward
    lh = unit.last_hits - unit_init.last_hits
    reward['lh'] = lh * 0.5

    # Deny reward
    denies = unit.denies - unit_init.denies
    reward['denies'] = denies * 0.05

    # Tower hp reward. Note: towers have 1900 hp.
    reward['tower_hp'] = (mid_tower.health - mid_tower_init.health) / 1900.
    #print("reward: ", reward)
    
    return reward


MultiHostSettings = collections.namedtuple('MultiHostSettings', 'strategy hosts training_strategy encode decode')
def init_learner_multi_host(num_training_tpus: int):
  """Performs common learner initialization including multi-host setting.

  In multi-host setting, this function will enter a loop for slave learners
  until the master signals end of training.

  Args:
    num_training_tpus: Number of training TPUs.

  Returns:
    A MultiHostSettings object.
  """
  tpu = ''
  job_name = None

  if tf.config.experimental.list_logical_devices('TPU'):
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu, job_name=job_name)
    topology = tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.experimental.TPUStrategy(resolver)

    assert num_training_tpus % topology.num_tasks == 0
    num_training_tpus_per_task = num_training_tpus // topology.num_tasks

    hosts = []
    training_coordinates = []
    for per_host_coordinates in topology.device_coordinates:
      host = topology.cpu_device_name_at_coordinates(per_host_coordinates[0], job=job_name)
      task_training_coordinates = (per_host_coordinates[:num_training_tpus_per_task])
      training_coordinates.extend([[c] for c in task_training_coordinates])

      inference_coordinates = per_host_coordinates[num_training_tpus_per_task:]
      hosts.append((host, [topology.tpu_device_name_at_coordinates(c, job=job_name) for c in inference_coordinates]))

    training_da = tf.tpu.experimental.DeviceAssignment(topology, training_coordinates)
    training_strategy = tf.distribute.experimental.TPUStrategy(resolver, device_assignment=training_da)

    return MultiHostSettings(strategy, hosts, training_strategy, tpu_encode, tpu_decode)
  else:
    tf.device('/cpu').__enter__()
    any_gpu = tf.config.experimental.list_logical_devices('GPU')
    device_name = '/device:GPU:0' if any_gpu else '/device:CPU:0'
    strategy = tf.distribute.OneDeviceStrategy(device=device_name)
    enc = lambda x: x
    dec = lambda x, s=None: x if s is None else tf.nest.pack_sequence_as(s, x)

    return MultiHostSettings(strategy, [('/cpu', [device_name])], strategy, enc, dec)


class UnrollStore(tf.Module):
  """Utility module for combining individual environment steps into unrolls."""
  def __init__(self, num_envs, unroll_length, timestep_specs, num_overlapping_steps=0, name='UnrollStore'):
    super(UnrollStore, self).__init__(name=name)
    with self.name_scope:
      self._full_length = num_overlapping_steps + unroll_length + 1

      def create_unroll_variable(spec):
        z = tf.zeros([num_envs, self._full_length] + spec.shape.dims, dtype=spec.dtype)
        return tf.Variable(z, trainable=False, name=spec.name)

      self._unroll_length = unroll_length
      self._num_overlapping_steps = num_overlapping_steps
      self._state = tf.nest.map_structure(create_unroll_variable, timestep_specs)

      # For each environment, the index into the environment dimension of the
      # tensors in self._state where we should add the next element.
      self._index = tf.Variable(tf.fill([num_envs], tf.constant(num_overlapping_steps, tf.int32)), trainable=False, name='index')

  @property
  def unroll_specs(self):
    return tf.nest.map_structure(lambda v: tf.TensorSpec(v.shape[1:], v.dtype), self._state)

  @tf.function
  @tf.Module.with_name_scope
  def append(self, env_ids, values):
    """Appends values and returns completed unrolls.

    Args:
      env_ids: 1D tensor with the list of environment IDs for which we append
        data.
        There must not be duplicates.
      values: Values to add for each environment. This is a structure
        (in the tf.nest sense) of tensors following "timestep_specs", with a
        batch front dimension which must be equal to the length of 'env_ids'.

    Returns:
      A pair of:
        - 1D tensor of the environment IDs of the completed unrolls.
        - Completed unrolls. This is a structure of tensors following
          'timestep_specs', with added front dimensions: [num_completed_unrolls,
          num_overlapping_steps + unroll_length + 1].
    """
    tf.debugging.assert_equal(tf.shape(env_ids), tf.shape(tf.unique(env_ids)[0]), 
                                 message=f'Duplicate environment ids in store {self.name}')
    tf.nest.map_structure(lambda s: tf.debugging.assert_equal(tf.shape(env_ids)[0], tf.shape(s)[0],
                            message=(f'Batch dimension must equal the number of environments in store {self.name}.')), values)
    
    curr_indices = self._index.sparse_read(env_ids)
    unroll_indices = tf.stack([env_ids, curr_indices], axis=-1)
    for s, v in zip(tf.nest.flatten(self._state), tf.nest.flatten(values)):
      #tf.print("unroll_indices: ", unroll_indices)
      #tf.print("v: ", v)
      s.scatter_nd_update(unroll_indices, v)

    # Intentionally not protecting against out-of-bounds to make it possible to
    # detect completed unrolls.
    self._index.scatter_add(tf.IndexedSlices(1, env_ids))

    return self._complete_unrolls(env_ids)

  @tf.function
  @tf.Module.with_name_scope
  def reset(self, env_ids):
    """Resets state.

    Note, this is only intended to be called when environments need to be reset
    after preemptions. Not at episode boundaries.

    Args:
      env_ids: The environments that need to have their state reset.
    """
    self._index.scatter_update(tf.IndexedSlices(self._num_overlapping_steps, env_ids))

    # The following code is the equivalent of:
    # s[env_ids, :j] = 0
    j = self._num_overlapping_steps
    repeated_env_ids = tf.reshape(tf.tile(tf.expand_dims(tf.cast(env_ids, tf.int64), -1), [1, j]), [-1])

    repeated_range = tf.tile(tf.range(j, dtype=tf.int64), [tf.shape(env_ids)[0]])
    indices = tf.stack([repeated_env_ids, repeated_range], axis=-1)

    for s in tf.nest.flatten(self._state):
      z = tf.zeros(tf.concat([tf.shape(repeated_env_ids), s.shape[2:]], axis=0), s.dtype)
      s.scatter_nd_update(indices, z)

  def _complete_unrolls(self, env_ids):
    # Environment with unrolls that are now complete and should be returned.
    env_indices = self._index.sparse_read(env_ids)

    env_ids = tf.gather(env_ids, tf.where(tf.equal(env_indices, self._full_length))[:, 0])
    env_ids = tf.cast(env_ids, tf.int64)
    unrolls = tf.nest.map_structure(lambda s: s.sparse_read(env_ids), self._state)

    # Store last transitions as the first in the next unroll.
    # The following code is the equivalent of:
    # s[env_ids, :j] = s[env_ids, -j:]
    j = self._num_overlapping_steps + 1
    repeated_start_range = tf.tile(tf.range(j, dtype=tf.int64), [tf.shape(env_ids)[0]])
    repeated_end_range = tf.tile(tf.range(self._full_length - j, self._full_length, dtype=tf.int64), [tf.shape(env_ids)[0]])
    repeated_env_ids = tf.reshape(tf.tile(tf.expand_dims(env_ids, -1), [1, j]), [-1])

    start_indices = tf.stack([repeated_env_ids, repeated_start_range], -1)
    end_indices = tf.stack([repeated_env_ids, repeated_end_range], -1)
    for s in tf.nest.flatten(self._state):
      s.scatter_nd_update(start_indices, s.gather_nd(end_indices))

    self._index.scatter_update(tf.IndexedSlices(1 + self._num_overlapping_steps, env_ids))

    return env_ids, unrolls


class Aggregator(tf.Module):
  """Utility module for keeping state for individual environments."""
  def __init__(self, num_envs, specs, name='Aggregator'):
    """Inits an Aggregator.
    Args:
      num_envs: int, number of environments.
      specs: Structure (as defined by tf.nest) of tf.TensorSpecs that will be
        stored for each environment.
      name: Name of the scope for the operations.
    """
    super(Aggregator, self).__init__(name=name)
    def create_variable(spec):
      z = tf.zeros([num_envs] + spec.shape.dims, dtype=spec.dtype)

      return tf.Variable(z, trainable=False, name=spec.name)

    self._state = tf.nest.map_structure(create_variable, specs)

  @tf.Module.with_name_scope
  def reset(self, env_ids):
    """Fills the tensors for the given environments with zeros."""
    with tf.name_scope('Aggregator_reset'):
      for s in tf.nest.flatten(self._state):
        s.scatter_update(tf.IndexedSlices(0, env_ids))

  @tf.Module.with_name_scope
  def add(self, env_ids, values):
    """In-place adds values to the state associated to the given environments.
    Args:
      env_ids: 1D tensor with the environment IDs we want to add values to.
      values: A structure of tensors following the input spec, with an added
        first dimension that must either have the same size as 'env_ids', or
        should not exist (in which case, the value is broadcasted to all
        environment ids).
    """
    tf.nest.assert_same_structure(values, self._state)
    for s, v in zip(tf.nest.flatten(self._state), tf.nest.flatten(values)):
      s.scatter_add(tf.IndexedSlices(v, env_ids))

  @tf.Module.with_name_scope
  def read(self, env_ids):
    """Reads the values corresponding to a list of environments.
    Args:
      env_ids: 1D tensor with the list of environment IDs we want to read.

    Returns:
      A structure of tensors with the same shapes as the input specs. A
      dimension is added in front of each tensor, with size equal to the number
      of env_ids provided.
    """
    return tf.nest.map_structure(lambda s: s.sparse_read(env_ids), self._state)

  @tf.Module.with_name_scope
  def replace(self, env_ids, values, debug_op_name='', debug_tensors=None):
    """Replaces the state associated to the given environments.
    Args:
      env_ids: 1D tensor with the list of environment IDs.
      values: A structure of tensors following the input spec, with an added
        first dimension that must either have the same size as 'env_ids', or
        should not exist (in which case, the value is broadcasted to all
        environment ids).
      debug_op_name: Debug name for the operation.
      debug_tensors: List of tensors to print when the assert fails.
    """
    tf.debugging.assert_rank(env_ids, 1, message=f'Invalid rank for aggregator {self.name}')

    tf.debugging.Assert(
      tf.reduce_all(tf.equal(tf.shape(env_ids), tf.shape(tf.unique(env_ids)[0]))),
        data=[env_ids, (f'Duplicate environment ids in Aggregator: {self.name} with op name "{debug_op_name}"')] + (debug_tensors or []),
        summarize=4096, name=f'assert_no_dups_{self.name}')

    tf.nest.assert_same_structure(values, self._state)

    for s, v in zip(tf.nest.flatten(self._state), tf.nest.flatten(values)):
      s.scatter_update(tf.IndexedSlices(v, env_ids))


class ProgressLogger(object):
  """Helper class for performing periodic logging of the training progress."""
  def __init__(self,
               summary_writer=None,
               initial_period=0.1,
               period_factor=1.01,
               max_period=10.0,
               starting_step=0):
    """Constructs ProgressLogger.
    Args:
      summary_writer: Tensorflow summary writer to use.
      initial_period: Initial logging period in seconds
        (how often logging happens).
      period_factor: Factor by which logging period is
        multiplied after each iteration (exponential back-off).
      max_period: Maximal logging period in seconds
        (the end of exponential back-off).
      starting_step: Step from which to start the summary writer.
    """
    # summary_writer, last_log_{time, step} are set in reset() function.
    self.summary_writer = None
    self.last_log_time = None
    self.last_log_step = 0
    self.period = initial_period
    self.period_factor = period_factor
    self.max_period = max_period
    # Array of strings with names of values to be logged.
    self.log_keys = []
    self.log_keys_set = set()
    self.step_cnt = tf.Variable(-1, dtype=tf.int64)
    self.ready_values = tf.Variable([-1.0],
                                    dtype=tf.float32,
                                    shape=tf.TensorShape(None))
    self.logger_thread = None
    self.logging_callback = None
    self.terminator = None
    self.reset(summary_writer, starting_step)

  def reset(self, summary_writer=None, starting_step=0):
    """Resets the progress logger.

    Args:
      summary_writer: Tensorflow summary writer to use.
      starting_step: Step from which to start the summary writer.
    """
    self.summary_writer = summary_writer
    self.step_cnt.assign(starting_step)
    self.ready_values.assign([-1.0])
    self.last_log_time = timeit.default_timer()
    self.last_log_step = starting_step

  def start(self, logging_callback=None):
    assert self.logger_thread is None
    self.logging_callback = logging_callback
    self.terminator = threading.Event()
    self.logger_thread = threading.Thread(target=self._logging_loop)
    self.logger_thread.start()

  def shutdown(self):
    assert self.logger_thread
    self.terminator.set()
    self.logger_thread.join()
    self.logger_thread = None

  def log_session(self):
    return []

  def log(self, session, name, value):
    # this is a python op so it happens only when this tf.function is compiled
    if name not in self.log_keys_set:
      self.log_keys.append(name)
      self.log_keys_set.add(name)
    # this is a TF op.
    session.append(value)

  def log_session_from_dict(self, dic):
    session = self.log_session()
    for key in dic:
      self.log(session, key, dic[key])

    return session

  def step_end(self, session, strategy=None, step_increment=1):
    logs = []
    for value in session:
      if strategy:
        value = tf.reduce_mean(tf.cast(strategy.experimental_local_results(value)[0], tf.float32))

      logs.append(value)

    self.ready_values.assign(logs)
    self.step_cnt.assign_add(step_increment)

  def _log(self):
    """Perform single round of logging."""
    logging_time = timeit.default_timer()
    step_cnt = self.step_cnt.read_value()
    if step_cnt == self.last_log_step:
      return

    values = self.ready_values.read_value().numpy()
    if values[0] == -1:
      return

    assert len(values) == len(self.log_keys), 'Mismatch between number of keys and values to log: %r vs %r' % (values, self.log_keys)

    if self.summary_writer:
      self.summary_writer.set_as_default()

    tf.summary.experimental.set_step(step_cnt.numpy())
    if self.logging_callback:
      self.logging_callback()

    for key, value in zip(self.log_keys, values):
      tf.summary.scalar(key, value)

    dt = logging_time - self.last_log_time
    df = tf.cast(step_cnt - self.last_log_step, tf.float32)
    tf.summary.scalar('speed/steps_per_sec', df / dt)
    self.last_log_time, self.last_log_step = logging_time, step_cnt

  def _logging_loop(self):
    """Loop being run in a separate thread."""
    last_log_try = timeit.default_timer()
    while not self.terminator.isSet():
      try:
        self._log()
      except Exception:  
        logging.fatal('Logging failed.', exc_info=True)

      now = timeit.default_timer()
      elapsed = now - last_log_try
      last_log_try = now
      self.period = min(self.period_factor * self.period, self.max_period)
      self.terminator.wait(timeout=max(0, self.period - elapsed))


class StructuredFIFOQueue(tf.queue.FIFOQueue):
  """A tf.queue.FIFOQueue that supports nests and tf.TensorSpec."""
  def __init__(self, capacity, specs, shared_name=None, name='structured_fifo_queue'):
    self._specs = specs
    self._flattened_specs = tf.nest.flatten(specs)
    dtypes = [ts.dtype for ts in self._flattened_specs]
    shapes = [ts.shape for ts in self._flattened_specs]
    super(StructuredFIFOQueue, self).__init__(capacity, dtypes, shapes)

  def dequeue(self, name=None):
    result = super(StructuredFIFOQueue, self).dequeue(name=name)
    return tf.nest.pack_sequence_as(self._specs, result)

  def dequeue_many(self, batch_size, name=None):
    result = super(StructuredFIFOQueue, self).dequeue_many(
        batch_size, name=name)
    return tf.nest.pack_sequence_as(self._specs, result)

  def enqueue(self, vals, name=None):
    tf.nest.assert_same_structure(vals, self._specs)
    return super(StructuredFIFOQueue, self).enqueue(
        tf.nest.flatten(vals), name=name)

  def enqueue_many(self, vals, name=None):
    tf.nest.assert_same_structure(vals, self._specs)
    return super(StructuredFIFOQueue, self).enqueue_many(
        tf.nest.flatten(vals), name=name)


def make_time_major(x):
  """Transposes the batch and time dimensions of a nest of Tensors.

  If an input tensor has rank < 2 it returns the original tensor. Retains as
  much of the static shape information as possible.

  Args:
    x: A nest of Tensors.

  Returns:
    x transposed along the first two dimensions.
  """
  def transpose(t):  
    t_static_shape = t.shape
    if t_static_shape.rank is not None and t_static_shape.rank < 2:
      return t

    t_rank = tf.rank(t)
    t_t = tf.transpose(t, tf.concat(([1, 0], tf.range(2, t_rank)), axis=0))
    t_t.set_shape(tf.TensorShape([t_static_shape[1], t_static_shape[0]]).concatenate(t_static_shape[2:]))
    return t_t

  return tf.nest.map_structure(lambda t: tf.xla.experimental.compile(transpose, [t])[0], x)


def batch_apply(fn, inputs):
  """Folds time into the batch dimension, runs fn() and unfolds the result.

  Args:
    fn: Function that takes as input the n tensors of the tf.nest structure,
      with shape [time*batch, <remaining shape>], and returns a tf.nest
      structure of batched tensors.
    inputs: tf.nest structure of n [time, batch, <remaining shape>] tensors.

  Returns:
    tf.nest structure of [time, batch, <fn output shape>]. Structure is
    determined by the output of fn.
  """
  time_to_batch_fn = lambda t: tf.reshape(t, [-1] + t.shape[2:].as_list())
  batched = tf.nest.map_structure(time_to_batch_fn, inputs)
  output = fn(*batched)
  prefix = [int(tf.nest.flatten(inputs)[0].shape[0]), -1]
  batch_to_time_fn = lambda t: tf.reshape(t, prefix + t.shape[1:].as_list())

  return tf.nest.map_structure(batch_to_time_fn, output)