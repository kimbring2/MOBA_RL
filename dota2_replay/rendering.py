import pygame
import numpy as np
import cv2
import requests
import math
import datetime

from pygame.locals import (
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_ESCAPE,
    KEYDOWN,
    QUIT,
    MOUSEBUTTONDOWN,
    MOUSEBUTTONUP
)

pygame.init()


def RotatePoint(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

    return int(qx), int(qy)


def GetHeroIconImage(hero_name):
    icon_image_url = "https://cdn.cloudflare.steamstatic.com/apps/dota2/images/heroes/" + hero_name + "_icon.png"
    icon_image_nparray = np.asarray(bytearray(requests.get(icon_image_url).content), dtype=np.uint8)
    icon_image = cv2.imdecode(icon_image_nparray, cv2.IMREAD_UNCHANGED)
    icon_image = cv2.resize(icon_image, dsize=(64, 64), interpolation=cv2.INTER_LINEAR)

    return icon_image


def GetHeroPortraitImage(hero_name):
    portrait_image_url = "https://cdn.cloudflare.steamstatic.com/apps/dota2/images/heroes/" + hero_name + "_vert.jpg"
    portrait_image_nparray = np.asarray(bytearray(requests.get(portrait_image_url).content), dtype=np.uint8)
    portrait_image = cv2.imdecode(portrait_image_nparray, cv2.IMREAD_UNCHANGED)
    portrait_image = cv2.resize(portrait_image, dsize=(200, 256), interpolation=cv2.INTER_LINEAR)

    return portrait_image


def GetRuneIconImage(rune_name):
    #icon_image_url = "https://cdn.cloudflare.steamstatic.com/apps/dota2/images/heroes/" + hero_name + "_icon.png"
    icon_image_nparray = np.asarray(bytearray(requests.get(icon_image_url).content), dtype=np.uint8)
    icon_image = cv2.imdecode(icon_image_nparray, cv2.IMREAD_UNCHANGED)
    icon_image = cv2.resize(icon_image, dsize=(64, 64), interpolation=cv2.INTER_LINEAR)

    return icon_image


def GetAbilityImage(ability_name):
	  #print("")
    hero_ability_image_url = "https://cdn.cloudflare.steamstatic.com/apps/dota2/images/dota_react/abilities/" + ability_name + ".png"               
    hero_ability_image_nparray = np.asarray(bytearray(requests.get(hero_ability_image_url).content), dtype=np.uint8)
    hero_ability_image = cv2.imdecode(hero_ability_image_nparray, cv2.IMREAD_UNCHANGED)
    try:
      hero_ability_image = cv2.resize(hero_ability_image, dsize=(100, 100), interpolation=cv2.INTER_LINEAR)
      hero_ability_image = cv2.cvtColor(hero_ability_image, cv2.COLOR_BGRA2BGR)
      return hero_ability_image
    except:
      return np.zeros((100,100,3))


def GetItemImage(item_name):
    hero_item_image_url = "https://cdn.cloudflare.steamstatic.com/apps/dota2/images/dota_react/items/" + item_name + ".png"               
    hero_item_image_nparray = np.asarray(bytearray(requests.get(hero_item_image_url).content), dtype=np.uint8)
    hero_item_image = cv2.imdecode(hero_item_image_nparray, cv2.IMREAD_UNCHANGED)
    hero_item_image = cv2.resize(hero_item_image, dsize=(90, 70), interpolation=cv2.INTER_LINEAR)
    hero_item_image = cv2.cvtColor(hero_item_image, cv2.COLOR_BGRA2BGR)

    return hero_item_image


def PutHeroIcon(parent_image, icon_image, hero_location_x, hero_location_y):
    parent_image[int(16384 / 4 - hero_location_y - 32):int(16384 / 4 - hero_location_y + 32),int(-hero_location_x - 32):int(-hero_location_x + 32),:] \
    = parent_image[int(16384 / 4 - hero_location_y - 32):int(16384 / 4 -  hero_location_y + 32),int(-hero_location_x - 32):int(-hero_location_x + 32),:] * \
        (1 - icon_image[:, :, 3:] / 255) + icon_image[:, :, :3] * (icon_image[:, :, 3:] / 255)


def PutNpcIcon(parent_image, icon_image, npc_location_x, npc_location_y):
    parent_image[int(16384 / 4 - npc_location_y - 16):int(16384 / 4 - npc_location_y + 16),int(-npc_location_x - 16):int(-npc_location_x + 16),:] \
    = parent_image[int(16384 / 4 - npc_location_y - 16):int(16384 / 4 -  npc_location_y + 16),int(-npc_location_x - 16):int(-npc_location_x + 16),:] * \
        (1 - icon_image[:, :, 3:] / 255) + icon_image[:, :, :3] * (icon_image[:, :, 3:] / 255)


def PutAngleArrow(parent_image, hero_location_x, hero_location_y, hero_angle, team_num):
    origin = (int(-hero_location_x), int(16384 / 4 - hero_location_y))
    pt1 = (int(-hero_location_x) + 20, int(16384 / 4 - hero_location_y + 0))
    pt1 = RotatePoint(origin, pt1, math.radians(360 - hero_angle))

    pt2 = (int(-hero_location_x) + 10, int(16384 / 4 - hero_location_y - 10))
    pt2 = RotatePoint(origin, pt2, math.radians(360 - hero_angle))

    pt3 = (int(-hero_location_x) + 10, int(16384 / 4 - hero_location_y + 10))
    pt3 = RotatePoint(origin, pt3, math.radians(360 - hero_angle))

    triangle_cnt = np.array([pt1, pt2, pt3])

    if team_num == 2:
    	cv2.drawContours(parent_image, [triangle_cnt], 0, colors['green'], -1)
    elif team_num == 3:
    	cv2.drawContours(parent_image, [triangle_cnt], 0, colors['red'], -1)


def PutHeroLevel(parent_image, hero_location_x, hero_location_y, hero_level):
    cv2.putText(parent_image, str(hero_level), (int(-hero_location_x - 71), int(16384 / 4 - hero_location_y - 32)), cv2.FONT_HERSHEY_SIMPLEX, 
                1, colors['yellow'], 1, cv2.LINE_AA)


def PutIconHpHero(parent_image, location_x, location_y, hp, hp_max):
    hp_percentage = hp / hp_max * 100.0
    cv2.rectangle(parent_image, (int(-location_x - 40), int(16384 / 4 - location_y - 47)), 
                                (int(-location_x + hp_percentage * 0.8 - 40), int(16384 / 4 - location_y - 52)), 
                                colors['green'], 3)


def PutIconHpNPC(parent_image, location_x, location_y, hp, hp_max):
    hp_percentage = hp / hp_max * 100.0
    cv2.rectangle(parent_image, (int(-location_x - 26), int(16384 / 4 - location_y - 32)), 
                                (int(-location_x + hp_percentage * 0.5 - 26), int(16384 / 4 - location_y - 33)), 
                                colors['green'], 3)


def PutIconMana(parent_image, hero_location_x, hero_location_y, hero_mana, hero_mana_max):
    if hero_mana != 0.0 and hero_mana_max != 0.0:
        if hero_mana / hero_mana_max * 100.0 <= 100:
            hero_mana_percentage = hero_mana / hero_mana_max * 100.0
        else:
            hero_mana_percentage = 100.0
    else:
        hero_mana_percentage = 100.0

    cv2.rectangle(parent_image, (int(-hero_location_x - 40), int(16384 / 4 - hero_location_y - 38)), 
                                (int(-hero_location_x + hero_mana_percentage * 0.8 - 40), int(16384 / 4 - hero_location_y - 40)), 
                                 colors['blue'], 3)


def PutSelectedCircle(parent_image, hero_location_x, hero_location_y):
	  cv2.circle(parent_image, (int(-hero_location_x), int(16384 / 4 - hero_location_y)), 64, colors['light_gray'], 1)


def PutClickArrow(parent_image, location_x, location_y, click_x, click_y):
		cv2.arrowedLine(parent_image, (int(-location_x), int(16384 / 4 - location_y)), 
                                  (int(-click_x), int(16384 / 4 - click_y)), colors['blue'], 1)


def PutPortraitName(parent_image, hero_name):
    hero_name_len = len(hero_name) * 17
    name_start_pos = int(500 + (756 - 500 - hero_name_len) / 2)
    cv2.putText(parent_image, hero_name.upper(), (name_start_pos, 1450), cv2.FONT_HERSHEY_DUPLEX, 1, colors['yellow'], 1, cv2.LINE_AA)


def PutPortraitNameNPC(parent_image, npc_name):
    npc_name_len = len(npc_name) * 17
    name_start_pos = int(500 + (756 - 500 - npc_name_len) / 2)
    cv2.putText(parent_image, npc_name.upper(), (name_start_pos, 1450), cv2.FONT_HERSHEY_DUPLEX, 1, colors['yellow'], 1, cv2.LINE_AA)


def PutPortraitImage(parent_image, portrait_image):
    parent_image[1500:1500+256,550:550+200,:] = portrait_image


def PutPortraitImageNPC(parent_image, portrait_image):
    parent_image[1500:1500+272,550:550+235,:] = portrait_image


def PutPortraitHp(parent_image, hero_hp, hero_hp_max):
    hero_hp_percentage = hero_hp / hero_hp_max * 100.0
    cv2.rectangle(parent_image, (800, 1650), (800 + int(hero_hp_percentage * 6.5), 1690), colors['green'], -1)
    cv2.putText(parent_image, str(hero_hp) + ' / ' + str(hero_hp_max), (1040, 1680), cv2.FONT_HERSHEY_DUPLEX, 1, colors['white'], 
    				    1, cv2.LINE_AA)


def PutPortraitHpNPC(parent_image, npc_hp, npc_hp_max):
    npc_hp_percentage = npc_hp / npc_hp_max * 100.0
    cv2.rectangle(parent_image, (800, 1650), (800 + int(npc_hp_percentage * 6.5), 1690), colors['green'], -1)


def PutBoundaryCircle(parent_image, location_x, location_y, color):
	  cv2.circle(parent_image, (int(-location_x), int(16384 / 4 - location_y)), 16, colors[color], -1)


def PutPortraitMana(parent_image, hero_mana, hero_mana_max):
    if hero_mana != 0.0 and hero_mana_max != 0.0:
        if hero_mana / hero_mana_max * 100.0 <= 100:
            hero_mana_percentage = hero_mana / hero_mana_max * 100.0
        else:
            hero_mana = 100
            hero_mana_max = hero_mana_max
            hero_mana_percentage = 100.0
    else:
        hero_mana = 100
        hero_mana_max = hero_mana_max
        hero_mana_percentage = 100.0

    cv2.rectangle(parent_image, (800, 1710), (800 + int(hero_mana_percentage * 6.5), 1750), colors['blue'], -1)
    cv2.putText(parent_image, str(int(hero_mana)) + ' / ' + str(int(hero_mana_max)), (1040, 1740), cv2.FONT_HERSHEY_DUPLEX, 1, colors['white'], 
    				    1, cv2.LINE_AA)


def PutPortraitAbility(parent_image, ability_image, ability_pos, abilites_info):
    if ability_pos == 1:
        parent_image[1500:1500+100,800:800+100,:] = ability_image
        cv2.putText(parent_image, str(abilites_info['ability_level']), (840, 1630), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, colors['white'], 2, cv2.LINE_4)
        cv2.putText(parent_image, str(int(abilites_info['ability_cool'])), (870, 1600), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, colors['magenta'], 2, cv2.LINE_4)
    elif ability_pos == 2:
        parent_image[1500:1500+100,910:910+100,:] = ability_image
        cv2.putText(parent_image, str(abilites_info['ability_level']), (950, 1630), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, colors['white'], 2, cv2.LINE_4)
        cv2.putText(parent_image, str(int(abilites_info['ability_cool'])), (980, 1600), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, colors['magenta'], 2, cv2.LINE_4)
    elif ability_pos == 3:
        parent_image[1500:1500+100,1020:1020+100,:] = ability_image
        cv2.putText(parent_image, str(abilites_info['ability_level']), (1060, 1630), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, colors['white'], 2, cv2.LINE_4)
        cv2.putText(parent_image, str(int(abilites_info['ability_cool'])), (1090, 1600), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, colors['magenta'], 2, cv2.LINE_4)
    elif ability_pos == 4:
        parent_image[1500:1500+100,1130:1130+100,:] = ability_image
        cv2.putText(parent_image, str(abilites_info['ability_level']), (1170, 1630), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, colors['white'], 2, cv2.LINE_4)
        cv2.putText(parent_image, str(int(abilites_info['ability_cool'])), (1200, 1600), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, colors['magenta'], 2, cv2.LINE_4)
    elif ability_pos == 5:
        parent_image[1500:1500+100,1240:1240+100,:] = ability_image
        cv2.putText(parent_image, str(abilites_info['ability_level']), (1290, 1630), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, colors['white'], 2, cv2.LINE_4)
        cv2.putText(parent_image, str(int(abilites_info['ability_cool'])), (1310, 1600), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, colors['magenta'], 2, cv2.LINE_4)
    elif ability_pos == 6:
        parent_image[1500:1500+100,1350:1350+100,:] = ability_image
        cv2.putText(parent_image, str(abilites_info['ability_level']), (1390, 1630), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, colors['white'], 2, cv2.LINE_4)
        cv2.putText(parent_image, str(int(abilites_info['ability_cool'])), (1420, 1600), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, colors['magenta'], 2, cv2.LINE_4)


#colors = {'blue': (255, 0, 0), 'green': (0, 255, 0), 'red': (0, 0, 255), 'yellow': (0, 255, 255), 'magenta': (255, 0, 255), 
#          'cyan': (255, 255, 0), 'white': (255, 255, 255), 'black': (0, 0, 0), 'gray': (125, 125, 125), 
#          'rand': np.random.randint(0, high=256, size=(3,)).tolist(), 'dark_gray': (50, 50, 50), 'light_gray': (220, 220, 220)}
def PutPortraitItem(parent_image, item_image, item_pos, items_info):
    if item_pos == 1:
        parent_image[1500:1500+70,1600:1600+90,:] = item_image
        cv2.putText(parent_image, str(items_info['item_num']), (1665, 1525), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, colors['white'], 1, cv2.LINE_4)
        cv2.putText(parent_image, str(int(items_info['item_cool'])), (1665, 1565), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, colors['magenta'], 1, cv2.LINE_4)
    elif item_pos == 2:
        parent_image[1500:1500+70,1700:1700+90,:] = item_image
        cv2.putText(parent_image, str(items_info['item_num']), (1765, 1525), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, colors['white'], 1, cv2.LINE_4)
        cv2.putText(parent_image, str(int(items_info['item_cool'])), (1765, 1565), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, colors['magenta'], 1, cv2.LINE_4)
    elif item_pos == 3:
        parent_image[1500:1500+70,1800:1800+90,:] = item_image
        cv2.putText(parent_image, str(items_info['item_num']), (1865, 1525), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, colors['white'], 1, cv2.LINE_4)
        cv2.putText(parent_image, str(int(items_info['item_cool'])), (1865, 1565), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, colors['magenta'], 1, cv2.LINE_4)
    elif item_pos == 4:
        parent_image[1580:1580+70,1600:1600+90,:] = item_image
        cv2.putText(parent_image, str(items_info['item_num']), (1665, 1605), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, colors['white'], 1, cv2.LINE_4)
        cv2.putText(parent_image, str(int(items_info['item_cool'])), (1665, 1645), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, colors['magenta'], 1, cv2.LINE_4)
    elif item_pos == 5:
        parent_image[1580:1580+70,1700:1700+90,:] = item_image
        cv2.putText(parent_image, str(items_info['item_num']), (1765, 1605), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, colors['white'], 1, cv2.LINE_4)
        cv2.putText(parent_image, str(int(items_info['item_cool'])), (1765, 1645), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, colors['magenta'], 1, cv2.LINE_4)
    elif item_pos == 6:
        parent_image[1580:1580+70,1800:1800+90,:] = item_image
        cv2.putText(parent_image, str(items_info['item_num']), (1865, 1605), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, colors['white'], 1, cv2.LINE_4)
        cv2.putText(parent_image, str(int(items_info['item_cool'])), (1865, 1645), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, colors['magenta'], 1, cv2.LINE_4)


entire_game = cv2.imread("dota2_map.jpg", cv2.IMREAD_COLOR)

melee_creep_icon = cv2.imread("npc_dota_creep_goodguys_melee.png", cv2.IMREAD_UNCHANGED)
range_creep_icon = cv2.imread("npc_dota_creep_goodguys_ranged.png", cv2.IMREAD_UNCHANGED)
tower_icon = cv2.imread("npc_dota_badguys_tower.png", cv2.IMREAD_UNCHANGED)
siege_icon = cv2.imread("npc_dota_badguys_siege.png", cv2.IMREAD_UNCHANGED)
super_melee_creep_icon = cv2.imread("npc_dota_creep_goodguys_melee_upgraded_mega.png", cv2.IMREAD_UNCHANGED)
super_range_creep_icon = cv2.imread("npc_dota_creep_goodguys_ranged_upgraded_mega.png", cv2.IMREAD_UNCHANGED)

melee_barrack_icon = cv2.imread("npc_dota_goodguys_melee_rax.png", cv2.IMREAD_UNCHANGED)
range_barrack_icon = cv2.imread("npc_dota_goodguys_range_rax.png", cv2.IMREAD_UNCHANGED)

ancient_icon = cv2.imread("npc_dota_goodguys_fort.png", cv2.IMREAD_UNCHANGED)

water_rune_icon = cv2.imread("/home/kimbring2/Downloads/Water_Rune_minimap_icon.webp", cv2.IMREAD_UNCHANGED)
water_rune_icon = cv2.resize(water_rune_icon, (128, 128), interpolation=cv2.INTER_CUBIC)

water_rune_icon = cv2.imread("/home/kimbring2/Downloads/Water_Rune_minimap_icon.webp", cv2.IMREAD_UNCHANGED)
water_rune_icon = cv2.resize(water_rune_icon, (128, 128), interpolation=cv2.INTER_CUBIC)

minimap_image = cv2.imread("dota2_minimap.jpg", cv2.IMREAD_COLOR)
minimap_image = cv2.cvtColor(minimap_image, cv2.COLOR_RGB2BGR)

colors = {'blue': (255, 0, 0), 'green': (0, 255, 0), 'red': (0, 0, 255), 'yellow': (0, 255, 255), 'magenta': (255, 0, 255), 
          'cyan': (255, 255, 0), 'white': (255, 255, 255), 'black': (0, 0, 0), 'gray': (125, 125, 125), 
          'rand': np.random.randint(0, high=256, size=(3,)).tolist(), 'dark_gray': (50, 50, 50), 'light_gray': (220, 220, 220)}


class RuneRendering(object):
  def __init__(self):
    self.location_x = None
    self.location_y = None
    self.name = None

  def rendering_map(self, parent_image, rune_info_dict):
    self.location_x = rune_info_dict['pos'][0]
    self.location_y = rune_info_dict['pos'][1]
    self.name = rune_info_dict['name']
    #print("rune name: ", name)
    if self.name != None:
    	cv2.putText(parent_image, str(self.name), (int(-self.location_x), int(16384 / 4 - self.location_y)), 
    							cv2.FONT_HERSHEY_SIMPLEX, 1, colors['magenta'], 1, cv2.LINE_4)
    else:
    	cv2.putText(parent_image, 'rune', (int(-self.location_x), int(16384 / 4 - self.location_y)), 
    							cv2.FONT_HERSHEY_SIMPLEX, 1, colors['magenta'], 1, cv2.LINE_4)
    #PutNpcIcon(parent_image, ancient_icon, self.location_x, self.location_y)


class NpcRendering(object):
  def __init__(self):
    self.location_x = None
    self.location_y = None
    self.team_num = None
    self.hp = None
    self.hp_max = None
    self.selected = False

  def rendering_map(self, parent_image, npc_info_dict):
    self.location_x = npc_info_dict['pos'][0]
    self.location_y = npc_info_dict['pos'][1]
    self.hp = npc_info_dict['hp']
    self.team_num = npc_info_dict['team_num']
    self.hp_max = npc_info_dict['hp_max']

    if 'selected' in npc_info_dict:
    	self.selected = npc_info_dict['selected']
    else:
    	self.selected = False

    if self.team_num == 2:
      PutBoundaryCircle(parent_image, self.location_x, self.location_y, 'green')
    elif self.team_num == 3:
      PutBoundaryCircle(parent_image, self.location_x, self.location_y, 'red')

    if self.hp_max == 550 or self.hp_max == 574 or self.hp_max == 586 or self.hp_max == 598:
    	PutNpcIcon(parent_image, melee_creep_icon, self.location_x, self.location_y)
    	PutIconHpNPC(parent_image, self.location_x, self.location_y, self.hp, self.hp_max)
    elif self.hp_max == 300 or self.hp_max == 324 or self.hp_max == 336 or self.hp_max == 348:
      PutNpcIcon(parent_image, range_creep_icon, self.location_x, self.location_y)
      PutIconHpNPC(parent_image, self.location_x, self.location_y, self.hp, self.hp_max)
    elif self.hp_max == 1800 or self.hp_max == 2500 or self.hp_max == 2600:
      PutNpcIcon(parent_image, tower_icon, self.location_x, self.location_y)
      PutIconHpNPC(parent_image, self.location_x, self.location_y, self.hp, self.hp_max)
    elif self.hp_max == 935:
      PutNpcIcon(parent_image, siege_icon, self.location_x, self.location_y)
      PutIconHpNPC(parent_image, self.location_x, self.location_y, self.hp, self.hp_max)
    elif self.hp_max == 757 or self.hp_max == 776 or self.hp_max == 1270:
      PutNpcIcon(parent_image, super_melee_creep_icon, self.location_x, self.location_y)
      PutIconHpNPC(parent_image, self.location_x, self.location_y, self.hp, self.hp_max)
    elif self.hp_max == 529 or self.hp_max == 547 or self.hp_max == 1015:
      PutNpcIcon(parent_image, super_range_creep_icon, self.location_x, self.location_y)
      PutIconHpNPC(parent_image, self.location_x, self.location_y, self.hp, self.hp_max)
    elif self.hp_max == 1300:
      PutNpcIcon(parent_image, range_barrack_icon, self.location_x, self.location_y)
      PutIconHpNPC(parent_image, self.location_x, self.location_y, self.hp, self.hp_max)
    elif self.hp_max == 2200:
      PutNpcIcon(parent_image, melee_barrack_icon, self.location_x, self.location_y)
      PutIconHpNPC(parent_image, self.location_x, self.location_y, self.hp, self.hp_max)
    elif self.hp_max == 4500:
      PutNpcIcon(parent_image, ancient_icon, self.location_x, self.location_y)
      PutIconHpNPC(parent_image, self.location_x, self.location_y, self.hp, self.hp_max)
    else:
      PutNpcIcon(parent_image, range_creep_icon, self.location_x, self.location_y)
      PutIconHpNPC(parent_image, self.location_x, self.location_y, self.hp, self.hp_max)


class HeroRendering(object):
    def __init__(self):
        self.name = None
        self.location_x = None
        self.location_y = None
        self.icon_image = None
        self.icon_image_minimap = None
        self.icon_flag = False
        self.level = None
        self.angle = None
        self.team_num = None

        self.portrait_image = None
        self.portrait_flag = False

        self.item_image1 = None
        self.item_image2 = None
        self.item_image3 = None
        self.item_image4 = None
        self.item_image5 = None
        self.item_image6 = None
        self.item_image1_flag = None
        self.item_image2_flag = None
        self.item_image3_flag = None
        self.item_image4_flag = None
        self.item_image5_flag = None
        self.item_image6_flag = None

        self.ability_image1 = None
        self.ability_image2 = None
        self.ability_image3 = None
        self.ability_image4 = None
        self.ability_image5 = None
        self.ability_image6 = None
        self.ability_image1_flag = False
        self.ability_image2_flag = False
        self.ability_image3_flag = False
        self.ability_image4_flag = False
        self.ability_image5_flag = False 
        self.ability_image6_flag = False

        self.pre_item_name1 = False 
        self.pre_item_name2 = False 
        self.pre_item_name3 = False 
        self.pre_item_name4 = False 
        self.pre_item_name5 = False 
        self.pre_item_name6 = False

        self.selected = False

        self.abilites_info = None
        self.items_info = None

        self.respawning = False 

    def rendering_map(self, parent_image, hero_name, hero_info_dict, move_to_position_x, move_to_position_y, no_target_ability, 
    								  purchase_item, train_ability, cast_target_tree, move_to_target_x, move_to_target_y, move_to_target_name,
    								  attack_target_x, attack_target_y, attack_target_name):
        if hero_name == 'vengeful_spirit':
          hero_name = "vengefulspirit"

        self.name = hero_name

        self.location_x = hero_info_dict['pos'][0]
        self.location_y = hero_info_dict['pos'][1]
        self.hp = hero_info_dict['hp']
        self.hp_max = hero_info_dict['hp_max']
        self.level = hero_info_dict['level']
        self.mana = hero_info_dict['mana']
        self.mana_max = hero_info_dict['mana_max']
        self.angle = hero_info_dict['angle']
        self.team_num = hero_info_dict['team_num']

        self.abilites_info = hero_info_dict['abilites_info']
        self.items_info = hero_info_dict['items_info']

        self.respawning = hero_info_dict['respawning']

        if self.respawning == True:
        	return 

        if 'selected' in hero_info_dict:
        	self.selected = hero_info_dict['selected']
        else:
        	self.selected = False

        if self.selected == True:
        	PutSelectedCircle(parent_image, self.location_x, self.location_y) 

        if self.name != None and self.icon_flag == False:
            self.icon_image = GetHeroIconImage(hero_name)
            #self.icon_image_minimap = cv2.cvtColor(self.icon_image, cv2.COLOR_RGB2BGR)
            self.icon_image_minimap = self.icon_image
            #print("self.icon_image_minimap.shape: ", self.icon_image_minimap.shape)
            self.icon_flag = True
    
        if self.icon_flag != False:
            PutHeroIcon(parent_image, self.icon_image, self.location_x, self.location_y)
            PutAngleArrow(parent_image, self.location_x, self.location_y, self.angle, self.team_num)
            PutHeroLevel(parent_image, self.location_x, self.location_y, self.level)
            PutIconHpHero(parent_image, self.location_x, self.location_y, self.hp, self.hp_max)
            PutIconMana(parent_image, self.location_x, self.location_y, self.mana, self.mana_max)

        if self.name == 'nevermore':
            if move_to_position_x != None:
              PutClickArrow(parent_image, self.location_x, self.location_y, move_to_position_x, move_to_position_y)
              cv2.putText(parent_image, "move_to_position", (int(-self.location_x - 50), int(16384 / 4 - self.location_y - 0)), 
            						  cv2.FONT_HERSHEY_SIMPLEX, 1, colors['yellow'], 1, cv2.LINE_AA)

            if no_target_ability != None:
              if str(no_target_ability) == "nevermore_shadowraze1":
                origin = (int(-self.location_x), int(16384 / 4 - self.location_y))
                pt = (int(-self.location_x + 200 / 4), int(16384 / 4 - self.location_y + 0))
                pt = RotatePoint(origin, pt, math.radians(360 - self.angle))
                cv2.circle(parent_image, (int(pt[0]), int(pt[1])), int(250 / 4), colors['blue'], 1)
                cv2.putText(parent_image, str(no_target_ability.split('_')[1:][0]), (int(pt[0]), int(pt[1])), 
            	  						cv2.FONT_HERSHEY_SIMPLEX, 1, colors['rand'], 1, cv2.LINE_4)
              elif str(no_target_ability) == "nevermore_shadowraze2":
                origin = (int(-self.location_x), int(16384 / 4 - self.location_y))
                pt = (int(-self.location_x + 400 / 4), int(16384 / 4 - self.location_y + 0))
                pt = RotatePoint(origin, pt, math.radians(360 - self.angle))
                cv2.circle(parent_image, (int(pt[0]), int(pt[1])), int(250 / 4), colors['blue'], 1)
                cv2.putText(parent_image, str(no_target_ability.split('_')[1:][0]), (int(pt[0]), int(pt[1])), 
            	  						cv2.FONT_HERSHEY_SIMPLEX, 1, colors['rand'], 1, cv2.LINE_4)
              elif str(no_target_ability) == "nevermore_shadowraze3":
                origin = (int(-self.location_x), int(16384 / 4 - self.location_y))
                pt = (int(-self.location_x + 700 / 4), int(16384 / 4 - self.location_y + 0))
                pt = RotatePoint(origin, pt, math.radians(360 - self.angle))
                cv2.circle(parent_image, (int(pt[0]), int(pt[1])), int(250 / 4), colors['blue'], 1)
                cv2.putText(parent_image, str(no_target_ability.split('_')[1:][0]), (int(pt[0]), int(pt[1])), 
            	  						cv2.FONT_HERSHEY_SIMPLEX, 1, colors['rand'], 1, cv2.LINE_4)

            if purchase_item != None:
            	cv2.arrowedLine(parent_image, (int(-self.location_x - 50), int(16384 / 4 - self.location_y - 50)), 
            									(int(-self.location_x), int(16384 / 4 - self.location_y)), colors['yellow'], 1)
            	cv2.putText(parent_image, "buy " + str(purchase_item), (int(-self.location_x - 100), int(16384 / 4 - self.location_y - 150)), 
            						  cv2.FONT_HERSHEY_SIMPLEX, 1, colors['yellow'], 1, cv2.LINE_AA)

            if train_ability != None:
            	cv2.arrowedLine(parent_image, (int(-self.location_x - 50), int(16384 / 4 - self.location_y - 100)), 
            									(int(-self.location_x), int(16384 / 4 - self.location_y)), colors['yellow'], 1)
            	cv2.putText(parent_image, "train " + str(train_ability.split('_')[1:][0]), (int(-self.location_x - 100), 
            							int(16384 / 4 - self.location_y - 150)), 
            						  cv2.FONT_HERSHEY_SIMPLEX, 1, colors['yellow'], 1, cv2.LINE_AA)

            if cast_target_tree != None:
            	cv2.arrowedLine(parent_image, (int(-self.location_x - 50), int(16384 / 4 - self.location_y - 150)), 
            									(int(-self.location_x), int(16384 / 4 - self.location_y)), colors['yellow'], 1)
            	cv2.putText(parent_image, "cast_target_tree", (int(-self.location_x - 100), int(16384 / 4 - self.location_y - 150)), 
            						  cv2.FONT_HERSHEY_SIMPLEX, 1, colors['yellow'], 1, cv2.LINE_AA)

            if move_to_target_x != None:
              PutClickArrow(parent_image, self.location_x, self.location_y, move_to_target_x, move_to_target_y)
              cv2.putText(parent_image, "move_to_target:" + str(move_to_target_name), 
              						(int(-self.location_x - 100), int(16384 / 4 - self.location_y - 50)), 
            						  cv2.FONT_HERSHEY_SIMPLEX, 1, colors['yellow'], 1, cv2.LINE_AA)

            if attack_target_x != None:
              PutClickArrow(parent_image, self.location_x, self.location_y, attack_target_x, attack_target_y)
              cv2.putText(parent_image, "attack_target:" + str(attack_target_name), 
                          (int(-self.location_x - 50), int(16384 / 4 - self.location_y - 100)), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, colors['yellow'], 1, cv2.LINE_AA)

    def rendering_portrait(self, parent_image, heros_modifiers):
        if self.abilites_info != None:
          if self.ability_image1_flag == False:
            if self.abilites_info['ability0']['ability_name'] != 'generic_hidden':
              self.ability_image1 = GetAbilityImage(self.abilites_info['ability0']['ability_name'])
              self.ability_image1_flag = True
          if self.ability_image2_flag == False:
            if self.abilites_info['ability1']['ability_name'] != 'generic_hidden':
              self.ability_image2 = GetAbilityImage(self.abilites_info['ability1']['ability_name'])
              self.ability_image2_flag = True
          if self.ability_image3_flag == False:
            if self.abilites_info['ability2']['ability_name'] != 'generic_hidden':
              self.ability_image3 = GetAbilityImage(self.abilites_info['ability2']['ability_name'])
              self.ability_image3_flag = True
          if self.ability_image4_flag == False:
            if self.abilites_info['ability3']['ability_name'] != 'generic_hidden':
              self.ability_image4 = GetAbilityImage(self.abilites_info['ability3']['ability_name'])
              self.ability_image4_flag = True
          if self.ability_image5_flag == False:
            if self.abilites_info['ability4']['ability_name'] != 'generic_hidden':
              self.ability_image5 = GetAbilityImage(self.abilites_info['ability4']['ability_name'])
              self.ability_image5_flag = True
          if self.ability_image6_flag == False:
            if self.abilites_info['ability5']['ability_name'] != 'generic_hidden':
              self.ability_image6 = GetAbilityImage(self.abilites_info['ability5']['ability_name'])
              self.ability_image6_flag = True

        if self.items_info != None:
          if self.items_info['item0']['item_name'] != None:
            if self.item_image1_flag == False:
              self.item_image1 = GetItemImage(self.items_info['item0']['item_name'])
              self.item_image1_flag = True
              self.pre_item_name1 = self.items_info['item0']['item_name']

          if self.items_info['item1']['item_name'] != None:
            if self.item_image2_flag == False:
              self.item_image2 = GetItemImage(self.items_info['item1']['item_name'])
              self.item_image2_flag = True
              self.pre_item_name2 = self.items_info['item1']['item_name']

          if self.items_info['item2']['item_name'] != None:
            if self.item_image3_flag == False:
              self.item_image3 = GetItemImage(self.items_info['item2']['item_name'])
              self.item_image3_flag = True
              self.pre_item_name3 = self.items_info['item2']['item_name']

          if self.items_info['item3']['item_name'] != None:
            if self.item_image4_flag == False:
              self.item_image4 = GetItemImage(self.items_info['item3']['item_name'])
              self.item_image4_flag = True
              self.pre_item_name4 = self.items_info['item3']['item_name']

          if self.items_info['item4']['item_name'] != None:
            if self.item_image5_flag == False:
              self.item_image5 = GetItemImage(self.items_info['item4']['item_name'])
              self.item_image5_flag = True
              self.pre_item_name5 = self.items_info['item4']['item_name']

          if self.items_info['item5']['item_name'] != None:
            if self.item_image6_flag == False:
              self.item_image6 = GetItemImage(self.items_info['item5']['item_name'])
              self.item_image6_flag = True
              self.pre_item_name6 = self.items_info['item5']['item_name']

          if self.items_info['item0']['item_name'] != self.pre_item_name1:
            self.item_image1_flag = False
          if self.items_info['item1']['item_name'] != self.pre_item_name2:
            self.item_image2_flag = False
          if self.items_info['item2']['item_name'] != self.pre_item_name3:
            self.item_image3_flag = False
          if self.items_info['item3']['item_name'] != self.pre_item_name4:
            self.item_image4_flag = False
          if self.items_info['item4']['item_name'] != self.pre_item_name5:
            self.item_image5_flag = False
          if self.items_info['item5']['item_name'] != self.pre_item_name6:
            self.item_image6_flag = False

        if self.portrait_flag == False:
            self.portrait_image = GetHeroPortraitImage(self.name)
            self.portrait_flag = True

        if self.name != None:
          PutPortraitName(parent_image, self.name)

        if self.portrait_flag:
          PutPortraitImage(parent_image, self.portrait_image)
          PutPortraitHp(parent_image, self.hp, self.hp_max)
          PutPortraitMana(parent_image, self.mana, self.mana_max)

        if self.ability_image1_flag:
          PutPortraitAbility(parent_image, self.ability_image1, 1, self.abilites_info['ability0'])

        if self.ability_image2_flag:
          PutPortraitAbility(parent_image, self.ability_image2, 2, self.abilites_info['ability1'])

        if self.ability_image3_flag:
          PutPortraitAbility(parent_image, self.ability_image3, 3, self.abilites_info['ability2'])

        if self.ability_image4_flag:
          PutPortraitAbility(parent_image, self.ability_image4, 4, self.abilites_info['ability3'])

        if self.ability_image5_flag:
          PutPortraitAbility(parent_image, self.ability_image5, 5, self.abilites_info['ability4'])

        if self.ability_image6_flag:
          PutPortraitAbility(parent_image, self.ability_image6, 6, self.abilites_info['ability5'])
        
        if self.item_image1_flag:
          PutPortraitItem(parent_image, self.item_image1, 1, self.items_info['item0'])

        if self.item_image2_flag:
          PutPortraitItem(parent_image, self.item_image2, 2, self.items_info['item1'])

        if self.item_image3_flag:
          PutPortraitItem(parent_image, self.item_image3, 3, self.items_info['item2'])

        if self.item_image4_flag:
          PutPortraitItem(parent_image, self.item_image4, 4, self.items_info['item3'])

        if self.item_image5_flag:
          PutPortraitItem(parent_image, self.item_image5, 5, self.items_info['item4'])

        if self.item_image6_flag:
          PutPortraitItem(parent_image, self.item_image6, 6, self.items_info['item5'])

        if self.name in heros_modifiers:
        	for i, modifiers in enumerate(heros_modifiers[self.name]):
        		cv2.putText(parent_image, str(modifiers), (50, 300 + 50 * i), cv2.FONT_HERSHEY_SIMPLEX, 1, 
        							  colors['yellow'], 2, cv2.LINE_AA)

    def rendering_minimap(self, parent_image, hero_info_dict):
        location_x = hero_info_dict['pos'][0]
        location_y = hero_info_dict['pos'][1]
        icon_image = self.icon_image_minimap
        icon_flag = self.icon_flag

        if icon_flag != False:
          #PutHeroIconMinimap(parent_image, self.icon_image, self.location_x, self.location_y)
          try:
            parent_image[int(512 - location_y / 8 - 32):int(512 - location_y / 8 + 32),int(-location_x / 8 - 32):int(-location_x / 8 + 32),:] \
            = parent_image[int(512 - location_y / 8 - 32):int(512 -  location_y / 8 + 32),int(-location_x / 8 - 32):int(-location_x / 8 + 32),:] * \
               (1 - icon_image[:, :, 3:] / 255) + icon_image[:, :, :3] * (icon_image[:, :, 3:] / 255)
          except:
            pass


radiant_hero_1 = HeroRendering()
radiant_hero_2 = HeroRendering()
radiant_hero_3 = HeroRendering()
radiant_hero_4 = HeroRendering()
radiant_hero_5 = HeroRendering()

dire_hero_1 = HeroRendering()
dire_hero_2 = HeroRendering()
dire_hero_3 = HeroRendering()
dire_hero_4 = HeroRendering()
dire_hero_5 = HeroRendering()

def RenderingFunction(screen, camera_middle_x, camera_middle_y, radiant_heros, dire_heros, npcs, buildings, runes, 
										  game_time, move_to_position_x, move_to_position_y, no_target_ability, mouse_click_x, mouse_click_y,
											purchase_item, train_ability, cast_target_tree, move_to_target_x, move_to_target_y, move_to_target_name,
											attack_target_x, attack_target_y, attack_target_name, radiant_heros_modifiers, dire_heros_modifiers):
    global entire_game
    global minimap_image

    global radiant_hero_1, radiant_hero_2, radiant_hero_3, radiant_hero_4, radiant_hero_5 

    global melee_creep_icon
    global range_creep_icon

    select_flag = 0
    selected_hero = 1
    selected_npc = 1

    screen.fill((255, 255, 255))

    #size = (280, 280)
    #minimap = pygame.Surface(size)
    #minimap.fill((0, 255, 0))

    # Initialing Color
    #pygame.draw.circle(minimap, (0, 0, 255), (int(hero1_location_y / 50), 280 - int(hero1_location_x / 50)), 3)

    # Create a black image
    entire_game_copy = cv2.resize(entire_game, dsize=(int(16384 / 4), int(16384 / 4)), interpolation=cv2.INTER_LINEAR)

    minimap_image_copy = cv2.resize(minimap_image, dsize=(512, 512), interpolation=cv2.INTER_LINEAR)

    if len(radiant_heros) == 5:
      hero_name_list = list(radiant_heros.keys())
      radiant_hero_1.rendering_map(entire_game_copy, hero_name_list[0], radiant_heros[hero_name_list[0]], 
      														 move_to_position_x, move_to_position_y, no_target_ability, purchase_item,
      														 train_ability, cast_target_tree, move_to_target_x, move_to_target_y,
      														 move_to_target_name, attack_target_x, attack_target_y, attack_target_name)
      radiant_hero_2.rendering_map(entire_game_copy, hero_name_list[1], radiant_heros[hero_name_list[1]], 
      														 move_to_position_x, move_to_position_y, no_target_ability, purchase_item,
      														 train_ability, cast_target_tree, move_to_target_x, move_to_target_y,
      														 move_to_target_name, attack_target_x, attack_target_y, attack_target_name)
      radiant_hero_3.rendering_map(entire_game_copy, hero_name_list[2], radiant_heros[hero_name_list[2]], 
      														 move_to_position_x, move_to_position_y, no_target_ability, purchase_item,
      														 train_ability, cast_target_tree, move_to_target_x, move_to_target_y,
      														 move_to_target_name, attack_target_x, attack_target_y, attack_target_name)
      radiant_hero_4.rendering_map(entire_game_copy, hero_name_list[3], radiant_heros[hero_name_list[3]], 
      														 move_to_position_x, move_to_position_y, no_target_ability, purchase_item,
      														 train_ability, cast_target_tree, move_to_target_x, move_to_target_y,
      														 move_to_target_name, attack_target_x, attack_target_y, attack_target_name)
      radiant_hero_5.rendering_map(entire_game_copy, hero_name_list[4], radiant_heros[hero_name_list[4]], 
      														 move_to_position_x, move_to_position_y, no_target_ability, purchase_item,
      														 train_ability, cast_target_tree, move_to_target_x, move_to_target_y,
      														 move_to_target_name, attack_target_x, attack_target_y, attack_target_name)

    if len(dire_heros) == 5:
      hero_name_list = list(dire_heros.keys())
      dire_hero_1.rendering_map(entire_game_copy, hero_name_list[0], dire_heros[hero_name_list[0]], 
      												  move_to_position_x, move_to_position_y, no_target_ability, purchase_item,
      												  train_ability, cast_target_tree, move_to_target_x, move_to_target_y,
      												  move_to_target_name, attack_target_x, attack_target_y, attack_target_name)
      dire_hero_2.rendering_map(entire_game_copy, hero_name_list[1], dire_heros[hero_name_list[1]], 
      												  move_to_position_x, move_to_position_y, no_target_ability, purchase_item,
      												  train_ability, cast_target_tree, move_to_target_x, move_to_target_y,
      												  move_to_target_name, attack_target_x, attack_target_y, attack_target_name)
      dire_hero_3.rendering_map(entire_game_copy, hero_name_list[2], dire_heros[hero_name_list[2]], 
      													move_to_position_x, move_to_position_y, no_target_ability, purchase_item,
      													train_ability, cast_target_tree, move_to_target_x, move_to_target_y,
      													move_to_target_name, attack_target_x, attack_target_y, attack_target_name)
      dire_hero_4.rendering_map(entire_game_copy, hero_name_list[3], dire_heros[hero_name_list[3]], 
      													move_to_position_x, move_to_position_y, no_target_ability, purchase_item,
      													train_ability, cast_target_tree, move_to_target_x, move_to_target_y,
      													move_to_target_name, attack_target_x, attack_target_y, attack_target_name)
      dire_hero_5.rendering_map(entire_game_copy, hero_name_list[4], dire_heros[hero_name_list[4]], 
      													move_to_position_x, move_to_position_y, no_target_ability, purchase_item,
      													train_ability, cast_target_tree, move_to_target_x, move_to_target_y,
      													move_to_target_name, attack_target_x, attack_target_y, attack_target_name)

    if len(list(npcs.keys())) > 10:
      for npc_key in npcs:
	       npc = npcs[npc_key]
	       NpcRendering().rendering_map(entire_game_copy, npc)
    
    if len(list(npcs.keys())) > 5:
      for building_key in buildings:
         building = buildings[building_key]
         NpcRendering().rendering_map(entire_game_copy, building)

    if len(list(runes.keys())) > 0:
      for rune_key in runes:
         rune = runes[rune_key]
         RuneRendering().rendering_map(entire_game_copy, rune)

    if mouse_click_x != None:
      cv2.circle(entire_game_copy, (int(mouse_click_y), int(mouse_click_x)), 10, colors['cyan'], 5)
      mouse_click_x = None
      mouse_click_y = None

    player_camera = entire_game_copy[-500 + camera_middle_x:500 + camera_middle_x, -500 + camera_middle_y:500 + camera_middle_y]
    player_camera = cv2.resize(player_camera, dsize=(2000, 2000), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)

    if game_time < 0:
      game_time = str(datetime.timedelta(seconds=-game_time))
      cv2.putText(player_camera, "-" + str(game_time), (1000, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, colors['white'], 1, cv2.LINE_AA)
    else:
      game_time = str(datetime.timedelta(seconds=game_time))
      cv2.putText(player_camera, "-" + str(game_time), (1000, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, colors['white'], 1, cv2.LINE_AA)

    for key in radiant_heros_modifiers:
    	modifiers = radiant_heros_modifiers[key]

    if len(radiant_heros) == 5:
      if radiant_hero_1.selected == True:
        radiant_hero_1.rendering_portrait(player_camera, radiant_heros_modifiers)
      elif radiant_hero_2.selected == True:
        radiant_hero_2.rendering_portrait(player_camera, radiant_heros_modifiers)
      elif radiant_hero_3.selected == True:
        radiant_hero_3.rendering_portrait(player_camera, radiant_heros_modifiers)
      elif radiant_hero_4.selected == True:
        radiant_hero_4.rendering_portrait(player_camera, radiant_heros_modifiers)
      elif radiant_hero_5.selected == True:
        radiant_hero_5.rendering_portrait(player_camera, radiant_heros_modifiers)

    if len(dire_heros) == 5:
      if dire_hero_1.selected == True:
        dire_hero_1.rendering_portrait(player_camera, dire_heros_modifiers)
      elif dire_hero_2.selected == True:
        dire_hero_2.rendering_portrait(player_camera, dire_heros_modifiers)
      elif dire_hero_3.selected == True:
        dire_hero_3.rendering_portrait(player_camera, dire_heros_modifiers)
      elif dire_hero_4.selected == True:
        dire_hero_4.rendering_portrait(player_camera, dire_heros_modifiers)
      elif dire_hero_5.selected == True:
        dire_hero_5.rendering_portrait(player_camera, dire_heros_modifiers)

    player_camera = cv2.cvtColor(player_camera, cv2.COLOR_BGR2RGB)
    main_screen = pygame.surfarray.make_surface(player_camera)

    main_screen = pygame.transform.rotate(main_screen, 270)
    main_screen = pygame.transform.flip(main_screen, True, False)
    screen.blit(main_screen, (0, 0))

    minimap_image_copy = cv2.cvtColor(minimap_image_copy, cv2.COLOR_BGR2RGB)
    if len(radiant_heros) == 5:
      hero_name_list = list(radiant_heros.keys())
      radiant_hero_1.rendering_minimap(minimap_image_copy, radiant_heros[hero_name_list[0]])
      radiant_hero_2.rendering_minimap(minimap_image_copy, radiant_heros[hero_name_list[1]])
      radiant_hero_3.rendering_minimap(minimap_image_copy, radiant_heros[hero_name_list[2]])
      radiant_hero_4.rendering_minimap(minimap_image_copy, radiant_heros[hero_name_list[3]])
      radiant_hero_5.rendering_minimap(minimap_image_copy, radiant_heros[hero_name_list[4]])

    if len(dire_heros) == 5:
    	hero_name_list = list(dire_heros.keys())
    	dire_hero_1.rendering_minimap(minimap_image_copy, dire_heros[hero_name_list[0]])
    	dire_hero_2.rendering_minimap(minimap_image_copy, dire_heros[hero_name_list[1]])
    	dire_hero_3.rendering_minimap(minimap_image_copy, dire_heros[hero_name_list[2]])
    	dire_hero_4.rendering_minimap(minimap_image_copy, dire_heros[hero_name_list[3]])
    	dire_hero_5.rendering_minimap(minimap_image_copy, dire_heros[hero_name_list[4]])

    x_min = int(-512/8 + camera_middle_x / 8)
    x_max = int(512/8 + camera_middle_y / 8)

    y_min = int(-512/8 + camera_middle_x / 8)
    y_max = int(512/8 + camera_middle_y / 8)

    sub_img = minimap_image_copy[int(-512/8 + camera_middle_x / 8):int(512/8 + camera_middle_x / 8), 
    														 int(-512/8 + camera_middle_y / 8):int(512/8 + camera_middle_y / 8)]
    white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
    res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)
    minimap_image_copy[int(-512/8 + camera_middle_x / 8):int(512/8 + camera_middle_x / 8), 
    									 int(-512/8 + camera_middle_y / 8):int(512/8 + camera_middle_y / 8)] = res

    minimap_image_copy = cv2.cvtColor(minimap_image_copy, cv2.COLOR_BGR2RGB)
    minimap = pygame.surfarray.make_surface(minimap_image_copy)
    minimap = pygame.transform.rotate(minimap, 270)
    minimap = pygame.transform.flip(minimap, True, False)
    screen.blit(minimap, (0, 2000 - 512))

    # Flip the display
    pygame.display.flip()