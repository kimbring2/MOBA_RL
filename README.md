# MOBA_RL
Deep Reinforcement Learning for Multiplayer Online Battle Arena

# Tabel of Content
- [Prerequisite](#prerequisite)
- [Reference](#reference)
- [Derk Environment](#derk-environment)
- [Dota2 Environment](#dota2-environment)
  * [1. Network Architecture](#1-network-architecture)
  * [2. Rendering Environment](#2-rendering-environment)
  * [3. Training Environment](#3-training-environment)
    + [Single Hero Training Result](#single-hero-training-result)
  * [4. Ability and Item](#4-ability-and-item)
    + [Buying and Using Item](#buying-and-using-item)
    + [Learning and Using Ability](#learning-and-using-ability)
    + [Upgrading Item](#upgrading-item)
    + [Using Town Portar Scroll](#using-town-portar-scroll)
    + [Using the Courier](#using-the-courier)
    + [Using Ability to Other Hero](#using-ability-to-other-hero)

# Prerequisite
1. Python 3
2. tmux
3. gym-derk 
4. Tensorflow 2.4.1
5. TensorFlow Probability 0.11.0
6. Dotaservice of TimZaman
7. Seed RL of Google
8. Ubuntu 20.04
11. No GPU, 30GB RAM Desktop mini Desktop is used to make multiple Dokcer container of Dotaservice
10. GPU, 46GB RAM Desktop is used to make the IMPALA RL agent

# Reference
1. Seed RL: https://github.com/google-research/seed_rl
2. Derk's Gym: https://gym.derkgame.com/ 
3. Dotaservice: https://github.com/TimZaman/dotaservice
4. Dotaclient: https://github.com/TimZaman/dotaclient
5. LastOrder-Dota2: https://github.com/bilibili/LastOrder-Dota2

# Derk Environment
We are going to train small MOBA environment called [Derk](https://gym.derkgame.com/).

<img src="image/derk_screen.gif " width="500">

First, move to [dr-derks-mutant-battlegrounds folder](https://github.com/kimbring2/MOBA_RL/tree/main/dr-derks-mutant-battlegrounds).

<img src="image/derk_network.png" width="800">

Run below command to run the 50 parallel environemnt. I modified [Seel_RL](https://github.com/google-research/seed_rl) of Google for my MOBA case. 

```
$ python learner_1.py --workspace_path [your path]/dr-derks-mutant-battlegrounds/
$ python learner_2.py --workspace_path [your path]/dr-derks-mutant-battlegrounds/
$ python run.py -p1 bot -p2 oldbot -n 50
```

You can check the training progress using Tensorboard log under tboard path of workspace.

<img src="image/reward_derk.png " width="300">

After training, one team choose to attack the opponent and the other team choose to defend.

[![Derk demo](https://img.youtube.com/vi/avQra5Wt-wI/sddefault.jpg)](https://www.youtube.com/watch?v=avQra5Wt-wI "Derk video - Click to Watch!")
<strong>Click to Watch!</strong>

# Dota2 Environment
In the case of Derk environment, you can render game and train agent at the same time on one PC. However, in the case of dota2, PC for rendering and a PC for training are required separately because of large size of network and game, and multiple docker containers. Additionally, I use the same user name for the rendering pc and the training pc for the convenience of path setting.  

## 1. Network Architecture
Unlike network for Derk game, which consists of one for observation processing and one for action selecting network, Dota2 agent needs a 6 observation processing networks and 5 action processing networks due to the large size of the game.

<img src="image/dota2_shadowfiend_network.png " width="1000">

Furthermore, each hero has different kind of ability. Some abilities need to be activated by user or other abilities are just passive. That means action part of network structure also should be different from hero to here. 

<img src="image/dota2_ability_type.png " width="1000">

In the case of Shadowfiend which has 4 non passive abilities, ability action network has 4 output. For Omniknight case, network output is three.

<img src="image/dota2_omniknight_network.png " width="1000">

## 2. Rendering Environment
You first need to install Dota 2 from Steam. After installation, please check there is Dota2 folder under /home/[your account]/.steam/steam/steamapps/common/dota 2 beta'. We are going to run Dota2 from terminal command.

Next, you need to download and install [dotaservice](https://github.com/TimZaman/dotaservice). In my case, I should modity the _run_dota function of [dotaservice.py](https://github.com/TimZaman/dotaservice/blob/master/dotaservice/dotaservice.py) like below.

```
async def _run_dota(self):
  script_path = os.path.join(self.dota_path, self.DOTA_SCRIPT_FILENAME)
  script_path = '/home/[your user name]/.local/share/Steam/ubuntu12_32/steam-runtime/run.sh'

  # TODO(tzaman): all these options should be put in a proto and parsed with gRPC Config.
  args = [
       script_path,
       '/home/[your user name]/.local/share/Steam/steamapps/common/dota 2 beta/game/dota.sh',
       '-botworldstatesocket_threaded',
       '-botworldstatetosocket_frames', '{}'.format(self.ticks_per_observation),
       '-botworldstatetosocket_radiant', '{}'.format(self.PORT_WORLDSTATES[TEAM_RADIANT]),
       '-botworldstatetosocket_dire', '{}'.format(self.PORT_WORLDSTATES[TEAM_DIRE]),
       '-con_logfile', 'scripts/vscripts/bots/{}'.format(self.CONSOLE_LOG_FILENAME),
       '-con_timestamp',
       '-console',
       '-dev',
       '-insecure',
       '-noip',
       '-nowatchdog',  # WatchDog will quit the game if e.g. the lua api takes a few seconds.
       '+clientport', '27006',  # Relates to steam client.
       '+dota_1v1_skip_strategy', '1',
       '+dota_surrender_on_disconnect', '0',
       '+host_timescale', '{}'.format(self.host_timescale),
       '+hostname dotaservice',
       '+sv_cheats', '1',
       '+sv_hibernate_when_empty', '0',
       '+tv_delay', '0',
       '+tv_enable', '1',
       '+tv_title', '{}'.format(self.game_id),
       '+tv_autorecord', '1',
       '+tv_transmitall', '1',  # TODO(tzaman): what does this do exactly?
  ]
```

If you enter the following command after modification, the Dota2 game will be launched.

```
$ python -m dotaservice
$ python env_test.py --render True
```

Dota 2 should be successfully launched and the hero selection screen should appear. When entering the main game, you can then use \ key to pop up the console. Then, try use the 'jointeam spec' command to see the hero, tower of entire map.

[![Dota launch demo](https://i.ytimg.com/vi/GzILbfRFnZE/sddefault.jpg)](https://www.youtube.com/watch?v=GzILbfRFnZE "Dota2 launch test video - Click to Watch!")
<strong>Click to Watch!</strong>

Now, you are ready to train Dota2 with Seed RL just as we did in the Derk game. Try to run Seed RL and Dota2 together on the rendering PC with the following command. The hero behaves randomly because model does not be trained yet.

```
$ python -m dotaservice
$ python learner_dota.py
$ python run.py --render True
```

To see proper behavior, you need to put the weight trained on training PC in the [model](https://github.com/kimbring2/MOBA_RL/tree/main/dota2/model) folder.

## 3. Training Environment
Unlike Derk game, each Dotaservice occupies more than 1GB of memory. Therefore, it is good to run them separately on a mini PC without a GPU. Then, Learner and Actor of IMPALA RL need to be ran on a PC with a GPU.

You need to build the Docker image of Dotaservice mentioned in [README](https://github.com/TimZaman/dotaservice/blob/master/docker/README.md).  

<img src="image/training_environment.png " width="800">

First, you need to run the Docker containers of Dotaservice using below command on no GPU pc.
```
$ ./run_dotaservice.sh 16
```

Next, you need to run the IMPALA RL at GPU PC using below command.
```
$ ./run_impala.sh 16
```

Addidinally, you can terminate both process using below command.
```
$ ./stop.sh
```

If you search through the tmux, you can see that 16 dotaservices is ran in one terminal and the other terminal runs 1 learner and 16 actors.

<img src="image/dota2_server_log.png " width="1000">

Run below command to see Tensorboard log of training PC from rendeirng PC remotely.

```
tensorboard --host 0.0.0.0 --logdir=./tensorboard
```

### Single Hero Training Result
In the case of 1V1MID game mode, which is the most basic type, you can confirm that training was done properly based on the reward graph.

<img src="image/dota2_single_reward.png " width="500">

After finishing training, you need a trained model from training pc to rendering pc. Please copy it to the model folder and run below command at terminal.

```
$ python -m dotaservice
$ python learner_dota.py
$ python run.py --render True
```

On the rendering PC, you can check the training result better than the graph as shown in the video below. The hero learns how to move to mid area and attack enermy creep.

[![Dota2 single hero demo](https://i.ytimg.com/vi/uc1Zyvg-jl0/sddefault.jpg)](https://www.youtube.com/watch?v=uc1Zyvg-jl0 "Dota2 single hero training video - Click to Watch!")
<strong>Click to Watch!</strong>

## 4. Ability and Item
Unlike the Derk game, where items and ability are chosen at the start of game, the hero of Dota2 can improve the them gradually during the playing time.

<img src="image/ability_item_inrto.png " width="800">

Therefore, hero need to decide what abilities and items to buy when leveling up or collecting ma certain amount of gold. The Rule Based method is used for that part because there is no suitable Learning-Based method can deal with it.

### Buying and Using Item
Unlike the Derk game, where items are given at the start, the hero of Dota2 must visit the item store to purchase the item. I will explain how to write Lua script for that because the dotaservice lacks this part.

The Tango is most basic item can be purchased at the store when start of game. Hero can use it on the near trees to regenerate the health.

<img src="image/tango_description.png" width="300">

Hero can purchase and use Tango items like a below video.

[![Dota2 Tango item demo](https://i.ytimg.com/vi/-Alt7TSRZVg/sddefault.jpg)](https://www.youtube.com/watch?v=-Alt7TSRZVg "Dota2 Tango item video - Click to Watch!")
<strong>Click to Watch!</strong>

### Learning and using ability
Unlike the Derk game, where ability are given at the start, the hero of Dota2 must learn the ability to use it. Furthermore, the method of selecting the target for each ability is slightly different. Target can be nothing, unit, and tree.

<img src="image/shadowraze_description.png" width="300">

The Shadowraze does not require the target. It would best to use that ability when an enemy hero or creep is within range of it like a below video. 

[![Dota2 Shadowraze ability demo](https://img.youtube.com/vi/OVScU7aLEpk/sddefault.jpg)](https://www.youtube.com/watch?v=OVScU7aLEpk "Dota2 Shadowraze ability video - Click to Watch!")
<strong>Click to Watch!</strong>

## 6. Upgrading item
Unlike the Derk game, where item are not changed until end of game, the hero of Dota2 can upgrade low level items to high level one by using recipe system.

For example, in the case of the Magic Stick, which is a very early game, it can be upgraded to the Magic Wand by using 2 Iron Branch and 1 Recipe like a below image.

<img src="image/dota2_magic_wand.png" width="600">

The video below shows how to use obtain the magic wand from recipe.

[![Dota2 upgrade item demo](https://img.youtube.com/vi/EbCzKKf4aao/sddefault.jpg)](https://www.youtube.com/watch?v=EbCzKKf4aao "Dota2 upgrade item video - Click to Watch!")
<strong>Click to Watch!</strong>

### Using Town Portar Scroll
Unlike the Derk game, where map size small, the range of Dota2 between starting and battle point are long. Therefore, the hero should use the Town Portal scroll to join and exit from battle. 

<img src="image/dota2_tp_scrool.png" width="300">

In the case of Town Portar scroll, it is given at start of game, and it is stored in the 15th slot of inventory. After using it, hero can buy it from store using gold. The Python code for using it is like below.

[![Dota2 use teloport scroll](https://img.youtube.com/vi/rudbbEhshIw/sddefault.jpg)](https://www.youtube.com/watch?v=rudbbEhshIw "Dota2 use teloport scroll video - Click to Watch!")
<strong>Click to Watch!</strong>

### Using the Courier
Unlike the Derk game, where map size small, the range of Dota2 between starting and battle point are long. Therefore, the hero of battle point can use the Courier to obtain the items without moving to starting point.

<img src="image/dota2_courier.png" width="600">

If hero purchase an item when there is no store around, that item in stored under stash. Below video shows how the hero of battle point obtains an item without moving to the starting point using Courier.

[![Dota2 use courier](https://img.youtube.com/vi/xSvZRYFXErg/sddefault.jpg)](https://www.youtube.com/watch?v=xSvZRYFXErg "Dota2 use courier video - Click to Watch!")
<strong>Click to Watch!</strong>

### Using Ability to Other Hero
In MOBA games, there is a hero who is mainly in charge of attacks, and there is a hero who assists it. Mainly units with abilities such as HP recovery and shield generation can take that position. 

<img src="image/purification_description.png" width="600">

The following shows an example of recovering HP of same team hero.

[![Dota2 use purification](https://img.youtube.com/vi/41a3XyxKlus/sddefault.jpg)](https://www.youtube.com/watch?v=xSvZRYFXErg "Dota2 use use purification video - Click to Watch!")
<strong>Click to Watch!</strong>
