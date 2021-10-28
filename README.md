# MOBA_RL
Deep Reinforcement Learning for Multiplayer Online Battle Arena

# Prerequisite
1. Python 3
2. tmux
3. gym-derk 
4. Tensorflow 2.4.1
5. Dotaservice of TimZaman
6. Seed RL of Google
7. Ubuntu 20.04
8. RTX 3060 GPU, 16GB RAM is used to run Dota2 environment with rendering
9. RTX 3080 GPU, 46GB RAM is used to training 16 number of headless Dota2 environment together in my case

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

<img src="image/dota2_network.png " width="1000">

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
You need to build the Docker image of Dotaservice mentioned in [README](https://github.com/TimZaman/dotaservice/blob/master/docker/README.md) of Docker of the dotaservice.

You can run the Seel RL for Dota2 using below command.
```
$ ./run_dotaservice.sh 16
$ ./run_impala.sh 16
```

Addidinally, you can terminate all process using below command.
```
$ ./stop.sh
```
If you search through the tmux, you can see that 16 dotaservices is ran in one terminal and the other terminal runs 1 learner and 16 actors.

<img src="image/dota2_server_log.png " width="1000">

### Single Hero Training Result

In the case of 1V1MID game mode, which is the most basic type, you can confirm that training was done properly based on the reward graph.

<img src="image/dota2_single_reward.png " width="300">

After finishing training, you need a trained model from training pc to rendering pc. Please copy it to the model folder and run below command at terminal.

```
$ python -m dotaservice
$ python learner_dota.py
$ python3.7 run.py --render True
```

On the rendering PC, you can check the training result better than the graph as shown in the video below. The hero learns how to move to mid area and attack enermy creep.

[![Dota single hero demo](https://i.ytimg.com/vi/uc1Zyvg-jl0/sddefault.jpg)](https://www.youtube.com/watch?v=uc1Zyvg-jl0 "Dota2 single hero training video - Click to Watch!")
<strong>Click to Watch!</strong>

## 4. Buying and using item
Unlike the Derk game, where items are given at the start, the hero of Dota2 must visit the item store to purchase the item. I will explain how to write Lua script for that because the dotaservice lacks this part.

To add the ability to purchase and use items, you must first add a below code line to the [action_processor.lua](https://github.com/TimZaman/dotaservice/blob/master/dotaservice/lua/action_processor.lua) of dotaservice.

```
local purchaseItem              = require( "bots/actions/purchase_item" )
```

Then add the following line to the [bot_generic.lua](https://github.com/TimZaman/dotaservice/blob/master/dotaservice/lua/bot_generic.lua) file.

```
elseif action.actionType == "DOTA_UNIT_ORDER_PURCHASE_ITEM" then
        action_table[action.actionType] = {{action.purchaseItem.item_name}}
```

Finally, make the purchase_item.lua file to your [actions](https://github.com/TimZaman/dotaservice/tree/master/dotaservice/lua/actions) folder and copy the following content.

```
local PurchaseItem = {}

PurchaseItem.Name = "Purchase Item"
PurchaseItem.NumArgs = 2

local tableItemsToBuy = { 
				"item_tango",
				"item_tango",
			};


----------------------------------------------------------------------------------------------------

function PurchaseItem:Call( hUnit, item_name )
	if ( #tableItemsToBuy == 0 )
	then
		hUnit:SetNextItemPurchaseValue( 0 );
		return;
	end

	local sNextItem = "item_tango"
  
	hUnit:SetNextItemPurchaseValue( GetItemCost( sNextItem ) );
	if ( hUnit:GetGold() >= GetItemCost( sNextItem ) )
	then
		hUnit:ActionImmediate_PurchaseItem( sNextItem );
		table.remove( tableItemsToBuy, 1 );
	end

end

----------------------------------------------------------------------------------------------------
return PurchaseItem
```
