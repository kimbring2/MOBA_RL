# MOBA_RL
Deep Reinforcement Learning for Multiplayer Online Battle Arena

# Prerequisite
1. Python 3
2. gym-derk 
3. Tensorflow 2.4.1
4. Dotaservice of TimZaman
5. Seed RL of Google
6. Ubuntu 20.04
7. RTX 3060 GPU, 16GB RAM is used to run Dota2 environment with rendering
8. RTX 3080 GPU, 64GB RAM is used to training 16 number of headless Dota2 environment together in my case

# Derk Environment
We are going to train small MOBA environment called [Derk](https://gym.derkgame.com/).

<img src="image/derk_screen.gif " width="500">

First, move to [dr-derks-mutant-battlegrounds folder](https://github.com/kimbring2/MOBA_RL/tree/main/dr-derks-mutant-battlegrounds).

Run below command to run the 50 parallel environemnt. I modified [Seel_RL](https://github.com/google-research/seed_rl) of Google for my MOBA case. 

```
$ python learner_1.py --workspace_path [your path]/dr-derks-mutant-battlegrounds/
$ python learner_2.py --workspace_path [your path]/dr-derks-mutant-battlegrounds/
$ python run.py -p1 bot -p2 oldbot -n 50
```

You can check the training progress using Tensorboard log under tboard path of workspace.

# Dota2 Environment
## Rendering Environment
You first need to install Dota2 from Steam. After installation, please check there is Dota2 folder under /home/[your account]/.steam/steam/steamapps/common/dota 2 beta'. We are going to run Dota2 from terminal command.

Next, you need to download and install [dotaservice](https://github.com/TimZaman/dotaservice). In my case, I should modity the _run_dota function of [dotaservice.py](https://github.com/TimZaman/dotaservice/blob/master/dotaservice/dotaservice.py) like below.

```
async def _run_dota(self):
  script_path = os.path.join(self.dota_path, self.DOTA_SCRIPT_FILENAME)
  script_path = '/home/kimbring2/.local/share/Steam/ubuntu12_32/steam-runtime/run.sh'

  # TODO(tzaman): all these options should be put in a proto and parsed with gRPC Config.
  args = [
       script_path,
       '/home/kimbring2/.local/share/Steam/steamapps/common/dota 2 beta/game/dota.sh',
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

  if self.host_mode == HOST_MODE_DEDICATED:
      args.append('-dedicated')
            
  if self.host_mode == HOST_MODE_DEDICATED or \
      self.host_mode == HOST_MODE_GUI:
      args.append('-fill_with_bots')
      args.extend(['+map', 'start', 'gamemode', '{}'.format(self.game_mode)])
      args.extend(['+sv_lan', '1'])
            
  if self.host_mode == HOST_MODE_GUI_MENU:
      args.extend(['+sv_lan', '0'])
            
  # Supress stdout if the logger level is info.
  stdout = None if logger.level == 'INFO' else asyncio.subprocess.PIPE
  
  create = asyncio.create_subprocess_exec(
       *args,
       stdin=asyncio.subprocess.PIPE, stdout=stdout, stderr=stdout,
  )
  self.process = await create

  task_monitor_log = asyncio.create_task(self.monitor_log())

  try:
    await self.process.wait()
  except asyncio.CancelledError:
    kill_processes_and_children(pid=self.process.pid)
    raise
```

## Training Environment
