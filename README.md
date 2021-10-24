# MOBA_RL
Deep Reinforcement Learning for Multiplayer Online Battle Arena

# Prerequisite
1. Python 3
2. gym-derk 
3. Tensorflow 2.4.1
4. Dotaservice
5. Ubuntu 20.04
6. RTX 3060 GPU, 16GB RAM is used to run Dota2 environment with rendering
7. RTX 3080 GPU, 64GB RAM is used to training 16 number of headless Dota2 environment together in my case

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
You first need to install Dota2 from Steam. After installation, please check there is Dota2 folder under /home/[your account]/.steam/steam/steamapps/common/dota 2 beta'. We are going to run Dota2 from terminal command.

Next, you need to download and install [dotaservice](https://github.com/TimZaman/dotaservice). 
