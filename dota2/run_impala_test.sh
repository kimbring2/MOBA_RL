tmux new-session -d -t impala_dota

tmux new-window -d -n learner_shadowfiend
COMMAND_SHADOWFIEND='python3.7 shadowfiend/learner_dota.py --train False'
echo $COMMAND_SHADOWFIEND

tmux send-keys -t "learner_shadowfiend" "$COMMAND_SHADOWFIEND" ENTER

tmux new-window -d -n learner_omninight
COMMAND_OMNINIGHT='python3.7 omninight/learner_dota.py --train False'
echo $COMMAND_OMNINIGHT

tmux send-keys -t "learner_omninight" "$COMMAND_OMNINIGHT" ENTER

tmux new-window -d -n "dotaservice"
COMMAND='python3.7 -m dotaservice'
tmux send-keys -t "dotaservice" "$COMMAND" ENTER

tmux new-window -d -n "actor"
COMMAND='python3.7 run.py --render True'
tmux send-keys -t "actor" "$COMMAND" ENTER

tmux attach -t impala_dota