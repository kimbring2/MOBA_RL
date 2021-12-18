NUM_ACTORS=$1
IP=$2

tmux new-session -d -t impala_dota

tmux new-window -d -n learner_shadowfiend
COMMAND_SHADOWFIEND='python3.7 shadowfiend/learner_dota.py --env_number '"${NUM_ACTORS}"''
echo $COMMAND_SHADOWFIEND

tmux send-keys -t "learner_shadowfiend" "$COMMAND_SHADOWFIEND" ENTER

tmux new-window -d -n learner_omninight
COMMAND_OMNINIGHT='python3.7 omninight/learner_dota.py --env_number '"${NUM_ACTORS}"''
echo $COMMAND_OMNINIGHT

tmux send-keys -t "learner_omninight" "$COMMAND_OMNINIGHT" ENTER

for ((id=0; id < $NUM_ACTORS; id++)); do
    tmux new-window -d -n "actor_${id}"
    COMMAND='python3.7 run.py --id '"${id}"' --ip '"${IP}"''
    tmux send-keys -t "actor_${id}" "$COMMAND" ENTER
done

tmux attach -t impala_dota