NUM_ACTORS=$1

tmux new-session -d -t impala_dota

tmux new-window -d -n learner
COMMAND='python3.7 learner_dota.py --env_number '"${NUM_ACTORS}"''
echo $COMMAND
tmux send-keys -t "learner" "$COMMAND" ENTER

for ((id=0; id < $NUM_ACTORS; id++)); do
    tmux new-window -d -n "actor_${id}"
    COMMAND='python3.7 run.py --id  '"${id}"''
    tmux send-keys -t "actor_${id}" "$COMMAND" ENTER
done

tmux attach -t impala_dota