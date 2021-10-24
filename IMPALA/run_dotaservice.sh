NUM_ENVS=$1

tmux new-session -d -t docker_dota

for ((id=0+13337; id < $NUM_ENVS+13337; id++)); do
    tmux new-window -d -n "env_${id}"
    COMMAND='docker run -itp '"${id}"':13337 dotaservice'
    tmux send-keys -t "env_${id}" "$COMMAND" ENTER
done

tmux attach -t docker_dota