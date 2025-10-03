#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

pid_file="scripts/local_judge.txt"

stop_server() {
    if [ -f $pid_file ]; then
        server_pid=$(cat $pid_file)
        kill $server_pid
        rm $pid_file
        echo "Server with PID $server_pid killed"
    else
        echo "PID file not found. Server might not be running."
    fi
}


if [ "$1" = "start" ]; then
    stop_server

    lmdeploy serve api_server internlm/internlm2_5-7b-chat --server-port 23333 &
    server_pid=$!
    echo $server_pid > $pid_file
    echo "Server started with PID: $server_pid"
elif [ "$1" = "stop" ]; then
    stop_server
else
    echo "Usage: $0 {start|stop}"
fi