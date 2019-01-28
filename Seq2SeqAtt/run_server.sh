#!/bin/bash
python_path="/home/andre/python_virtualenv/tf-nightly/bin/python3"
script_path="/home/andre/Git/CodeRecommendations/Seq2SeqAtt/generate_process.py"
experiment_path="/home/andre/Git/CodeRecommendations/Seq2SeqAtt/att_prebuild_dataset"
experiment_config_path="/home/andre/Git/CodeRecommendations/Seq2SeqAtt/Config/Input/exp01.conf"
checkpoint_path="/home/andre/Git/CodeRecommendations/Seq2SeqAtt/att_prebuild_dataset/results/checkpoints/2018-10-25 00:46:37.991859"

echo "$python_path $script_path $experiment_path $experiment_config_path '$checkpoint_path'"

"$python_path" "$script_path" "$experiment_path" "$experiment_config_path" "$checkpoint_path"
