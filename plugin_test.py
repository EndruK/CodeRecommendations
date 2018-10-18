# python3 ./main.py ./Seq2Seq/Config/Machine/zuse_json.conf ./Seq2Seq/Config/Output/output.conf ./Seq2Seq/Config/Input/exp01.conf --experiment_path ./Seq2Seq/test_json3

import os, sys
args = sys.argv
string = args[-1]
os.system('python3 ./main.py ./Seq2Seq/Config/Machine/zuse_json.conf ./Seq2Seq/Config/Output/output.conf ./Seq2Seq/Config/Input/exp01.conf --experiment_path ./Seq2Seq/test_json3 --input "' + string + '"')