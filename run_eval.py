import numpy as np
import argparse
import os
import sys
import subprocess
import shutil

sys.path.append(os.getcwd())
from data.dataloader import data_generator
from utils.torch import *
from utils.config import Config
from model.model_lib import model_dict
from utils.utils import prepare_seed, print_log, mkdir_if_missing

# results_dir = 'results/Oracle/epoch_0035/val/samples' # oracle
# results_dir = 'results/CV/epoch_0035/val/samples' # cv
results_dir = 'results/nuscenes_5sample_agentformer/results/epoch_0035/val/samples' #AF
# results_dir = 'results/CTT/epoch_0035/val/samples' # CTT

cmd = f'python eval.py --dataset nuscenes_pred --results_dir {results_dir} --data val --log results/nuscenes_5sample_agentformer/log/log_eval.txt'
subprocess.run(cmd.split(' '))
