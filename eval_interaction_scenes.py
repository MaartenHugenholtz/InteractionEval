import numpy as np
import argparse
import os
import sys
import subprocess
import shutil
sys.path.append(os.getcwd())
from data.dataloader_debug import data_generator
from utils.torch import *
from utils.config import Config
from model.model_lib import model_dict
from utils.utils import prepare_seed, print_log, mkdir_if_missing
from eval_utils import *
from utils.homotopy import *
import plotly.graph_objects as go
import plotly.express as px
from agent_class import Agent
import time
from tqdm import tqdm

start_time = time.time()

""" setup """
cfg = Config('nuscenes_5sample_agentformer' )
epochs = [cfg.get_last_epoch()]
epoch = epochs[0]

torch.set_default_dtype(torch.float32)
device = torch.device('cuda', index=0) if 0 >= 0 and torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available(): torch.cuda.set_device(0)
torch.set_grad_enabled(False)
log = open(os.path.join(cfg.log_dir, 'log_test.txt'), 'w')




""" Get predictions and compute metrics """

split = 'val'

generator = data_generator(cfg, log, split=split, phase='testing')
scene_preprocessors = generator.sequence
df_modemetrics = pd.DataFrame()

for scene in tqdm(scene_preprocessors):
    gt = scene.gt
    pred_frames = scene.pred_frames
    df_scene = Agent.process_data(gt)
    agents_scene = list(df_scene.agent_id.unique()) # definitive order for agent ids in all tensors

    path_intersection_bool, inframes_bool, df_modemetrics_scene = calc_path_intersections(df_scene, agents_scene, pred_frames)


    df_modemetrics_scene['scene'] = scene.seq_name
    # append to main df:
    df_modemetrics = pd.concat([df_modemetrics, df_modemetrics_scene])


end_time = time.time()
execution_time = end_time - start_time

print("Execution Time:", execution_time, "seconds")

# save result:
print(df_modemetrics)
df_modemetrics.to_csv(f'interaction_metrics_{split}.csv', index = False)