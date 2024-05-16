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


""" Get predictions and compute metrics """

split = 'val'

df_modemetrics = pd.DataFrame()

data_path = 'datasets/interaction/val_reformatted'
scenes = os.listdir(data_path)

min_past_frames = 10 # full history horizon 
min_future_frames = 5 # corresponds to 0.5s, so 1 frame equivalent for nuscenes

for scene in tqdm(scenes):
    df_scene = pd.read_csv(data_path + '/' + scene)
    df_scene = df_scene.rename(columns = {'frame_id': 'frame', 'track_id': 'agent_id'})
    pred_frames = np.arange(df_scene.frame.min() + min_past_frames -1, df_scene.frame.max() - min_future_frames +1)
    df_scene = df_scene[(df_scene.frame >= pred_frames.min())*(df_scene.frame <= pred_frames.max())]
    agents_scene = list(df_scene.agent_id.unique()) # definitive order for agent ids in all tensors

    path_intersection_bool, inframes_bool, df_modemetrics_scene = calc_path_intersections(df_scene, agents_scene, pred_frames)


    df_modemetrics_scene['scene'] = scene[:-4]
    # append to main df:
    df_modemetrics = pd.concat([df_modemetrics, df_modemetrics_scene])


end_time = time.time()
execution_time = end_time - start_time

print("Execution Time:", execution_time, "seconds")

# save result:
print(df_modemetrics)
df_modemetrics.to_csv(f'interaction_metrics__INTERACTION_{split}.csv', index = False)