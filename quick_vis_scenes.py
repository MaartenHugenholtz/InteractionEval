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


model_id = cfg.get('model_id', 'agentformer')
model = model_dict[model_id](cfg)
model.set_device(device)
model.eval()
cp_path = cfg.model_path % epoch
print_log(f'loading model from checkpoint: {cp_path}', log, display=True)
model_cp = torch.load(cp_path, map_location='cpu')
model.load_state_dict(model_cp['model_dict'], strict=False)

""""  #################  """


def get_model_prediction(data, sample_k):
    model.set_data(data)
    recon_motion_3D, _ = model.inference(mode='recon', sample_num=sample_k)
    sample_motion_3D, data = model.inference(mode='infer', sample_num=sample_k, need_weights=False)
    sample_motion_3D = sample_motion_3D.transpose(0, 1).contiguous()
    return recon_motion_3D, sample_motion_3D



""" Get predictions and compute metrics """

split = 'val'
focus_scene = 'scene-0103'


generator = data_generator(cfg, log, split=split, phase='testing')
scene_preprocessors = generator.sequence
df_modemetrics = pd.DataFrame()

for scene in scene_preprocessors:

    if scene.seq_name == focus_scene:

        gt = scene.gt
        pred_frames = scene.pred_frames


        for frame_idx, frame in enumerate(pred_frames):
            # frame corresponds to the current timestep, i.e. the last of pre_motion
            data = scene(frame)
            if data is None:
                print('Frame skipped in loop')
                continue

            seq_name, frame = data['seq'], data['frame']
            frame = int(frame)
            sys.stdout.write('testing seq: %s, frame: %06d                \r' % (seq_name, frame))  
            sys.stdout.flush()

            with torch.no_grad():
                recon_motion_3D, sample_motion_3D = get_model_prediction(data, cfg.sample_k)
            recon_motion_3D, sample_motion_3D = recon_motion_3D * cfg.traj_scale, sample_motion_3D * cfg.traj_scale

            data['scene_vis_map'].visualize_trajs(data, sample_motion_3D)


