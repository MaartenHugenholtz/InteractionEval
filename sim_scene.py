import numpy as np
import argparse
import os
import sys
import subprocess
import shutil
from data.preprocessor_modify import preprocess_modify
from data.preprocessor import preprocess
from utils.homotopy import *
sys.path.append(os.getcwd())
from data.dataloader import data_generator
from utils.torch import *
from utils.config import Config
from model.model_lib import model_dict
from utils.utils import prepare_seed, print_log, mkdir_if_missing
from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.prediction import PredictHelper
from nuscenes.map_expansion.map_api import NuScenesMap
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from GIDM import GIDM, simulate_scene

DATAROOT = '/home/maarten/Documents/NuScenes_mini'
SCENE = 'scene-0103'
SPLIT = 'val'

# prepare log files and model #
sys.path.append(DATAROOT)
cfg = Config('nuscenes_5sample_agentformer' )
torch.set_default_dtype(torch.float32)
device = torch.device('cuda', index=0) if 0 >= 0 and torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available(): torch.cuda.set_device(0)
torch.set_grad_enabled(False)
log = open(os.path.join(cfg.log_dir, 'log_test.txt'), 'w')
prepare_seed(cfg.seed)
""" model """
model_id = cfg.get('model_id', 'agentformer')
model = model_dict[model_id](cfg)
model.set_device(device)
model.eval()
epoch = cfg.get_last_epoch()
cp_path = cfg.model_path % epoch
print_log(f'loading model from checkpoint: {cp_path}', log, display=True)
model_cp = torch.load(cp_path, map_location='cpu')
model.load_state_dict(model_cp['model_dict'], strict=False)
#######################

def min_distance_to_path(row, path):
    point = np.array([row['x'], row['y']])
    distances = np.sqrt(np.sum((path - point[:, np.newaxis])**2, axis=0))
    return np.min(distances)

def distance_along_path(row, xy_path, s_path):
    point = np.array([row['x'], row['y']])
    distances = np.sqrt(np.sum((xy_path - point[:, np.newaxis])**2, axis=0))
    idx_path = np.argmin(distances)
    return s_path[idx_path]

def project_trajectories(df, projected_id = 99, interp_points = 1000, path_threshold = 3):
    """
    project trajectories to path of the agent
    return a filtered df with s, t coordaintes
    """
    ego_df = df[df['agent_id']==projected_id]
    t_interp = np.linspace(ego_df['t'].values[0], ego_df['t'].values[-1], interp_points)
    x_path = np.interp(t_interp, ego_df['t'], ego_df['x'])
    y_path = np.interp(t_interp, ego_df['t'], ego_df['y'])
    xy_path = np.array([x_path, y_path])

    s_path = np.zeros_like(x_path)
    ds = np.sqrt(np.diff(x_path)**2 + np.diff(y_path)**2)
    s_path[1:] = np.cumsum(ds)

    df['min_distance_to_path'] = df.apply(lambda row: min_distance_to_path(row, xy_path), axis=1)
    df['on_path'] = df['min_distance_to_path'] < path_threshold
    df['distance_along_path'] = df[df['on_path']].apply(lambda row: distance_along_path(row, xy_path, s_path), axis = 1)
    df['rb_along_path'] = df['distance_along_path'] - 0.5*df['length'] # rear bound vehicle along path
    df['fb_along_path'] = df['distance_along_path'] + 0.5*df['length'] # front bound vehicle along path


    color_scale = px.colors.qualitative.Plotly + px.colors.qualitative.Plotly
    # fig_line = px.line(df, x='distance_along_path', y='t', color='agent_id', color_discrete_sequence=color_scale)
    # fig_line.show()
    # #plot xy raw data
    # fig_scatter = px.scatter(df, x='x', y='y', color=df['agent_id'].astype(str), color_discrete_sequence=color_scale)
    #     # fig_scatter = px.scatter(df, x='x', y='y', color= [color_scale[-1] if agent_id == 99 else color_scale[int(agent_id)] for agent_id in df['agent_id']], )
    # fig_scatter.show()
    color_map = {str(agent_id): (color_scale[-1] if agent_id == 99 else color_scale[int(agent_id)]) for agent_id in df['agent_id']}
    # fig_scatter = px.scatter(df, x='x', y='y',animation_frame = 't', color=df['agent_id'].astype(str), color_discrete_map=color_map)
    # fig_scatter.show()

    fig_scatter = px.scatter(df, x='x', y='y', color=df['agent_id'].astype(str), color_discrete_map=color_map)
    fig_scatter.show()

    fig = go.Figure()
    df_onpath = df[df['on_path']]
    for i, agent_id in enumerate(df['agent_id'].unique()):
        agent_df = df_onpath[df_onpath['agent_id']==agent_id]
        if not agent_df.empty:
            x_points = list(agent_df['rb_along_path'].values) + list(agent_df['fb_along_path'].values[::-1]) + [agent_df['rb_along_path'].values[0]]
            y_points = list(agent_df['t'].values) + list(agent_df['t'].values[::-1]) + [agent_df['t'].values[0]]
            fig.add_trace(
                go.Scatter(x=y_points, y=x_points, 
                        fill="toself",
                        mode = 'lines',
                        legendgroup='gt',
                        name = f'gt_agent_{agent_id}',
                        showlegend=True,
                        line=dict(color=color_scale[-1] if agent_id == 99 else color_scale[int(agent_id)]),
                        ))
    fig.update_layout(
        xaxis_title="frame",
        yaxis_title="distance along ego path"
    )
    fig.show()
    # return processed df
    return df

def get_prediction(data, sample_k = cfg.sample_k):
    with torch.no_grad():
        model.set_data(data)
        recon_motion_3D, _ = model.inference(mode='recon', sample_num=sample_k)
        sample_motion_3D, data = model.inference(mode='infer', sample_num=sample_k, need_weights=False)
        sample_motion_3D = sample_motion_3D.transpose(0, 1).contiguous()
    recon_motion_3D, sample_motion_3D = recon_motion_3D * cfg.traj_scale, sample_motion_3D * cfg.traj_scale
    return sample_motion_3D

def calc_mode_metrics(data, sample_motion_3D, verbose = True):
    """
    calculate mode matrices
    """
    fut_motion = np.stack(data['fut_motion_3D']) * data['traj_scale']
    fut_motion_batch = torch.from_numpy(fut_motion).unsqueeze(0)
    angle_diff_gt, homotopy_gt = identify_pairwise_homotopy(fut_motion_batch)
    angle_diff_pred, homotopy_pred = identify_pairwise_homotopy(sample_motion_3D)

    modes_correct_matrix = (homotopy_gt==homotopy_pred[0,:,:]) # first entry corresponds to ML mode.
    k_sample = homotopy_pred.shape[0]
    modes_covered_matrix = (homotopy_gt.repeat(k_sample, 1, 1)==homotopy_pred[:,:,:]).max(axis=0).values

    modes_correct = modes_correct_matrix.all().item()
    modes_covered = modes_covered_matrix.all().item()


    if verbose:
        print(f'modes correct: {modes_correct}. modes covered: {modes_covered}.')
        print('mode correct matrix:')
        print(modes_correct_matrix)
        print('mode covered matrix:')
        print(modes_covered_matrix)

    return modes_correct, modes_covered, angle_diff_pred, angle_diff_gt

def plot_homotopy(df, fut_frames = 12):
    
    for t in range(int(df.t.min()), int(df.t.max()-fut_frames+1)):
        df_t = df[(df.t>t)*(df.t<=(t+fut_frames))]
        # input shape: agents x time x 2
        ego_agent_xy = df_t[df_t['agent_id'] == 99][['x', 'y']].values
        agents_xy = [ego_agent_xy] # intialize list with ego agent xy
        ids_list = []
        agents_ids = list(df_t['agent_id'].unique())
        agents_ids.remove(99) # remove ego
        for agent_id in agents_ids:
            agent_xy = df_t[df_t['agent_id'] == agent_id][['x', 'y']].values
            if agent_xy.shape[0] == fut_frames: # for now only calculate for agents that are lon engough in the data
                agents_xy.append(agent_xy)
                ids_list.append(agent_id)
        fut_motion = np.stack(agents_xy, axis = 0)
        fut_motion_batch = torch.from_numpy(fut_motion).unsqueeze(0)
        angle_diff_gt, homotopy_gt = identify_pairwise_homotopy(fut_motion_batch)

        # add data to df for ego agent 99:
        for idx, agent_id in enumerate(ids_list):
            df.loc[(df['agent_id'] == 99)*(df['t']==t), f'angle_diff_ego_{int(agent_id)}'] = angle_diff_gt[0, 0, idx+1].item() # interactions with ego agent

    
    fig = go.Figure()
    color_scale = px.colors.qualitative.Plotly
    homotopy_cols = [col for col in df.keys() if 'angle_diff_ego' in col]
    for i, col in enumerate(homotopy_cols):
        df_col = df.dropna(subset = [col])
        fig.add_trace(go.Scatter(x = df_col['t'], y = df_col[col],
                        mode = 'lines',
                        name = col,
                        showlegend=True,
                        line=dict(color=color_scale[i+1]),
                        )
        )

    fig.show()
    pass

def time_shift_modify(gt, shift_args):
        dt_shift = shift_args
        df = pd.DataFrame(gt)
        df[0] = df[0].astype(float) # time axis
        tmin = df[0].min()
        tmax = df[0].max()
        df.loc[df[1] == '99.0', 0] = df.loc[df[1] == '99.0', 0] + dt_shift
        df = df[(df[0] >= tmin)*(df[0] <= tmax)] # remove values that are not within timeframe
        matrix = df.astype(str).values.astype('<U20') # convert back to string and matrix format
        return matrix

# checkout gt first:
# processor = preprocess(cfg.data_root_nuscenes_pred, seq_name= SCENE, parser = cfg,
#                             log=log, split = SPLIT)
processor = preprocess_modify(cfg.data_root_nuscenes_pred, seq_name= SCENE, parser = cfg,
                            log=log, split = SPLIT, modify_func = simulate_scene,
                            modify_args=(None))

df = pd.DataFrame(data = processor.gt[:,[0,1,2,13,15,10,11,12,16]],
                columns = ['t','agent_id','agent_type','x','y','width','height','length','heading']
)
df['agent_id'] = df['agent_id'].astype(str)

fig = px.scatter(df, x='x', y='y', animation_frame='t', hover_data = ['agent_id', 't'], color='agent_id')
fig.update_layout(
    xaxis=dict(
        range=[df.x.min(), df.x.max()],  # Set the x-axis range
        ),
    yaxis=dict(
        range=[df.y.min(), df.y.max()], # Set the y-axis range
        )
)
fig.show()

fig = px.scatter(df, x='x', y='y',  hover_data = ['agent_id', 't'], color='agent_id')
fig.update_layout(
    xaxis=dict(
        range=[df.x.min(), df.x.max()],  # Set the x-axis range
        ),
    yaxis=dict(
        range=[df.y.min(), df.y.max()], # Set the y-axis range
        )
)
fig.show()

projected_trajs = project_trajectories(df)

