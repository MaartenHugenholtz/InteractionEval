# Init NuScenes. Requires the dataset to be stored on disk.
from nuscenes.nuscenes import NuScenes
import tqdm
import numpy as np
from typing import List
import sys
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.bitmap import BitMap
import matplotlib.pyplot as plt
from plotly.tools import mpl_to_plotly
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.prediction import PredictHelper
from pyquaternion import Quaternion
import json
import os
from itertools import chain
import cv2
import argparse
import pandas as pd


past_frames = 4
future_frames = 12
NUM_IN_TRAIN_VAL = 200

DATAROOT = '/home/maarten/Documents/NuScenes_mini'
sys.path.append(DATAROOT)

### Load classes ###
nuscenes = NuScenes('v1.0-mini', dataroot=DATAROOT)
helper = PredictHelper(nuscenes)
######################

def get_prediction_challenge_split(split: str, dataroot: str) -> List[str]:
    """
    Gets a list of {instance_token}_{sample_token} strings for each split.
    :param split: One of 'mini_train', 'mini_val', 'train', 'val'.
    :param dataroot: Path to the nuScenes dataset.
    :return: List of tokens belonging to the split. Format {instance_token}_{sample_token}.
    """
    if split not in {'mini_train', 'mini_val', 'train', 'train_val', 'val'}:
        raise ValueError("split must be one of (mini_train, mini_val, train, train_val, val)")
    
    if split == 'train_val':
        split_name = 'train'
    else:
        split_name = split

    path_to_file = os.path.join(dataroot, "maps", "prediction", "prediction_scenes.json")
    prediction_scenes = json.load(open(path_to_file, "r"))
    scenes = create_splits_scenes()
    scenes_for_split = scenes[split_name]
    
    if split == 'train':
        scenes_for_split = scenes_for_split[NUM_IN_TRAIN_VAL:]
    if split == 'train_val':
        scenes_for_split = scenes_for_split[:NUM_IN_TRAIN_VAL]

    token_list_for_scenes = map(lambda scene: prediction_scenes.get(scene, []), scenes_for_split)

    return prediction_scenes, scenes_for_split, list(chain.from_iterable(token_list_for_scenes))


def plot_scene(scene_name, split, plot_on_map = False):
        prediction_scenes, split_scenes, split_data = get_prediction_challenge_split(split, dataroot=DATAROOT)
        scene_token = nuscenes.field2token('scene', 'name', scene_name)[0]
        scene = nuscenes.get('scene', scene_token)
        scene_log_token = scene['log_token']
        log_data = nuscenes.get('log', scene_log_token)
        location = log_data['location']

        scene_data_orig = prediction_scenes.get(scene_name, [])
        scene_data_orig_set = set(scene_data_orig)
        scene_data = set(scene_data_orig)
        for data in scene_data_orig:
            cur_sample = helper.get_sample_annotation(*data.split('_'))
            sample = cur_sample
            for i in range(past_frames - 1):
                if sample['prev'] == '':
                    break
                sample = nuscenes.get('sample_annotation', sample['prev'])
                cur_data = sample['instance_token'] + '_' + sample['sample_token']
                scene_data.add(cur_data)
            sample = cur_sample
            for i in range(future_frames):
                sample = nuscenes.get('sample_annotation', sample['next'])
                cur_data = sample['instance_token'] + '_' + sample['sample_token']
                scene_data.add(cur_data)

        all_tokens = np.array([x.split("_") for x in scene_data])
        all_samples = set(np.unique(all_tokens[:, 1]).tolist())
        all_instances = np.unique(all_tokens[:, 0]).tolist()
        first_sample_token = scene['first_sample_token']
        first_sample = nuscenes.get('sample', first_sample_token)
        while first_sample['token'] not in all_samples:
            first_sample = nuscenes.get('sample', first_sample['next'])

        frame_id = 0
        sample = first_sample
        cvt_data = []
        while True:
            if sample['token'] in all_samples:
                instances_in_frame = []

                # first get ego pose:
                ego_pose = nuscenes.get('ego_pose', sample['data']['LIDAR_TOP'])
                data = np.ones(18) * -1.0
                data[0] = frame_id
                data[1] = 99.0 # unique id for ego vehicle    
                # ego vehicle dimensions from:  https://forum.nuscenes.org/t/dimensions-of-the-ego-vehicle-used-to-gather-data/550
                data[10] = 1.730 # with
                data[11] = 1.562  # height
                data[12] = 4.084 # length
                #############################
                data[13] = ego_pose['translation'][0]
                data[14] = ego_pose['translation'][2]
                data[15] = ego_pose['translation'][1]
                data[16] = Quaternion(ego_pose['rotation']).yaw_pitch_roll[0]
                data[17] = 1 # ego vehicle constant in data
                data = data.astype(str)
                data[2] = 'Car' # ego vehicle is car

                cvt_data.append(data)

                for ann_token in sample['anns']:
                    annotation = nuscenes.get('sample_annotation', ann_token)
                    category = annotation['category_name']
                    instance = annotation['instance_token']
                    cur_data = instance + '_' + annotation['sample_token']
                    if cur_data not in scene_data:
                        continue
                    instances_in_frame.append(instance)
                    # get data
                    data = np.ones(18) * -1.0
                    data[0] = frame_id
                    data[1] = all_instances.index(instance)
                    data[10] = annotation['size'][0]
                    data[11] = annotation['size'][2]
                    data[12] = annotation['size'][1]
                    data[13] = annotation['translation'][0]
                    data[14] = annotation['translation'][2]
                    data[15] = annotation['translation'][1]
                    data[16] = Quaternion(annotation['rotation']).yaw_pitch_roll[0]
                    data[17] = 1 if cur_data in scene_data_orig_set else 0
                    data = data.astype(str)
                    if 'car' in category:
                        data[2] = 'Car'
                    elif 'bus' in category:
                        data[2] = 'Bus'
                    elif 'truck' in category:
                        data[2] = 'Truck'
                    elif 'emergency' in category:
                        data[2] = 'Emergency'
                    elif 'construction' in category:
                        data[2] = 'Construction'
                    else:
                        raise ValueError(f'wrong category {category}')
                    cvt_data.append(data)

            frame_id += 1
            if sample['next'] != '':
                sample = nuscenes.get('sample', sample['next'])
            else:
                break
            
        cvt_data = np.stack(cvt_data)
        df = pd.DataFrame({
            't': cvt_data[:,0].astype(np.float32),
            'agent_id': cvt_data[:,1].astype(np.float32),
            'agent_type':cvt_data[:,2],
            'x': cvt_data[:,13].astype(np.float32),
            'y': cvt_data[:,15].astype(np.float32),
            'width': cvt_data[:,10].astype(np.float32),
            'height': cvt_data[:,11].astype(np.float32),
            'length': cvt_data[:,12].astype(np.float32),
            'heading': cvt_data[:,16].astype(np.float32),
        })


        if plot_on_map:
            # Generate Maps
            map_name = nuscenes.get('log', scene['log_token'])['location']
            nusc_map = NuScenesMap(dataroot=DATAROOT, map_name=map_name)
            margin = 50
            xy = cvt_data[:, [13, 15]].astype(np.float32)
            x_min = np.round(xy[:, 0].min() - margin)
            x_max = np.round(xy[:, 0].max() + margin)
            y_min = np.round(xy[:, 1].min() - margin)
            y_max = np.round(xy[:, 1].max() + margin)

            fig, ax = nusc_map.render_map_patch((x_min, y_min, x_max, y_max), render_egoposes_range = False)
            xyit = cvt_data[:, [13, 15, 1, 0]].astype(np.float32)

            colors = px.colors.qualitative.Plotly
            agent_ids = np.unique(xyit[:,2])

            projected_trajs = project_trajectories(df)

            for agent_idx, agent_id in enumerate(agent_ids):
                agent_traj = df[df['agent_id'] == agent_id]
                plt.scatter(agent_traj['x'], agent_traj['y'], 
                            c = colors[agent_idx], label =f'gt_agent_{agent_id}' )
            plt.legend()
            plt.show()
                
        else:
            colors = px.colors.qualitative.Plotly
            projected_trajs = project_trajectories(df) # plot projected trajectories




        
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


    color_scale = px.colors.qualitative.Plotly
    fig_line = px.line(df, x='distance_along_path', y='t', color='agent_id', color_discrete_sequence=color_scale)
    fig_line.show()
    #plot xy raw data
    fig_scatter = px.scatter(df, x='x', y='y', color=df['agent_id'].astype(str), color_discrete_sequence=color_scale)
    fig_scatter.show()

    fig = go.Figure()
    df_onpath = df[df['on_path']]
    for i, agent_id in enumerate(df['agent_id'].unique()):
        agent_df = df_onpath[df_onpath['agent_id']==agent_id]
        if not agent_df.empty:
            x_points = list(agent_df['rb_along_path'].values) + list(agent_df['fb_along_path'].values[::-1]) + [agent_df['rb_along_path'].values[0]]
            y_points = list(agent_df['t'].values) + list(agent_df['t'].values[::-1]) + [agent_df['t'].values[0]]
            fig.add_trace(
                go.Scatter(x=x_points, y=y_points, 
                        fill="toself",
                        mode = 'lines',
                        legendgroup='gt',
                        name = f'gt_agent_{agent_id}',
                        showlegend=True,
                        line=dict(color=color_scale[i]),
                        ))
    
    fig.show()
    # return processed df
    return df


plot_scene('scene-0103', 'val')

