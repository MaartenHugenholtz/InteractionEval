import numpy as np
import torch
import itertools
from eval_utils import calc_collision_matrix, calc_travelled_distance
import os
import glob
import pandas as pd

def get_files_starting_with(directory, prefix):
    """
    Get a list of file paths in the specified directory that start with the given prefix.
    :param directory: The path of the directory to search in.
    :param prefix: The prefix string to match file names against.
    :return: A list of file paths that start with the given prefix.
    """
    matching_files = []

    # Traverse the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith(prefix):
                # Get the full path of the file
                full_path = os.path.join(root, file)
                matching_files.append(full_path)

    return matching_files

def get_model_prediction(data, sample_k,
                         gt_path = 'datasets/nuscenes_pred_ctt/val_interaction',
                         pred_path = 'CTT_predictions/pred',
                         interp_factor = 2, # 4 hz predictions -> 2 Hz data
                         output_frames = 6,
                         ):
    """"
    Plan:
    1. find matching scene, gt file --> not found, raise error
    2. based on ego/99 agent, find corresponding curr_frame 
    3. make matching agent_id dict for agents based on their locations, if agent not found --> keep curr as prediction
    4. get predictions from the correct file
    5. interpolate predictions to 2Hz

    """
    frame_curr = data['frame']
    ids = data['valid_id']
    scene = data['seq']

    # FIND MATCHING SCENE AND GT FILE
    gt_file = glob.glob(f'{gt_path}//{scene}*.csv')
    if not gt_file:
        raise FileNotFoundError('gt file not found')
    elif len(gt_file) > 1:
        raise FileNotFoundError('multiple files found')
    
    df_gt = pd.read_csv(gt_file[0])

    # FIND CURR CTTT FRAME
    df_ego = df_gt[df_gt['agent_name']=='ego'].reset_index()
    ego_idx = data['valid_id'].index(99)
    agents_pre = torch.stack(data['pre_motion_3D'], axis = 0) * data['traj_scale']# agents x time x 2
    agents_curr = agents_pre[:,-1,:]
    ego_curr = agents_curr[ego_idx, :]
    df_ego['dist_to_curr'] = np.sqrt((df_ego['x'] - ego_curr[0].item())**2 + (df_ego['y'] - ego_curr[1].item())**2)
    curr_frame = df_ego.loc[df_ego['dist_to_curr'].idxmin()]
    dist_to_curr = curr_frame.dist_to_curr
    ctt_frame = curr_frame.frame
    if dist_to_curr > 0.5:
        raise ValueError('could not match frame with gt')
    
    # MATCH AGENT_IDS
    df_gt_curr = df_gt[df_gt.frame == ctt_frame]
    agent_id_ctt_dict = {}
    for i, id in enumerate(ids):
        agent_curr = agents_curr[i, :]
        df_gt_curr['dist_to_curr'] = np.sqrt((df_gt_curr['x'].values - agent_curr[0].item())**2 + (df_gt_curr['y'].values - agent_curr[1].item())**2)
        gt_cur_agent = df_gt_curr.loc[df_gt_curr['dist_to_curr'].idxmin()]
        match_dist = gt_cur_agent.dist_to_curr
        if match_dist > 0.5: # no match:
            print(f'NO MATCH FOUND, MATCH DIST = {match_dist}')
            ctt_agent_name = None
        else:
            ctt_agent_name = gt_cur_agent.agent_name

        agent_id_ctt_dict.update({id: ctt_agent_name})

    # GET CTT PREDICTIONS FROM FILE AND INTERPOLATE
    pred_file = glob.glob(f'{pred_path}//{scene}_ts_{str(ctt_frame)}.csv')
    if not pred_file:
        raise FileNotFoundError('pred file not found')
    elif len(pred_file) > 1:
        raise FileNotFoundError('multiple files found')

    df_pred = pd.read_csv(pred_file[0])
    df_pred = df_pred[df_pred.mode_k == 0].reset_index() # all modes indentical

    #TODO: merge prediction files?? in some new frames there might not be predictions, 
    # but we can still use part of the old ones? --> nope... those are old predictions, and could alson be worse than the current state of the agent....

    try:
        predictions = []
        for id, ctt_agent_name in agent_id_ctt_dict.items():

            pred_agent = df_pred[df_pred.agent_name == ctt_agent_name]

            if ctt_agent_name is None or len(pred_agent) == 0: # keep static true position if no prediciton available
                idx = ids.index(id)
                pred_agent = np.zeros((output_frames, 2))
                pred_agent[:,0] = agents_curr[idx,0] # x
                pred_agent[:,1] = agents_curr[idx,1] # y
            else:
                pred_agent = pred_agent[pred_agent.frame >= ctt_frame] 
                pred_agent = pred_agent[['x', 'y']].values

                # interpolate to go back to 2Hz
                pred_agent_interp = pred_agent[::interp_factor, :]
                pred_agent = pred_agent_interp[1:] # remove curr frame frome file

                if pred_agent.shape[0] < output_frames: # repeat last prediction points to maintain shape
                    pred_agent_repeat = np.zeros((output_frames, 2))
                    pred_agent_repeat[:,0] = pred_agent[-1,0]
                    pred_agent_repeat[:,1] = pred_agent[-1,1]
                    pred_agent_repeat[0:pred_agent.shape[0],:] = pred_agent
                    pred_agent = pred_agent_repeat

            predictions.append(torch.from_numpy(pred_agent))

    except IndexError:
        print()


    recon_motion_3D = torch.stack(predictions, axis = 0)
    sample_motion_3D = recon_motion_3D.unsqueeze(0).repeat(sample_k, 1, 1, 1)
    
    # tensor output shapes
    # recon_motion_3D.shape = (N_agent, fut_stepes, 2)
    # sample_motion_3D.shape = (sample_k, N_agent, fut_stepes, 2)

    return recon_motion_3D, sample_motion_3D  