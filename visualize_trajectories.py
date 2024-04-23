import numpy as np
import argparse
import os
import sys
import subprocess
import shutil
import plotly.graph_objects as go
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
from utils.homotopy import *
from plotly.subplots import make_subplots
import pandas as pd
import plotly.express as px

# PREDICTION AND METRIC VARS:
PRED_FRAMES = 12
MIN_INTERACTION_FRAMES = 1
MIN_PRED_FRAMES = 1


DATAROOT = '/home/maarten/Documents/NuScenes_mini'

sys.path.append(DATAROOT)

def get_model_prediction(data, sample_k):
    model.set_data(data)
    recon_motion_3D, _ = model.inference(mode='recon', sample_num=sample_k)
    sample_motion_3D, data = model.inference(mode='infer', sample_num=sample_k, need_weights=False)
    sample_motion_3D = sample_motion_3D.transpose(0, 1).contiguous()
    return recon_motion_3D, sample_motion_3D

def save_prediction(pred, data, suffix, save_dir):
    pred_num = 0
    pred_arr = []
    fut_data, seq_name, frame, valid_id, pred_mask = data['fut_data'], data['seq'], data['frame'], data['valid_id'], data['pred_mask']

    for i in range(len(valid_id)):    # number of agents
        identity = valid_id[i]
        if pred_mask is not None and pred_mask[i] != 1.0:
            continue

        """future frames"""
        for j in range(cfg.future_frames):
            cur_data = fut_data[j]
            if len(cur_data) > 0 and identity in cur_data[:, 1]:
                data = cur_data[cur_data[:, 1] == identity].squeeze()
            else:
                data = most_recent_data.copy()
                data[0] = frame + j + 1
            data[[13, 15]] = pred[i, j].cpu().numpy()   # [13, 15] corresponds to 2D pos
            most_recent_data = data.copy()
            pred_arr.append(data)
        pred_num += 1

    if len(pred_arr) > 0:
        pred_arr = np.vstack(pred_arr)
        indices = [0, 1, 13, 15]            # frame, ID, x, z (remove y which is the height)
        pred_arr = pred_arr[:, indices]
        # save results
        fname = f'{save_dir}/{seq_name}/frame_{int(frame):06d}{suffix}.txt'
        mkdir_if_missing(fname)
        np.savetxt(fname, pred_arr, fmt="%.3f")
    return pred_num


def calc_ML_mode_metrics(data, sample_motion_3D, verbose = True,
                      ):
    """
    calculate mode matrices
    """

    sorted_ids = data['ids_list_scene'].tolist()

    # reshape motions here, such that we can calculate the matrix for the full shape of the scene
    fut_motion = np.stack(data['fut_motion_3D']) * data['traj_scale']
    fut_motion_batch = torch.from_numpy(fut_motion).unsqueeze(0)

    idx = [sorted_ids.index(x) for x in data['valid_id'] if x in sorted_ids]
    no_data_idx = [i for i in range(len(sorted_ids)) if i not in idx]

    assert( sample_motion_3D[0,...].shape == fut_motion.shape)
    # Reshape the tensor
    new_shape = (fut_motion_batch.shape[0], len(sorted_ids), fut_motion_batch.shape[2], fut_motion_batch.shape[3])

    # Create a new tensor with NaN/None values
    new_tensor = torch.empty(new_shape, dtype=fut_motion_batch.dtype)
    new_tensor.fill_(float('nan'))  # You can also use None if your data type allows
    new_tensor[:, idx, :, :] = fut_motion_batch
    fut_motion_batch = new_tensor

    new_shape = (sample_motion_3D.shape[0], len(sorted_ids), sample_motion_3D.shape[2], sample_motion_3D.shape[3])

    # Create a new tensor with NaN/None values
    new_tensor = torch.empty(new_shape, dtype=sample_motion_3D.dtype)
    new_tensor.fill_(float('nan'))  # You can also use None if your data type allows
    new_tensor[:, idx, :, :] = sample_motion_3D
    sample_motion_3D = new_tensor

    angle_diff_gt, homotopy_gt = identify_pairwise_homotopy(fut_motion_batch)
    angle_diff_pred, homotopy_pred = identify_pairwise_homotopy(sample_motion_3D)

    # fill empty data back to nan
    homotopy_gt[:, no_data_idx, :] = float('nan')
    homotopy_gt[:, :, no_data_idx] = float('nan')

    # fill empty data back to nan
    homotopy_pred[:, no_data_idx, :] = float('nan')
    homotopy_pred[:, :, no_data_idx] = float('nan')

    modes_correct_matrix = (homotopy_gt==homotopy_pred[0,:,:]) # first entry corresponds to ML mode.
    k_sample = homotopy_pred.shape[0]
    modes_covered_matrix = (homotopy_gt.repeat(k_sample, 1, 1)==homotopy_pred[:,:,:]).max(axis=0).values

    modes_correct = modes_correct_matrix.all().item()
    modes_covered = modes_covered_matrix.all().item()

    # works for now, but nan==nan will return false... important for stats


    if verbose:
        print(f"frame: {data['frame']}, with ids: {data['valid_id']} and corresponding idx: {idx}")
        print(f'prediction: \n {homotopy_pred[0,...]}')
        print(f'gt: \n {homotopy_gt}')

        # print(f'modes correct: {modes_correct}. modes covered: {modes_covered}.')
        # print('mode correct matrix:')
        # print(modes_correct_matrix)
        # print('mode covered matrix:')
        # print(modes_covered_matrix)

    return homotopy_gt, homotopy_pred


def calc_closest_interaction_frame(data, distances_threshold = 10):

    gt = data['gt_scene']

    df = pd.DataFrame(data = gt[:,[0,1,2,13,15,10,11,12,16]],
                    columns = ['t','agent_id','agent_type','x','y','width','height','length','heading']
    )

    fig = px.line_3d(df, x = 'x', y = 'y', z = 't', color = 'agent_id')
    fig.show()
    
    
    ids_list = data['ids_list_scene'] # important to use this one, to make sure order is the same as in homotoyp matrices!
    N = len(ids_list)

    matrix = np.empty((N,N, 7)) # last dimension: min_distance, argmin_frame, interaction_bool, start_frame, end_frame, agent1, agent2
    matrix.fill(np.nan)

    for i, agent1_id in enumerate(ids_list):
        for j, agent2_id in enumerate(ids_list):
            if i > j: # only fill upper diagonal
                df_i = df[df['agent_id'].isin([agent1_id])]
                df_j = df[df['agent_id'].isin([agent2_id])]
                t_min = max(df_i['t'].min(), df_j['t'].min())
                t_max = min(df_i['t'].max(), df_j['t'].max())
                df_i = df_i[(df_i['t']>= t_min) * (df_i['t']<= t_max)]
                df_j = df_j[(df_j['t']>= t_min) * (df_j['t']<= t_max)]
                distance_ij = np.sqrt((df_i['x'].values - df_j['x'].values)**2 + (df_i['y'].values - df_j['y'].values)**2)
                if len(distance_ij) > 0: # do interaction length filtering later on!
                    # only for overlapping timestamps of agent interaction pair
                    min_distance = min(distance_ij)
                    argmin_distance = np.argmin(distance_ij)
                    interaction_frame = df_j['t'].iloc[argmin_distance]
                    interaction_bool = min_distance < distances_threshold
                    matrix[i, j, 0] = min_distance
                    matrix[i, j, 1] = interaction_frame
                    matrix[i, j, 2] = interaction_bool
                    matrix[i, j, 3] = t_min
                    matrix[i, j, 4] = t_max
                    matrix[i, j, 5] = agent1_id
                    matrix[i, j, 6] = agent2_id

    return matrix


def calc_t2cmp(interaction_matrix, ML_mode_correct_matrix,ids_list, seq_name, fps = 2, plot = True):

    min_distance= interaction_matrix[:,:, 0]
    min_indices= interaction_matrix[:,:, 1]
    interaction_bool = interaction_matrix[:,:, 2]
    start_frames = interaction_matrix[:,:, 3]
    end_frames = interaction_matrix[:,:, 4]
    agent1_ids = interaction_matrix[:,:, 5]
    agent2_ids = interaction_matrix[:,:, 6]

    N = interaction_bool.shape[0]

    T2CMP = np.empty((N,N, 2)) # time, time / pred_time 
    T2CMP.fill(np.nan)

    # calculate for all, afterwards visualize relevant pairs and matrices 

    for i in range(N):
        for j in range(N):
            if ~np.isnan(min_indices[i,j]): # only check for non nan interactions; overlapping timeframes with min idx
                if True: #interaction_bool[i, j]: # only calculate relevant interacion pairs:
                    closest_interaction_idx = int(min_indices[i,j]) # is equal to frame! ML mode matrix indices should correspond to the frame!
                    start_interaction_idx = int(start_frames[i,j])
                    end_interaction_idx = int(end_frames[i,j])

                    # implement length limitations here: minimium steps + relative ihstory step! + future step behavior!
                    if closest_interaction_idx < (end_interaction_idx - MIN_PRED_FRAMES): # min framesPRED_FRAMES): # ensures there is a valid homotopy prediction at the last point before interaction
                        
                        mode_prediction_pairs = ML_mode_correct_matrix[start_interaction_idx:closest_interaction_idx+0, i, j] # use +1 or not???

                        if len(mode_prediction_pairs) >= MIN_INTERACTION_FRAMES: # only calculate metric for sufficient data

                            # limit length of mode_prediction_pairs to ensure a max for the prediction metric == correct from start of prediction
                            if len(mode_prediction_pairs) > PRED_FRAMES:
                                mode_prediction_pairs = mode_prediction_pairs[-PRED_FRAMES:]

                            frames_cmpd = len(mode_prediction_pairs) # number of frames that can be used for calculation, used for relative metric! 
                            time_cmpd = frames_cmpd / fps
                            # make code here for cases where it is shorter! And probably best to express in %

                            if mode_prediction_pairs.min(): # all correct predictions
                                t2cmpd = frames_cmpd / fps

                            elif mode_prediction_pairs.max(): # correct prediction at some point # how to handle inconsistent predictions?
                                # homotopy classes also false if no interaction.  ---> should get beginning frame too in interaction_matrix! 
                                idx_correct = np.argmin(mode_prediction_pairs[::-1]) 
                                t2cmpd = (idx_correct + 1) / fps  # +1 here? #TODO CHECK

                            else: # only wrong mode predictions
                                t2cmpd = 0 

                            # HOW TO handle inconsistent mode predictions???

                            T2CMP[i, j, 0] = t2cmpd
                            T2CMP[i, j, 1] = t2cmpd / time_cmpd # [%] wrpt prediction time

                    pass


    T2CMP_interaction = T2CMP.copy()
    T2CMP_interaction[np.logical_not(np.nan_to_num(interaction_bool).astype(int))] = np.nan
    if plot:
        plot_idx = 0 # 0 for time , 1 for %
        cbar_title = 'T2CMP [s]' if plot_idx == 0 else 'rT2CMP [%]'
        zmin = 0
        zmax = 1 if plot_idx == 1 else PRED_FRAMES/fps

        customdata= np.stack([min_distance, min_indices, start_frames, end_frames, agent1_ids, agent2_ids], axis = -1)

        # Create heatmap
        fig = make_subplots(rows=1, cols=2, subplot_titles=('All' + seq_name, 'Close interactions' + seq_name))

        # Add heatmaps to subplots
        fig.add_trace(go.Heatmap(z=T2CMP[:,:, plot_idx],customdata= customdata, 
                                    hovertemplate='Value: %{z}<br>' +
                                'Min Distance: %{customdata[0]}<br>' +
                                'Min Indices: %{customdata[1]}<br>' +
                                'Start Frames: %{customdata[2]}<br>' +
                                'End Frames: %{customdata[3]}<br>' +
                                'Agent 1 IDs: %{customdata[4]}<br>' +
                                'Agent 2 IDs: %{customdata[5]}<extra></extra>',
                                 colorscale='Viridis', zmin = zmin, zmax = zmax, colorbar=dict(title=cbar_title)), row=1, col=1)
        fig.add_trace(go.Heatmap(z=T2CMP_interaction[:,:, plot_idx], 
                                 customdata= customdata, 
                                    hovertemplate='Value: %{z}<br>' +
                                'Min Distance: %{customdata[0]}<br>' +
                                'Min Indices: %{customdata[1]}<br>' +
                                'Start Frames: %{customdata[2]}<br>' +
                                'End Frames: %{customdata[3]}<br>' +
                                'Agent 1 IDs: %{customdata[4]}<br>' +
                                'Agent 2 IDs: %{customdata[5]}<extra></extra>',
                                 colorscale='Viridis', zmin = zmin, zmax = zmax, colorbar=dict(title=cbar_title)), row=1, col=2)

        # Add title
        fig.update_layout(title='Time to correct mode prediction matrix')

        # Make axis equal
        fig.update_layout(xaxis=dict(scaleanchor="y", scaleratio=1),
                        yaxis=dict(scaleanchor="x", scaleratio=1))

        # Set custom tick values
        tickvals = np.arange(0, N) # Example tick values
        ticktext = ids_list
        fig.update_layout(xaxis=dict(tickvals=tickvals, ticktext=ticktext),
                        yaxis=dict(tickvals=tickvals, ticktext=ticktext),
                        xaxis2=dict(tickvals=tickvals, ticktext=ticktext),
                        yaxis2=dict(tickvals=tickvals, ticktext=ticktext),
                        autosize = False,
                            width=1800,
                            height=900,)
        fig.show()

    return T2CMP


def test_model(generator, save_dir, cfg):
    total_num_pred = 0


    seq_name_curr = None
    

    while not generator.is_epoch_end():
        


        data = generator()
        if data is None:
            continue
        seq_name, frame = data['seq'], data['frame']
        frame = int(frame)
        sys.stdout.write('testing seq: %s, frame: %06d                \r' % (seq_name, frame))  
        sys.stdout.flush()

        if seq_name_curr != seq_name:

            # calculate homotopy time metric here 

            seq_name_curr = seq_name
            ML_mode_correct_list = []
            ML_mode_covered_list = []

        #### put stuff here to get lane ids: # make seperate data modify function!!!
        # nuscenes = NuScenes('v1.0-mini', dataroot=DATAROOT)
        # scene_token = nuscenes.field2token('scene', 'name', seq_name)[0]
        # scene = nuscenes.get('scene', scene_token)
        # scene_log_token = scene['log_token']
        # log_data = nuscenes.get('log', scene_log_token)
        # location = log_data['location'],
        # nusc_map = NuScenesMap(dataroot=DATAROOT, map_name=location)
        # lane_ids = nusc_map.get_closest_lane(x = 500, y = 1740, radius  =5)


        # modify_scene(data)

        if True: #seq_name in SCENES:
            # gt_motion_3D = torch.stack(data['fut_motion_3D'], dim=0).to(device) * cfg.traj_scale
            with torch.no_grad():
                recon_motion_3D, sample_motion_3D = get_model_prediction(data, cfg.sample_k)
                assert (sample_motion_3D[0] == recon_motion_3D).min().item() # check that ML is at idx 0 
            recon_motion_3D, sample_motion_3D = recon_motion_3D * cfg.traj_scale, sample_motion_3D * cfg.traj_scale



            homotopy_gt, homotopy_pred = calc_ML_mode_metrics(data, sample_motion_3D)

            ML_covered = (homotopy_gt.repeat(homotopy_pred.shape[0], 1, 1)== homotopy_pred[:,:,:]).max(axis=0).values.numpy()
            ML_correct = homotopy_gt[0,...].numpy() == homotopy_pred[0,...].numpy()

            nan_indices = np.isnan(homotopy_gt[0,...].numpy())
            # put nan values back in place as safety measure: nan == nan --> false
            ML_correct = ML_correct.astype(float)
            ML_correct[nan_indices] = np.nan
            ML_covered = ML_covered.astype(float)
            ML_covered[nan_indices] = np.nan
        
            ML_mode_correct_list.append(ML_correct)
            ML_mode_covered_list.append(ML_covered)

            last_frame = int( data['gt_scene'][:,0].max() - PRED_FRAMES )
            if frame == last_frame:
                ids_list = data['ids_list_scene'].tolist()
                interaction_matrix= calc_closest_interaction_frame(data)
                ML_mode_correct_matrix = np.stack(ML_mode_correct_list, axis = 0)
                ML_mode_covered_matrix = np.stack(ML_mode_covered_list, axis = 0)

                # for each relevant interaction pair
                T2CMP = calc_t2cmp(interaction_matrix, ML_mode_correct_matrix, ids_list, seq_name)
                # T2CMP_covered = calc_t2cmp(interaction_matrix, ML_mode_covered_matrix, ids_list, seq_name)

                pass

            # implement plotting here
            # if data['seq'] == 'scene-0553':
            # data['scene_vis_map'].visualize_trajs(data, sample_motion_3D)
            # print()

            # store ids in list, together with matrices in list. Later figure out how to build one matrix from it




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--data_eval', default='test')
    parser.add_argument('--epochs', default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--cached', action='store_true', default=False)
    parser.add_argument('--cleanup', action='store_true', default=False)
    args = parser.parse_args()

    

    """ setup """
    cfg = Config('nuscenes_5sample_agentformer' )
    if args.epochs is None:
        epochs = [cfg.get_last_epoch()]
    else:
        epochs = [int(x) for x in args.epochs.split(',')]

    torch.set_default_dtype(torch.float32)
    device = torch.device('cuda', index=0) if 0 >= 0 and torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available(): torch.cuda.set_device(0)
    torch.set_grad_enabled(False)
    log = open(os.path.join(cfg.log_dir, 'log_test.txt'), 'w')

    for epoch in epochs:
        prepare_seed(cfg.seed)
        """ model """
        if not args.cached:
            model_id = cfg.get('model_id', 'agentformer')
            model = model_dict[model_id](cfg)
            model.set_device(device)
            model.eval()
            if epoch > 0:
                cp_path = cfg.model_path % epoch
                print_log(f'loading model from checkpoint: {cp_path}', log, display=True)
                model_cp = torch.load(cp_path, map_location='cpu')
                model.load_state_dict(model_cp['model_dict'], strict=False)

        """ save results and compute metrics """
        data_splits = ['val']
        SCENES = ['scene-1100']

        for split in data_splits:  
            generator = data_generator(cfg, log, split=split, phase='testing')
            save_dir = f'{cfg.result_dir}/epoch_{epoch:04d}/{split}'; mkdir_if_missing(save_dir)
            eval_dir = f'{save_dir}/samples'
            if not args.cached:
                test_model(generator, save_dir, cfg)

            log_file = os.path.join(cfg.log_dir, 'log_eval.txt')
            cmd = f"python eval.py --dataset {cfg.dataset} --results_dir {eval_dir} --data {split} --log {log_file}"
            subprocess.run(cmd.split(' '))



