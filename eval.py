import os
import numpy as np
import argparse
from data.nuscenes_pred_split import get_nuscenes_pred_split
from data.ethucy_split import get_ethucy_split
from utils.utils import print_log, AverageMeter, isfile, print_log, AverageMeter, isfile, isfolder, find_unique_common_from_lists, load_list_from_folder, load_txt_file
import pandas as pd
from utils.homotopy import *
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
""" Metrics """

def compute_ADE(pred_arr, gt_arr):
    ade = 0.0
    for pred, gt in zip(pred_arr, gt_arr): # iterates over agents, a.k.a. the minimum is taken for each agent seperately!
        diff = pred - np.expand_dims(gt, axis=0)        # samples x frames x 2
        dist = np.linalg.norm(diff, axis=-1)            # samples x frames
        dist = dist.mean(axis=-1)                       # samples
        ade += dist.min(axis=0)                         # (1, )
    ade /= len(pred_arr)
    return ade


def compute_FDE(pred_arr, gt_arr):
    fde = 0.0
    for pred, gt in zip(pred_arr, gt_arr):
        diff = pred - np.expand_dims(gt, axis=0)        # samples x frames x 2
        dist = np.linalg.norm(diff, axis=-1)            # samples x frames
        dist = dist[..., -1]                            # samples 
        fde += dist.min(axis=0)                         # (1, )
    fde /= len(pred_arr)
    return fde


def align_gt(pred, gt):
    frame_from_data = pred[0, :, 0].astype('int64').tolist()
    frame_from_gt = gt[:, 0].astype('int64').tolist()
    common_frames, index_list1, index_list2 = find_unique_common_from_lists(frame_from_gt, frame_from_data)
    # assert len(common_frames) == len(frame_from_data) # assert not needed?!?!?!
    assert len(index_list1) == len(index_list2)
    gt_new = gt[index_list1, 2:]
    pred_new = pred[:, index_list2, 2:]
    return pred_new, gt_new

def calc_closest_interaction_frame(gt, distances_threshold = 10):
    df = pd.DataFrame(data = gt,
                    columns = ['frame','agent_id','x','y']
    )

    # ids_list convention: make sorted everywhere
    ids_list = sorted(df['agent_id'].unique())
    N = len(ids_list)

    matrix = np.empty((N,N, 7)) # last dimension: min_distance, argmin_frame, interaction_bool, start_frame, end_frame, agent1, agent2
    matrix.fill(np.nan)

    for i, agent1_id in enumerate(ids_list):
        for j, agent2_id in enumerate(ids_list):
            if i > j: # only fill upper diagonal
                df_i = df[df['agent_id'].isin([agent1_id])]
                df_j = df[df['agent_id'].isin([agent2_id])]
                frame_min = max(df_i['frame'].min(), df_j['frame'].min())
                frame_max = min(df_i['frame'].max(), df_j['frame'].max())
                df_i = df_i[(df_i['frame']>= frame_min) * (df_i['frame']<= frame_max)]
                df_j = df_j[(df_j['frame']>= frame_min) * (df_j['frame']<= frame_max)]
                distance_ij = np.sqrt((df_i['x'].values - df_j['x'].values)**2 + (df_i['y'].values - df_j['y'].values)**2)
                if len(distance_ij) > 0: # do interaction length filtering later on!
                    # only for overlapping timestamps of agent interaction pair
                    min_distance = min(distance_ij)
                    argmin_distance = np.argmin(distance_ij)
                    interaction_frame = df_j['frame'].iloc[argmin_distance]
                    interaction_bool = min_distance < distances_threshold
                    matrix[i, j, 0] = min_distance
                    matrix[i, j, 1] = interaction_frame
                    matrix[i, j, 2] = interaction_bool
                    matrix[i, j, 3] = frame_min
                    matrix[i, j, 4] = frame_max
                    matrix[i, j, 5] = agent1_id
                    matrix[i, j, 6] = agent2_id

    return matrix, ids_list

def get_scene_homotopy_classes(gt_batch, pred_samples, id_list, ids_list_scene):
    """" Calcaulate homotopy classes, and return matrix in shape of entire scene """

    gt_batch = torch.from_numpy(gt_batch).unsqueeze(0)
    pred_samples = torch.from_numpy(pred_samples)

    idx = [ids_list_scene.index(x) for x in id_list if x in ids_list_scene]
    no_data_idx = [i for i in range(len(ids_list_scene)) if i not in idx]

    # Create a new tensor with NaN/None values
    new_shape = (gt_batch.shape[0], len(ids_list_scene), gt_batch.shape[2], gt_batch.shape[3])    
    new_tensor = torch.empty(new_shape, dtype=gt_batch.dtype)
    new_tensor.fill_(float('nan'))  # You can also use None if your data type allows
    new_tensor[:, idx, :, :] = gt_batch
    gt_batch = new_tensor

    # Create a new tensor with NaN/None values 
    new_shape = (pred_samples.shape[0], len(ids_list_scene), pred_samples.shape[2], pred_samples.shape[3])  
    new_tensor = torch.empty(new_shape, dtype=pred_samples.dtype)
    new_tensor.fill_(float('nan'))  # You can also use None if your data type allows
    new_tensor[:, idx, :, :] = pred_samples
    pred_samples = new_tensor


    # input: tensor  B x N x T x 2
    angle_diff_gt, homotopy_gt = identify_pairwise_homotopy(gt_batch)
    angle_diff_pred, homotopy_pred = identify_pairwise_homotopy(pred_samples)


    # fill empty data back to nan
    homotopy_gt[:, no_data_idx, :] = float('nan')
    homotopy_gt[:, :, no_data_idx] = float('nan')

    # fill empty data back to nan
    homotopy_pred[:, no_data_idx, :] = float('nan')
    homotopy_pred[:, :, no_data_idx] = float('nan')

    # # calculate mode coverage/correctness metric
    # modes_correct_matrix = (homotopy_gt[:, idx, idx]==homotopy_pred[0, idx, idx]) # first entry corresponds to ML mode.
    # k_sample = homotopy_pred.shape[0]
    # modes_covered_matrix = (homotopy_gt.repeat(k_sample, 1, 1)[:, idx, idx]==homotopy_pred[:,idx, idx]).max(axis=0).values


    # modes_correct = modes_correct_matrix.all().item()
    # modes_covered = modes_covered_matrix.all().item()

    return homotopy_gt, homotopy_pred

def calc_t2cmp(interaction_matrix, ML_mode_correct_matrix,ids_list, homotopy_frame0_list, seq_name, fps = 2, plot = True):

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
                    start_interaction_idx = max(int(start_frames[i,j]), min(homotopy_frame0_list)) # can only start once we have homotopy predictions!
                    end_interaction_idx = int(end_frames[i,j])

                    # implement length limitations here: minimium steps + relative ihstory step! + future step behavior!
                    if closest_interaction_idx < (end_interaction_idx - MIN_PRED_FRAMES): # min framesPRED_FRAMES): # ensures there is a valid homotopy prediction at the last point before interaction
                        
                        pred_start_frame = homotopy_frame0_list.index(start_interaction_idx)
                        pred_end_frame = homotopy_frame0_list.index(closest_interaction_idx+0) # use +1 or not???

                        mode_prediction_pairs = ML_mode_correct_matrix[pred_start_frame:pred_end_frame, i, j] 

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
        fig = make_subplots(rows=1, cols=2, subplot_titles=('All ' + seq_name, 'Close interactions ' + seq_name))

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


if __name__ == '__main__':

    # DATA = 'train'
    DATA = 'val'
    # PREDICTION AND METRIC VARS:
    #TODO: read these form config!!!!
    PRED_FRAMES = 12
    MIN_INTERACTION_FRAMES = 1
    MIN_PRED_FRAMES = 1


    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='nuscenes_pred')
    parser.add_argument('--results_dir', default=f'results/nuscenes_5sample_agentformer/results/epoch_0035/{DATA}/samples')
    parser.add_argument('--data', default=DATA)
    parser.add_argument('--log_file', default='results/nuscenes_5sample_agentformer/log/log_eval.txt')
    args = parser.parse_args()

    dataset = args.dataset.lower()
    results_dir = args.results_dir
    
    if dataset == 'nuscenes_pred':   # nuscenes
        data_root = f'datasets/nuscenes_pred'
        gt_dir = f'{data_root}/label/{args.data}'
        seq_train, seq_val, seq_test = get_nuscenes_pred_split(data_root)
        seq_eval = globals()[f'seq_{args.data}']
    else:                            # ETH/UCY
        gt_dir = f'datasets/eth_ucy/{args.dataset}'
        seq_train, seq_val, seq_test = get_ethucy_split(args.dataset)
        seq_eval = globals()[f'seq_{args.data}']

    if args.log_file is None:
        log_file = os.path.join(results_dir, 'log_eval.txt')
    else:
        log_file = args.log_file
    log_file = open(log_file, 'a+')
    print_log('loading results from %s' % results_dir, log_file)
    print_log('loading GT from %s' % gt_dir, log_file)

    stats_func = {
        'ADE': compute_ADE,
        'FDE': compute_FDE
    }

    stats_meter = {x: AverageMeter() for x in stats_func.keys()}

    seq_list, num_seq = load_list_from_folder(gt_dir)
    print_log('\n\nnumber of sequences to evaluate is %d' % len(seq_eval), log_file)
    for seq_name in seq_eval:
        # load GT raw data
        gt_data, _ = load_txt_file(os.path.join(gt_dir, seq_name+'.txt'))
        gt_raw = []
        for line_data in gt_data:
            line_data = np.array([line_data.split(' ')])[:, [0, 1, 13, 15]][0].astype('float32')
            if line_data[1] == -1: continue
            gt_raw.append(line_data)
        gt_raw = np.stack(gt_raw)

        # find closest interaction indices here
        scene_matrix, ids_list_scene = calc_closest_interaction_frame(gt_raw)
        ML_mode_correct_list = []
        ML_mode_covered_list = []
        homotopy_frame0_list = []   # list with start frame wrpt homotopy calculation

        data_filelist, _ = load_list_from_folder(os.path.join(results_dir, seq_name))    
            
        for data_file in data_filelist:      # each example e.g., seq_0001 - frame_000009
            # for reconsutrction or deterministic
            if isfile(data_file):
                all_traj = np.loadtxt(data_file, delimiter=' ', dtype='float32')        # (frames x agents) x 4
                all_traj = np.expand_dims(all_traj, axis=0)                             # 1 x (frames x agents) x 4
            # for stochastic with multiple samples
            elif isfolder(data_file):
                sample_list, _ = load_list_from_folder(data_file)
                sample_all = []
                for sample in sample_list:
                    sample = np.loadtxt(sample, delimiter=' ', dtype='float32')        # (frames x agents) x 4
                    sample_all.append(sample)
                all_traj = np.stack(sample_all, axis=0)                                # samples x (framex x agents) x 4
            else:
                assert False, 'error'

            # convert raw data to our format for evaluation
            id_list = sorted(np.unique(all_traj[:, :, 1])) # sort to be consistent everywhere
            frame_list = np.unique(all_traj[:, :, 0])
            agent_traj = []
            gt_traj = []
            for idx in id_list:
                # GT traj
                gt_idx = gt_raw[gt_raw[:, 1] == idx]                          # frames x 4
                # predicted traj
                ind = np.unique(np.where(all_traj[:, :, 1] == idx)[1].tolist())
                pred_idx = all_traj[:, ind, :]                                # sample x frames x 4
                # filter data
                pred_idx, gt_idx = align_gt(pred_idx, gt_idx)
                # append
                agent_traj.append(pred_idx)
                gt_traj.append(gt_idx)

            """Calculate homotopy classes here"""
            gt_matrix = np.stack(gt_traj)
            pred_matrix = np.stack(agent_traj, axis = 1)

            homotopy_gt, homotopy_pred = get_scene_homotopy_classes(gt_matrix, pred_matrix, id_list, ids_list_scene)
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
            homotopy_frame0_list.append(min(frame_list))

            """compute stats"""
            for stats_name, meter in stats_meter.items():
                func = stats_func[stats_name]
                value = func(agent_traj, gt_traj)
                meter.update(value, n=len(agent_traj))

            stats_str = ' '.join([f'{x}: {y.val:.4f} ({y.avg:.4f})' for x, y in stats_meter.items()])
            print_log(f'evaluating seq {seq_name:s}, forecasting frame {int(frame_list[0]):06d} to {int(frame_list[-1]):06d} {stats_str}', log_file)

        # evaluate scene homotoyp metrics here:
        ML_mode_correct_matrix = np.stack(ML_mode_correct_list, axis = 0)
        ML_mode_covered_matrix = np.stack(ML_mode_covered_list, axis = 0)

        # for each relevant interaction pair
        T2CMP_correct = calc_t2cmp(scene_matrix, ML_mode_correct_matrix, ids_list_scene, homotopy_frame0_list,  seq_name)
        # T2CMP_covered = calc_t2cmp(scene_matrix, ML_mode_covered_matrix, ids_list_scene, seq_name)




    print_log('-' * 30 + ' STATS ' + '-' * 30, log_file)
    for name, meter in stats_meter.items():
        print_log(f'{meter.count} {name}: {meter.avg:.4f}', log_file)
    print_log('-' * 67, log_file)
    log_file.close()
