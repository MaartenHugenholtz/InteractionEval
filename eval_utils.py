import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import math
import torch
from plotly.subplots import make_subplots
from torch.nn.functional import interpolate as tensor_interpolate

def get_rollout_combinations(fut_mod_decel, fut_mod_accel):
    """ Calculate all feasible rollout combinations for all agent pairs 
    and return a tensor with all feasible combinations"""
    fut_mod_rollout_combinations_list = []
    N_agents = fut_mod_decel.shape[1]
    for i in range(N_agents):
        others_idx = [idx for idx in range(N_agents) if idx != i]
        # stacking the rollouts, will change the original order. Thus we need this indexing to reverse back to the orginal
        stack_order = [i] + others_idx 
        corrected_order = [stack_order.index(j) for j in range(N_agents)]

        i_go = fut_mod_accel[:, [i],...]
        others_yield = fut_mod_decel[:, others_idx,...]
        combined_rollout1 = torch.cat([i_go, others_yield], axis = 1)[:, corrected_order, ...]
        fut_mod_rollout_combinations_list.append(combined_rollout1)      

        i_yield = fut_mod_decel[:, [i],...]
        others_go = fut_mod_accel[:, others_idx,...]
        combined_rollout2 = torch.cat([i_yield, others_go], axis = 1)[:, corrected_order, ...]
        fut_mod_rollout_combinations_list.append(combined_rollout2)      

    fut_mod_rollout_combinations = torch.cat(fut_mod_rollout_combinations_list)

    assert(not fut_mod_rollout_combinations.isnan().any())

    return fut_mod_rollout_combinations

def calc_collision_matrix2(motion_tensor):
    """" 
    advanced collision calculation:
    1. calculate x and y position differences
    2. calculate headings
    3. using heading and vehicle sizes 
    """

def calc_collision_matrix(motion_tensor):
    """Calculates boolean matrices to show which pairs are in collision for the given rollouts"""

    # Use constant collision distance index for now. #TODO: use car size parameters
    # DO efficiently: based on sizes an heading vectors, vary what the real distance is + respected collision distance 
    collision_distance = 3  # m

    # Extract necessary dimensions
    num_simulations = motion_tensor.size(0)
    num_agents = motion_tensor.size(1)

    # Reshape the tensor to facilitate broadcasting
    agent_positions = motion_tensor.unsqueeze(1)  # Shape: (num_simulations, 1, num_agents, timesteps, 2)
    positions_diff = agent_positions - agent_positions.permute(0, 2, 1, 3, 4)  # Shape: (num_simulations, num_agents, num_agents, timesteps, 2)
    
    # Calculate squared distances
    squared_distances = torch.sum(positions_diff ** 2, dim=-1)  # Shape: (num_simulations, num_agents, num_agents, timesteps)
    distances = torch.sqrt(squared_distances)  # Shape: (num_simulations, num_agents, num_agents, timesteps)
    
    # Find minimum distances for each agent pair over all timesteps
    min_distances, min_indices = torch.min(distances, dim=-1)  # Shape: (num_simulations, num_agents, num_agents)
    
    # Determine collision matrix for each simulation
    collision_matrices = min_distances < collision_distance  # Shape: (num_simulations, num_agents, num_agents)
    
    # No self-collisions
    diag_indices = torch.arange(num_agents)
    collision_matrices[:, diag_indices, diag_indices] = False
    
    return collision_matrices, min_distances, min_indices, distances



# PREDICTION AND METRIC VARS:
PRED_FRAMES = 12
MIN_INTERACTION_FRAMES = 1
MIN_PRED_FRAMES = 1
FPS = 2

def calc_time_based_metric(mode_vector):
    """ Calculate the time before the inevitable homotopy point at which the mode is precicted correctly"""
    if len(mode_vector) > PRED_FRAMES:
        mode_vector = mode_vector[-PRED_FRAMES:]

    frames_cmpd = len(mode_vector) # number of frames that can be used for calculation, used for relative metric! 
    time_cmpd = frames_cmpd / FPS
    # make code here for cases where it is shorter! And probably best to express in %

    if mode_vector.min(): # all correct predictions
        t2cmpd = frames_cmpd / FPS

    elif mode_vector.max(): # correct prediction at some point # how to handle inconsistent predictions?
        # homotopy classes also false if no interaction.  ---> should get beginning frame too in interaction_matrix! 
        idx_correct = np.argmin(mode_vector[::-1]) 
        t2cmpd = (idx_correct) / FPS  

    else: # only wrong mode predictions
        t2cmpd = 0 

    return t2cmpd, time_cmpd 

def calc_scene_mode_metrics(homotopy_scene_tensor, agents_scene, plot = True):
    _, N_frames, N_agents,_ = homotopy_scene_tensor.shape

    df_modes = pd.DataFrame(columns = ['agent1', 'agent2', 'gt_mode', 'h_final_frame', 'time_pred', 't2cor', 't2cov', 'pred_consistency', 'r_mode_collapse'])

    for i in range(N_agents):
        for j in range(N_agents):
            if j>i:
                not_nan_idx = torch.logical_not(np.isnan(homotopy_scene_tensor[0,:,i, j]))
                if not_nan_idx.max().item(): # there must be some non nan values in order to calculate metrics:
                    a2a_mode_collapse = homotopy_scene_tensor[0,not_nan_idx,i, j]
                    homotopy_class_final = homotopy_scene_tensor[1,not_nan_idx,i, j]
                    modes_correct_matrix = homotopy_scene_tensor[2,not_nan_idx,i, j]
                    modes_covered_matrix = homotopy_scene_tensor[3,not_nan_idx,i, j]
                    converging_trajectories_bool = homotopy_scene_tensor[4,not_nan_idx,i, j]
                    interaction_bool = homotopy_scene_tensor[5, not_nan_idx,i, j]
                    distance_path = homotopy_scene_tensor[6, not_nan_idx,i, j]
                    homotopy_gt = homotopy_scene_tensor[7, not_nan_idx,i, j]
                    homotopy_ml = homotopy_scene_tensor[8, not_nan_idx,i, j]
                    homotopy_pred0 = homotopy_scene_tensor[9, not_nan_idx,i, j]
                    homotopy_pred1 = homotopy_scene_tensor[10, not_nan_idx,i, j]
                    homotopy_pred2 = homotopy_scene_tensor[11, not_nan_idx,i, j]
                    homotopy_mod_feasible_0 = homotopy_scene_tensor[12, not_nan_idx,i, j]
                    homotopy_mod_feasible_1 = homotopy_scene_tensor[13, not_nan_idx,i, j]
                    homotopy_mod_feasible_2 = homotopy_scene_tensor[14, not_nan_idx,i, j]
                    N_feasible_mods = homotopy_mod_feasible_1*1 + homotopy_mod_feasible_2*1 # only consider determined interactions
                    N_pred_mods = homotopy_pred1*1 + homotopy_pred2*1

                    agent1_id = agents_scene[i]
                    agent2_id = agents_scene[j]

                    # TODO: find better way to check if homotopy_class_final converges! For now, just beunfix by checking length!
                    if agent1_id == '1' and (agent2_id == '3' or agent2_id == '9'):
                        print()
                    if interaction_bool.max().item() and not homotopy_class_final.min(): # interaction_bool should look for entire future?? yes, only 1 interaction needed. 
                        # calculate mode metrics here! 

                        h_final_frame = homotopy_class_final.argmax().item()
                        # if h_final_frame == 0: # 
                        #     h_notfinal_frame = homotopy_class_final.argmin().item()
                        #     h_final_frame = h_notfinal_frame + homotopy_class_final[h_notfinal_frame:].argmax().item()
                        h_final_frame_scene = (not_nan_idx*1).argmax().item() + h_final_frame  # for stats, taking into account nan values
                        mode_correct_predictions = modes_correct_matrix[:h_final_frame].numpy()
                        mode_covered_predictions = modes_covered_matrix[:h_final_frame].numpy()
                        gt_mode = homotopy_gt[h_final_frame].item()

                        if len(mode_correct_predictions) > 0:
                            t2cor, pred_time = calc_time_based_metric(mode_correct_predictions)
                            t2cov, _ = calc_time_based_metric(mode_covered_predictions)
                            prediction_consistentcy = check_consistency(mode_correct_predictions)
                            mode_collapse = N_pred_mods[:h_final_frame] < N_feasible_mods[:h_final_frame]
                            r_mode_collapse = (sum(mode_collapse)/len(mode_collapse)).item()


                            # HOW to save / verification visualization. h_final_frame vis. prediction correct time, and one before 
                            # 'agent1', 'agent2', 'gt_mode', 'h_final_frame', 'time_pred', 't2cor', 't2cov', 'pred_consistency', 'r_mode_collapse'])

                            df_modes.loc[len(df_modes.index)] = [agent1_id, agent2_id, gt_mode, h_final_frame_scene, pred_time, t2cor, t2cov, prediction_consistentcy, r_mode_collapse]


                        else:
                            print("final homotopy class detection failed")
    return df_modes

def get_path_crossing_point(path1, path2, crossing_threshold = 1):
    # inter paths 
    distances = cdist(np.array(path1).T, np.array(path2).T)
    min_distance = np.min(distances)
    min_indices = np.argwhere(distances == min_distance)
    intersect_bool = min_distance < crossing_threshold 
    idx1, idx2 = min_indices[0,[0,1]]
    return intersect_bool, idx1, idx2

def check_consistency(predictions):
    # Convert predictions tensor to NumPy array
    predictions = np.array(predictions)
    
    # Get unique classes from predictions
    unique_classes = np.unique(predictions)
    
    # Check if there are only consecutive occurrences of each class
    for cls in unique_classes:
        indices = np.where(predictions == cls)[0]
        if not np.all(np.diff(indices) == 1):
            return False
    return True


def calc_path_homotopy(motion_tensor, agents_scene, threshold_distance = 1):
    num_simulations = motion_tensor.size(0)
    num_agents = motion_tensor.size(1)
    homotopy_classes = torch.zeros((num_simulations, num_agents, num_agents))

    #TODO: 1 make efficient 2 interpolate for accuracy (and getting threshold right) (not needed anymore?)
     
    for s in range(num_simulations):
        for i in range(num_agents):
            for j in range(num_agents):
                if j > i:  # only fill upper triangular part
                    agent1  = motion_tensor[s, i, ...].unsqueeze(1) # shape = (time x 1 x 2)
                    agent2  = motion_tensor[s, j, ...].unsqueeze(1).permute(1, 0, 2) # shape = (1 x time x 2)
                    positions_diff = agent1 - agent2
                    squared_distances = torch.sum(positions_diff ** 2, dim=-1)  # Shape: (num_simulations, num_agents, num_agents, timesteps)
                    distances = torch.sqrt(squared_distances).numpy()  # Shape: (num_simulations, num_agents, num_agents, timesteps)
                    min_distance = distances.min()
                    indices = np.where(distances == min_distance)
                    idx1, idx2 = indices[0][-1], indices[1][-1] # if there are multiple indices, take the last one (in case of stationary vehicles)
                    homotopy_class = 0 if idx1 == idx2 else (1 if idx1 < idx2 else 2) # 0 == undetermined, 1==agent1 first, 2==agent2 first
                    # if homotopy_class == 0:
                    #     print()
                        # new problem: stationary (end of time trajectories)
                    homotopy_classes[s, i, j] = homotopy_class
    
    return homotopy_classes

def calc_intersections(motion_tensor, interp_factor = 100, threshold_distance = 0.5) :
    motion_tensor = tensor_interpolate(motion_tensor, size = (motion_tensor.shape[2]*interp_factor, 2))
    num_simulations = motion_tensor.size(0)
    num_agents = motion_tensor.size(1)
    distances_matrix = np.ones((num_simulations, num_agents, num_agents)) * 9999
    #TODO: 1 make efficient 2 interpolate for accuracy (and getting threshold right) (not needed anymore?)
     
    for s in range(num_simulations):
        for i in range(num_agents):
            for j in range(num_agents):
                if j > i:  # only fill upper triangular part
                    agent1  = motion_tensor[s, i, ...].unsqueeze(1) # shape = (time x 1 x 2)
                    agent2  = motion_tensor[s, j, ...].unsqueeze(1).permute(1, 0, 2) # shape = (1 x time x 2)
                    positions_diff = agent1 - agent2
                    squared_distances = torch.sum(positions_diff ** 2, dim=-1)  # Shape: (num_simulations, num_agents, num_agents, timesteps)
                    distances = torch.sqrt(squared_distances).numpy()  # Shape: (num_simulations, num_agents, num_agents, timesteps)
                    min_distance = distances.min()
                    distances_matrix[s, i, j] = min_distance

    interaction_bool = torch.from_numpy(distances_matrix) < threshold_distance
    return interaction_bool, torch.from_numpy(distances_matrix)