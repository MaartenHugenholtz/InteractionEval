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

    return fut_mod_rollout_combinations


def calc_collision_matrix(motion_tensor):
    """Calculates boolean matrices to show which pairs are in collision for the given rollouts"""

    # Use constant collision distance index for now. #TODO: use car size parameters
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

def calc_scene_mode_metrics(homotopy_scene_tensor):
    _, N_frames, N_agents,_ = homotopy_scene_tensor.shape
    T2CMP = np.empty((N_agents,N_agents, 2)) # time, time / pred_time 
    T2CMP.fill(np.nan)

    for i in range(N_agents):
        for j in range(N_agents):
            if i>j:
                not_nan_idx = np.logical_not(np.isnan(homotopy_scene_tensor[0,:,i, j]))
                a2a_mode_collapse = homotopy_scene_tensor[0,not_nan_idx,i, j]
                homotopy_class_final = homotopy_scene_tensor[1,not_nan_idx,i, j]
                modes_correct_matrix = homotopy_scene_tensor[2,not_nan_idx,i, j]
                modes_covered_matrix = homotopy_scene_tensor[3,not_nan_idx,i, j]
                converging_trajectories_bool = homotopy_scene_tensor[4,not_nan_idx,i, j]
                
                print()
                


def get_path_crossing_point(path1, path2, crossing_threshold = 1):
    # inter paths 
    distances = cdist(np.array(path1).T, np.array(path2).T)
    min_distance = np.min(distances)
    min_indices = np.argwhere(distances == min_distance)
    intersect_bool = min_distance < crossing_threshold 
    idx1, idx2 = min_indices[0,[0,1]]
    return intersect_bool, idx1, idx2


def calc_path_homotopy(motion_tensor, threshold_distance = 1):
    num_simulations = motion_tensor.size(0)
    num_agents = motion_tensor.size(1)
    homotopy_classes = np.zeros((num_simulations, num_agents, num_agents))

    #TODO: 1 make efficient 2 interpolate for accuracy (and getting threshold right)
     
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
                    idx1, idx2 = indices[0][0], indices[1][0]
                    homotopy_class = 0 if min_distance > threshold_distance else (1 if idx1 < idx2 else 2) 
                    homotopy_classes[s, i, j] = homotopy_class
    
    return homotopy_classes

