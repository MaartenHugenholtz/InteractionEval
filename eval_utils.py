import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import math
import torch

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
    min_distances, _ = torch.min(distances, dim=-1)  # Shape: (num_simulations, num_agents, num_agents)
    
    # Determine collision matrix for each simulation
    collision_matrices = min_distances < collision_distance  # Shape: (num_simulations, num_agents, num_agents)
    
    # No self-collisions
    diag_indices = torch.arange(num_agents)
    collision_matrices[:, diag_indices, diag_indices] = False
    
    return collision_matrices, min_distances
