import numpy as np
import torch
import itertools
from eval_utils import calc_collision_matrix, calc_travelled_distance


def get_model_prediction(data, sample_k, agent_dict, use_gt_path = True):

    frame_curr = data['frame']
    futures_constant_v = []
    futures_accel = []
    futures_decel = []

    for agent_id in data['valid_id']:
        agent = agent_dict[str(int(agent_id))]
        fut_agent_constant_v = agent.rollout_future(frame_curr, direction = 'constant', use_gt_path = use_gt_path)
        fut_agent_accel = agent.rollout_future(frame_curr, direction = 'accel', use_gt_path = use_gt_path)
        fut_agent_decel = agent.rollout_future(frame_curr, direction = 'decel', use_gt_path = use_gt_path)
        futures_constant_v.append(fut_agent_constant_v)
        futures_accel.append(fut_agent_accel)
        futures_decel.append(fut_agent_decel)

    
    constant_v = torch.from_numpy(np.stack(futures_constant_v)).unsqueeze(0)[...,0:2]
    accel = torch.from_numpy(np.stack(futures_accel)).unsqueeze(0)[...,0:2]
    decel = torch.from_numpy(np.stack(futures_decel)).unsqueeze(0)[...,0:2]
    # constant_v = torch.from_numpy(np.stack(futures_constant_v))[...,0:2]
    # accel = torch.from_numpy(np.stack(futures_accel))[...,0:2]
    # decel = torch.from_numpy(np.stack(futures_decel))[...,0:2]

    # make combinations
    fut_rollout_combinations = get_rollout_combinations(constant_v, accel, decel)

    # check collisions (basic distance based one, efficient)
    # collision_matrices, min_distances, _, _ = calc_collision_matrix(fut_rollout_combinations, collision_distance= 3)
    # collision_bool_sims = torch.amax(collision_matrices, dim = (1,2))
    # feasible_rollouts = fut_rollout_combinations[torch.logical_not(collision_bool_sims),...]

    feasible_rollouts = fut_rollout_combinations # collision checker moved within rollout combiner for memory issues
    
    # calculate average speed/distance travelled (utility)
    distances_agents = calc_travelled_distance(feasible_rollouts)
    distances_sims = torch.sum(distances_agents, axis = -1)

    #TODO: Check, maybe also use too close almost collisions in likelihood?

    # sort simulations from highest total distance to lowest (as a measure for utility -> probability)
    sorted_sim_indices = torch.argsort(distances_sims, descending=True)
    ML_modes_idx =  sorted_sim_indices[:sample_k] # only predict k modes, sorted in descending likelihood
    sample_motion_3D = fut_rollout_combinations[ML_modes_idx,...]
    recon_motion_3D = sample_motion_3D[[0],...] # ML prediction at first index
    
    # tensor output shapes
    # recon_motion_3D.shape = (N_agent, fut_stepes, 2)
    # sample_motion_3D.shape = (sample_k, N_agent, fut_stepes, 2)

    return recon_motion_3D, sample_motion_3D  


def get_rollout_combinations(constant_v, accel, decel):
    combinations = []

    roll_out_matrix = torch.cat([constant_v, accel, decel])
    N_sims, N_agents, _, _ = roll_out_matrix.shape

    # Generate all combinations of simulations for all agents
    sim_combinations = itertools.product(range(N_sims), repeat=N_agents)
    indices = torch.tensor(list(sim_combinations))

    
    for idx_comb in range(indices.shape[0]):
        sim = torch.stack(
            [roll_out_matrix[indices[idx_comb, idx_agent], idx_agent,...] for idx_agent in range(N_agents)])
        collision_matrices, min_distances, _, _ = calc_collision_matrix(sim.unsqueeze(0), collision_distance= 3)
        collision_bool_sims = torch.amax(collision_matrices, dim = (1,2))
        if collision_bool_sims:
            continue
        else:
            combinations.append(sim)


    fut_rollout_combinations = torch.stack(combinations)

    return fut_rollout_combinations