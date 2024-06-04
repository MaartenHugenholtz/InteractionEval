import numpy as np
import torch
import itertools
from eval_utils import calc_collision_matrix, calc_travelled_distance
from agent_class import Agent
from eval_utils import *

def get_model_prediction(data, sample_k, agent_dict = None, path_intersection_bool_frame = None, use_gt_path = True,
                         H_PRED = 12):
    
    if agent_dict is None: # model prediction case, build agent dict from data
        gt = data['gt_scene']
        df_scene = Agent.process_data(gt)
        vmax_scene = df_scene.v.max()
        
        agent_dict = {}
        agents_scene = list(df_scene.agent_id.unique()) # definitive order for agent ids in all tensors
        for agent in agents_scene:
            df_agent = df_scene[df_scene.agent_id == agent]
            agent_class = Agent(df_agent, v_max = vmax_scene, fut_steps=H_PRED) # impose vmax based on gt scene velocities
            agent_dict.update({agent: agent_class})

    if path_intersection_bool_frame is None:
        min_past_frames = 2
        min_future_frames = 1

        pred_frames = np.arange(df_scene.frame.min() + min_past_frames -1, df_scene.frame.max() - min_future_frames +1)

        path_intersection_bool, _, _ = calc_path_intersections(df_scene, agents_scene, pred_frames)

        # path_intersction for current agents in frame (for cv/oracle)
        idx_scene_agents = [agents_scene.index(str(int(agent_id))) for agent_id in data['valid_id']]
        path_intersection_bool_frame = path_intersection_bool[idx_scene_agents,:][:,idx_scene_agents] # re-order according to data[valid_id]
        


    frame_curr = data['frame']
    futures_constant_v = []
    futures_accel = []
    futures_decel = []
    agents_gt = []

    for agent_id in data['valid_id']:
        agent = agent_dict[str(int(agent_id))]
        fut_agent_constant_v = agent.rollout_future(frame_curr, direction = 'constant', use_gt_path = use_gt_path)
        fut_agent_accel = agent.rollout_future(frame_curr, direction = 'accel', use_gt_path = use_gt_path)
        fut_agent_decel = agent.rollout_future(frame_curr, direction = 'decel', use_gt_path = use_gt_path)
        futures_constant_v.append(fut_agent_constant_v)
        futures_accel.append(fut_agent_accel)
        futures_decel.append(fut_agent_decel)
        agents_gt.append(agent.get_gt_agent(frame_curr))
    
    constant_v = torch.from_numpy(np.stack(futures_constant_v)).unsqueeze(0)[...,0:2]
    accel = torch.from_numpy(np.stack(futures_accel)).unsqueeze(0)[...,0:2]
    decel = torch.from_numpy(np.stack(futures_decel)).unsqueeze(0)[...,0:2]
    # constant_v = torch.from_numpy(np.stack(futures_constant_v))[...,0:2]
    # accel = torch.from_numpy(np.stack(futures_accel))[...,0:2]
    # decel = torch.from_numpy(np.stack(futures_decel))[...,0:2]

    # make combinations
    fut_rollout_combinations = get_interacting_combinations(constant_v, accel, decel, path_intersection_bool_frame)

    # # check collisions (basic distance based one, efficient)
    # collision_matrices, min_distances, _, _ = calc_collision_matrix(fut_rollout_combinations, collision_distance= 3)
    # collision_bool_sims = torch.amax(collision_matrices, dim = (1,2))
    # feasible_rollouts = fut_rollout_combinations[torch.logical_not(collision_bool_sims),...]
    #TODO: with current plan there could be collisions still, and rejected sims...

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


def get_interacting_combinations(constant_v, accel, decel, path_intersection_bool_frame):
    combinations = []

    roll_out_matrix = torch.cat([constant_v, accel, decel])
    N_sims, N_agents, _, _ = roll_out_matrix.shape

    # first do interacting rollouts / the rest constant velocity... 
    sim_constant_v = roll_out_matrix[[0],...] 

    # get interaction agents:
    agent1_idx, agent2_idx = torch.where(torch.from_numpy(path_intersection_bool_frame))
    N__interactions = len(agent1_idx)
    indices_interacting_agents = np.unique(list(agent1_idx) + list(agent2_idx))
    N_interacting_agents = len(indices_interacting_agents)

    # only sim accel/decel for interacting pairs!!!
    sim_combinations = itertools.product(range(1, N_sims), repeat=N_interacting_agents) 
    indices = torch.tensor(list(sim_combinations),  dtype=torch.int) # just interacting agents

    indices_all = torch.zeros((indices.shape[0], N_agents),  dtype=torch.int) # constant velocity as default
    indices_all[:,indices_interacting_agents] = indices

    for idx_comb in range(indices_all.shape[0]):
        sim = torch.stack(
            [roll_out_matrix[indices_all[idx_comb, idx_agent], idx_agent,...] for idx_agent in range(N_agents)])
        
        # # only check collision for interacting pairs (the rest could theoretically brake/accelerate without colliding)
        # sim_interacting_agents = sim[list(indices_interacting_agents), ...]
        # collision_matrices, min_distances, _, _ = calc_collision_matrix(sim_interacting_agents.unsqueeze(0), collision_distance= 3)
        # collision_bool_sims = torch.amax(collision_matrices, dim = (1,2))
        # if collision_bool_sims:
        #     continue
        # else:

        # assume no collisiosn for now (evaluation stops anyway once a rollout is infeasible...what about constant vel>>?)    
        combinations.append(sim)

    fut_rollout_combinations = torch.stack(combinations)
    # then check agian for collisions, and adjust rollouts accordingly ??
    
    return fut_rollout_combinations