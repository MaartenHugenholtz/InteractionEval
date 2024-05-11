import numpy as np
import torch
import itertools
from eval_utils import calc_collision_matrix, calc_travelled_distance

def get_model_prediction(data, sample_k, agent_dict, path_intersection_bool_frame, use_gt_path = True):

    frame_curr = data['frame']
    futures_constant_v = []

    for agent_id in data['valid_id']:
        agent = agent_dict[str(int(agent_id))]
        fut_agent_constant_v = agent.rollout_future(frame_curr, direction = 'constant', use_gt_path = use_gt_path)
        futures_constant_v.append(fut_agent_constant_v)
    
    constant_v = torch.from_numpy(np.stack(futures_constant_v)).unsqueeze(0)[...,0:2]

    sample_motion_3D = constant_v
    recon_motion_3D = sample_motion_3D[[0],...] # ML prediction at first index
    
    # tensor output shapes
    # recon_motion_3D.shape = (N_agent, fut_stepes, 2)
    # sample_motion_3D.shape = (sample_k, N_agent, fut_stepes, 2)

    return recon_motion_3D, sample_motion_3D  

