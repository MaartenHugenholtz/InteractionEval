import numpy as np
import torch
import itertools
from eval_utils import calc_collision_matrix, calc_travelled_distance
from agent_class import Agent

def get_model_prediction(data, sample_k, agent_dict = None, path_intersection_bool_frame = None, use_gt_path = False,
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

