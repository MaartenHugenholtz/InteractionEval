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


def calc_headings(motion_tensor):
    """
    Function does not work for rollouts --> cannot determine headings for completely stationary vehicles
    Can still be used for predictions
    """
    # Calculate differences between consecutive points
    dx = torch.diff(motion_tensor[..., 0], dim=-1)  # Differences in x coordinates
    dy = torch.diff(motion_tensor[..., 1], dim=-1)  # Differences in y coordinates
    
    # Calculate headings using arctan2
    headings = torch.atan2(dy, dx)
    headings_corrected = headings.clone()
    assert len(headings.shape) == 3, 'Wrong input dimension in calc_headings'
    stationary_bool = (dx == 0) & (dy == 0)


    for i in range(headings.shape[0]):
        for j in range(headings.shape[1]):
            heading_agent = headings[i,j]
            stationary_bool_agent = stationary_bool[i,j]
            if stationary_bool_agent.any():
                idx_before_stationary = (1*stationary_bool_agent).argmax() - 1
                # assert idx_before_stationary >= 0, 'Stationary form beginning, invalid heading calculation'
                if idx_before_stationary:
                    print()
                heading_before_stationary = heading_agent[idx_before_stationary]
                headings_corrected[i,j,stationary_bool_agent] = heading_before_stationary

    # Append the last heading to match the length of trajectory
    last_headings = headings_corrected[..., -1].unsqueeze(-1)
    headings = torch.cat([headings_corrected, last_headings], dim=-1).unsqueeze(-1)
    
    return headings

def calc_collision_matrix_agentpair(motion_tensor, heading_tensor, lengths, widths, 
                                    extra_collision_threshold = 0.3, debug = False):
    """" 
    advanced collision calculation:
    1. fit circles to bumpers and calcaulate bumper positions
    2. calculate new xy positions for all radii and reshape, e.g. if 2 agents, 3 circles --> 9 agents
    3. calculate distance matrices and check for collisions: dmin > r1 + r2
    """
    num_simulations = motion_tensor.size(0) # should be one. Make 2*3
    num_agents = motion_tensor.size(1) # should be 2
    num_timesteps = motion_tensor.size(2)
    assert num_agents == 2, 'TOO MUCH AGENTS'

    # heading_tensor = calc_headings(motion_tensor) # cannot calculate for stationary vehicles, take as input instead

    disk_radii = []
    motion_tensor_bumpers = []

    partition_mode = 'inner'

    for i in range(num_agents): 
        l = lengths[i]
        w = widths[i]

        if partition_mode == 'outer': # use disks covering the whole bumper + outer parts
            r = 0.5 * np.sqrt(2*w**2)
            center_offset = 0.5*l - 0.5*w
            if 3*w < l: 
                print(f'l/w ratio, off: {l/w}. should be < 3. circle_gap: {center_offset - 2*r} (should be <0)') # assume 3 circles
        elif partition_mode == 'inner': # use disks covering only the bumper parts inside the car area
            r = 0.5*w
            center_offset = 0.5*l - 0.5*w
        else:
            raise NameError
        
        disk_radii.append(r)

        motion_i = motion_tensor[:,i,:,:].unsqueeze(0)
        heading_i = heading_tensor[:,i,:,:]
        headings_expanded = heading_i.unsqueeze(0)  # Shape: 1x2x12x1

        bumpers = torch.tensor([-center_offset, 0, center_offset])
        bumpers_broadcasted = bumpers.view(-1, 1, 1, 1)  # Shape: 3x1x1x1

        delta_x_bumpers = torch.cos(headings_expanded) * bumpers_broadcasted # shape: 3x2x12x1
        delta_y_bumpers = torch.sin(headings_expanded) * bumpers_broadcasted # shape: 3x2x12x1
        delta_pos_bumpers = torch.cat([delta_x_bumpers, delta_y_bumpers], dim =-1) # shape: 3x2x12x2
        pos_bumpers = motion_i + delta_pos_bumpers
        motion_tensor_bumpers.append(pos_bumpers)

    collision_threshold = sum(disk_radii) 

    if debug:
        for i in range(2):
            df = pd.DataFrame({
                'x': np.concatenate([motion_tensor_bumpers[0][:, i,:,0].flatten().numpy(), motion_tensor_bumpers[1][:, i,:,0].flatten().numpy()]),
                'y': np.concatenate([motion_tensor_bumpers[0][:, i,:,1].flatten().numpy(), motion_tensor_bumpers[1][:, i,:,1].flatten().numpy()]),
                'id': len(motion_tensor_bumpers[0][:, 0,:,0].flatten().numpy())*['1'] + len(motion_tensor_bumpers[1][:, 0,:,0].flatten().numpy())*['2']
            })
            px.scatter(df, x='x', y = 'y', color = 'id').show()


    motion_tensor1 = motion_tensor_bumpers[0].unsqueeze(0)  # shape: 3x2x12x2  (bumpers x sims x time x 2) ---> 1x3x2x12x2
    motion_tensor2 = motion_tensor_bumpers[1].unsqueeze(0)
    positions_diff = motion_tensor1 - motion_tensor2.permute(1, 0, 2, 3, 4)
    squared_distances = torch.sum(positions_diff ** 2, dim=-1)  # Shape: 3 x 3 x 2 x 12
    distances = torch.sqrt(squared_distances)
    assert distances.shape == (3,3, num_simulations, num_timesteps), 'WRONG SHAPES'
    min_distances = torch.amin(distances, dim=(0,1, -1))  # take minimum of bumper dimensions and time dimension
    
    collision_margins = min_distances - collision_threshold
    collision_bool = collision_margins < extra_collision_threshold
    return collision_margins, collision_bool
    


    

def calc_collision_matrix(motion_tensor, collision_distance = 3):
    """Calculates boolean matrices to show which pairs are in collision for the given rollouts"""

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

def calc_travelled_distance(motion_tensor):
    distances_steps = torch.sqrt(torch.square(motion_tensor[..., 0].diff()) + torch.square(motion_tensor[..., 1].diff()))
    distances_total = torch.sum(distances_steps, dim = -1)

    return distances_total # total travelled distance for each sim by each agent, so shape = (sims, agents)


def calc_time_based_metric(mode_vector, Hpred = 12, fps = 2):
    """ Calculate the time before the inevitable homotopy point at which the mode is precicted correctly"""
    if len(mode_vector) > Hpred:
        mode_vector = mode_vector[-Hpred:]

    frames_cmpd = len(mode_vector) # number of frames that can be used for calculation, used for relative metric! 
    time_cmpd = frames_cmpd / fps
    # make code here for cases where it is shorter! And probably best to express in %

    if mode_vector.min(): # all correct predictions
        t2cmpd = frames_cmpd / fps

    elif mode_vector.max(): # correct prediction at some point # how to handle inconsistent predictions?
        # homotopy classes also false if no interaction.  ---> should get beginning frame too in interaction_matrix! 
        idx_correct = np.argmin(mode_vector[::-1]) 
        t2cmpd = (idx_correct) / fps  

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
                    path_intersection_bool = homotopy_scene_tensor[5, :,i, j]
                    both_inframe_bool = homotopy_scene_tensor[6, :,i, j].to(torch.bool)
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


                    if len(path_intersection_bool[both_inframe_bool]) > 2 and path_intersection_bool[both_inframe_bool].argmax() > 0 and not homotopy_class_final.min(): # interaction_bool should look for entire future?? yes, only 1 interaction needed. 
                        # calculate mode metrics here! 

                        h_final_frame = homotopy_class_final.argmax().item()
                        # if h_final_frame == 0: # 
                        #     h_notfinal_frame = homotopy_class_final.argmin().item()
                        #     h_final_frame = h_notfinal_frame + homotopy_class_final[h_notfinal_frame:].argmax().item()
                        h_final_frame_scene = (not_nan_idx*1).argmax().item() + h_final_frame  # for stats, taking into account nan values
                        mode_correct_predictions = modes_correct_matrix[:h_final_frame].numpy()
                        mode_covered_predictions = modes_covered_matrix[:h_final_frame].numpy()
                        gt_mode = homotopy_gt[h_final_frame].item()

                        # iumrpove h_final handling

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


def calc_path_homotopy(motion_tensor, agents_scene, threshold_distance = 1, path_crossing_threshold = 1):
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
                    # homotopy_class = 0 if min_distance > path_crossing_threshold else ( 0 if idx1 == idx2 else (1 if idx1 < idx2 else 2)) # 0 == undetermined/not crossing, 1==agent1 first, 2==agent2 first

                    # if homotopy_class == 0:
                    #     print()
                        # new problem: stationary (end of time trajectories)
                    homotopy_classes[s, i, j] = homotopy_class
    
    return homotopy_classes

def calc_intersections(motion_tensor, interp_factor = 100, threshold_distance = 0.5) :
    motion_tensor = tensor_interpolate(motion_tensor, size = (motion_tensor.shape[2]*interp_factor, 2))
    assert(True, 'wrong use function above')
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


def calc_path_intersections(df_scene, agents_scene, pred_frames, interp_factor = 10, onpath_threshold = 1.5, 
                            interaction_threshold = 20,
                            start_path_sharing_frame_difference_theshold = 12,
                            use_distance_criterion = False) :
    df_scene = df_scene[(df_scene.frame >= pred_frames.min())*(df_scene.frame <= pred_frames.max())]
    num_agents = len(agents_scene)
    num_frames = len(pred_frames)
    motion_tensor = torch.full((num_frames, num_agents, 2), float('nan')) # frames x agents x [x,y]
    for i, agent in enumerate(agents_scene):
        df_agent = df_scene[df_scene.agent_id == agent]
        idx0 = list(pred_frames).index(df_agent.frame.values[0])
        idx1 = list(pred_frames).index(df_agent.frame.values[-1])
        try:
            motion_tensor[idx0:idx1+1, i,:] = torch.tensor(df_agent[['x', 'y']].values)
        except:
            print('Missing frame for agent detected')


    motion_tensor = motion_tensor.permute(1,0,2).unsqueeze(0)
    # motion_tensor_interpx = tensor_interpolate(motion_tensor[:,:,:,0], scale_factor = interp_factor, mode = 'linear', align_corners=True)
    # motion_tensor_interpy = tensor_interpolate(motion_tensor[:,:,:,1], scale_factor = interp_factor, mode = 'linear', align_corners=True)
    # motion_tensor_interp = torch.stack((motion_tensor_interpx, motion_tensor_interpy)).permute(1,2,3,0)

    path_intersection_bool = np.zeros((num_agents, num_agents)) * False
    inframes_bool = np.zeros((num_frames, num_agents, num_agents)) # both agents in frame

    # need interpolation here! FIX PROPER INTERPOLATION!!!!! This should resolve proper on_path bools!!! #TODO use np
    pred_frames_interp = np.arange(pred_frames[0], pred_frames[-1]+1/interp_factor, 1/interp_factor)
    
    df_modes = pd.DataFrame(columns = ['agent1', 'agent2', 'total_num_agents', 'common_start_frame', 'common_end_frame'])

    for i in range(num_agents):
        for j in range(num_agents):
            if j > i:  # only fill upper triangular part; calcaulate each pair once
                agent1_id = agents_scene[i]
                agent2_id = agents_scene[j]
                # interpolate paths + extrpolate outside of range 
                # agent1_interp_path = 
                agent1 = torch.tensor(
                        np.stack([np.interp(pred_frames_interp, pred_frames, motion_tensor[0,i,:,0]),
                                  np.interp(pred_frames_interp, pred_frames, motion_tensor[0,i,:,1])],).T
                )
                agent2 = torch.tensor(
                        np.stack([np.interp(pred_frames_interp, pred_frames, motion_tensor[0,j,:,0]),
                                  np.interp(pred_frames_interp, pred_frames, motion_tensor[0,j,:,1])],).T
                )
                agent1  = agent1.unsqueeze(1) # shape = (time x 1 x 2)
                agent2  = agent2.unsqueeze(1).permute(1, 0, 2) # shape = (1 x time x 2)
                positions_diff = agent1 - agent2
                squared_distances = torch.sum(positions_diff ** 2, dim=-1)  # Shape: (num_simulations, num_agents, num_agents, timesteps)
                distances = torch.sqrt(squared_distances).numpy()  # Shape: (num_simulations, num_agents, num_agents, timesteps)
                distances = np.nan_to_num(distances, nan = 9999)
                
                inframes = distances.diagonal()[::interp_factor] < 9999
                agent1_onpath = (distances.min(axis=1) < onpath_threshold)[::interp_factor]
                # agent1_onpath_idx = np.argmin(distances[::interp_factor,::interp_factor], axis=1)
                # agent1_inframes_bool = np.array([abs(i-agent1_onpath_idx[i]) for i in range(len(agent1_onpath_idx))]) <= 12

                agent2_onpath = (distances.min(axis=0) < onpath_threshold)[::interp_factor]
                # agent2_onpath_idx = np.argmin(distances[::interp_factor,::interp_factor], axis=0)
                # agent2_inframes_bool = np.array([abs(i-agent2_onpath_idx[i]) for i in range(len(agent2_onpath_idx))]) <= 12
                # path_intersection_bool[:,i, j] = agent1_onpath[::interp_factor] # agent i on shared path (i,j)
                # path_intersection_bool[:,j, i] = agent2_onpath[::interp_factor] # agent j on shared path (i,j)
                # onpath_frames = np.maximum(agent1_inframes_bool*agent1_onpath, agent2_inframes_bool*agent2_onpath)[inframes]
                onpath_frames = np.maximum(agent1_onpath, agent2_onpath)[inframes]
                real_time_closest_distance = distances.diagonal().min()
                start_path_sharing_frame_difference = abs(agent2_onpath.argmax() - agent1_onpath.argmax())
                
                # take maximum, to get timestep path sharing boolean. Problem maximum: very big time horizon differences.. Solution: Real time difference bool.
                # assert(not (agent1_id=='10')*(agent2_id=='5'))
                if use_distance_criterion:
                    interaction = (real_time_closest_distance<=interaction_threshold)*(start_path_sharing_frame_difference<=start_path_sharing_frame_difference_theshold)*onpath_frames #*agent1_onpath.any()*agent2_onpath.any()
                else:
                    interaction = (start_path_sharing_frame_difference<=start_path_sharing_frame_difference_theshold)*onpath_frames #*agent1_onpath.any()*agent2_onpath.any()


                pathcrossing_interaction = len(interaction) > 2 and interaction.argmax() > 0
                if pathcrossing_interaction:
                    path_intersection_bool[i,j] = True
                    common_start_frame = pred_frames[inframes].min()
                    common_end_frame = pred_frames[inframes].max()
                    df_modes.loc[len(df_modes.index)] = [agent1_id, agent2_id, num_agents, common_start_frame, common_end_frame]

                


    # return motion tensor shape with pred_frames. check for each overlapping frame, if it is a common waypoint@
    return path_intersection_bool, inframes_bool, df_modes

