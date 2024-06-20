"""
Code borrowed from Trajectron++: https://github.com/StanfordASL/Trajectron-plus-plus/blob/ef0165a93ee5ba8cdc14f9b999b3e00070cd8588/trajectron/environment/map.py
"""

import torch
import numpy as np
import cv2
import os
from .homography_warper import get_rotation_matrix2d, warp_affine_crop
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils.homotopy import *
from plotly.subplots import make_subplots

class Map(object):
    def __init__(self, data, homography, description=None):
        self.data = data
        self.homography = homography
        self.description = description

    def as_image(self):
        raise NotImplementedError

    def get_cropped_maps(self, world_pts, patch_size, rotation=None, device='cpu'):
        raise NotImplementedError

    def to_map_points(self, scene_pts):
        raise NotImplementedError


class GeometricMap(Map):
    """
    A Geometric Map is a int tensor of shape [layers, x, y]. The homography must transform a point in scene
    coordinates to the respective point in map coordinates.

    :param data: Numpy array of shape [layers, x, y]
    :param homography: Numpy array of shape [3, 3]
    """
    def __init__(self, data, homography, origin=None, description=None):
        #assert isinstance(data.dtype, np.floating), "Geometric Maps must be float values."
        super(GeometricMap, self).__init__(data, homography, description=description)

        if origin is None:
            self.origin = np.zeros(2)
        else:
            self.origin = origin
        self._last_padding = None
        self._last_padded_map = None
        self._torch_map = None

    def torch_map(self, device):
        if self._torch_map is not None:
            return self._torch_map
        self._torch_map = torch.tensor(self.data, dtype=torch.uint8, device=device)
        return self._torch_map

    def as_image(self):
        # We have to transpose x and y to rows and columns. Assumes origin is lower left for image
        # Also we move the channels to the last dimension
        return (np.transpose(self.data, (2, 1, 0))).astype(np.uint)

    def get_padded_map(self, padding_x, padding_y, device):
        if self._last_padding == (padding_x, padding_y):
            return self._last_padded_map
        else:
            self._last_padding = (padding_x, padding_y)
            self._last_padded_map = torch.full((self.data.shape[0],
                                                self.data.shape[1] + 2 * padding_x,
                                                self.data.shape[2] + 2 * padding_y),
                                               False, dtype=torch.uint8)
            self._last_padded_map[..., padding_x:-padding_x, padding_y:-padding_y] = self.torch_map(device)
            return self._last_padded_map

    @staticmethod
    def batch_rotate(map_batched, centers, angles, out_height, out_width):
        """
        As the input is a map and the warp_affine works on an image coordinate system we would have to
        flip the y axis updown, negate the angles, and flip it back after transformation.
        This, however, is the same as not flipping at and not negating the radian.

        :param map_batched:
        :param centers:
        :param angles:
        :param out_height:
        :param out_width:
        :return:
        """
        M = get_rotation_matrix2d(centers, angles, torch.ones_like(angles))
        rotated_map_batched = warp_affine_crop(map_batched, centers, M,
                                               dsize=(out_height, out_width), padding_mode='zeros')

        return rotated_map_batched

    @classmethod
    def get_cropped_maps_from_scene_map_batch(cls, maps, scene_pts, patch_size, rotation=None, device='cpu'):
        """
        Returns rotated patches of each map around the transformed scene points.
        ___________________
        |       |          |
        |       |ps[3]     |
        |       |          |
        |       |          |
        |      o|__________|
        |       |    ps[2] |
        |       |          |
        |_______|__________|
        ps = patch_size

        :param maps: List of GeometricMap objects [bs]
        :param scene_pts: Scene points: [bs, 2]
        :param patch_size: Extracted Patch size after rotation: [-x, -y, +x, +y]
        :param rotation: Rotations in degrees: [bs]
        :param device: Device on which the rotated tensors should be returned.
        :return: Rotated and cropped tensor patches.
        """
        batch_size = scene_pts.shape[0]
        lat_size = 2 * np.max((patch_size[0], patch_size[2]))
        long_size = 2 * np.max((patch_size[1], patch_size[3]))
        assert lat_size % 2 == 0, "Patch width must be divisible by 2"
        assert long_size % 2 == 0, "Patch length must be divisible by 2"
        lat_size_half = lat_size // 2
        long_size_half = long_size // 2

        context_padding_x = int(np.ceil(np.sqrt(2) * long_size))
        context_padding_y = int(np.ceil(np.sqrt(2) * long_size))

        centers = torch.tensor([s_map.to_map_points(scene_pts[np.newaxis, i]) for i, s_map in enumerate(maps)],
                               dtype=torch.long, device=device).squeeze(dim=1) \
                  + torch.tensor([context_padding_x, context_padding_y], device=device, dtype=torch.long)

        padded_map = [s_map.get_padded_map(context_padding_x, context_padding_y, device=device) for s_map in maps]

        padded_map_batched = torch.stack([padded_map[i][...,
                                          centers[i, 0] - context_padding_x: centers[i, 0] + context_padding_x,
                                          centers[i, 1] - context_padding_y: centers[i, 1] + context_padding_y]
                                          for i in range(centers.shape[0])], dim=0)

        center_patches = torch.tensor([[context_padding_y, context_padding_x]],
                                      dtype=torch.int,
                                      device=device).repeat(batch_size, 1)

        if rotation is not None:
            angles = torch.Tensor(rotation)
        else:
            angles = torch.zeros(batch_size)

        rotated_map_batched = cls.batch_rotate(padded_map_batched/255.,
                                                center_patches.float(),
                                                angles,
                                                long_size,
                                                lat_size)

        del padded_map_batched

        return rotated_map_batched[...,
               long_size_half - patch_size[1]:(long_size_half + patch_size[3]),
               lat_size_half - patch_size[0]:(lat_size_half + patch_size[2])]

    def get_cropped_maps(self, scene_pts, patch_size, rotation=None, device='cpu'):
        """
        Returns rotated patches of the map around the transformed scene points.
        ___________________
        |       |          |
        |       |ps[3]     |
        |       |          |
        |       |          |
        |      o|__________|
        |       |    ps[2] |
        |       |          |
        |_______|__________|
        ps = patch_size

        :param scene_pts: Scene points: [bs, 2]
        :param patch_size: Extracted Patch size after rotation: [-lat, -long, +lat, +long]
        :param rotation: Rotations in degrees: [bs]
        :param device: Device on which the rotated tensors should be returned.
        :return: Rotated and cropped tensor patches.
        """
        return self.get_cropped_maps_from_scene_map_batch([self]*scene_pts.shape[0], scene_pts,
                                                          patch_size, rotation=rotation, device=device)

    def to_map_points(self, scene_pts):
        org_shape = None
        if len(scene_pts.shape) != 2:
            org_shape = scene_pts.shape
            scene_pts = scene_pts.reshape((-1, 2))
        scene_pts = scene_pts - self.origin[None, :]
        N, dims = scene_pts.shape
        points_with_one = np.ones((dims + 1, N))
        points_with_one[:dims] = scene_pts.T
        map_points = (self.homography @ points_with_one).T[..., :dims]
        if org_shape is not None:
            map_points = map_points.reshape(org_shape)
        return map_points


    def visualize_data(self, data):
        pre_motion = np.stack(data['pre_motion_3D']) * data['traj_scale']
        fut_motion = np.stack(data['fut_motion_3D']) * data['traj_scale']
        heading = data['heading']
        img = np.transpose(self.data, (1, 2, 0))
        for i in range(pre_motion.shape[0]):
            cur_pos = pre_motion[i, -1]
            # draw agent
            cur_pos = np.round(self.to_map_points(cur_pos)).astype(int)
            img = cv2.circle(img, (cur_pos[1], cur_pos[0]), 3, (0, 255, 0), -1)
            prev_pos = cur_pos
            # draw fut traj
            for t in range(fut_motion.shape[0]):
                pos = fut_motion[i, t]
                pos = np.round(self.to_map_points(pos)).astype(int)
                img = cv2.line(img, (prev_pos[1], prev_pos[0]), (pos[1], pos[0]), (0, 255, 0), 2) 

            # draw heading
            theta = heading[i]
            v= np.array([5.0, 0.0])
            v_new = v.copy()
            v_new[0] = v[0] * np.cos(theta) - v[1] * np.sin(theta)
            v_new[1] = v[0] * np.sin(theta) + v[1] * np.cos(theta)
            vend = pre_motion[i, -1] + v_new
            vend = np.round(self.to_map_points(vend)).astype(int)
            img = cv2.line(img, (cur_pos[1], cur_pos[0]), (vend[1], vend[0]), (0, 255, 255), 2) 

        fname = f'out/agent_maps/{data["seq"]}_{data["frame"]}_vis.png'
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        cv2.imwrite(fname, img)

    def calc_mode_metrics(self, homotopy_gt, homotopy_pred, verbose = True):
        """
        calculate mode matrices
        """
        modes_correct_matrix = (homotopy_gt==homotopy_pred[0,:,:]) # first entry corresponds to ML mode.
        k_sample = homotopy_pred.shape[0]
        modes_covered_matrix = (homotopy_gt.repeat(k_sample, 1, 1)==homotopy_pred[:,:,:]).max(axis=0).values

        modes_correct = modes_correct_matrix.all().item()
        modes_covered = modes_covered_matrix.all().item()


        if verbose:
            print(f'modes correct: {modes_correct}. modes covered: {modes_covered}.')
            print('mode correct matrix:')
            print(modes_correct_matrix)
            print('mode covered matrix:')
            print(modes_covered_matrix)

        return modes_correct, modes_covered

    def rotate_car(self, xc, yc, l, w , heading):
        rot_matrix = np.array([[np.cos(heading), -np.sin(heading)],
                               [np.sin(heading), np.cos(heading)]])
        fl_corner = rot_matrix @ np.array([0.5*l, 0.5*w])
        fr_corner = rot_matrix @ np.array([0.5*l, -0.5*w])
        rl_corner = rot_matrix @ np.array([-0.5*l, 0.5*w])
        rr_corner = rot_matrix @ np.array([-0.5*l, -0.5*w]) 

        x_points = [fl_corner[0], fr_corner[0], rr_corner[0], rl_corner[0], fl_corner[0]] + xc
        y_points = [fl_corner[1], fr_corner[1], rr_corner[1], rl_corner[1], fl_corner[1]] + yc
        return x_points, y_points
    
    def visualize_pair_gt_scene(self, df_scene, agent_pair):
        """
        Plots GT trajectories (full)
        And predictions for all agents (grouped per scene prediction)
        """
        str_list_pair = [str(agent_pair[0]),str(agent_pair[1])]
        df_pair = df_scene[df_scene.agent_id.isin(str_list_pair)]
        df1 = df_scene[df_scene['agent_id'] == str(int(agent_pair[0]))]
        df2 = df_scene[df_scene['agent_id'] == str(int(agent_pair[1]))]
        common_start_frame = max(df1.frame.min(), df2.frame.min())
        common_end_frame = min(df1.frame.max(), df2.frame.max())
        # filter on common frames:
        df1 = df1[(df1.frame >= common_start_frame)*(df1.frame <= common_end_frame)]
        df2 = df2[(df2.frame >= common_start_frame)*(df2.frame <= common_end_frame)]

        # get common frames and put in similar format


        all_motion = np.stack([df1[['x', 'y']].values, df2[['x', 'y']].values]) # shape: 2 x T x 2 ?
        motion_map = self.to_map_points(all_motion)

        margin = 20
        # x and y indices reveresd because of motion mapping (same for plotting indices)
        all_x = motion_map[:,:,1].flatten()
        all_y = motion_map[:,:,0].flatten()

        # index of current timestamp:
        idx_cur = 0
        agent_ids = str_list_pair
        agent_headings = [df1.heading.values[0], df2.heading.values[0]]
        agent_lengths =[df1.length.values[0], df2.length.values[0]]
        agent_widths = [df1.width.values[0], df2.width.values[0]]

        img = np.transpose(self.data, (1, 2, 0))  

        # Create a Plotly figure
        fig = go.Figure()

        fig = px.imshow(img)

        # colors agents 

        colors = px.colors.qualitative.Plotly + px.colors.qualitative.Alphabet +  px.colors.qualitative.Dark24

        # Plot the GT trajectories and predictions
        for agent_idx, agent_id in enumerate(agent_ids):
            fig.add_trace(go.Scatter(
                x=motion_map[agent_idx,:,1],
                y=motion_map[agent_idx,:,0],  # x and y reversed for image
                mode='lines+markers',
                name = f'gt_agent_{agent_id}',
                # showlegend= (agent_idx==0),
                legendgroup='gt',
                legendgrouptitle_text='gt',
                line=dict(color=colors[agent_idx])
            ))

            # add vehicle shapes:
            x_points, y_points = self.rotate_car(xc = motion_map[agent_idx,idx_cur,0], 
                                             yc = motion_map[agent_idx,idx_cur,1], 
                                             l = agent_lengths[agent_idx]*3, # size still needs to be scaled
                                             w = agent_widths[agent_idx]*3, # size still needs to be scaled
                                             heading = agent_headings[agent_idx])
            fig.add_trace(
                go.Scatter(x=y_points, y=x_points, 
                           fill="toself",
                           mode = 'lines',
                            legendgroup='gt',
                            name = f'gt_agent_{agent_id}',
                            showlegend=False,
                            line=dict(color=colors[agent_idx]),
                            ))



        # Update layout to remove axes
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            # title=dict(text = data['seq'] + ', frame-' + str(data['frame'])),
        )

        # Calculate the range for both axes
        x_range = all_x.max() - all_x.min()
        y_range = all_y.max() - all_y.min()

        # Determine the maximum range
        max_range = max(x_range, y_range)
        extra_margin_x = (max_range - x_range)/2
        extra_margin_y = (max_range - y_range)/2

        # Update the axes with the new ranges
        fig.update_xaxes(range=[all_x.min() - margin - extra_margin_x, all_x.max() + margin + extra_margin_x], visible = False, scaleanchor="y", scaleratio=1)
        fig.update_yaxes(range=[all_y.max() + margin + extra_margin_y, all_y.min() - margin - extra_margin_y], visible = False, scaleanchor="x", scaleratio=1) #
        fig.update_layout(showlegend=False)

        # Set margins to zero
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        # Show the Plotly figure
        # fig.show()
        return fig

    def visualize_trajs(self, data, prediction, show_map = True, show_hist= True, show_fut = True, show_pred = True, show_fig = True):
        """
        Plots GT trajectories (full)
        And predictions for all agents (grouped per scene prediction)
        """
        pre_motion = np.stack(data['pre_motion_3D']) * data['traj_scale']
        fut_motion = np.stack(data['fut_motion_3D']) * data['traj_scale']
        all_motion = np.concatenate((pre_motion, fut_motion), axis=1)
        motion_map = self.to_map_points(all_motion)
        motion_pre = self.to_map_points(pre_motion)
        H_steps = pre_motion.shape[1]
        motion_fut = self.to_map_points(all_motion[:,H_steps-1:,:])
        curr_t = torch.from_numpy(pre_motion[:,[-1],:]).unsqueeze(0).repeat(prediction.shape[0], 1, 1, 1)
        prediction_t = torch.cat((curr_t, prediction), 2)
        pred_map =self.to_map_points(prediction_t)

        # index of current timestamp:
        idx_cur = pre_motion.shape[1] - 1

        fut_motion_batch = torch.from_numpy(fut_motion).unsqueeze(0)
        angle_diff_gt, homotopy_gt = identify_pairwise_homotopy(fut_motion_batch)
        angle_diff_pred, homotopy_pred = identify_pairwise_homotopy(prediction)
        modes_correct, modes_covered = self.calc_mode_metrics(homotopy_gt, homotopy_pred)


        agent_ids = data['valid_id']
        agent_headings = data['heading']
        agent_lengths = data['pre_data'][0][:,12] # idx = 12, and first timestep of predata
        agent_widths = data['pre_data'][0][:,10]   # idx = 10

        img = np.transpose(self.data, (1, 2, 0))  

        # Create a Plotly figure
        fig = go.Figure()

        if show_map:
            fig = px.imshow(img)
        else:
            img_blank = img.copy()
            img_blank.fill(255)
            fig = px.imshow(img_blank)


        # colors agents 

        colors = px.colors.qualitative.Plotly + px.colors.qualitative.Alphabet +  px.colors.qualitative.Dark24

        # Plot the GT trajectories and predictions
        for agent_idx, agent_id in enumerate(agent_ids):
            fig.add_trace(go.Scatter(
                x=motion_pre[agent_idx,:,1],
                y=motion_pre[agent_idx,:,0],  # x and y reversed for image
                mode='lines+markers',
                name = f'gt_agent_{agent_id}',
                # showlegend= (agent_idx==0),
                legendgroup='gt_pre',
                legendgrouptitle_text='gt_pre',
                visible =  True if show_hist else 'legendonly',
                line=dict(color=colors[agent_idx],
                )
            ))

            fig.add_trace(go.Scatter(
                x=motion_fut[agent_idx,:,1],
                y=motion_fut[agent_idx,:,0],  # x and y reversed for image
                mode='lines+markers',
                name = f'gt_agent_{agent_id}',
                # showlegend= (agent_idx==0),
                legendgroup='gt_fut',
                legendgrouptitle_text='gt_fut',
                visible =True if show_fut else 'legendonly',
                line=dict(color=colors[agent_idx])
            ))

            # add vehicle shapes:
            x_points, y_points = self.rotate_car(xc = motion_map[agent_idx,idx_cur,0], 
                                             yc = motion_map[agent_idx,idx_cur,1], 
                                             l = agent_lengths[agent_idx]*3, # size still needs to be scaled
                                             w = agent_widths[agent_idx]*3, # size still needs to be scaled
                                             heading = agent_headings[agent_idx])
            fig.add_trace(
                go.Scatter(x=y_points, y=x_points, 
                           fill="toself",
                           mode = 'lines',
                            legendgroup='vehicle',
                            legendgrouptitle_text='vehicle',
                            name = f'gt_agent_{agent_id}',
                            showlegend=True,
                            line=dict(color=colors[agent_idx]),
                            ))
            

            for pred in range(pred_map.shape[0]):
                fig.add_trace(go.Scatter(
                    x=pred_map[pred,agent_idx,:,1],
                    y=pred_map[pred,agent_idx,:,0],  # x and y reversed for image
                    mode='lines + markers',
                    name = f'pred{pred+1}_agent{agent_id}',
                    # showlegend= (agent_idx==0),
                    legendgroup=f'pred{pred+1}',
                    legendgrouptitle_text=f'pred{pred+1}',
                    opacity=0.3,
                    visible = True if show_pred else 'legendonly',
                    line=dict(color=colors[agent_idx], dash='dash')
                    ))


        margin = 10
        # x and y indices reveresd because of motion mapping (same for plotting indices)
        all_x = np.concatenate([motion_map[...,1].flatten(),pred_map[...,1].flatten()])
        all_y = np.concatenate([motion_map[...,0].flatten(),pred_map[...,0].flatten()])
        # Update layout to remove axes
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            # title=dict(text = data['seq'] + ', frame-' + str(data['frame'])),
        )
        fig.update_xaxes(range=[all_x.min() - margin, all_x.max() + margin], visible = False, scaleanchor="y", scaleratio=1)
        fig.update_yaxes(range=[all_y.max() + margin, all_y.min() - margin], visible = False, scaleanchor="x", scaleratio=1) #


        # Show the Plotly figure
        if show_fig:
            fig.show()
        return fig

    def calc_pathhomotopy_pair(self, motion_agent1, motion_agent2):
        if isinstance(motion_agent1, np.ndarray):
            motion_agent1 = torch.tensor(motion_agent1)
            motion_agent2 = torch.tensor(motion_agent2)

        agent1  = motion_agent1.unsqueeze(1) # shape = (time x 1 x 2)
        agent2  = motion_agent2.unsqueeze(1).permute(1, 0, 2) # shape = (1 x time x 2)
        positions_diff = agent1 - agent2
        squared_distances = torch.sum(positions_diff ** 2, dim=-1)  # Shape: (num_simulations, num_agents, num_agents, timesteps)
        distances = torch.sqrt(squared_distances).numpy()  # Shape: (num_simulations, num_agents, num_agents, timesteps)
        min_distance = distances.min()
        indices = np.where(distances == min_distance)
        idx1, idx2 = indices[0][-1], indices[1][-1] # if there are multiple indices, take the last one (in case of stationary vehicles)
        homotopy_class = 0 if idx1 == idx2 else (1 if idx1 < idx2 else 2) 
        return homotopy_class

    def visualize_interactionpair(self, data, prediction, rollout, rollout_collisions, agent_pair):
        """
        Plots GT trajectories (full)
        And predictions for all agents (grouped per scene prediction)
        """
        agent1_id, agent2_id  = agent_pair
        agent1_idx = data['valid_id'].index(agent1_id)
        agent2_idx = data['valid_id'].index(agent2_id)

        pre_motion = np.stack(data['pre_motion_3D']) * data['traj_scale']
        fut_motion = np.stack(data['fut_motion_3D']) * data['traj_scale']
        all_motion = np.concatenate((pre_motion, fut_motion), axis=1)

        # index agent_pair
        # all_motion = all_motion[[agent1_idx,agent2_idx]]
        # prediction = prediction[:,[agent1_idx,agent2_idx]]
        # rollout = rollout # rollout already indexed for [agent1_idx,agent2_idx]
        # # calculate homotopy classes 
        # gt_class = self.calc_pathhomotopy_pair(all_motion[0,...], all_motion[1,...])
        # pred_classes = [self.calc_pathhomotopy_pair(prediction[pred, 0,...], prediction[pred, 1,...]) for pred in range(prediction.shape[0])]
        # rollout_classes = [self.calc_pathhomotopy_pair(rollout[r, 0,...], rollout[r, 1,...]) for r in range(rollout.shape[0])]

        # get minimum fut mask to make sure homotopy classes stay valid and fair timestep comparisons can be made
        pre_motion_mask_valid_pair = torch.minimum(data['pre_motion_mask'][agent1_idx], data['pre_motion_mask'][agent2_idx]).bool()
        fut_motion_mask_valid_pair = torch.minimum(data['fut_motion_mask'][agent1_idx], data['fut_motion_mask'][agent2_idx]).bool()
        all_motion_mask_valid_pair = torch.cat([pre_motion_mask_valid_pair, fut_motion_mask_valid_pair])
        Npred_frames = sum(fut_motion_mask_valid_pair).item()

        # focus on pairs only; only use valid data masks
        all_motion_pair = all_motion[[agent1_idx,agent2_idx]][:,all_motion_mask_valid_pair,:]  # for plotting
        fut_motion_pair = fut_motion[[agent1_idx,agent2_idx]][:,fut_motion_mask_valid_pair,:] 
        prediciton_pair = prediction[:,[agent1_idx,agent2_idx]][:,:,fut_motion_mask_valid_pair,:] 
        rollout_pair = rollout[:,:,fut_motion_mask_valid_pair,:]  # already only focus agent pair, just index valid timesteps


        # calc homotopy classes with free-end angles
        fut_motion_pair_batch = torch.from_numpy(fut_motion_pair).unsqueeze(0)
        angle_diff_gt, homotopy_gt = identify_pairwise_homotopy(fut_motion_pair_batch)  
        angle_diff_pred, homotopy_pred = identify_pairwise_homotopy(prediciton_pair)
        angle_diff_rollout, homotopy_rollout = identify_pairwise_homotopy(rollout_pair)

        homotopy_dict = {0: 'STATIC',
                         1: 'CW',
                         2: 'CCW',}

        gt_class = [homotopy_dict[homotopy_gt[:,0,1].item()]]
        pred_classes = [homotopy_dict[c.item()] for c in homotopy_pred[:,0,1]]
        rollout_classes = [homotopy_dict[c.item()] for c in homotopy_rollout[:,0,1]]

        # check rollout collisions:
        rollout_collisions_bool = rollout_collisions.tolist()
        rollout_feasible_bool = np.logical_not(rollout_collisions_bool)

        # do some checks
        # assert not min(rollout_collisions_bool), 'no roll-outs possible without collisiosn' # at least one class possible without collisions
        # if min(rollout_feasible_bool): # if both rollouts are feasible
        #     assert ('CW' in rollout_classes)*('CCW' in rollout_classes), 'roll-outs are same class' # both classes should be covered with rollouts 
        
        if sum(rollout_feasible_bool) == 0:
            print()
        feasible_rollout_classes = np.unique(np.array(rollout_classes)[rollout_feasible_bool])
        N_feasible_rollouts = len(feasible_rollout_classes)
        N_modes_covered = len(np.unique(pred_classes))
        h_final = N_feasible_rollouts < 2

        # calculate relevant mode dict:
        mode_dict = {
            'frame': data['frame'],
            'gt_mode': gt_class[0],
            'ml_mode': pred_classes[0],
            'K_modes': np.unique(pred_classes),
            'sim_modes': list(feasible_rollout_classes),
            'mode_correct': pred_classes[0] == gt_class[0], # ML == gt
            'mode_covered': max([pred_class == gt_class[0] for pred_class in pred_classes]), # any pred == gt
            'mode_collapse': N_modes_covered < N_feasible_rollouts,
            'N_modes_covered': N_modes_covered,
            'N_feasible_rollouts': N_feasible_rollouts,
            'h_final': h_final,
            'Npred_frames': Npred_frames, 
        }
        
        # transform points to map 
        motion_map = self.to_map_points(all_motion_pair)
        pred_map = self.to_map_points(prediciton_pair)
        rollout_map = self.to_map_points(rollout_pair)

        margin = 5
        # x and y indices reveresd because of motion mapping (same for plotting indices)
        all_x = np.concatenate([motion_map[...,1].flatten(),pred_map[...,1].flatten(), rollout_map[...,1].flatten()])
        all_y = np.concatenate([motion_map[...,0].flatten(),pred_map[...,0].flatten(), rollout_map[...,0].flatten()])

        # index of current timestamp:
        idx_cur = pre_motion.shape[1] - 1

        agent_ids = [data['valid_id'][agent1_idx], data['valid_id'][agent2_idx]]
        agent_headings = [data['heading'][agent1_idx], data['heading'][agent2_idx]]
        agent_lengths = data['pre_data'][0][[agent1_idx,agent2_idx], :][:,12] # idx = 12, and first timestep of predata
        agent_widths = data['pre_data'][0][[agent1_idx,agent2_idx], :][:,10]   # idx = 10

        img = np.transpose(self.data, (1, 2, 0))  

        # Create a Plotly figure
        fig = go.Figure()

        fig = px.imshow(img)

        # colors agents 

        colors = px.colors.qualitative.Plotly + px.colors.qualitative.Alphabet +  px.colors.qualitative.Dark24

        # Plot the GT trajectories and predictions
        for agent_idx, agent_id in enumerate(agent_ids):
            fig.add_trace(go.Scatter(
                x=motion_map[agent_idx,:,1],
                y=motion_map[agent_idx,:,0],  # x and y reversed for image
                mode='lines+markers',
                name = f'gt_agent_{agent_id}',
                # showlegend= (agent_idx==0),
                legendgroup='gt',
                legendgrouptitle_text=f'gt, h_class: {gt_class[0]}',
                line=dict(color=colors[agent_idx])
            ))

            # add vehicle shapes:
            x_points, y_points = self.rotate_car(xc = motion_map[agent_idx,idx_cur,0], 
                                             yc = motion_map[agent_idx,idx_cur,1], 
                                             l = agent_lengths[agent_idx]*3, # size still needs to be scaled
                                             w = agent_widths[agent_idx]*3, # size still needs to be scaled
                                             heading = agent_headings[agent_idx])
            fig.add_trace(
                go.Scatter(x=y_points, y=x_points, 
                           fill="toself",
                           mode = 'lines',
                            legendgroup='gt',
                            name = f'gt_agent_{agent_id}',
                            showlegend=False,
                            line=dict(color=colors[agent_idx]),
                            ))
            

            for pred in range(pred_map.shape[0]):
                fig.add_trace(go.Scatter(
                    x=pred_map[pred,agent_idx,:,1],
                    y=pred_map[pred,agent_idx,:,0],  # x and y reversed for image
                    mode='lines + markers',
                    name = f'pred{pred+1}_agent{agent_id}',
                    # showlegend= (agent_idx==0),
                    legendgroup=f'pred{pred+1}',
                    legendgrouptitle_text=f'pred{pred+1}, h_class: {pred_classes[pred]}',
                    opacity=0.5,
                    line=dict(color=colors[agent_idx], dash='dash'),
                    visible = True if pred == 0 else 'legendonly'
                    ))
                
            rollout_symbols = ['square', 'x']
            r_markersize = 6
            for r in range(rollout_map.shape[0]):
                fig.add_trace(go.Scatter(
                    x=rollout_map[r,agent_idx,:,1],
                    y=rollout_map[r,agent_idx,:,0],  # x and y reversed for image
                    mode='lines + markers',
                    name = f'rollout{r+1}_agent{agent_id}',
                    marker_symbol= rollout_symbols[r],
                    marker_size = r_markersize,
                    # showlegend= (agent_idx==0),
                    legendgroup=f'rollout{r+1}',
                    legendgrouptitle_text=f'rollout{r+1}, h_class: {rollout_classes[r]}, collision: {rollout_collisions_bool[r]}',
                    opacity=0.5,
                    line=dict(color=colors[agent_idx], dash='dot')
                    ))


        # Update layout to remove axes
        fig.update_layout(
            title=dict(text = data['seq'] + ', frame-' + str(data['frame']) + '-' + str(Npred_frames + data['frame'])),
        )
        fig.update_xaxes(range=[all_x.min() - margin, all_x.max() + margin], visible = False, scaleanchor="y", scaleratio=1)
        fig.update_yaxes(range=[all_y.max() + margin, all_y.min() - margin], visible = False, scaleanchor="x", scaleratio=1) #

        # # Show the Plotly figure
        # fig.show()
        # print()
        return fig, mode_dict


    def visualize_interactionpair_splitplot(self, data, prediction, rollout, rollout_collisions, agent_pair,
                                            plot_square = True):
        """
        Plots GT trajectories (full)
        And predictions for all agents (grouped per scene prediction)
        """
        agent1_id, agent2_id  = agent_pair
        agent1_idx = data['valid_id'].index(agent1_id)
        agent2_idx = data['valid_id'].index(agent2_id)

        pre_motion = np.stack(data['pre_motion_3D']) * data['traj_scale']
        fut_motion = np.stack(data['fut_motion_3D']) * data['traj_scale']
        all_motion = np.concatenate((pre_motion, fut_motion), axis=1)

        # index agent_pair
        # all_motion = all_motion[[agent1_idx,agent2_idx]]
        # prediction = prediction[:,[agent1_idx,agent2_idx]]
        # rollout = rollout # rollout already indexed for [agent1_idx,agent2_idx]
        # # calculate homotopy classes 
        # gt_class = self.calc_pathhomotopy_pair(all_motion[0,...], all_motion[1,...])
        # pred_classes = [self.calc_pathhomotopy_pair(prediction[pred, 0,...], prediction[pred, 1,...]) for pred in range(prediction.shape[0])]
        # rollout_classes = [self.calc_pathhomotopy_pair(rollout[r, 0,...], rollout[r, 1,...]) for r in range(rollout.shape[0])]

        # get minimum fut mask to make sure homotopy classes stay valid and fair timestep comparisons can be made
        pre_motion_mask_valid_pair = torch.minimum(data['pre_motion_mask'][agent1_idx], data['pre_motion_mask'][agent2_idx]).bool()
        fut_motion_mask_valid_pair = torch.minimum(data['fut_motion_mask'][agent1_idx], data['fut_motion_mask'][agent2_idx]).bool()
        all_motion_mask_valid_pair = torch.cat([pre_motion_mask_valid_pair, fut_motion_mask_valid_pair])
        Npred_frames = sum(fut_motion_mask_valid_pair).item()

        # focus on pairs only; only use valid data masks
        all_motion_pair = all_motion[[agent1_idx,agent2_idx]][:,all_motion_mask_valid_pair,:]  # for plotting
        pre_motion_pair = pre_motion[[agent1_idx,agent2_idx]][:,pre_motion_mask_valid_pair,:]
        fut_motion_pair = fut_motion[[agent1_idx,agent2_idx]][:,fut_motion_mask_valid_pair,:] 
        prediciton_pair = prediction[:,[agent1_idx,agent2_idx]][:,:,fut_motion_mask_valid_pair,:] 
        rollout_pair = rollout[:,:,fut_motion_mask_valid_pair,:]  # already only focus agent pair, just index valid timesteps


        # calc homotopy classes with free-end angles
        fut_motion_pair_batch = torch.from_numpy(fut_motion_pair).unsqueeze(0)
        angle_diff_gt, homotopy_gt = identify_pairwise_homotopy(fut_motion_pair_batch)  
        angle_diff_pred, homotopy_pred = identify_pairwise_homotopy(prediciton_pair)
        angle_diff_rollout, homotopy_rollout = identify_pairwise_homotopy(rollout_pair)

        homotopy_dict = {0: 'STATIC',
                         1: 'CW',
                         2: 'CCW',}

        gt_class = [homotopy_dict[homotopy_gt[:,0,1].item()]]
        pred_classes = [homotopy_dict[c.item()] for c in homotopy_pred[:,0,1]]
        rollout_classes = [homotopy_dict[c.item()] for c in homotopy_rollout[:,0,1]]

        # check rollout collisions:
        rollout_collisions_bool = rollout_collisions.tolist()
        rollout_feasible_bool = np.logical_not(rollout_collisions_bool)

        # do some checks
        # assert not min(rollout_collisions_bool), 'no roll-outs possible without collisiosn' # at least one class possible without collisions
        # if min(rollout_feasible_bool): # if both rollouts are feasible
        #     assert ('CW' in rollout_classes)*('CCW' in rollout_classes), 'roll-outs are same class' # both classes should be covered with rollouts 
        
        if sum(rollout_feasible_bool) == 0:
            print()
        feasible_rollout_classes = np.unique(np.array(rollout_classes)[rollout_feasible_bool])
        N_feasible_rollouts = len(feasible_rollout_classes)
        N_modes_covered = len(np.unique(pred_classes))
        h_final = N_feasible_rollouts < 2

        # calculate relevant mode dict:
        mode_dict = {
            'frame': data['frame'],
            'gt_mode': gt_class[0],
            'ml_mode': pred_classes[0],
            'K_modes': np.unique(pred_classes),
            'sim_modes': list(feasible_rollout_classes),
            'mode_correct': pred_classes[0] == gt_class[0], # ML == gt
            'mode_covered': max([pred_class == gt_class[0] for pred_class in pred_classes]), # any pred == gt
            'mode_collapse': N_modes_covered < N_feasible_rollouts,
            'N_modes_covered': N_modes_covered,
            'N_feasible_rollouts': N_feasible_rollouts,
            'h_final': h_final,
            'Npred_frames': Npred_frames, 
            'h_angle_gt': angle_diff_gt,
            
        }
        
        curr_motion_pair = pre_motion_pair[:,[-1],:]
        curr_motion_pair_pred= np.tile(curr_motion_pair, (prediciton_pair.shape[0],1,1,1))# extra dimension for multimodal arrays
        curr_motion_pair_roll= np.tile(curr_motion_pair, (rollout_pair.shape[0],1,1,1))# extra dimension for multimodal arrays

        # transform points to map; curr + fut steps
        motion_map = self.to_map_points(np.concatenate([curr_motion_pair, fut_motion_pair], axis = 1))
        pred_map = self.to_map_points(np.concatenate([curr_motion_pair_pred, prediciton_pair], axis = 2))
        rollout_map = self.to_map_points(np.concatenate([curr_motion_pair_roll, rollout_pair], axis = 2))

        margin = 10
        # x and y indices reveresd because of motion mapping (same for plotting indices)
        all_x = np.concatenate([motion_map[...,1].flatten(),pred_map[...,1].flatten(), rollout_map[...,1].flatten()])
        all_y = np.concatenate([motion_map[...,0].flatten(),pred_map[...,0].flatten(), rollout_map[...,0].flatten()])

        # index of current timestamp:
        idx_cur = 0 # ALL MOTIONS START WITH CURR STEP NOW

        agent_ids = [data['valid_id'][agent1_idx], data['valid_id'][agent2_idx]]
        agent_headings = [data['heading'][agent1_idx], data['heading'][agent2_idx]]
        agent_lengths = data['pre_data'][0][[agent1_idx,agent2_idx], :][:,12] # idx = 12, and first timestep of predata
        agent_widths = data['pre_data'][0][[agent1_idx,agent2_idx], :][:,10]   # idx = 10

        img = np.transpose(self.data, (1, 2, 0))  

        # Create a Plotly figure with 1 row and 3 columns
        fig = make_subplots(rows=1, cols=4, shared_yaxes=True, 
                            subplot_titles=("Ground truth", "Predictions", "Roll-out 1", "Roll-out 2"),
                            horizontal_spacing = 0.0,
                            vertical_spacing = 0.0)
        
        # Plot the image in all three subplots
        fig.add_trace(go.Image(z=img), row=1, col=1)
        fig.add_trace(go.Image(z=img), row=1, col=2)
        fig.add_trace(go.Image(z=img), row=1, col=3)
        fig.add_trace(go.Image(z=img), row=1, col=4)

        # colors agents 
        colors = px.colors.qualitative.Plotly + px.colors.qualitative.Alphabet +  px.colors.qualitative.Dark24

        # Plot the GT trajectories
        for agent_idx, agent_id in enumerate(agent_ids):
            # GT
            fig.add_trace(go.Scatter(
                x=motion_map[agent_idx,:,1],
                y=motion_map[agent_idx,:,0],  # x and y reversed for image
                mode='lines+markers',
                name = f'agent_{int(agent_id)}',
                # showlegend= (agent_idx==0),
                legendgroup='gt',
                legendgrouptitle_text=f'gt<br>h_class: {gt_class[0]}<br>',
                line=dict(color=colors[agent_idx]),
                showlegend=True,
                ),
                row=1, col=1,
            )

            # add vehicle shapes:
            x_points, y_points = self.rotate_car(xc = motion_map[agent_idx,idx_cur,0], 
                                            yc = motion_map[agent_idx,idx_cur,1], 
                                            l = agent_lengths[agent_idx]*3, # size still needs to be scaled
                                            w = agent_widths[agent_idx]*3, # size still needs to be scaled
                                            heading = agent_headings[agent_idx])
            
            # add to all plots:
            for col_num in range(1, 5):
                fig.add_trace(
                    go.Scatter(x=y_points, y=x_points, 
                            fill="toself",
                            mode = 'lines',
                                # legendgroup='gt',
                                # # name = f'gt_agent_{agent_id}',
                                showlegend=False,
                                line=dict(color=colors[agent_idx]),
                                ),
                    row=1, col=col_num,
                )

        # Plot the predictions
        for agent_idx, agent_id in enumerate(agent_ids):
            for pred in range(pred_map.shape[0]):
                fig.add_trace(go.Scatter(
                    x=pred_map[pred,agent_idx,:,1],
                    y=pred_map[pred,agent_idx,:,0],  # x and y reversed for image
                    mode='lines + markers',
                    name = f'agent_{int(agent_id)}',
                    # showlegend= (agent_idx==0),
                    legendgroup=f'pred{pred+1}',
                    legendgrouptitle_text=f'pred{pred+1}<br>h_class: {pred_classes[pred]}<br>',
                    opacity= 1.0 if pred == 0 else 0.2, # ML prediction better visible
                    line=dict(color=colors[agent_idx], dash='dash'),
                    visible = True, # if pred == 0 else 'legendonly',
                    showlegend=True,
                    ),
                    row=1, col=2,
                )

        # Plot the roll-outs
        for agent_idx, agent_id in enumerate(agent_ids):
            rollout_symbols = ['square', 'x']
            r_markersize = 6
            for r in range(rollout_map.shape[0]):
                fig.add_trace(go.Scatter(
                    x=rollout_map[r,agent_idx,:,1],
                    y=rollout_map[r,agent_idx,:,0],  # x and y reversed for image
                    mode='lines + markers',
                    name = f'agent_{int(agent_id)}',
                    marker_symbol= rollout_symbols[r],
                    marker_size = r_markersize,
                    # showlegend= (agent_idx==0),
                    legendgroup=f'rollout{r+1}',
                    legendgrouptitle_text=f'rollout{r+1}<br>h_class: {rollout_classes[r]}<br>collision: {rollout_collisions_bool[r]}',
                    opacity=1,
                    line=dict(color=colors[agent_idx], dash='dot'),
                    showlegend=True,
                    ),
                    row=1, col=3 + r,
                )

        # Update layout to remove axes
        fig.update_layout(
            title=dict(text = data['seq'] + ', frame ' + str(data['frame']) + '-' + str(Npred_frames + data['frame'])),
        )
        

        if plot_square:
            # Calculate the range for both axes
            x_range = all_x.max() - all_x.min()
            y_range = all_y.max() - all_y.min()

            # Determine the maximum range
            max_range = max(x_range, y_range)
            extra_margin_x = (max_range - x_range)/2
            extra_margin_y = (max_range - y_range)/2
        else:
            extra_margin_x = 0
            extra_margin_y = 0



        fig.update_xaxes(range=[all_x.min() - margin - extra_margin_x, all_x.max() + margin + extra_margin_x], visible = False, scaleanchor="y", scaleratio=1)
        fig.update_yaxes(range=[all_y.max() + margin + extra_margin_y, all_y.min() - margin - extra_margin_y], visible = False, scaleanchor="x", scaleratio=1) #        
        fig.update_layout(legend=dict(
                orientation="h"))



        # # Show the Plotly figure
        # fig.show()
        # print()
        return fig, mode_dict