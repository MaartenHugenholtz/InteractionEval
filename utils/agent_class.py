import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import math




class Agent():
    def __init__(self, 
                 df_agent,
                 dt = 0.5, # s
                 fut_steps = 12,
                 v_min = 0, # m/s
                 v_max = 50/3.6, # m/s
                 ax_min = -3, # m/s^2
                 ax_max = 2,  # m/s^2
                 ay_abs_max = 5, # m/s^2
                 ) -> None:
        """
        Implementation of agent modifier class
        """

        # init agent states and simulation settings
        self.df_agent = df_agent
        self.frame_end = df_agent.frame.max()
        self.dt = dt
        self.id = df_agent['agent_id'].values[0]
        assert len(df_agent['agent_id'].unique()) == 1, 'More than 1 agent'
        # assert np.diff(df_agent['t'])[0] == dt, 'Timesteps missing in data'
        

        self.fut_steps = fut_steps
        self.v_min = v_min
        self.v_max = v_max
        self.ax_min = ax_min
        self.ax_max = ax_max
        self.ay_abs_max = ay_abs_max

        # remove stationary points for interpolation functions
        idx_denoise = df_agent['v'].values > 1
        idx_denoise[-1] = True
        idx_denoise[0]  = True

        # also affects distance based variables
        self.gt_x = df_agent['x'].values[idx_denoise]
        self.gt_y = df_agent['y'].values[idx_denoise]
        self.gt_k = df_agent['k'].values[idx_denoise]
        self.gt_heading = df_agent['heading'].values[idx_denoise]
        self.gt_s = np.zeros_like(self.gt_x)
        self.ds = np.sqrt(np.diff(self.gt_x)**2 + np.diff(self.gt_y)**2)
        self.gt_s[1:] = np.cumsum(self.ds)

        # edgde case: if vehicle completely stationary, we need to add a small offset to gt_s for interpolation 
        if self.gt_s[0] == self.gt_s[-1] == 0:
            self.gt_s[-1] += 1e-3
        
        # distance dependent interpolation functions
        self.f_x_s = interpolate.interp1d(self.gt_s, self.gt_x, fill_value='extrapolate', assume_sorted=False) # NO UTURNS
        self.f_y_s  = interpolate.interp1d(self.gt_s, self.gt_y, fill_value='extrapolate', assume_sorted=False) # NO UTURNS
        self.f_k_s  = interpolate.interp1d(self.gt_s, self.gt_k, fill_value='extrapolate', assume_sorted=True) # NO UTURNS  
        self.f_heading_s  = interpolate.interp1d(self.gt_s, self.gt_heading, fill_value=(self.gt_heading[0], self.gt_heading[-1]), assume_sorted=True, bounds_error= False) # DO NOT EXTRAPOLATE HEADING
        
    def get_gt_agent(self, frame_curr):
        #TODO: might still need to make option for longer futsteps
        df_agent_fut = self.df_agent.copy().reset_index() # reinitlaize new copy of gt
        pred_idx = (df_agent_fut['frame'] > frame_curr) * (df_agent_fut['frame'] <= frame_curr + self.fut_steps) 
        fut_rollout = df_agent_fut[pred_idx][['x', 'y']].values

        if fut_rollout.shape[0] < self.fut_steps:
            # fut_rollout_repeat = np.zeros((self.fut_steps, 3))
            fut_rollout_repeat = np.zeros((self.fut_steps, 2))
            fut_rollout_repeat[:,0] = fut_rollout[-1,0]
            fut_rollout_repeat[:,1] = fut_rollout[-1,1]
            # fut_rollout_repeat[:,2] = fut_rollout[-1,2]
            fut_rollout_repeat[0:fut_rollout.shape[0],:] = fut_rollout
            fut_rollout = fut_rollout_repeat
        
        gt_agent = fut_rollout
        return gt_agent

    def rollout_future(self, frame_curr, direction = 'accel', use_gt_path = True):
        df_agent_fut = self.df_agent.copy().reset_index() # reinitlaize new copy of gt

        for idx in range(len(df_agent_fut)):
            row = df_agent_fut.loc[idx]
            if row['frame'] >= frame_curr and row['frame'] < self.frame_end:
                # get updated row from df

                # calculate travelled distance ds and new s/x/y/k 
                v_curr, s_curr, heading_curr, x_curr, y_curr = row['v'], row['s'], row['heading'], row['x'], row['y']
                ds = v_curr * self.dt
                s_new = s_curr + ds

                if use_gt_path:  # Future rollouts and Oracle model
                    x_new = self.f_x_s(s_new).item()
                    y_new = self.f_y_s(s_new).item()
                    k_new = self.f_k_s(s_new).item()
                    heading_new = self.f_heading_s(s_new).item()

                else: # constant heading model
                    x_new = x_curr + np.cos(heading_curr) * ds
                    y_new = y_curr + np.sin(heading_curr) * ds
                    k_new = 0 # constant heading, no corner
                    heading_new = heading_curr

                df_agent_fut.loc[idx + 1, ['s', 'x', 'y', 'k', 'heading']] = s_new, x_new, y_new, k_new, heading_new

                # calculate acceleration and velocity for new frame
                if direction == 'decel':
                    ax_min_stationary = (self.v_min - v_curr) / self.dt
                    ax = max(ax_min_stationary, self.ax_min)
                elif direction == 'accel':
                    try:
                        vmax_corner = np.sqrt(self.ay_abs_max / abs(k_new))
                        vmax = min(self.v_max, vmax_corner)
                    except ZeroDivisionError:
                        vmax = self.v_max
                    vmax = self.v_max
                    ax_max_limit = (vmax - v_curr) / self.dt
                    assert ax_max_limit >= 0, 'ax <0 for acceleration rollout'
                    ax = min(ax_max_limit, self.ax_max)
                elif direction == 'constant':
                    ax = 0 # constant velocity 
                else:
                    raise NameError('direction mode does not exist')
                
                assert not np.isnan(ax), 'ax nan'
                assert not np.isnan(v_curr), 'vcurr nan'

                v_new = v_curr + ax*self.dt
                df_agent_fut.loc[idx + 1, 'v'] = v_new

        pred_idx = (df_agent_fut['frame'] > frame_curr) * (df_agent_fut['frame'] <= frame_curr + self.fut_steps) 
        fut_rollout = df_agent_fut[pred_idx][['x', 'y', 'heading']].values

        # assert(not (frame_curr == 16 and self.id =='2'))

        if fut_rollout.shape[0] < self.fut_steps:
            fut_rollout_repeat = np.zeros((self.fut_steps, 3))
            fut_rollout_repeat[:,0] = fut_rollout[-1,0]
            fut_rollout_repeat[:,1] = fut_rollout[-1,1]
            fut_rollout_repeat[:,2] = fut_rollout[-1,2]
            fut_rollout_repeat[0:fut_rollout.shape[0],:] = fut_rollout
            fut_rollout = fut_rollout_repeat

        assert not np.isnan(fut_rollout).any(), 'rollout nan'

        return fut_rollout #includes heading now

    @staticmethod
    def calc_path_distance(group, v_var = 'v', dt = 0.5):
        s= np.zeros_like(group[v_var])
        ds = group[v_var] * dt
        s[1:] = np.cumsum(ds[:-1])
        return pd.Series(s, index=group.index)
    
    @staticmethod
    def calc_velocity_vector(group, xvar = 'x', yvar = 'y', tvar = 't'):
        vx = np.gradient(group[xvar], group[tvar])
        vy = np.gradient(group[yvar], group[tvar])
        v = np.sqrt(vx**2 + vy**2)
        return pd.Series(v, index=group.index)
    
    @staticmethod
    def calc_acceleration(group, v_var = 'v', t_var = 't'):
        ax = np.gradient(group[v_var], group[t_var])
        return pd.Series(ax, index=group.index)
    
    @staticmethod
    def calc_yaw_rate(group):
        yaw_rate = np.gradient(group['heading'], group['t'])
        return pd.Series(yaw_rate, index=group.index)
    
    @staticmethod
    def process_data(gt, ego_id = '99', fps_gt=2):
        df = pd.DataFrame(data=gt[:, [0, 1, 13, 15, 10, 11, 12, 16, 18]],
                            columns=['frame', 'agent_id', 'x', 'y', 'width', 'height', 'length', 'heading', 'lane_num']).astype(float)
        df['agent_type'] = gt[:, 2]
        df['agent_id'] = df['agent_id'].astype(int).astype(str)# for plotting; categorical
        df['t'] = df['frame'] / fps_gt

        # Apply calculation functions to each group of agent_id
        df['v'] = df.groupby('agent_id').apply(Agent.calc_velocity_vector).reset_index(level=0, drop=True)
        df['ax'] = df.groupby('agent_id').apply(Agent.calc_acceleration).reset_index(level=0, drop=True)
        df['yaw_rate'] = df.groupby('agent_id').apply(Agent.calc_yaw_rate).reset_index(level=0, drop=True)
        df['k'] = df['yaw_rate'] / df['v'] # curvature [1/m]
        df.loc[df['v']<1, 'k'] = 1e-6 # no curvature at low speeds, because of noise at stationary points
        df['ay'] = df['v']**2 * df['k']
        df['s'] = df.groupby('agent_id').apply(Agent.calc_path_distance).reset_index(level=0, drop=True)

        return df
    
    @staticmethod
    def get_path_crossing_point(path1, path2, crossing_threshold = 1):
        distances = cdist(np.array(path1).T, np.array(path2).T)
        min_distance = np.min(distances)
        min_indices = np.argwhere(distances == min_distance)
        intersect_bool = min_distance < crossing_threshold 
        idx1, idx2 = min_indices[0,[0,1]]
        return intersect_bool, idx1, idx2
