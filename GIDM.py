import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import math

class GIDM():
    def __init__(self, 
                 df_agent,
                 df_all_gt = None,
                 replay = True,
                 time_replay = True, 
                 use_vref = True,
                 t_shift = None,
                 delta = 2, # free-drive  exponent [-]
                 d_min = 1, # min. distance at rest [m]
                 v_star = 30 / 3.6, # target velocity [m/s]
                 T = 1.8, # safety time gab [s]
                 a_IDM = 2, # max. IDM acceleration [m/s^2]
                 b_comf = 3, # comfort deceleration [m/s^2]
                 ) -> None:
        """
        Implementation of the generalized intelligent driver model
        """
        # init model parameters:
        self.delta = delta
        self.d_min = d_min
        self.v_star = v_star
        self.T = T
        self.a_IDM = a_IDM
        self.b_comf = b_comf
        self.b_emergency = 8

        # init agent states and simulation settings
        self.df_all_gt = df_all_gt
        self.replay = replay
        self.time_replay = time_replay
        self.use_vref = use_vref
        self.df_agent = df_agent
        self.id = df_agent['agent_id'].values[0]
        if not t_shift is None:
            self.gt_t = df_agent['t'].values + t_shift
        else:
            self.gt_t = df_agent['t'].values
        self.gt_x = df_agent['x'].values
        self.gt_y = df_agent['y'].values
        self.gt_heading = df_agent['heading'].values
        self.t_start = min(self.gt_t)
        self.t_end = max(self.gt_t)
        self.dt = np.diff(self.gt_t)[0]

        self.gt_s = np.zeros_like(self.gt_x)
        self.ds = np.sqrt(np.diff(self.gt_x)**2 + np.diff(self.gt_y)**2)
        self.gt_s[1:] = np.cumsum(self.ds)
        self.gt_v = np.gradient(self.gt_s, self.gt_t)

        # check stationary points here
        # make the last point v = 0 if v<1
        # remove all x/ypoints that are almost stationary, and only keep the last
        if not self.replay and (self.gt_v[-1] < 1):
            idx_denoise = self.gt_v >1
            idx_denoise[-1] = True
            idx_denoise[0]  = True
            # overwrite variables and recalculate
            self.gt_x = self.gt_x[idx_denoise]
            self.gt_y = self.gt_y[idx_denoise]
            self.gt_t = self.gt_t[idx_denoise]
            self.gt_heading = self.gt_heading[idx_denoise]

            self.gt_s = np.zeros_like(self.gt_x)
            self.ds = np.sqrt(np.diff(self.gt_x)**2 + np.diff(self.gt_y)**2)
            self.gt_s[1:] = np.cumsum(self.ds)
            try:
                self.gt_v = np.gradient(self.gt_s, self.gt_t)
            except IndexError:
                self.gt_v = np.zeros_like(self.gt_t)

            self.gt_v[-1] = 0 # stationary points are noisy, so to avoid incorrect extrapolation when not replaying, set last v to 0


        # intialize current distance and velocity variables (changes dynamically during simulation)
        self.x_curr = self.gt_x[0]
        self.y_curr = self.gt_y[0]
        self.heading_curr = self.gt_heading[0]
        self.s_curr = self.gt_s[0]
        self.v_curr = self.gt_v[0]

        #TODO: fix interpolation functions, to handle stationary behavior correctly, basically only using forward points!
        # time dependent interpolation functions
        self.f_x_t = interpolate.interp1d(self.gt_t, self.gt_x, fill_value='extrapolate', assume_sorted=True)
        self.f_y_t  = interpolate.interp1d(self.gt_t, self.gt_y, fill_value='extrapolate', assume_sorted=True)
        self.f_heading_t = interpolate.interp1d(self.gt_t, self.gt_heading, 
                                              fill_value=(self.gt_heading[0],self.gt_heading[-1]), # don't extrapolate heading!
                                              bounds_error=False, assume_sorted=True)
        self.f_v_t = interpolate.interp1d(self.gt_t, self.gt_v, 
                                              fill_value=(self.gt_v[0],self.gt_v[-1]), # don't extrapolate velocity!
                                              bounds_error=False, assume_sorted=True)
        
        # distance dependent interpolation functions
        self.f_x_s = interpolate.interp1d(self.gt_s, self.gt_x, fill_value='extrapolate', assume_sorted=False) # NO UTURNS
        self.f_y_s  = interpolate.interp1d(self.gt_s, self.gt_y, fill_value='extrapolate', assume_sorted=False) # NO UTURNS
        self.f_heading_s = interpolate.interp1d(self.gt_s, self.gt_heading, 
                                              fill_value=(self.gt_heading[0],self.gt_heading[-1]), # don't extrapolate heading!
                                              bounds_error=False, assume_sorted=True)
        self.f_v_s = interpolate.interp1d(self.gt_s, self.gt_v, 
                                              fill_value=(self.gt_v[0],self.gt_v[-1]), # don't extrapolate velocity!
                                              bounds_error=False, assume_sorted=True)

    def sim_step(self, t, df_scene = None):
        if self.replay:
            if (t >= self.t_start)*(t <= self.t_end):
                if self.time_replay:             # time based replay
                    x_curr = self.f_x_t(t).item()
                    y_curr = self.f_y_t(t).item()
                    heading_curr = self.f_heading_t(t).item()
                    v_curr = self.f_v_t(t).item()

                    # calculate new position and set as current one for next iteration
                    s_new = self.s_curr + v_curr * self.dt
                    self.s_curr = s_new
                    return (x_curr, y_curr, v_curr, heading_curr)
                else:                       # distance based replay
                    x_curr = self.f_x_s(self.s_curr).item()
                    y_curr = self.f_y_s(self.s_curr).item()
                    heading_curr = self.f_heading_s(self.s_curr).item()
                    v_curr = self.f_v_s(self.s_curr).item()

                    # solution1: recognize stationary point and map v to 0 --> easier 
                    # solution2: use heading for extrapolation of position

                    # calculate new position and set as current one for next iteration
                    s_new = self.s_curr + v_curr * self.dt
                    self.s_curr = s_new
                    return (x_curr, y_curr, v_curr, heading_curr)
            else:
                return None
            
        else: 
            if (t >= self.t_start)*(t <= self.t_end):
                # get distance based congfiguration
                x_curr = self.f_x_s(self.s_curr).item()
                y_curr = self.f_y_s(self.s_curr).item()
                heading_curr = self.f_heading_s(self.s_curr).item()
                v_curr = self.v_curr

                if self.use_vref:
                    v_ref = self.f_v_s(self.s_curr).item() # use distance based reference velocity
                    self.v_star = v_ref

                # calculate idm acceleration
                d_01, v1 = self.get_interacting_agent_vars(df_scene)
                dv_dt = self.calc_acceleration(d_01, v_curr, v1)   

                # calculate new velocity and position
                v_new = v_curr + dv_dt * self.dt
                s_new = self.s_curr + self.v_curr * self.dt

                # only set  new velocity and position here!
                self.x_curr = x_curr
                self.y_curr = y_curr
                self.heading_curr = heading_curr
                self.v_curr = v_new
                self.s_curr = s_new

                assert not math.isinf(dv_dt)

                return (x_curr, y_curr, v_curr, heading_curr)
            else:
                return None
        
    def get_interacting_agent_vars(self,  df_scene, d_01_empty = 1000, v1_empty = 1000,
                                   interp_points = 100):
        # find closest interacting agent and calculate velocity and projected distance
        if not df_scene.empty:
            s_fut = self.gt_s[self.gt_s >= self.s_curr]

            if len(s_fut) == 0:
                return d_01_empty, v1_empty # no interaction when agent is at the end of its path???

            s_fut = np.interp(np.linspace(s_fut[0], s_fut[-1], interp_points),
                              s_fut, s_fut)
            x_fut = self.f_x_s(s_fut)
            y_fut = self.f_y_s(s_fut)
            path1 = x_fut, y_fut

            ego_row = pd.DataFrame({'x':[ self.x_curr], 'y': [self.y_curr], 'heading':[ self.heading_curr]})

            # calculate agent projections
            for i, row in df_scene.iterrows():
                x_proj, y_proj = self.project_trajectory(row, use_gt= False, projection_var='t')  
                df_scene.loc[i, 'x_proj'] = x_proj
                df_scene.loc[i, 'y_proj'] = y_proj

                if row.agent_id != self.id:
                    path2 = np.linspace(row.x, x_proj, interp_points), np.linspace(row.y, y_proj, interp_points)
                    intersect_bool, idx1, idx2 = GIDM.get_path_crossing_point(path1, path2)
                    if intersect_bool:
                        rel_x_local, rel_y_local = GIDM.relative_position(row, ego_row)
                        df_scene.loc[i, 'intersect_bool'] = True
                        df_scene.loc[i, 'rel_x_local'] = rel_x_local.item()
                        df_scene.loc[i, 'rel_y_local'] = rel_y_local.item()
                        df_scene.loc[i, 'in_front_bool'] = rel_x_local.item() > 0
 
            # calculate closest intersecting agent
            # for df_scene
            try:
            # calculate projection distance, and get velocity of closest  interacting agent
                df_interacting_in_front = df_scene[df_scene['in_front_bool'] == True]
            except KeyError:
                return d_01_empty, v1_empty

            if df_interacting_in_front.empty:
                return d_01_empty, v1_empty
            else:  
                # find closest agent in front, and return its variables
                idx_closest = df_interacting_in_front['rel_x_local'].idxmin()
                row_closest = df_interacting_in_front.loc[idx_closest]
                d_01 = row_closest.rel_x_local # TODO: add sizes later, + option to project
                v1 = row_closest.v
                return d_01,  v1
        else:
            return d_01_empty, v1_empty

    def calc_acceleration(self, 
                          d_01, # long bumper-to-bumper distance two vehicles
                          v0, # velocity vehicle 0 (ego)
                          v1, # velocity vehicle 1 (in front)
                          ):
        delta_v_01 = v1 - v0
        d_01_star = self.d_min + v0*self.T - v0*delta_v_01/(2*np.sqrt(self.a_IDM*self.b_comf)) # desired long bumper-to-bumper distance two vehicles
        d_01_star = 0 if d_01_star < 0 else d_01_star

        dv_dt = self.a_IDM*(1-(v0/self.v_star)**self.delta) - self.a_IDM*(d_01_star/d_01)**2 # acceleration ego vehicle

        if self.v_star == 0 and v0 == 0:
            dv_dt = 0

        # put limit to make sure going reverse is not possible:
        dv_dt_min = max((0-v0)/self.dt, -self.b_emergency)
        dv_dt = max(dv_dt, dv_dt_min)

        assert(dv_dt < 5)
        assert(dv_dt > -10)

        return dv_dt


    def project_trajectory(self, df_row, projection_var = 's', projection_time = 1, projection_distance = 100, use_gt = True):
        if use_gt:
            x_proj = self.df_all_gt[self.df_all_gt['agent_id'] == df_row.agent_id].iloc[-1]['x']
            y_proj = self.df_all_gt[self.df_all_gt['agent_id'] == df_row.agent_id].iloc[-1]['y']
        else:
            if projection_var == 't':
                projection_distance = df_row.v * projection_time
            x_proj = df_row.x + np.cos(df_row.heading) * projection_distance
            y_proj = df_row.y + np.sin(df_row.heading) * projection_distance
        return x_proj, y_proj
        
    @staticmethod
    def get_path_crossing_point(path1, path2, crossing_threshold = 1):
        distances = cdist(np.array(path1).T, np.array(path2).T)
        min_distance = np.min(distances)
        min_indices = np.argwhere(distances == min_distance)
        intersect_bool = min_distance < crossing_threshold 
        idx1, idx2 = min_indices[0,[0,1]]
        return intersect_bool, idx1, idx2

    @staticmethod
    def calc_velocity_vector(group):
        vx = np.gradient(group['x'], group['t'])
        vy = np.gradient(group['y'], group['t'])
        v = np.sqrt(vx**2 + vy**2)
        return pd.Series(v, index=group.index)
    
    @staticmethod
    def min_distance_to_path(row, path):
        point = np.array([row['x'], row['y']])
        distances = np.sqrt(np.sum((path - point[:, np.newaxis])**2, axis=0))
        return np.min(distances)

    @staticmethod
    def distance_along_path(row, xy_path, s_path):
        point = np.array([row['x'], row['y']])
        distances = np.sqrt(np.sum((xy_path - point[:, np.newaxis])**2, axis=0))
        idx_path = np.argmin(distances)
        return s_path[idx_path]
    
    @staticmethod
    def relative_position(row, ego_row):
        # Calculate relative position in global reference frame
        rel_x = row['x'] - ego_row['x']
        rel_y = row['y'] - ego_row['y']

        # Rotate relative position based on ego vehicle's heading
        theta = - ego_row['heading']
        rel_x_local = rel_x * np.cos(theta) + rel_y * - np.sin(theta)
        rel_y_local = rel_x * np.sin(theta) + rel_y *   np.cos(theta)

        return rel_x_local, rel_y_local
    
    @staticmethod
    def project_trajectories(df, projected_id = '99', interp_points = 1000, path_threshold = 3,
                             plot = False):
        ego_df = df[df['agent_id']==projected_id]
        t_interp = np.linspace(ego_df['t'].values[0], ego_df['t'].values[-1], interp_points)
        x_path = np.interp(t_interp, ego_df['t'], ego_df['x'])
        y_path = np.interp(t_interp, ego_df['t'], ego_df['y'])
        xy_path = np.array([x_path, y_path])

        s_path = np.zeros_like(x_path)
        ds = np.sqrt(np.diff(x_path)**2 + np.diff(y_path)**2)
        s_path[1:] = np.cumsum(ds)

        df['min_distance_to_path'] = df.apply(lambda row: GIDM.min_distance_to_path(row, xy_path), axis=1)
        df['on_path'] = df['min_distance_to_path'] < path_threshold
        df['distance_along_path'] = df[df['on_path']].apply(lambda row: GIDM.distance_along_path(row, xy_path, s_path), axis = 1)
        df['rb_along_path'] = df['distance_along_path'] - 0.5*df['length'] # rear bound vehicle along path
        df['fb_along_path'] = df['distance_along_path'] + 0.5*df['length'] # front bound vehicle along path
        
        if plot:
            color_scale = px.colors.qualitative.Plotly + px.colors.qualitative.Plotly
            color_map = {str(agent_id): (color_scale[-1] if agent_id == '99' else color_scale[int(agent_id)]) for agent_id in df['agent_id']}
            fig_scatter = px.scatter(df, x='x', y='y', color=df['agent_id'].astype(str), color_discrete_map=color_map)
            fig_scatter.show()

            fig = go.Figure()
            df_onpath = df[df['on_path']]
            for i, agent_id in enumerate(df['agent_id'].unique()):
                agent_df = df_onpath[df_onpath['agent_id']==agent_id]
                if not agent_df.empty:
                    x_points = list(agent_df['rb_along_path'].values) + list(agent_df['fb_along_path'].values[::-1]) + [agent_df['rb_along_path'].values[0]]
                    y_points = list(agent_df['t'].values) + list(agent_df['t'].values[::-1]) + [agent_df['t'].values[0]]
                    fig.add_trace(
                        go.Scatter(x=y_points, y=x_points, 
                                fill="toself",
                                mode = 'lines',
                                legendgroup='gt',
                                name = f'gt_agent_{agent_id}',
                                showlegend=True,
                                line=dict(color=color_scale[-1] if agent_id == '99' else color_scale[int(agent_id)]),
                                ))
            fig.update_layout(
                xaxis_title="frame",
                yaxis_title="distance along ego path"
            )
            fig.show()

        return df

    @staticmethod
    def process_data(gt, ego_id = '99', fps_gt=2):
        df = pd.DataFrame(data=gt[:, [0, 1, 13, 15, 10, 11, 12, 16]],
                          columns=['frame', 'agent_id', 'x', 'y', 'width', 'height', 'length', 'heading']).astype(float)
        df['agent_type'] = gt[:, 2]
        df['agent_id'] = df['agent_id'].astype(int).astype(str)# for plotting; categorical
        df['t'] = df['frame'] / fps_gt

        # Apply the velocity calculation function to each group of agent_id
        df['v'] = df.groupby('agent_id').apply(GIDM.calc_velocity_vector).reset_index(level=0, drop=True)

        # project trajectories to ego vehicle path
        df = GIDM.project_trajectories(df, projected_id= ego_id)

        return df


def simulate_scene(gt, modify_args, plot_mods = True):

    b_comf = modify_args

    df = GIDM.process_data(gt)

    agents = []
    for agent_id in df.agent_id.unique():
        df_agent = df[df['agent_id']==agent_id]
        # if True:
        #     agent_sim = GIDM(df_agent, df_all_gt = df, replay = True, time_replay = True, use_vref = True,
        #                      t_shift = None)
        if agent_id == '99':
            agent_sim = GIDM(df_agent, df_all_gt = df, replay = False, time_replay = False, use_vref = False
                             )
            # agent_sim.b_comf = b_comf
            agent_sim.v_star = b_comf/3.6

        else:
            agent_sim = GIDM(df_agent, df_all_gt = df, replay = False, time_replay = False, use_vref = True,
                             t_shift = None)
        agents.append(agent_sim)

    df_mod = pd.DataFrame()
    df_curr = pd.DataFrame() #TODO intialize with start values
    # loop through time (min - max) and do .sim_step for all agents for each frame
    for frame, t in enumerate(df.t.unique()):
        for agent in agents:
            curr_states = agent.sim_step(t, df_curr)
            if curr_states:
                x, y, v, heading = curr_states
                data_row = {'frame': frame, 't': t, 'agent_id': agent.id, 'x': x, 'y': y, 'v': v, 'heading': heading}
                df_mod = df_mod.append(data_row, ignore_index = True)
        
        df_curr = df_mod[df_mod['frame'] == frame] # current agent states

                
    if plot_mods:
        fig = px.scatter(df_mod, x='x', y='y',  hover_data = ['agent_id', 't'], color='agent_id')
        fig.update_layout(
            xaxis=dict(
                range=[df.x.min(), df.x.max()],  # Set the x-axis range
                ),
            yaxis=dict(
                range=[df.y.min(), df.y.max()], # Set the y-axis range
                )
        )
        fig.show()

        fig = px.line_3d(df_mod, x = 'x', y = 'y', z = 't', color = 'agent_id')
        fig.show()

        #NOTE: not exactlt the same... Reason: noise in annations (and only forward velocity). Avoid by not extrapolating??
        # or due to rounding errors? Compare speeds,.. Problem likely with sorting in interpolation!


    # add car type and dimensions to new df:
    copy_keys = ['width', 'height', 'length', 'agent_type']
    for agent_id in df['agent_id'].unique():
        for key in copy_keys:
            df_mod.loc[df_mod['agent_id']==agent_id, key] = df.loc[df['agent_id']==agent_id, key].values[0]
        
    # return modified gt:
    gt_mod = np.ones((len(df_mod), gt.shape[1]), dtype = '<U19')
    gt_mod.fill(-1)
    gt_mod[:,-1] = '1.0' # is in frame col?
    gt_mod[:, 0] = df_mod['frame'].values
    gt_mod[:, 1] = df_mod['agent_id'].values
    gt_mod[:, 2] = df_mod['agent_type'].values
    gt_mod[:, 13] = df_mod['x'].values
    gt_mod[:, 14] = np.zeros_like(df_mod['x'].values)  # z value
    gt_mod[:, 15] = df_mod['y'].values
    gt_mod[:, 10] = df_mod['width'].values
    gt_mod[:, 11] = df_mod['height'].values
    gt_mod[:, 12] = df_mod['length'].values
    gt_mod[:, 16] = df_mod['heading'].values

    return gt_mod