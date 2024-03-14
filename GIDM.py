import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy import interpolate
import matplotlib.pyplot as plt

class GIDM():
    def __init__(self, 
                 df_agent,
                 replay = True,
                 delta = 2, # free-drive  exponent [-]
                 d_min = 1, # min. distance at rest [m]
                 v_star = 50 / 3.6, # target velocity [m/s]
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

        # init agent states and simulation settings
        self.replay = replay
        self.df_agent = df_agent
        self.id = df_agent['agent_id'].values[0]
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

        # intialize current distance and velocity variables (changes dynamically during simulation)
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

    def sim_step(self, t, df_scene = None, time_replay = False, use_vref = False):
        if self.replay:
            if (t >= self.t_start)*(t <= self.t_end):
                if time_replay:             # time based replay
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

                if use_vref:
                    v_ref = self.f_v_s(self.s_curr).item() # use distance based reference velocity
                    self.v_star = v_ref

                # calculate idm acceleration
                d_01, v1 = self.get_interacting_agent_vars(df_scene)
                dv_dt = self.calc_acceleration(d_01, v_curr, v1)   

                # calculate new velocity and position
                v_new = v_curr + dv_dt * self.dt
                s_new = self.s_curr + self.v_curr * self.dt

                # only set  new velocity and position here!
                self.v_curr = v_new
                self.s_curr = s_new

                return (x_curr, y_curr, v_curr, heading_curr)
            else:
                return None
        
    def get_interacting_agent_vars(self,  df_scene, d_01_empty = 1000, v1_empty = 1000,
                                   projection_distance = 100):
        # find closest interacting agent and calculate velocity and projected distance
        if not df_scene.empty:
            s_fut = self.gt_s[self.gt_s >= self.s_curr]
            x_fut = self.f_x_s(s_fut)
            y_fut = self.f_y_s(s_fut)

            # calculate agent projections

            # calculate closest intersecting agent

            # calculate projection distance, and get velocity of closest  interacting agent


            
            d_01 = 0
            v1 = 0
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
        return dv_dt

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

    def sim_agent(self, gt, sim_args, ego_id = '99', fps_gt = 2, project_lat_distance = True):
        mod_pars, mod_values, start_frame_sim = sim_args
        # modify model parameters:
        for par, value in zip(mod_pars, mod_values):
            self.__setattr__(par, value) 

        df = GIDM.process_data(gt, ego_id = ego_id, fps_gt=fps_gt)
        dt = 1/ fps_gt
        gt_mod = gt.copy()

        agents_on_ego_path = [id for id in list(df[df['on_path']]['agent_id'].unique()) if id != ego_id]

        accels = []
        x_list = []

        df_ego_scene = df[df['agent_id']=='99'] # keep unchanged for path interpolation. Only df modified for ego
        s_next = 0

        # interpolation/extrapolation functions for modificaitons:
        f_x = interpolate.interp1d(df_ego_scene['distance_along_path'].values, df_ego_scene['x'].values, fill_value='extrapolate')
        f_y = interpolate.interp1d(df_ego_scene['distance_along_path'].values, df_ego_scene['y'].values, fill_value='extrapolate')
        # f_heading = interpolate.interp1d(df_ego_scene['distance_along_path'].values, df_ego_scene['heading'].values, 
        #                                  fill_value = (df_ego_scene['heading'].values[0], df_ego_scene['heading'].values[-1])) # don't extrapolate heading, keep constant if out of range

        
        for frame in range(start_frame_sim, int(df.frame.max())+1):
            df_frame = df[df['frame']==frame]
            df_agents = df_frame[df_frame['agent_id'].isin(agents_on_ego_path)]
            df_ego = df_frame[df_frame['agent_id'].isin([ego_id])]
            df_agent_ahead = pd.DataFrame() # intialize empty df
            
            if not df_agents.empty:
                relative_positions_x = []
                relative_positions_y = []
                ids_infront = []
                for index, row in df_agents.iterrows():
                    rel_x_local, rel_y_local = GIDM.relative_position(row, df_ego.iloc[0])
                    if rel_x_local > 0:  # vehicle must be in front
                        relative_positions_x.append(rel_x_local)
                        relative_positions_y.append(rel_y_local)
                        ids_infront.append(row['agent_id'])

                if relative_positions_x:
                    # get closest vehicle ahead:
                    idx_closest_ahead = np.argmin(relative_positions_x)
                    agent_ahead = ids_infront[idx_closest_ahead]
                    df_agent_ahead = df_frame[df_frame['agent_id']==agent_ahead]
                    df_agent_ahead['rel_x_local'] = relative_positions_x[idx_closest_ahead]
                    df_agent_ahead['rel_y_local'] = relative_positions_y[idx_closest_ahead]

                    
            ego_row = df_ego.iloc[0]
            v0 = ego_row['v']

            if df_agent_ahead.empty:
                v1 = 999
                d_01 = 999
            else:
                agent_ahead_row = df_agent_ahead.iloc[0]
                v1 = agent_ahead_row['v']
                d_01 = agent_ahead_row['rel_x_local'] + 0.5*(agent_ahead_row['length'] + ego_row['length']) + project_lat_distance * abs(agent_ahead_row['rel_y_local'])  # optionally project y distance
            
            # calculate new position based on current velocity, and new velocity based on current acceleration
            dv_dt = self.calc_acceleration(d_01, v0, v1)
            v_next = v0 + dv_dt*dt
            s_next += v0*dt # keep path the same, interpolate based on changing distance 
            x_next = f_x(s_next)  #   np.interp(s_next, df_ego_scene['distance_along_path'].values, df_ego_scene['x'].values)
            y_next = f_y(s_next)  #   np.interp(s_next, df_ego_scene['distance_along_path'].values, df_ego_scene['y'].values)
            heading_next = np.interp(s_next, df_ego_scene['distance_along_path'].values, df_ego_scene['heading'].values) # does not extrapolate, but keeps constant
            
            idx_next_ego = (df['frame']==frame+1) * (df['agent_id'].isin([ego_id]))
            df.loc[idx_next_ego, 'v'] = v_next
            df.loc[idx_next_ego, 'x'] = x_next
            df.loc[idx_next_ego, 'y'] = y_next
            df.loc[idx_next_ego, 'heading'] = heading_next

            accels.append(dv_dt)
            x_list.append(x_next)


        # order of rows should be the same, so simply overwrite the changed columns:
        gt_mod[:,[13, 15, 16]] = df[['x', 'y', 'heading']].to_numpy()


        # # compare accels with real accel
        # import matplotlib.pyplot as plt
        # accels_real = np.gradient(df[df['agent_id']=='99']['v'].values, df[df['agent_id']=='99']['t'].values)
        # plt.figure()
        # plt.plot(accels_real)
        # plt.plot(accels)

        # GIDM.project_trajectories(df, plot =True)

        return gt_mod
    




def simulate_scene(gt, modify_args, plot_mods = True):

    
    df = GIDM.process_data(gt)

    agents = []
    for agent_id in df.agent_id.unique():
        df_agent = df[df['agent_id']==agent_id]
        agent_sim = GIDM(df_agent, replay = False)
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

        #NOTE: not exactlt the same... Reason: noise in annations (and only forward velocity). Avoid by not extrapolating??
        # or due to rounding errors? Compare speeds,.. Problem likely with sorting in interpolation!


    return gt