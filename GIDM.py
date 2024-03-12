import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy import interpolate

class GIDM():
    def __init__(self, 
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
    def project_trajectories(df, projected_id = 99, interp_points = 1000, path_threshold = 3,
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
            color_map = {str(agent_id): (color_scale[-1] if agent_id == 99 else color_scale[int(agent_id)]) for agent_id in df['agent_id']}
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
                                line=dict(color=color_scale[-1] if agent_id == 99 else color_scale[int(agent_id)]),
                                ))
            fig.update_layout(
                xaxis_title="frame",
                yaxis_title="distance along ego path"
            )
            fig.show()

        return df

    @staticmethod
    def process_data(gt, ego_id = 99, fps_gt=2):
        df = pd.DataFrame(data=gt[:, [0, 1, 13, 15, 10, 11, 12, 16]],
                          columns=['frame', 'agent_id', 'x', 'y', 'width', 'height', 'length', 'heading']).astype(float)
        df['agent_type'] = gt[:, 2]
        df['t'] = df['frame'] / fps_gt

        # Apply the velocity calculation function to each group of agent_id
        df['v'] = df.groupby('agent_id').apply(GIDM.calc_velocity_vector).reset_index(level=0, drop=True)

        # project trajectories to ego vehicle path
        df = GIDM.project_trajectories(df, projected_id= ego_id)

        return df

    def sim_agent(self, gt, sim_args, ego_id = 99, fps_gt = 2, project_lat_distance = True):
        mod_pars, mod_values, start_frame_sim = sim_args
        # modify model parameters:
        for par, value in zip(mod_pars, mod_values):
            self.__setattr__(par, value) 

        df = self.process_data(gt, ego_id = ego_id, fps_gt=fps_gt)
        dt = 1/ fps_gt
        gt_mod = gt.copy()

        agents_on_ego_path = [id for id in list(df[df['on_path']]['agent_id'].unique()) if id != ego_id]

        accels = []
        x_list = []

        df_ego_scene = df[df['agent_id']==99] # keep unchanged for path interpolation. Only df modified for ego
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
        # accels_real = np.gradient(df[df['agent_id']==99]['v'].values, df[df['agent_id']==99]['t'].values)
        # plt.figure()
        # plt.plot(accels_real)
        # plt.plot(accels)

        # GIDM.project_trajectories(df, plot =True)

        return gt_mod
    




