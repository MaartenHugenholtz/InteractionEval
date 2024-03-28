import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import math

"""" Roll-out parameters """
# AY_MAX =  5  # m/s^2
# AX_MIN = -3 # m/s^2
# AX_MAX =  2 # m/s^2
# VMIN = 0      / 3.6 # m/s
# VMAX = 50     / 3.6 # m/s
# FUT_STEPS = 12
# DT = 0.5


def calc_path_distance(group, v_var = 'v', dt = 0.5):
    s= np.zeros_like(group[v_var])
    ds = group[v_var] * dt
    s[1:] = np.cumsum(ds[:-1])
    return pd.Series(s, index=group.index)

def calc_velocity_vector(group, xvar = 'x', yvar = 'y', tvar = 't'):
    vx = np.gradient(group[xvar], group[tvar])
    vy = np.gradient(group[yvar], group[tvar])
    v = np.sqrt(vx**2 + vy**2)
    return pd.Series(v, index=group.index)

def calc_acceleration(group, v_var = 'v', t_var = 't'):
    ax = np.gradient(group[v_var], group[t_var])
    return pd.Series(ax, index=group.index)

def calc_yaw_rate(group):
    yaw_rate = np.gradient(group['heading'], group['t'])
    return pd.Series(yaw_rate, index=group.index)




def process_data(gt, ego_id = '99', fps_gt=2):
    df = pd.DataFrame(data=gt[:, [0, 1, 13, 15, 10, 11, 12, 16]],
                        columns=['frame', 'agent_id', 'x', 'y', 'width', 'height', 'length', 'heading']).astype(float)
    df['agent_type'] = gt[:, 2]
    df['agent_id'] = df['agent_id'].astype(int).astype(str)# for plotting; categorical
    df['t'] = df['frame'] / fps_gt

    # Apply calculation functions to each group of agent_id
    df['v'] = df.groupby('agent_id').apply(calc_velocity_vector).reset_index(level=0, drop=True)
    df['ax'] = df.groupby('agent_id').apply(calc_acceleration).reset_index(level=0, drop=True)
    df['yaw_rate'] = df.groupby('agent_id').apply(calc_yaw_rate).reset_index(level=0, drop=True)
    df['k'] = df['yaw_rate'] / df['v'] # curvature [1/m]
    df['ay'] = df['v']**2 * df['k']
    df['s'] = df.groupby('agent_id').apply(calc_path_distance).reset_index(level=0, drop=True)

    return df

def get_path_crossing_point(path1, path2, crossing_threshold = 1):
    distances = cdist(np.array(path1).T, np.array(path2).T)
    min_distance = np.min(distances)
    min_indices = np.argwhere(distances == min_distance)
    intersect_bool = min_distance < crossing_threshold 
    idx1, idx2 = min_indices[0,[0,1]]
    return intersect_bool, idx1, idx2

def decelerate_path(df, frame_curr):
    df_fut = df.copy().reset_index()
    frame_end = df_fut.frame.max()

    # interpolation functions:


    for idx, row in df_fut.iterrows():
        if row['frame'] >= frame_curr and row['frame'] < frame_end:
            

            v_curr = v_decel[idx]
            ax_min_stationary = (VMIN - v_curr) / DT
            ax = max(ax_min_stationary, AX_MIN)
            v_next = v_curr + ax*DT
            idx_next = group.iloc[group.index.get_loc(idx) + 1].name
            v_decel[idx_next] = v_next

    return pd.Series(v_decel, index=group.index)

def accelerate_path(group, frame_curr):
    frame_end = group.frame.max()
    v_accel = group['v'] # intialize with gt velocity
    for idx, row in group.iterrows():
        if row['frame'] >= frame_curr and row['frame'] < frame_end:
            v_curr = v_accel[idx]
            vmax_corner = VMAX # fix later: either remove trajectories exceeding ay limit or change whole modification loop?
            vmax = min(VMAX, vmax_corner)
            ax_max_limit = (VMAX - v_curr) / DT
            ax = min(ax_max_limit, AX_MAX)
            v_next = v_curr + ax*DT
            idx_next = group.iloc[group.index.get_loc(idx) + 1].name
            v_accel[idx_next] = v_next

    return pd.Series(v_accel, index=group.index)

def interp_path(group, x_var, xp_var, fp_var):
    # remove stationary point noise
    idx_denoise = group['v'].values >1
    idx_denoise[-1] = True
    idx_denoise[0]  = True

    # only interpoloate non-stationary points
    f_x = interpolate.interp1d(group[xp_var].values[idx_denoise], group[fp_var].values[idx_denoise], fill_value='extrapolate', assume_sorted=True) # NO UTURNS
    f = f_x(group[x_var])
    return pd.Series(f, index=group.index)


