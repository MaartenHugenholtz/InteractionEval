import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from scipy.interpolate import interp1d
from itertools import product
import random

# np.random.seed(42)

T_PRED = 5
K_PRED = 6
START_POINTS = [(1, -8), (-1, 8), (8, 1), (-8, -1)]
END_POINTS = [(-1, -8), (1, 8), (-8, 1), (8, -1)]
SPEED_CHOICES = [3, 6]

def calc_distance(x, y):
    # return array of distances travelled between the points:
    s = np.zeros_like(x)
    s[1:] = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    return s

def generate_trajectory(start, end, speed = 8, dt = 0.1):
    x0, y0 = start
    x1, y1 = end
    
    if x0 == x1: # going straight vertical:
        x_mid = x0
        y_mid = (y0 + y1)/2
    elif y0 == y1: # going straight horizontal
        y_mid = y0
        x_mid = (x0 + x1)/2
    else:
        x_points = np.array([x0, x1])
        y_points = np.array([y0, y1])
        xmid_point_idx = np.argmin(abs(x_points))
        ymid_point_idx = np.argmin(abs(y_points))
        x_mid = x_points[xmid_point_idx]
        y_mid = y_points[ymid_point_idx]
    
    x_path = np.array([x0, x_mid, x1])
    y_path = np.array([y0, y_mid, y1])
    ds_path = calc_distance(x_path, y_path)
    dt_path = ds_path / speed
    t_path = np.cumsum(dt_path)
    
    # interpolate to real time to get trajectories and extrapolate
    t_traj = np.arange(0, T_PRED + dt, dt)
    fx_path = interp1d(t_path, x_path, fill_value = 'extrapolate')
    fy_path = interp1d(t_path, y_path, fill_value = 'extrapolate')
    x_traj = fx_path(t_traj)
    y_traj = fy_path(t_traj)
    return x_traj, y_traj, t_traj

def trajectory_trace(start, end, speed, car_id=1, prob = 1):
    x_traj, y_traj, t_traj = generate_trajectory(start, end, speed)
    
    

    hover_text = [f'Time: {t:.2f}' for t in t_traj]  # Format time values in hover text

    trace = go.Scatter(
        x=x_traj,
        y=y_traj,
        mode='markers+lines',
        name='car_' + str(car_id),
        opacity= prob,
        marker=dict(
            size=10,
            color=t_traj,
            symbol=car_id,
            colorscale='Viridis',  # You can choose a different colorscale
            cmin = 0,
            cmax = T_PRED
        ),
        line=dict(color='rgba(255,255,255,0.5)'),
        text=hover_text,  # Set the hover text
        hoverinfo='text'  # Show only text in hover info
    )
    return trace, (x_traj, y_traj)

def trajectory_traces_combined(start_list, end_list, speed_list, car_ids, name, prob = 1):
    x_traj_list = []
    y_traj_list = []
    t_traj_list = []
    ids_list = []
    for start, end, speed, id in zip(start_list, end_list, speed_list, car_ids):   
        x_traj, y_traj, t_traj = generate_trajectory(start, end, speed)
        x_traj_list.extend(x_traj)
        y_traj_list.extend(y_traj)
        t_traj_list.extend(t_traj)
        ids_list.extend([id]*len(x_traj))

    hover_text = [f'Time: {t:.2f}' for t in t_traj_list]  # Format time values in hover text

    trace = go.Scatter(
        x=x_traj_list,
        y=y_traj_list,
        mode='markers',
        name=name,
        opacity= prob,
        marker=dict(
            size=10,
            color=t_traj_list,
            symbol=ids_list,
            colorscale='Viridis',  # You can choose a different colorscale
            cmin = 0,
            cmax = T_PRED
        ),
        line=dict(color='rgba(255,255,255,0.5)'),
        text=hover_text,  # Set the hover text
        hoverinfo='text'  # Show only text in hover info
    )
    return trace

def predict_trajectories(start_list, id_list, predict_all = True):
    # make list with all combinations; assume just two cars for now
    endpoints_choices_1 = [point for point in END_POINTS if \
            np.sqrt((start_list[0][0]-point[0])**2 + (start_list[0][1]-point[1])**2) > 2]   # remove start point from possible endpoint choices
    endpoints_choices_2 = [point for point in END_POINTS if \
            np.sqrt((start_list[1][0]-point[0])**2 + (start_list[1][1]-point[1])**2) > 2]   # remove start point from possible endpoint choices
    speed_choices_1 = SPEED_CHOICES
    speed_choices_2 = SPEED_CHOICES
    
    joint_prediction_combinations = list(product(endpoints_choices_1, endpoints_choices_2, speed_choices_1, speed_choices_2))
    # sample from list to generate predictions
    sample_idx_predictions = random.sample(range(0,len(joint_prediction_combinations)), K_PRED)
    
    if predict_all:
        sample_idx_predictions = np.arange(0,len(joint_prediction_combinations))
    
    traces = []
    for i, sample_idx in enumerate(sample_idx_predictions):
        prediction = joint_prediction_combinations[sample_idx]
        # get trajectories and homotopy class based on prediction
        end1, end2, speed1, speed2 = prediction
        start1, start2 = start_list
        x_traj, y_traj, t_traj = generate_trajectory(start1, end1, speed1)
        traj1 = (x_traj, y_traj)
        x_traj, y_traj, t_traj = generate_trajectory(start2, end2, speed2)
        traj2 = (x_traj, y_traj)
        homotopy, angle_diff = get_homotopy(traj1, traj2)
        name = 'pred_' + str(i) + ': ' + str(end1) +', '+ str(end2) +', ' + homotopy
        trace = trajectory_traces_combined(start_list, prediction[0:2], prediction[2:], 
                                           id_list, name, prob=0.5)
        traces.append(trace)
        
    return traces
         
def generate_intersection():
    # Coordinates for solid rectangles
    solid1_x = [-2, -10, -10, -2, -2]
    solid1_y = [-2, -2, -10, -10, -2]

    solid2_x = [2, 10, 10, 2, 2]
    solid2_y = [2, 2, 10, 10, 2]

    solid3_x = [-2, -10, -10, -2, -2]
    solid3_y = [2, 2, 10, 10, 2]

    solid4_x = [2, 10, 10, 2, 2]
    solid4_y = [-2, -2, -10, -10, -2]

    # Create traces for the solid rectangles
    trace_solid1 = go.Scatter(
        x=solid1_x,
        y=solid1_y,
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        mode='none'
    )

    trace_solid2 = go.Scatter(
        x=solid2_x,
        y=solid2_y,
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        mode='none'
    )

    trace_solid3 = go.Scatter(
        x=solid3_x,
        y=solid3_y,
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        mode='none'
    )

    trace_solid4 = go.Scatter(
        x=solid4_x,
        y=solid4_y,
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        mode='none'
    )
    traces = [trace_solid1,trace_solid2,trace_solid3,trace_solid4] 
    return traces

def get_homotopy(traj1, traj2, homotopy_threshold = np.pi / 6, plot = False):
    # apply free-end homotopy, assume traj1 to be ego  
    x1, y1 = traj1
    x2, y2 = traj2
    
    a1=np.arctan2(y1[1:]-y2[1:],x1[1:]-x2[1:])
    a0=np.arctan2(y1[:-1]-y2[:-1],x1[:-1]-x2[:-1])
    diff = np.arctan2(np.sin(a1-a0), np.cos(a1-a0)) # calculate angle difference like this to avoid jumps because of [-pi,pi]-interval
    angle_diff = np.sum(diff)
    
    if plot:
        dt = T_PRED / (len(x1)-1)
        t = np.arange(0, T_PRED, dt)
        fig = px.line(x = t, y = diff)
        fig.show()
    
    if angle_diff < -homotopy_threshold:
        return 'CW', angle_diff
    elif angle_diff > homotopy_threshold:
        return 'CCW', angle_diff
    else:
        return 'S', angle_diff

# Layout settings with equal axis scale and range limit
layout = go.Layout(
    title='4-way intersection',
    xaxis=dict(range=[-8, 8], scaleanchor='y', scaleratio=1),
    yaxis=dict(range=[-8, 8]),
    yaxis_range=[-8,8],
    xaxis_range=[-8,8],
    width=700,  # Set the width of the figure (in pixels)
    height=700  # Set the height of the figure (in pixels)
)


traces_intersection = generate_intersection()

# start_car1 = (1, -4)
# end_car1 = (-8, 1)
# speed1= 3
# start_car2 = (-1, 4)
# end_car2 = (8, -1)
# speed2 = 3

start_car1 = (1, -8)
end_car1 = (-8, 1)
speed1= 5
start_car2 = (-1, 8)
end_car2 = (8, -1)
speed2 = 5

GT_car1, traj1 = trajectory_trace(start_car1, end_car1, speed = speed1, car_id= 1)
GT_car2, traj2 = trajectory_trace(start_car2, end_car2, speed= speed2, car_id= 2)
GT_cars = [GT_car1, GT_car2]

# calc homotopy class of GT:
h, angle = get_homotopy(traj2, traj1)
print(f"Homotopy class {h}, with angle diff {round(angle, 3)}")

# make predictions:
predictions_cars = predict_trajectories([start_car1, start_car2], [1, 2])

# Create the figure
fig = go.Figure(data=traces_intersection + GT_cars, layout=layout)

# Show the plot
fig.show()

fig = go.Figure(data=traces_intersection  + predictions_cars, layout=layout)

# Show the plot
fig.show()
