import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.io as pio
df_train = pd.read_csv('interaction_scenes/interaction_metrics_train_all.csv')
df_val = pd.read_csv('interaction_scenes/interaction_metrics_val_all.csv')
df  = pd.concat([df_train, df_val])

df['start_path_sharing_time_difference'] = df['start_path_sharing_frame_difference']  / 2 # 2 Hz

df_path_sharing = df[df['path_sharing_bool']]  # path sharing at the end
df_interaction = df[df['interaction_bool']]

N_scenes = len(df['scene'].unique())
df_agents = df[['num_agents_scene', 'scene']].drop_duplicates()
assert len(df_agents) == N_scenes
mean_N_agents = df_agents.num_agents_scene.mean()
total_N_agents = df_agents.num_agents_scene.sum()
df_agents['num_interactions_theoretical'] = df_agents['num_agents_scene']*(df_agents['num_agents_scene'] - 1)/2  #  theroetical number of interactions: N · (N − 1)/2
N_interactions_theoretical_total = sum(df_agents['num_interactions_theoretical'])
N_interactions_real_total = len(df) # this df contains all agent-pairs of all scenes. Only added if there are enough common frames in the data
N_interactions_pathsharing_total = len(df_path_sharing)
N_interactions_critical_total = len(df_interaction)
N_interactions_final_total = None # from results; missing interactions due to pred length...

print('N_scenes: ', N_scenes)
print('mean_N_agents: ', mean_N_agents)
print('total_N_agents', total_N_agents)
print('N_interactions_theoretical_total: ', N_interactions_theoretical_total)
print('N_interactions_real_total', N_interactions_real_total)
print('N_interactions_pathsharing_total', N_interactions_pathsharing_total)
print('N_interactions_critical_total', N_interactions_critical_total)
print('N_interactions_final_total', N_interactions_final_total)

# px.scatter(df_path_sharing, x = 'start_path_sharing_frame_difference', y = 'real_time_closest_distance').show()
fig = px.density_heatmap(df_path_sharing, 
                   x = 'start_path_sharing_time_difference', y = 'real_time_closest_distance',
                   color_continuous_scale='deep')
# px.histogram(df_path_sharing, x = 'start_path_sharing_frame_difference').show()


fig.update_layout(margin=dict(l=5, r=5, t=5, b=5))
fig.update_layout(width=500, height=400)
fig.update_xaxes(title_text=r"$\Delta t_{\text{path-sharing}} \, \text{(s)}$",title_font={"size": 40})
# fig.update_yaxes(title_text=r"$\min D$",title_font={"size": 30}) 
fig.update_yaxes(title_text=r"$d_{\text{min}} \, \text{(m)}$",title_font={"size": 40})


pio.write_image(fig, 'interaction_scenes/path_sharing_density.png')

fig.show()