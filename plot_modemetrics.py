import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import numpy as np
import pandas as pd


split = 'val'
df_interactions = pd.read_csv(f'interaction_mode_metrics_{split}.csv')

df_interactions = df_interactions.dropna()

# calculate metrics
df_interactions['r_t2cor'] = df_interactions['t2cor'] / df_interactions['pred_time']
df_interactions['r_t2cov'] = df_interactions['t2cov'] / df_interactions['pred_time']

# get prediciton consistency metric:
r_pred_consistency = len(df_interactions[df_interactions.prediction_consistency == True]) / len(df_interactions)
print(f'Prediction consistency ratio = {r_pred_consistency}')

# Plot histogram using Plotly Express
fig = px.histogram(df_interactions, x='r_mode_collapse', nbins=10, title='mode collapse ratio')
fig.show()

# find better definition for relative metrics! 

fig = px.histogram(df_interactions, x='r_t2cor', nbins=10, title='relative time to correct prediction [%]')
fig.show()

fig = px.histogram(df_interactions, x='r_t2cov', nbins=10, title='relative time to covered prediction [%]')
fig.show()

fig = px.histogram(df_interactions, x='t2cor', nbins=10, title='time to correct prediction [s]')
fig.show()

fig = px.histogram(df_interactions, x='t2cov', nbins=10, title='time to covered prediction [s]')
fig.show()
