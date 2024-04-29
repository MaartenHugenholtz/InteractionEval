import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import numpy as np
import pandas as pd


split = 'val'
df_interactions = pd.read_csv(f'interaction_mode_metrics_{split}.csv')

df_interactions = df_interactions.dropna()

print(f'{len(df_interactions)} interactions in {split} dataset')

# calculate metrics
# df_interactions['metric t2cor'] = df_interactions.apply(lambda row: str(row['t2cor']) if row['t2cor'] < row['pred_time'] else 'correct', axis=1)
# df_interactions['metric t2cov'] = df_interactions.apply(lambda row: str(row['t2cov']) if row['t2cov'] < row['pred_time'] else 'covered', axis=1)
Hpred = 6
df_interactions['metric t2cor'] = df_interactions.apply(lambda row: row['t2cor'] if row['t2cor'] < row['pred_time'] else Hpred, axis=1)
df_interactions['metric t2cov'] = df_interactions.apply(lambda row: row['t2cov'] if row['t2cov'] < row['pred_time'] else Hpred, axis=1)

df_notcorrect = df_interactions[df_interactions['metric t2cor'] < Hpred]
df_0scor = df_interactions[df_interactions['metric t2cor'] ==0]
print(f"{round(100 * len(df_notcorrect) / len(df_interactions),1)}% of the intentions not correct, with {df_notcorrect['t2cor'].mean()}s as average time-to-correct-mode-prediction")
print(f"{round(100*len(df_0scor)/len(df_interactions), 1)}% of the intentions not correctly predicted before inevitable homotopy collapse (t2cor = 0s)")

df_notcovered = df_interactions[df_interactions['metric t2cov'] < Hpred]
df_0scov = df_interactions[df_interactions['metric t2cov'] ==0]
print(f"{round(100 * len(df_notcovered) / len(df_interactions),1)}% of the intentions not covered, with {df_notcovered['t2cov'].mean()}s as average time-to-covered-mode-prediction")
print(f"{round(100*len(df_0scov)/len(df_interactions), 1)}% of the intentions not covered predicted before inevitable homotopy collapse (t2cov = 0s)")


# get prediciton consistency metric:
r_pred_consistency = len(df_interactions[df_interactions.prediction_consistency == True]) / len(df_interactions)
print(f'Prediction consistency = {round(100*r_pred_consistency,1)}%')

print(f"Average mode-collapse = {round(df_interactions['r_mode_collapse'].mean(),1)}%")

# Plot histogram using Plotly Express
fig = px.histogram(df_interactions, x='r_mode_collapse', nbins=11, title='mode collapse ratio')
fig.show()

# find better definition for relative metrics! 

fig = px.histogram(df_interactions, x='metric t2cor', nbins=10, title='time to correct prediction [s]')
fig.show()

fig = px.histogram(df_interactions, x='metric t2cov', nbins=10, title='time to covered prediction [s]')
fig.show()
