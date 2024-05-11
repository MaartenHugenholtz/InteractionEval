import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots

Hpred = 6

models = ['AgentFormer',
           'Oracle',
           'CV model'
           ]
models_result_paths = ['interaction_mode_metrics_val.csv',
                        'interaction_mode_metrics_oracle_val.csv',
                        'interaction_mode_metrics_cv_val.csv'
                        ]

# combine data into one df:
dfs = []
dfs_stats = []
consistencies = []

for model, path in zip(models, models_result_paths):
    df_temp = pd.read_csv(path).dropna()
    df_temp['model'] = model
    df_temp['metric t2cor'] = df_temp.apply(lambda row: row['t2cor'] if row['t2cor'] < row['pred_time'] else Hpred, axis=1)
    df_temp['metric t2cov'] = df_temp.apply(lambda row: row['t2cov'] if row['t2cov'] < row['pred_time'] else Hpred, axis=1)
    dfs.append(df_temp)

    # print stats
    df_notcorrect = df_temp[df_temp['metric t2cor'] < Hpred]
    df_0scor = df_temp[df_temp['metric t2cor'] ==0]
    df_notcovered = df_temp[df_temp['metric t2cov'] < Hpred]
    df_0scov = df_temp[df_temp['metric t2cov'] ==0]

    print(f'MODEL: {model}')
    print(f'{len(df_temp)} interactions in dataset')
    print(f"{round(100 * len(df_notcorrect) / len(df_temp),1)}% of the intentions not correct, with {df_notcorrect['t2cor'].mean()}s as average time-to-correct-mode-prediction")
    print(f"{round(100*len(df_0scor)/len(df_temp), 1)}% of the intentions not correctly predicted before inevitable homotopy collapse (t2cor = 0s)")
    r_pred_consistency = len(df_temp[df_temp.prediction_consistency == True]) / len(df_temp)
    consistencies.append(r_pred_consistency)
    print(f'Prediction consistency = {round(100*r_pred_consistency,1)}%')
    print(f"Average mode-collapse = {round(df_temp['r_mode_collapse'].mean(),1)}%")
    print()

    df_stats_model = pd.DataFrame({
        'model': [model],
        'ML prediction consistency': [round(100*r_pred_consistency,1)],
        'mode-collapse ratio': [round(df_temp['r_mode_collapse'].mean(),1)],
        'incorrect a2a predictions':[ round(100 * len(df_notcorrect) / len(df_temp),1)],
        'mean time correct a2a mode': [df_notcorrect['t2cor'].mean()],
        'percentage predictions @ t2cor = 0s': round(100*len(df_0scor)/len(df_temp), 1),
        'uncovered a2a predictions': [ round(100 * len(df_notcovered) / len(df_temp),1)],
        'mean time covered a2a mode': [df_notcovered['t2cov'].mean()],
        'percentage predictions @ t2cov = 0s': [round(100*len(df_0scov)/len(df_temp), 1)],
    })
    dfs_stats.append(df_stats_model)

df_combined = pd.concat(dfs)
df_stats = pd.concat(dfs_stats)



# fig = px.histogram(df_combined, x='r_mode_collapse', color = 'model', barmode= 'group',nbins=11, title='mode collapse ratio')
# fig.show()

# fig = px.histogram(df_combined, x='metric t2cor',color = 'model', barmode= 'group',nbins=10, title='time to correct prediction [s]')
# fig.show()

# fig = px.histogram(df_combined, x='metric t2cov',color = 'model', barmode= 'group',nbins=10, title='time to covered prediction [s]')
# fig.show()

# # Create subplots
# fig = make_subplots(rows=1, cols=4, subplot_titles=('Time to Correct Prediction [s]', 'Time to Covered Prediction [s]', 'Mode Collapse Ratio', 'Prediction Consistency'))

# # Add histograms to subplots
# for i, model in enumerate(models):
#     fig.add_trace(go.Histogram(x = df_combined[df_combined['model']==model]['metric t2cor'].values,legendgroup = model, legendgrouptitle=model, bingroup = model), row=1, col=1)
#     fig.add_trace(go.Histogram(x = df_combined[df_combined['model']==model]['metric t2cov'].values,legendgroup = model, legendgrouptitle=model, bingroup = model), row=1, col=2)
#     fig.add_trace(go.Histogram(x = df_combined[df_combined['model']==model]['r_mode_collapse'].values,legendgroup = model, legendgrouptitle=model, bingroup = model), row=1, col=3)
#     # fig.add_trace(go.Histogram(x = df_combined[df_combined['model']==model]['metric t2cor'].values,legendgroup = model, legendgrouptitle=model, bingroup = model), row=1, col=4)


# fig.show()

fig = make_subplots(rows=1, cols=3)
colors = px.colors.qualitative.Plotly

# Iterate through unique model values and create histogram for each
for i, model in enumerate(df_combined['model'].unique(), start=1):
    # Filter data for current model
    data_model = df_combined[df_combined['model'] == model]
    
    # Add histograms to subplots
    fig.add_trace(
        go.Histogram(x=data_model['metric t2cor'], nbinsx=10, histnorm='percent',
                     name=model, marker_color=colors[i-1], legendgroup=model, showlegend=False),
        row=1, col=1
    )
    fig.add_trace(
        go.Histogram(x=data_model['metric t2cov'], nbinsx=10, histnorm='percent',
                     name=model, marker_color=colors[i-1], legendgroup=model, showlegend=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Histogram(x=data_model['r_mode_collapse'], nbinsx=11, histnorm='percent',
                     name=model, marker_color=colors[i-1], legendgroup=model),
        row=1, col=3
    )


fig.update_xaxes(title_text='Time to Covered Prediction [s]', row=1, col=1)
fig.update_xaxes(title_text='Time to Correct Prediction [s]', row=1, col=2)
fig.update_xaxes(title_text='Mode Collapse Ratio [%]', row=1, col=3)
fig.update_yaxes(title_text='Percentage [%]', row=1, col=1)

fig.show()