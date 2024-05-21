import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.io as pio

Hpred_time = 6 # TIME NOT FRAMES!

save_plot = True
save_path = f'mode_metric_results/mode_metrics_{Hpred_time}s.png'

Title = f'Model prediction results @ {Hpred_time}s'


if Hpred_time == 6: # time not frames, so 6s == 12f
    models = [
           'AgentFormer',
           'Oracle',
           'CV model',
           ]
    models_result_paths = ['mode_metric_results/interaction_mode_metrics_AF_val_Tpred_12f.csv',
                        'mode_metric_results/interaction_mode_metrics_oracle_val_Tpred_12f.csv',
                        'mode_metric_results/interaction_mode_metrics_cv_val_Tpred_12f.csv',
                        ]
elif Hpred_time ==3: # time not frames, so 3s == 6f
    models = [
           'AgentFormer',
           'Oracle',
           'CV model',
           'CTT'
           ]
    models_result_paths = ['mode_metric_results/interaction_mode_metrics_AF_val_Tpred_6f.csv',
                        'mode_metric_results/interaction_mode_metrics_oracle_val_Tpred_6f.csv',
                        'mode_metric_results/interaction_mode_metrics_cv_val_Tpred_6f.csv',
                        'mode_metric_results/interaction_mode_metrics_CTT_val_Tpred_6f.csv',
                        ]
else:
    raise NameError

# combine data into one df:
dfs = []
dfs_stats = []
consistencies = []

for model, path in zip(models, models_result_paths):
    df_temp = pd.read_csv(path).dropna()
    df_temp['model'] = model
    df_temp['metric t2cor'] = df_temp.apply(lambda row: row['t2cor'] if row['t2cor'] < row['pred_time'] else Hpred_time, axis=1)
    df_temp['metric t2cov'] = df_temp.apply(lambda row: row['t2cov'] if row['t2cov'] < row['pred_time'] else Hpred_time, axis=1)
    dfs.append(df_temp)

    # calc stats
    df_notcorrect = df_temp[df_temp['metric t2cor'] < Hpred_time]
    df_0scor = df_temp[df_temp['metric t2cor'] ==0]
    df_notcovered = df_temp[df_temp['metric t2cov'] < Hpred_time]
    df_0scov = df_temp[df_temp['metric t2cov'] ==0]

    N_interactions = len(df_temp)
    perc_not_correct = round(100 * len(df_notcorrect) / len(df_temp),1)
    perc_correct = round(100 - perc_not_correct, 1)
    avg_t2cor = df_notcorrect['t2cor'].mean()
    perc_t2cor_0s = round(100*len(df_0scor)/len(df_temp), 1)
    perc_not_covered = round(100 * len(df_notcovered) / len(df_temp),1)
    perc_covered = round(100 - perc_not_covered, 1)
    avg_t2cov = df_notcovered['t2cov'].mean()
    perc_t2cov_0s = round(100*len(df_0scov)/len(df_temp), 1)

    r_pred_consistency = len(df_temp[df_temp.prediction_consistency == True]) / len(df_temp)
    perc_pred_consistency = round(100*r_pred_consistency,1)
    perc_mode_collapse = round(df_temp['r_mode_collapse'].mean(),1)
    consistencies.append(r_pred_consistency)

    # print stats
    print(f'MODEL: {model}')
    print(f'{N_interactions} interactions in dataset')
    print(f"{perc_not_correct}% of the intentions not correct, with {avg_t2cor}s as average time-to-correct-mode-prediction")
    print(f"{perc_t2cor_0s}% of the intentions not correctly predicted before inevitable homotopy collapse (t2cor = 0s)")

    print(f'Prediction consistency = {perc_pred_consistency}%')
    print(f"Average mode-collapse = {perc_mode_collapse}%")
    print(f'Latex table str: {model} & {avg_t2cor:.1f} & {perc_t2cor_0s:.1f} & {perc_correct:.1f} & {avg_t2cov:.1f} & {perc_t2cov_0s:.1f} & {perc_covered:.1f} & {perc_mode_collapse:.1f} & {perc_pred_consistency:.1f}  \\')
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



fig = make_subplots(rows=1, cols=3)
fig.update_layout(title_text = Title)
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


fig.update_xaxes(title_text='Time to Correct Prediction [s]', row=1, col=1)
fig.update_xaxes(title_text='Time to Covered Prediction [s]', row=1, col=2)
fig.update_xaxes(title_text='Mode Collapse Ratio [%]', row=1, col=3)
fig.update_yaxes(title_text='Percentage [%]', range=[0, 105], row=1, col=1)
fig.update_yaxes(range=[0, 105], row=1, col=2)
fig.update_yaxes(range=[0, 105], row=1, col=3)

fig.show()

if save_plot:
    pio.write_image(fig, save_path, width=0.8*1*1700/1.1, height=0.8*0.8*800/1.2)