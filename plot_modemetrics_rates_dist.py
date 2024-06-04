import plotly.graph_objects as go
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

def angle_difference(angle1, angle2):
    diff = angle2 - angle1
    while diff <= -np.pi:
        diff += 2 * np.pi
    while diff > np.pi:
        diff -= 2 * np.pi
    return diff

def normalize_frames(group, freq=2):
    max_frame = group['frame'].max()
    group['frame_till_h_final'] = 1 + max_frame - group['frame']
    group['Dt_till_h_final'] = group['frame_till_h_final'] / freq
    return group

def calculate_percentage(df, x_axis_var, bins=None):
    if bins is not None:
        df[f'binned_{x_axis_var}'] = pd.cut(df[x_axis_var], bins=bins)
        x_axis_var = f'binned_{x_axis_var}'



    df['dh'] = df.apply(lambda row: abs(angle_difference(row['heading1'], row['heading2'])), axis=1)

    percentage_correct = df.groupby([x_axis_var, 'model'])['mode_correct'].mean() * 100
    percentage_covered = df.groupby([x_axis_var, 'model'])['mode_covered'].mean() * 100
    percentage_collapse = df.groupby([x_axis_var, 'model'])['mode_collapse'].mean() * 100

    percentage_df_correct = percentage_correct.reset_index().rename(columns={'mode_correct': 'Rate [%]'})
    percentage_df_covered = percentage_covered.reset_index().rename(columns={'mode_covered': 'Rate [%]'})
    percentage_df_collapse = percentage_collapse.reset_index().rename(columns={'mode_collapse': 'Rate [%]'})

    # use midpoints as bins:
    if bins is not None:
        percentage_df_correct[x_axis_var[7:]] = percentage_df_correct.apply(lambda row: row[x_axis_var].mid , axis = 1)
        percentage_df_covered[x_axis_var[7:]] = percentage_df_covered.apply(lambda row: row[x_axis_var].mid , axis = 1)
        percentage_df_collapse[x_axis_var[7:]] = percentage_df_collapse.apply(lambda row: row[x_axis_var].mid , axis = 1)

    return percentage_df_correct, percentage_df_covered, percentage_df_collapse

Hpred_time = 6  # TIME NOT FRAMES!
K_Modes = 5

save_plot = True
save_path = f'mode_metric_results/mode_metrics_data_{Hpred_time}s.png'

Title = f'Model prediction results @ {Hpred_time}s'

# Load data
df_af = pd.read_csv(f'mode_metric_results/interaction_mode_metrics_data_AF_val_Tpred_{2*Hpred_time}f_{K_Modes}samples.csv')
df_oracle = pd.read_csv(f'mode_metric_results/interaction_mode_metrics_data_oracle_val_Tpred_{2*Hpred_time}f_{K_Modes}samples.csv')
df_cv = pd.read_csv(f'mode_metric_results/interaction_mode_metrics_data_cv_val_Tpred_{2*Hpred_time}f_{K_Modes}samples.csv')
if Hpred_time == 3:
    df_ctt = pd.read_csv(f'mode_metric_results/interaction_mode_metrics_data_CTT_val_Tpred_{2*Hpred_time}f_{K_Modes}samples.csv')
    df_ctt['model'] = 'CTT'

# Add model identifier
df_af['model'] = 'AF'
df_oracle['model'] = 'Oracle'
df_cv['model'] = 'CV'

# Combine data
if Hpred_time == 3:
    df = pd.concat([df_af, df_oracle, df_cv, df_ctt])
    models = ['AF', 'Oracle', 'CV', 'CTT']
else:
    df = pd.concat([df_af, df_oracle, df_cv])
    models = ['AF', 'Oracle', 'CV']

# Calculate dv and dh
df['dv'] = abs(df['v1'] - df['v2'])
df['dh'] = df.apply(lambda row: abs(angle_difference(row['heading1'], row['heading2'])), axis=1)
df['interaction_id'] = df['scene'] + '_' + df['agent1'].astype(str) + '_' + df['agent2'].astype(str)

# Normalize frames
df = df.groupby(['interaction_id', 'model']).apply(normalize_frames)

# Select x-axis variable
x_axis_vars = ['Dt_till_h_final', 'dv', 'dh']
x_axis_var = 'Dt_till_h_final'  # Change this to 'Dt_till_h_final', 'dv', or 'dh' as needed

# Define bins for continuous variables
Nbins = 10
if x_axis_var == 'dv':
    bins = np.linspace(df['dv'].min(), df['dv'].max(), Nbins)  # 20 bins for dv
elif x_axis_var == 'dh':
    bins = np.linspace(df['dh'].min(), df['dh'].max(), Nbins)  # 20 bins for dh
else:
    bins = None

# Calculate percentages for each model and mode based on selected x-axis variable
percentage_df_correct, percentage_df_covered, percentage_df_collapse = calculate_percentage(df, x_axis_var, bins)

# Create subplots
fig = make_subplots(rows=1, cols=3, subplot_titles=('Mode Correct', 'Mode Covered', 'Mode Collapse'), 
                    shared_yaxes=False)

colors = px.colors.qualitative.Plotly
mode = 'lines+markers'
# Add traces for each mode
for i, model in enumerate(models):
    color = colors[i]
    fig.add_trace(go.Scatter(x=percentage_df_correct[percentage_df_correct['model'] == model][x_axis_var],
                             y=percentage_df_correct[percentage_df_correct['model'] == model]['Rate [%]'],
                             mode=mode, name=model, line=dict(color=color), legendgroup=model),
                  row=1, col=1)

    fig.add_trace(go.Scatter(x=percentage_df_covered[percentage_df_covered['model'] == model][x_axis_var],
                             y=percentage_df_covered[percentage_df_covered['model'] == model]['Rate [%]'],
                             mode=mode, name=model, line=dict(color=color), legendgroup=model, showlegend=False),
                  row=1, col=2)

    fig.add_trace(go.Scatter(x=percentage_df_collapse[percentage_df_collapse['model'] == model][x_axis_var],
                             y=percentage_df_collapse[percentage_df_collapse['model'] == model]['Rate [%]'],
                             mode=mode, name=model, line=dict(color=color), legendgroup=model, showlegend=False),
                  row=1, col=3)

    mode_correct_rate = df[df['model']==model]['mode_correct'].mean() * 100
    mode_covered_rate = df[df['model']==model]['mode_covered'].mean()* 100
    mode_collapse_rate = df[df['model']==model]['mode_collapse'].mean()* 100
    print(f'Latex table str {model}: & {mode_correct_rate:.1f} & {mode_covered_rate:.1f} & {mode_collapse_rate:.1f}')

# Update layout for better readability
fig.update_layout(showlegend=True, legend_title_text='Model')

# Set y-axis range for all subplots and ensure y-axis titles are visible
for i in range(1, 4):
    fig.update_yaxes(range=[0, 105], title_text=r"$\text{Rate } (\%)$" if i==1 else '', row=1, col=i)

# Apply x-axis title to all subplots
x_axis_titles = {
    'Dt_till_h_final': r"$\Delta t_{\text{h,collapse}} \, \text{(s)}$",
    'dv': r"$\Delta v \, \text{(m/s)}$",
    'dh': r"$\Delta h \, \text{(rad)}$"
}
for i in range(1, 4):
    fig.update_xaxes(title_text=x_axis_titles[x_axis_var], title_font={"size": 20}, row=1, col=i)

fig.update_layout(margin=dict(l=5, r=5, t=50, b=5))

# Show the figure
# fig.update_layout(title_text=Title)
fig.show()


pio.write_image(fig, save_path, width=0.8*1*1700/1.1, height=0.8*0.8*800/1.2)


fig = px.histogram(df[df['model']=='AF'], x = 'Dt_till_h_final')
fig.update_layout(
    bargap=0.2,  # Adjust the bargap value as needed
    xaxis=dict(
        tickvals = np.arange(0.5, 6.5, 0.5),
        title_text =  r"$\Delta t_{\text{h,collapse}} \, \text{(s)}$",
        title_font={"size": 40}),
    yaxis = dict(title_text = r"$\text{Count}$",
    title_font={"size": 40})
)
fig.update_layout(margin=dict(l=5, r=5, t=5, b=5))
fig.update_layout(width=500, height=400)

fig.show()

pio.write_image(fig, f'mode_metric_results/histogram_dt_h_collapse_{Hpred_time}s.png')

