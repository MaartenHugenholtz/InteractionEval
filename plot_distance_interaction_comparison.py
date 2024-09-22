import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.io as pio

# metric1 = 'Mode Covered Rate [%]'
# metric2 = 'minJoint ADE (m)'

# metric1 = 'Mode Correct Rate [%]'
# metric2 = 'ML ADE (m)'

metric1 = 'dT covered @ Tpred [%]'
metric2 = 'minJoint ADE (m)'

# metric1 = 'dT correct @ Tpred [%]'
# metric2 = 'ML ADE (m)'

plot_title = 'ComparisonModels_' + metric1 + metric2 + '.png'


# Create the DataFrame
df = pd.DataFrame({
    'Model': ['AF', 'Oracle', 'CV'],
    'Tpred': [6, 6, 6],
    'Mode Correct Rate [%]': [74, 86, 80.6],
    'Mode Covered Rate [%]': [89.3, 100, 80.6],
    'dT correct (s)': [1.9, 9, 2.4],
    'dT covered (s)': [1.8, None, 2.3],
    'dT correct @ Tpred [%]': [56.1, 73.2, 78],
    'dT covered @ Tpred [%]': [80.5, 100, 78],
    'ML ADE (m)': [3.88, 3.84, 3.64],
    'minJoint ADE (m)': [2.86, 3.56, 3.64],
    'ML FDE (m)': [9.10, 9.12, 9.04],
    'minJoint FDE (m)': [6.48, 8.41, 9.04],
})




colors = {
    'AF': px.colors.qualitative.Plotly[0],
    'Oracle': px.colors.qualitative.Plotly[1],
    'CV': px.colors.qualitative.Plotly[2]
}

# Create subplots
fig = make_subplots(rows=1, cols=2, subplot_titles=(metric1, metric2))

# Bar chart for "Mode Covered Rate [%]"
for model in df['Model']:
    fig.add_trace(go.Bar(
        x=[model],
        y=df[df['Model'] == model][metric1],
        name=model,
        marker_color=colors[model]
    ), row=1, col=1)

# Bar chart for "minJoint ADE (m)"
for model in df['Model']:
    fig.add_trace(go.Bar(
        x=[model],
        y=df[df['Model'] == model][metric2],
        name=model,
        marker_color=colors[model]
    ), row=1, col=2)

# Update layout
fig.update_layout(
    # title='Model Comparison on Mode Covered Rate and minJoint ADE',
    showlegend=False
)

fig.update_layout(margin=dict(l=5, r=5, t=50, b=5))
fig.update_yaxes(range=[0, 105], row=1, col=1)
fig.update_yaxes(range=[0, 4], row=1, col=2)
# Show plot
# fig.show()

pio.write_image(fig, 'mode_metric_results/' + plot_title,
                width=1600/1.5, height=800/1.7)