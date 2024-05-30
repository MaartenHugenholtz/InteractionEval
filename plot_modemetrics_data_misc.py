import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.io as pio

def angle_difference(angle1, angle2):
    diff = angle2 - angle1
    while diff <= -np.pi:
        diff += 2 * np.pi
    while diff > np.pi:
        diff -= 2 * np.pi
    return diff

def normalize_frames(group):
    max_frame = group['frame'].max()
    group['frame_till_h_final'] = max_frame - group['frame']
    return group

# Apply the function to each group
df = pd.read_csv('mode_metric_results/interaction_mode_metrics_data_AF_val_Tpred_12f.csv')

df['dv'] = abs(df['v1'] - df['v2'])
df['dh'] = df.apply(lambda row: abs(angle_difference(row['heading1'], row['heading2'])), axis=1)
df['interaction_id'] = df['scene'] + '_'+ df['agent1'].astype(str) + '_'+ df['agent2'].astype(str)


df = df.groupby('interaction_id').apply(normalize_frames)

# make these relative, to see percentage / group and calc average?
px.histogram(df, x = 'dh', color = 'mode_correct').show()
px.histogram(df, x = 'dv', color = 'mode_correct').show()
px.histogram(df, x = 'frame_till_h_final', color = 'mode_correct').show()


# px.scatter(df, x = 'dh', y = 'dv', color = 'mode_correct').show()
print()


percentage_true = df.groupby('frame_till_h_final')['mode_correct'].mean() * 100
percentage_true = percentage_true.reset_index()

# Rename columns for clarity
percentage_true.columns = ['Frames before homotopy collapse', 'Mode correct rate [%]']

# Create the bar plot
# px.bar(percentage_true, x='Frames before homotopy collapse', y='Mode correct rate [%]').show()

# Calculate the percentage of True values for each column
percentage_correct = df.groupby('frame_till_h_final')['mode_correct'].mean() * 100
percentage_covered = df.groupby('frame_till_h_final')['mode_covered'].mean() * 100
percentage_collapse = df.groupby('frame_till_h_final')['mode_collapse'].mean() * 100


# Combine the results into a single DataFrame
percentage_df = pd.DataFrame({
    'Frames before homotopy collapse': percentage_correct.index,
    'Mode correct rate [%]': percentage_correct.values,
    'Mode covered rate [%]': percentage_covered.values,
    'Mode collapse rate [%]': percentage_collapse.values
})

# Melt the DataFrame for Plotly
melted_df = percentage_df.melt(id_vars=['Frames before homotopy collapse'], 
                               value_vars=['Mode correct rate [%]', 'Mode covered rate [%]', 'Mode collapse rate [%]'],
                               var_name='Mode', 
                               value_name='Rate [%]')

# Create the bar plot
fig = px.bar(melted_df, x='Frames before homotopy collapse', y='Rate [%]', color='Mode', 
             barmode='group', 
             labels={'Frames before homotopy collapse': 'Frames before homotopy collapse', 'Rate [%]': 'Rate [%]'},
             title='Rates of Mode Correct, Covered, and Collapse')

# Customize the layout for better readability
fig.update_layout(xaxis_title='Frames before homotopy collapse', yaxis_title='Rate [%]')

# Show the figure
fig.show()

